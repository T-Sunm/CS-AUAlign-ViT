"""Export manifest-keyed 7x7 AU heatmap masks for offline training use."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_aualign_vit.data.au_guidance import AU_IDS, heatmaps_to_masks
from cs_aualign_vit.data.manifests import read_jsonl, validate_rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--peak-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_existing(output):
    if not output.exists():
        return [], np.empty((0, 49, 12), np.float32), np.empty((0, 12), np.float32), {}
    with (output / "index.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if any(int(record["row"]) != index for index, record in enumerate(records)):
        raise ValueError("Existing heatmap cache index rows are not contiguous")
    masks = np.load(output / "spatial_masks.npy")
    peaks = np.load(output / "peaks.npy")
    metadata = json.loads((output / "metadata.json").read_text())
    if masks.shape != (len(records), 49, 12) or peaks.shape != (len(records), 12):
        raise ValueError("Existing heatmap cache has incompatible arrays")
    return records, masks.astype(np.float32), peaks.astype(np.float32), metadata


def publish(output, records, masks, peaks, metadata):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backup = output.with_name(output.name + ".previous")
    try:
        np.save(temporary / "spatial_masks.npy", np.asarray(masks, dtype=np.float32))
        np.save(temporary / "peaks.npy", np.asarray(peaks, dtype=np.float32))
        with (temporary / "index.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "row", "status"])
            writer.writeheader()
            writer.writerows(records)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            os.replace(output, backup)
        os.replace(temporary, output)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(args):
    if args.peak_threshold < 0:
        raise ValueError("peak threshold must be non-negative")
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    tool = PROJECT_ROOT / "tools" / "au_heatmap"
    sys.path.insert(0, str(tool))
    from au_heatmap import AUHeatmapDetector

    records, masks, peaks, old_metadata = load_existing(output)
    if old_metadata:
        if old_metadata.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError("Existing heatmap cache uses a different checkpoint")
        if old_metadata.get("peak_threshold") != args.peak_threshold:
            raise ValueError("Existing heatmap cache uses a different peak threshold")
        if old_metadata.get("split", args.split) != args.split:
            raise ValueError("Existing heatmap cache uses a different split")
    positions = {record["sample_id"]: int(record["row"]) for record in records}
    detector = AUHeatmapDetector(checkpoint, device=args.device)
    rows = [
        row
        for row in validate_rows(read_jsonl(manifest))
        if row["usable"] and row["split"] == args.split
    ]
    for row in rows:
        sample_id = row["sample_id"]
        position = positions.get(sample_id)
        if position is not None and records[position]["status"] == "success":
            continue
        mask = np.zeros((49, 12), dtype=np.float32)
        peak = np.zeros(12, dtype=np.float32)
        status = "failed"
        try:
            image_path = (manifest.parent / row["processed_path"]).resolve()
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            heatmaps, _ = detector([rgb])
            converted, raw_peaks, valid = heatmaps_to_masks(
                heatmaps[0], peak_threshold=args.peak_threshold
            )
            if valid:
                mask = converted.numpy()
                peak = raw_peaks.numpy()
                status = "success"
        except (OSError, RuntimeError, ValueError):
            pass
        if position is None:
            position = len(records)
            positions[sample_id] = position
            records.append({"sample_id": sample_id, "row": str(position), "status": status})
            masks = np.concatenate((masks, mask[None]), axis=0)
            peaks = np.concatenate((peaks, peak[None]), axis=0)
        else:
            records[position]["status"] = status
            masks[position] = mask
            peaks[position] = peak

    manifest_ids = set(old_metadata.get("manifest_sha256s", ()))
    if old_metadata.get("manifest_sha256"):
        manifest_ids.add(old_metadata["manifest_sha256"])
    manifest_ids.add(sha256(manifest))
    metadata = {
        "au_order": list(AU_IDS),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256s": sorted(manifest_ids),
        "manifest_path": str(manifest),
        "split": args.split,
        "peak_threshold": args.peak_threshold,
        "resize": {"size": [256, 256], "interpolation": "bilinear"},
        "input_normalization": "per-image min-max [0,1]",
        "pooling": "adaptive-average 7x7",
        "mask_normalization": "per-channel max",
        "epsilon": 1.0e-8,
        "dtype": "float32",
        "shape": list(masks.shape),
        "record_count": len(records),
        "success_count": sum(record["status"] == "success" for record in records),
        "failure_count": sum(record["status"] == "failed" for record in records),
    }
    publish(output, records, masks, peaks, metadata)


if __name__ == "__main__":
    run(parse_args())
