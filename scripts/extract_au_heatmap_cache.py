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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    return parser.parse_args()


def load_existing(output):
    backup = output.with_name(output.name + ".previous")
    if not output.exists() and backup.exists():
        os.replace(backup, output)
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
    batch_size = getattr(args, "batch_size", 16)
    checkpoint_every = getattr(args, "checkpoint_every", 5000)
    if args.peak_threshold < 0 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("threshold must be non-negative; batch sizes must be positive")
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    tool = PROJECT_ROOT / "tools" / "au_heatmap"
    sys.path.insert(0, str(tool))
    from au_heatmap import AUHeatmapDetector

    records, old_masks, old_peaks, old_metadata = load_existing(output)
    checkpoint_sha256 = sha256(checkpoint)
    if old_metadata:
        if old_metadata.get("checkpoint_sha256") != checkpoint_sha256:
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
    manifest_ids = set(old_metadata.get("manifest_sha256s", ()))
    if old_metadata.get("manifest_sha256"):
        manifest_ids.add(old_metadata["manifest_sha256"])
    manifest_ids.add(sha256(manifest))
    work = [
        row
        for row in rows
        if row["sample_id"] not in positions
        or records[positions[row["sample_id"]]]["status"] == "failed"
    ]
    capacity = len(records) + sum(row["sample_id"] not in positions for row in work)
    masks = np.zeros((capacity, 49, 12), dtype=np.float32)
    peaks = np.zeros((capacity, 12), dtype=np.float32)
    masks[: len(old_masks)] = old_masks
    peaks[: len(old_peaks)] = old_peaks

    def metadata():
        return {
            "au_order": list(AU_IDS),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
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
            "shape": [len(records), 49, 12],
            "record_count": len(records),
            "success_count": sum(record["status"] == "success" for record in records),
            "failure_count": sum(record["status"] == "failed" for record in records),
            "batch_size": batch_size,
        }

    processed = 0
    next_checkpoint = checkpoint_every
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        images = []
        loadable = []
        results = {}
        for row in batch:
            try:
                image_path = (manifest.parent / row["processed_path"]).resolve()
                with Image.open(image_path) as image:
                    images.append(np.array(image.convert("RGB"), copy=True))
                loadable.append(row)
            except (OSError, ValueError):
                results[row["sample_id"]] = None
        if images:
            heatmaps, _ = detector(images)
            if len(heatmaps) != len(loadable):
                raise RuntimeError("Heatmap model returned the wrong batch size")
            for row, heatmap in zip(loadable, heatmaps, strict=True):
                converted, raw_peaks, valid = heatmaps_to_masks(
                    heatmap, peak_threshold=args.peak_threshold
                )
                results[row["sample_id"]] = (
                    converted.numpy(),
                    raw_peaks.numpy(),
                ) if valid else None
        for row in batch:
            sample_id = row["sample_id"]
            position = positions.get(sample_id)
            if position is None:
                position = len(records)
                positions[sample_id] = position
                records.append({"sample_id": sample_id, "row": str(position), "status": "failed"})
            result = results[sample_id]
            records[position]["status"] = "success" if result is not None else "failed"
            masks[position].fill(0)
            peaks[position].fill(0)
            if result is not None:
                masks[position], peaks[position] = result
        processed += len(batch)
        if processed >= next_checkpoint:
            publish(output, records, masks[: len(records)], peaks[: len(records)], metadata())
            print(f"checkpointed {processed}/{len(work)} pending records", flush=True)
            next_checkpoint += checkpoint_every

    publish(output, records, masks[: len(records)], peaks[: len(records)], metadata())
    print(
        f"complete records={len(records)} success={metadata()['success_count']} "
        f"failed={metadata()['failure_count']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
