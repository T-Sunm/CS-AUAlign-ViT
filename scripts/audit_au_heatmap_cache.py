"""Audit learned AU spatial caches on train manifests without using labels."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs_aualign_vit.data.au_guidance import AU_COLUMNS, AU_IDS, build_au_mask
from cs_aualign_vit.data.manifests import read_jsonl, validate_rows, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for domain in ("source", "target"):
        parser.add_argument(f"--{domain}-manifest", type=Path, required=True)
        parser.add_argument(f"--{domain}-au-cache", type=Path, required=True)
        parser.add_argument(f"--{domain}-heatmap-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantiles(values, points):
    values = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(values, point)) if values.size else None
        for name, point in points.items()
    }


def read_au_cache(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "status", *AU_COLUMNS}
        if required - set(reader.fieldnames or ()):
            raise ValueError(f"Invalid AU cache schema: {path}")
        result = {}
        for row in reader:
            if row["sample_id"] in result:
                raise ValueError(f"Duplicate AU sample_id: {row['sample_id']}")
            values = None
            if row["status"] == "success":
                values = np.array([float(row[column]) / 5 for column in AU_COLUMNS])
            elif row["status"] != "failed":
                raise ValueError(f"Invalid AU status: {row['status']}")
            result[row["sample_id"]] = values
    return result


def read_heatmap_cache(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if tuple(metadata.get("au_order", ())) != AU_IDS:
        raise ValueError("Heatmap cache AU order differs from the training contract")
    masks = np.load(path / "spatial_masks.npy", mmap_mode="r")
    peaks = np.load(path / "peaks.npy", mmap_mode="r")
    with (path / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if masks.shape != (len(rows), 49, 12) or peaks.shape != (len(rows), 12):
        raise ValueError("Heatmap cache arrays and index disagree")
    index = {}
    for row in rows:
        sample_id = row["sample_id"]
        position = int(row["row"])
        if sample_id in index or not 0 <= position < len(rows):
            raise ValueError("Invalid or duplicate heatmap cache index")
        index[sample_id] = (position, row["status"] == "success")
    return metadata, masks, peaks, index


def audit_domain(manifest, au_path, heatmap_path):
    rows = [
        row
        for row in validate_rows(read_jsonl(manifest))
        if row["usable"] and row["split"] == "train"
    ]
    au = read_au_cache(au_path)
    metadata, masks, peaks, heatmaps = read_heatmap_cache(heatmap_path)
    threshold = float(metadata["peak_threshold"])
    sample_ids = [row["sample_id"] for row in rows]
    missing_heatmap = [sample_id for sample_id in sample_ids if sample_id not in heatmaps]
    missing_au = [sample_id for sample_id in sample_ids if sample_id not in au]
    landmark_valid = {}
    for row in rows:
        size = row.get("processed_size")
        landmark_valid[row["sample_id"]] = bool(
            size and build_au_mask(row["landmarks_crop"], size)[1]
        )
    heatmap_success = {
        sample_id: sample_id in heatmaps and heatmaps[sample_id][1]
        for sample_id in sample_ids
    }
    au_valid = {sample_id: au.get(sample_id) is not None for sample_id in sample_ids}
    g0 = {
        sample_id: au_valid[sample_id] and landmark_valid[sample_id]
        for sample_id in sample_ids
    }
    g1 = {
        sample_id: g0[sample_id] and heatmap_success[sample_id]
        for sample_id in sample_ids
    }
    success_positions = [
        heatmaps[sample_id][0]
        for sample_id in sample_ids
        if heatmap_success[sample_id]
    ]
    success_positions = np.asarray(success_positions, dtype=np.int64)
    channel_rows = []
    for channel, au_id in enumerate(AU_IDS):
        channel_peaks = peaks[success_positions, channel] if len(success_positions) else []
        masses = masks[success_positions, :, channel].sum(axis=1) if len(success_positions) else []
        evidence = [
            sample_id
            for sample_id in sample_ids
            if au_valid[sample_id] and au[sample_id][channel] >= 0.5 and heatmap_success[sample_id]
        ]
        evidence_below = sum(
            peaks[heatmaps[sample_id][0], channel] < threshold for sample_id in evidence
        )
        channel_rows.append(
            {
                "au": au_id,
                "peak": quantiles(
                    channel_peaks,
                    {"min": 0, "median": 0.5, "p90": 0.9, "p99": 0.99, "max": 1},
                ),
                "peak_below_threshold_rate": (
                    float(np.mean(np.asarray(channel_peaks) < threshold))
                    if len(channel_peaks)
                    else None
                ),
                "strong_evidence_count": len(evidence),
                "strong_evidence_below_threshold_rate": (
                    evidence_below / len(evidence) if evidence else None
                ),
                "mask_mass": quantiles(
                    masses, {"min": 0, "p1": 0.01, "p5": 0.05, "median": 0.5, "p95": 0.95}
                ),
                "zero_mask_rate": (
                    float(np.mean(np.asarray(masses) == 0)) if len(masses) else None
                ),
            }
        )
    cosine = {}
    for left, right in ((1, 2), (12, 25), (25, 26)):
        left_index, right_index = AU_IDS.index(left), AU_IDS.index(right)
        values = []
        for position in success_positions:
            a, b = masks[position, :, left_index], masks[position, :, right_index]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator > 0:
                values.append(float(np.dot(a, b) / denominator))
        cosine[f"AU{left}_AU{right}"] = {
            "count": len(values),
            **quantiles(values, {"p5": 0.05, "median": 0.5, "p95": 0.95}),
        }
    count = len(rows)
    return {
        "sample_count": count,
        "missing_au_count": len(missing_au),
        "missing_heatmap_count": len(missing_heatmap),
        "au_valid_rate": sum(au_valid.values()) / count,
        "landmark_valid_rate": sum(landmark_valid.values()) / count,
        "heatmap_success_rate": sum(heatmap_success.values()) / count,
        "g0_valid_rate": sum(g0.values()) / count,
        "g1_valid_rate": sum(g1.values()) / count,
        "g0_valid_g1_excluded_count": sum(g0[sid] and not g1[sid] for sid in sample_ids),
        "channels": channel_rows,
        "cosine_similarity": cosine,
    }


def run(args):
    report = {
        domain: audit_domain(
            getattr(args, f"{domain}_manifest"),
            getattr(args, f"{domain}_au_cache"),
            getattr(args, f"{domain}_heatmap_cache"),
        )
        for domain in ("source", "target")
    }
    write_json(args.output, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
