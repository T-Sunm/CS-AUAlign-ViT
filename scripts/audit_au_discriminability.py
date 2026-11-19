"""Probe LibreFace intensities and FANAU spatial information across domains.

Run from the repository root with uv run python scripts/audit_au_discriminability.py.
Target labels are used only for the final evaluation.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "au_heatmap"))

from au_heatmap import AUHeatmapDetector  # noqa: E402
from cs_aualign_vit.data.au_guidance import AU_COLUMNS, AU_IDS  # noqa: E402
from cs_aualign_vit.data.manifests import CLASSES, read_jsonl, validate_rows  # noqa: E402

RESOLUTIONS = (64, 56, 28, 14, 7)
PAIRS = (("fear", "surprise"), ("disgust", "neutral"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_rows(path, split):
    return [row for row in validate_rows(read_jsonl(path)) if row["usable"] and row["split"] == split]


def intensities(path, rows):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if {"sample_id", "status", *AU_COLUMNS} - set(reader.fieldnames or ()):
            raise ValueError(f"Invalid LibreFace cache: {path}")
        records = {}
        for record in reader:
            sid = record["sample_id"]
            if sid in records:
                raise ValueError(f"Duplicate AU cache ID: {sid}")
            values = np.full(12, np.nan, np.float32)
            if record["status"] == "success":
                values = np.asarray([float(record[c]) for c in AU_COLUMNS], np.float32) / 5
                if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError(f"Invalid AU intensities: {sid}")
            elif record["status"] != "failed":
                raise ValueError(f"Invalid AU status: {sid}")
            records[sid] = values
    return np.asarray([records.get(row["sample_id"], np.full(12, np.nan)) for row in rows], np.float32)


def spatial_descriptors(heatmap):
    """One 4x4 max-normalized spatial grid per AU, at every input resolution."""
    value = torch.as_tensor(heatmap, dtype=torch.float32)
    if value.shape != (12, 64, 64) or not torch.isfinite(value).all():
        return None
    value = value.clamp_min(0)
    if value.amax().item() == 0:
        return None
    result = []
    for size in RESOLUTIONS:
        scaled = F.adaptive_avg_pool2d(value[:, None], (size, size))
        scaled = scaled / (scaled.amax(dim=(2, 3), keepdim=True) + 1e-8)
        result.append(F.adaptive_avg_pool2d(scaled, (4, 4))[:, 0].flatten(1).numpy())
    return np.stack(result).astype(np.float32)


def load_or_build_spatial(manifest, rows, checkpoint, cache, device, batch_size):
    key = {"manifest_sha256": digest(manifest), "checkpoint_sha256": digest(checkpoint),
           "sample_ids": [row["sample_id"] for row in rows], "resolutions": RESOLUTIONS}
    cache = Path(cache)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata != json.loads(json.dumps(key)):
                raise ValueError(f"Spatial cache identity differs: {cache}")
            if "heatmaps" in saved:
                heatmaps = saved["heatmaps"].astype(np.float32)
                valid = saved["valid"].copy()
                descriptors = np.zeros((len(rows), len(RESOLUTIONS), 12, 16), np.float32)
                for index in np.flatnonzero(valid):
                    descriptors[index] = spatial_descriptors(heatmaps[index])
                return descriptors, valid
    detector = AUHeatmapDetector(checkpoint, device=device)
    features = np.zeros((len(rows), len(RESOLUTIONS), 12, 16), np.float32)
    original_heatmaps = np.zeros((len(rows), 12, 64, 64), np.float16)
    valid = np.zeros(len(rows), bool)
    for start in range(0, len(rows), batch_size):
        images, positions = [], []
        for index in range(start, min(start + batch_size, len(rows))):
            path = (Path(manifest).parent / rows[index]["processed_path"]).resolve()
            try:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB")).copy())
                positions.append(index)
            except (OSError, ValueError):
                continue
        if images:
            heatmaps, _ = detector(images)
            for index, heatmap in zip(positions, heatmaps, strict=True):
                descriptor = spatial_descriptors(heatmap)
                if descriptor is not None:
                    features[index] = descriptor
                    original_heatmaps[index] = heatmap.astype(np.float16)
                    valid[index] = True
        print(f"FANAU {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + ".tmp.npz")
    np.savez_compressed(temporary, metadata=json.dumps(key), descriptors=features,
                        heatmaps=original_heatmaps, valid=valid)
    temporary.replace(cache)
    return features, valid


def quality(rows, p, spatial_valid, h):
    okay = np.isfinite(p).all(axis=1)
    return {
        "rows": len(rows), "class_counts": dict(Counter(CLASSES[r["label"]] for r in rows)),
        "duplicate_group_rows": len(rows) - len({r["group_id"] for r in rows}),
        "libreface_valid": int(okay.sum()), "fanau_valid": int(spatial_valid.sum()),
        "both_valid": int((okay & spatial_valid).sum()),
        "intensity_quantiles_by_au": {
            str(au): np.quantile(p[okay, i], [0.1, 0.5, 0.9]).tolist() if okay.any() else None
            for i, au in enumerate(AU_IDS)
        },
        "heatmap_nonzero_fraction_by_resolution_au": {
            f"H{size}": {
                str(au): float((h[spatial_valid, j, i].max(axis=1) > 0).mean())
                if spatial_valid.any() else None
                for i, au in enumerate(AU_IDS)
            }
            for j, size in enumerate(RESOLUTIONS)
        },
    }


def score(y, probability, threshold):
    predicted = probability >= threshold
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0))}


def bootstrap(y, probabilities, threshold, repeats, seed):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    draws = {name: [] for name in probabilities}
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        for name, probability in probabilities.items():
            draws[name].append(score(y[indices], probability[indices], threshold[name]))
    result = {name: {metric: list(np.quantile([item[metric] for item in values], [0.025, 0.975]))
                     for metric in values[0]} for name, values in draws.items()}
    result["P+H_minus_P"] = {
        metric: list(np.quantile([a[metric] - b[metric] for a, b in zip(draws["P+H"], draws["P"])], [0.025, 0.975]))
        for metric in draws["P"][0]
    }
    return result


def probe(train, validation, heldout, target, pair, repeats, seed):
    a, b = [CLASSES.index(name) for name in pair]
    def choose(data):
        keep = np.isin(data["y"], [a, b]) & data["valid"]
        return {k: v[keep] for k, v in data.items()}
    tr, va, te, ex = (choose(data) for data in (train, validation, heldout, target))
    if any(len(np.unique(data["y"])) != 2 for data in (tr, va, te, ex)):
        raise ValueError(f"Both classes are required in each split for {pair}")
    result = {}
    for size_index, size in enumerate(RESOLUTIONS):
        selected_c = {}
        predictions = {"raf_heldout": {}, "expw": {}}
        thresholds = {}
        for kind in ("P", "H", "P+H"):
            def matrix(data):
                p = data["p"].reshape(len(data["p"]), 12, 1)
                h = data["h"][:, size_index]
                blocks = p if kind == "P" else h if kind == "H" else np.concatenate((p, h), axis=2)
                return blocks.reshape(len(blocks), -1)
            x, xv = matrix(tr), matrix(va)
            choices = []
            for c in (0.01, 0.1, 1.0, 10.0):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
                model.fit(x, (tr["y"] == b).astype(int))
                pv = model.predict_proba(xv)[:, 1]
                choices.append((roc_auc_score(va["y"] == b, pv), c, model, pv))
            _, chosen_c, model, pv = max(choices, key=lambda row: (row[0], -row[1]))
            thresholds[kind] = max((0.3, 0.4, 0.5, 0.6, 0.7),
                                   key=lambda t: balanced_accuracy_score(va["y"] == b, pv >= t))
            selected_c[kind] = chosen_c
            for label, data in (("raf_heldout", te), ("expw", ex)):
                predictions[label][kind] = model.predict_proba(matrix(data))[:, 1]
        for label, data in (("raf_heldout", te), ("expw", ex)):
            y = (data["y"] == b).astype(int)
            metrics = {kind: score(y, p, thresholds[kind]) for kind, p in predictions[label].items()}
            ci = bootstrap(y, predictions[label], thresholds, repeats, seed)
            result[f"H{size}_{label}"] = {"n": len(y), "metrics": metrics,
                                           "confidence_intervals": ci, "thresholds": thresholds,
                                           "selected_c": selected_c}
        # Refit without one AU group, using the selected P+H regularization contract.
        # Ablation is evaluated on RAF held-out; ExpW labels never select a model.
        full = result[f"H{size}_raf_heldout"]["metrics"]["P+H"]["roc_auc"]
        group_scores = {}
        for index, au in enumerate(AU_IDS):
            columns = np.ones(12, bool)
            columns[index] = False
            def without(data):
                return np.concatenate((data["p"][:, :, None], data["h"][:, size_index]), axis=2)[:, columns].reshape(len(data["y"]), -1)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=selected_c["P+H"], max_iter=1000))
            model.fit(without(tr), (tr["y"] == b).astype(int))
            auc = roc_auc_score(te["y"] == b, model.predict_proba(without(te))[:, 1])
            group_scores[str(au)] = float(full - auc)
        result[f"H{size}_raf_heldout"]["au_group_auc_drop"] = group_scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for domain in ("source", "target"):
        parser.add_argument(f"--{domain}-manifest", required=True, type=Path)
        parser.add_argument(f"--{domain}-au-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error("batch-size and bootstrap must be positive")
    source_rows = selected_rows(args.source_manifest, "train")
    target_rows = selected_rows(args.target_manifest, args.target_split)
    if not source_rows or not target_rows:
        raise ValueError("Selected source or target split is empty")
    domains = {}
    for name, manifest, cache, rows in (
        ("source", args.source_manifest, args.source_au_cache, source_rows),
        ("target", args.target_manifest, args.target_au_cache, target_rows),
    ):
        p = intensities(cache, rows)
        h, h_valid = load_or_build_spatial(manifest, rows, args.checkpoint,
                                           args.output / f"{name}_{'train' if name == 'source' else args.target_split}_spatial.npz",
                                           args.device, args.batch_size)
        domains[name] = {"y": np.asarray([row["label"] for row in rows]), "p": p,
                         "h": h, "valid": np.isfinite(p).all(axis=1) & h_valid}
        domains[name + "_quality"] = quality(rows, p, h_valid, h)
    y = domains["source"]["y"]
    groups = np.asarray([row["group_id"] for row in source_rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_val, test = next(splitter.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed + 1)
    inner_train, inner_val = next(inner.split(np.zeros(len(train_val)), y[train_val], groups[train_val]))
    train, validation = train_val[inner_train], train_val[inner_val]
    def subset(indices):
        return {key: value[indices] for key, value in domains["source"].items()}
    report = {"protocol": {"source_split_counts": {"train": len(train), "validation": len(validation), "heldout": len(test)},
                            "target_split": args.target_split, "seed": args.seed,
                            "descriptor": "per-AU max-normalized 4x4 adaptive-average grid",
                            "selection": "RAF validation only; complete cases shared by P, H and P+H"},
              "quality": {"source": domains["source_quality"], "target": domains["target_quality"]},
              "pairs": {}}
    for pair in PAIRS:
        report["pairs"]["-".join(pair)] = probe(subset(train), subset(validation), subset(test),
                                                  domains["target"], pair, args.bootstrap, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
