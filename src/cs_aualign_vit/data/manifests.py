"""Versioned, portable JSONL manifests; no training-time random splitting."""

import hashlib
import json
import os
from pathlib import Path

CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_rows(rows):
    seen = set()
    groups = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen.add(sid)
        if row["dataset"] not in {"raf_db", "expw"}:
            raise ValueError("Unknown dataset")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        if type(row["label"]) is not int or not 0 <= row["label"] < len(CLASSES):
            raise ValueError("Invalid canonical label")
        if row["class_name"] != CLASSES[row["label"]]:
            raise ValueError("Class name/ID mismatch")
        key = (row["dataset"], row["group_id"])
        if key in groups and groups[key] != row["split"]:
            raise ValueError("Photograph/content group crosses splits")
        groups[key] = row["split"]
    return rows


def group_split(group_id, seed, ratios):
    """Label-blind seeded hash split, stable under annotation order changes."""
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Expected positive train/validation/test ratios summing to one")
    bucket = (
        int.from_bytes(
            hashlib.sha256(f"post-expw-v1:{seed}:{group_id}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return (
        "train"
        if bucket < ratios[0]
        else ("validation" if bucket < ratios[0] + ratios[1] else "test")
    )
