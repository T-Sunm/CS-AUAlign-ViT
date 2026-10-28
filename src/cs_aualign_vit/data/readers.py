"""Read native annotations, preserving annotated-face identity."""

import hashlib
import json
import math
from pathlib import Path

from .manifests import CLASSES, group_split, sha256, validate_rows

RAF_LABELS = {1: 6, 2: 2, 3: 1, 4: 3, 5: 5, 6: 0, 7: 4}
EXPW_LABELS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 4}


def image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Annotation path escapes image root")
    return path


def base_row(dataset, sid, relative, split, label, group_id, reference_bbox=None):
    return {
        "sample_id": sid,
        "dataset": dataset,
        "image_path": relative,
        "split": split,
        "label": label,
        "class_name": CLASSES[label],
        "group_id": group_id,
        "reference_bbox": reference_bbox,
    }


def read_raf(annotation, root, bounding_boxes=None):
    """Native list_partition_label.txt (including distribution's 'patition' typo)."""
    rows = []
    box_digest = hashlib.sha256()
    for line in Path(annotation).read_text().splitlines():
        if not line.strip():
            continue
        name, raw_label = line.split()
        if not name.startswith(("train_", "test_")):
            raise ValueError(f"RAF official role missing: {name}")
        image_path(root, name)
        split = "train" if name.startswith("train_") else "test"
        label = RAF_LABELS[int(raw_label)]
        reference = None
        if bounding_boxes is not None:
            box_file = image_path(bounding_boxes, Path(name).stem + "_boundingbox.txt")
            content = box_file.read_bytes()
            reference = [float(value) for value in content.split()]
            if (
                len(reference) != 4
                or not all(math.isfinite(x) for x in reference)
                or reference[2] <= reference[0]
                or reference[3] <= reference[1]
            ):
                raise ValueError(f"Invalid RAF reference box: {box_file.name}")
            box_digest.update(box_file.name.encode() + b"\0" + content + b"\0")
        rows.append(base_row("raf_db", f"raf_db:{name}", name, split, label, name, reference))
    info = {"annotation_rows": len(rows), "duplicates_removed": 0}
    if bounding_boxes is not None:
        info["bounding_boxes_sha256"] = box_digest.hexdigest()
    return validate_rows(rows), info


def read_expw(annotation, root, seed, ratios, benchmark_split=None):
    """label.lst: filename face_id top left right bottom confidence expression."""
    records = {}
    duplicates = 0
    for line in Path(annotation).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        bits = line.split()
        if len(bits) != 8:
            raise ValueError("ExpW annotation must have eight fields")
        name, face, top, left, right, bottom, confidence, raw_label = bits
        image_path(root, name)
        sid = f"expw:{name}:{face}"
        bbox = [int(left), int(top), int(right), int(bottom)]
        signature = (name, face, *bbox, float(confidence), int(raw_label))
        if sid in records:
            if records[sid][0] != signature:
                raise ValueError(f"Conflicting ExpW annotation: {sid}")
            duplicates += 1
            continue
        label = EXPW_LABELS[int(raw_label)]
        records[sid] = (signature, base_row("expw", sid, name, "train", label, name, bbox))

    # Group byte-identical photographs across different filenames as well.
    # Missing images retain a filename group and are recorded as unusable later.
    hashes = {}
    for _, row in records.values():
        name = row["image_path"]
        if name not in hashes:
            path = image_path(root, name)
            hashes[name] = sha256(path) if path.is_file() else f"missing:{name}"
        row["group_id"] = hashes[name]

    supplied = None
    if benchmark_split:
        supplied = json.loads(Path(benchmark_split).read_text())
        if set(supplied) != set(records):
            raise ValueError("Benchmark split must cover every unique ExpW sample_id exactly")
    rows = []
    for sid, (_, row) in sorted(records.items()):
        row["split"] = supplied[sid] if supplied else group_split(row["group_id"], seed, ratios)
        rows.append(row)
    return validate_rows(rows), {
        "annotation_rows": len(rows) + duplicates,
        "duplicates_removed": duplicates,
        "photograph_filenames": len(hashes),
        "content_groups": len(set(hashes.values())),
        "split_policy": "supplied" if supplied else "seeded-sha256-photo-content-v1",
        "seed": seed,
        "ratios": list(ratios),
    }
