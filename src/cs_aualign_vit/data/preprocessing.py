"""MTCNN detection only; native-resolution PIL crops, resumable JSONL journal."""

import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .deduplication import exclude_raf_test_duplicates
from .manifests import CLASSES, SCHEMA_VERSION, read_jsonl, sha256, write_json, write_jsonl
from .readers import image_path


def intersection_over_union(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def select_detection(boxes, probabilities, points, size, reference=None, min_size=20, min_iou=0.3):
    """Match ExpW annotated identity; otherwise choose the largest detected face."""
    if boxes is None or len(boxes) == 0:
        return None, "no_face"
    candidates = []
    width, height = size
    for box, probability, landmarks in zip(boxes, probabilities, points, strict=True):
        box, landmarks = np.asarray(box, dtype=float), np.asarray(landmarks, dtype=float)
        if box.shape != (4,) or landmarks.shape != (5, 2):
            raise ValueError("Unexpected detector output shape")
        if not np.isfinite(box).all() or not np.isfinite(landmarks).all():
            continue
        clipped = [
            max(0, math.floor(box[0])),
            max(0, math.floor(box[1])),
            min(width, math.ceil(box[2])),
            min(height, math.ceil(box[3])),
        ]
        area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
        score = intersection_over_union(clipped, reference) if reference else area
        candidates.append((score, float(probability), clipped, landmarks.tolist()))
    if not candidates:
        return None, "invalid_detection"
    score, probability, box, landmarks = max(candidates, key=lambda r: (r[0], r[1]))
    if reference and score < min_iou:
        return None, "annotated_face_unmatched"
    if box[2] - box[0] < min_size or box[3] - box[1] < min_size:
        return None, "face_below_minimum_size"
    if not np.isfinite(probability):
        return None, "invalid_detection"
    return {
        "bbox": box,
        "landmarks_original": landmarks,
        "landmarks_crop": [[x - box[0], y - box[1]] for x, y in landmarks],
        "detection_probability": probability,
        "match_iou": score if reference else None,
    }, None


class MTCNNDetector:
    def __init__(self, device="cpu", min_face_size=20, thresholds=(0.6, 0.7, 0.7)):
        import torch
        from facenet_pytorch import MTCNN

        torch.set_num_threads(2)
        self.model = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            post_process=False,
        )
        self.identity = {
            "implementation": "facenet-pytorch",
            "version": importlib.metadata.version("facenet-pytorch"),
            "device": device,
            "min_face_size": min_face_size,
            "thresholds": list(thresholds),
            "factor": 0.709,
            "weights": {},
        }
        import facenet_pytorch

        data = Path(facenet_pytorch.__file__).parent / "data"
        for name in ("pnet.pt", "rnet.pt", "onet.pt"):
            self.identity["weights"][name] = sha256(data / name)

    def detect(self, image):
        # detect(), never forward()/extract(), which resize the face internally.
        return self.model.detect(image, landmarks=True)


def summarize(rows, source_info):
    splits = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            continue
        usable = [r for r in subset if r["usable"]]
        counts = Counter(r["class_name"] for r in usable)
        splits[split] = {
            "input_faces": len(subset),
            "usable": len(usable),
            "rejected": len(subset) - len(usable),
            "input_photographs": len({r["image_path"] for r in subset}),
            "usable_class_counts": {name: counts[name] for name in CLASSES},
            "reject_reasons": dict(Counter(r["reject_reason"] for r in subset if not r["usable"])),
        }
    return {
        "canonical_classes": list(CLASSES),
        "source_info": source_info,
        "input_faces": len(rows),
        "input_photographs": len({r["image_path"] for r in rows}),
        "usable": sum(r["usable"] for r in rows),
        "rejected": sum(not r["usable"] for r in rows),
        "splits": splits,
    }


def preprocess(rows, image_root, output, detector, source_info, min_size=20, min_iou=0.3):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "classes": list(CLASSES),
        "image_root": str(Path(image_root).resolve()),
        "source_info": source_info,
        "detector": detector.identity,
        "min_crop_size": min_size,
        "min_match_iou": min_iou,
        "box_convention": "xyxy pixel edges, floor left/top, ceil right/bottom, clipped",
        "landmark_order": [
            "left_eye_image",
            "right_eye_image",
            "nose",
            "left_mouth_image",
            "right_mouth_image",
        ],
        "face_selection": "max IoU to annotation when available; otherwise largest clipped box",
        "offline_resize": False,
        "crop_format": "PNG RGB, lossless",
    }
    identity_path = output / "preparation.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Output belongs to a different preparation configuration; choose new output"
        )
    if (output / "splits.jsonl").exists() and read_jsonl(output / "splits.jsonl") != rows:
        raise ValueError("Saved split manifest differs from requested inputs")
    write_json(identity_path, identity)
    write_jsonl(output / "splits.jsonl", rows)
    journal = output / "journal.jsonl"
    # Only an interrupted final line is repairable; never discard a valid record.
    if journal.exists():
        data = journal.read_bytes()
        if data and not data.endswith(b"\n"):
            journal.write_bytes(data[: data.rfind(b"\n") + 1])
    completed = read_jsonl(journal) if journal.exists() else []
    done = {r["sample_id"] for r in completed}
    if len(done) != len(completed) or not done.issubset({r["sample_id"] for r in rows}):
        raise ValueError("Invalid preparation journal")
    source_by_id = {r["sample_id"]: r for r in rows}
    for row in completed:
        if any(row[key] != value for key, value in source_by_id[row["sample_id"]].items()):
            raise ValueError("Resume journal changed input annotation fields")
        if row["usable"]:
            crop = output / row["processed_path"]
            if not crop.is_file() or sha256(crop) != row["processed_sha256"]:
                raise ValueError(
                    "Resume crop missing/corrupt; restore artifact or choose new output"
                )
        if row["raw_sha256"]:
            raw = image_path(image_root, row["image_path"])
            if not raw.is_file() or sha256(raw) != row["raw_sha256"]:
                raise ValueError("Resume raw image changed")
    started = time.time()
    write_json(
        output / "status.json",
        {"state": "running", "total": len(rows), "completed": len(done), "pid": os.getpid()},
    )
    last_name, image, detections, image_digest = None, None, None, None
    try:
        with journal.open("a") as stream:
            for original in rows:
                if original["sample_id"] in done:
                    continue
                row = dict(
                    original,
                    usable=False,
                    processed_path=None,
                    bbox=None,
                    landmarks_original=None,
                    landmarks_crop=None,
                    reject_reason=None,
                    original_size=None,
                    processed_size=None,
                    raw_sha256=None,
                )
                name = row["image_path"]
                path = image_path(image_root, name)
                if not path.is_file():
                    row["reject_reason"] = "missing_image"
                elif row["reference_bbox"] and (
                    row["reference_bbox"][2] <= row["reference_bbox"][0]
                    or row["reference_bbox"][3] <= row["reference_bbox"][1]
                ):
                    row["reject_reason"] = "invalid_reference_bbox"
                else:
                    decode_error = False
                    if name != last_name:
                        try:
                            with Image.open(path) as raw:
                                image = raw.convert("RGB")
                        except (UnidentifiedImageError, OSError):
                            decode_error = True
                            image = None
                            last_name = None
                    if decode_error:
                        row["reject_reason"] = "image_decode_error"
                    else:
                        if name != last_name:
                            image_digest = sha256(path)
                            detections = detector.detect(image)
                            last_name = name
                        row["raw_sha256"] = image_digest
                        row["original_size"] = list(image.size)
                        selected, reason = select_detection(
                            *detections, image.size, row["reference_bbox"], min_size, min_iou
                        )
                        row["reject_reason"] = reason
                        if selected:
                            row.update(selected)
                            filename = hashlib.sha256(row["sample_id"].encode()).hexdigest()
                            relative = f"crops/{filename}.png"
                            crop = image.crop(selected["bbox"])
                            temp = output / f"crops/{filename}.tmp"
                            crop.save(temp, format="PNG")
                            os.replace(temp, output / relative)
                            row.update(
                                usable=True,
                                processed_path=relative,
                                processed_size=list(crop.size),
                                processed_sha256=sha256(output / relative),
                            )
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                completed.append(row)
                done.add(row["sample_id"])
                if len(done) % 100 == 0:
                    os.fsync(stream.fileno())
                    status = {
                        "state": "running",
                        "total": len(rows),
                        "completed": len(done),
                        "elapsed_seconds": time.time() - started,
                        "pid": os.getpid(),
                    }
                    write_json(output / "status.json", status)
                    print(json.dumps(status), flush=True)
        by_id = {r["sample_id"]: r for r in completed}
        ordered = [by_id[r["sample_id"]] for r in rows]
        ordered, excluded = exclude_raf_test_duplicates(ordered)
        write_json(output / "excluded_duplicates.json", excluded)
        write_jsonl(output / "processed.jsonl", ordered)
        write_json(output / "summary.json", summarize(ordered, source_info))
        write_json(
            output / "status.json",
            {
                "state": "completed",
                "total": len(rows),
                "completed": len(done),
                "manifest_sha256": sha256(output / "processed.jsonl"),
            },
        )
    except BaseException as error:
        write_json(
            output / "status.json",
            {"state": "failed", "completed": len(done), "error": repr(error), "pid": os.getpid()},
        )
        raise
