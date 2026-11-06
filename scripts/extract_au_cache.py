"""Extract a resumable sample-id keyed LibreFace AU cache outside training."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

AU_IDS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
AU_COLUMNS = tuple(f"au_{au}_intensity" for au in AU_IDS)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--retry-error-type",
        action="append",
        default=[],
        help="Retry only failed records whose reason starts with this exception type; repeatable",
    )
    return parser.parse_args()


def _existing(path):
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "status", *AU_COLUMNS}
        if required - set(reader.fieldnames or ()):
            raise ValueError("Existing AU cache has an incompatible schema")
        records = {}
        for row in reader:
            if not row["sample_id"] or row["sample_id"] in records:
                raise ValueError("Existing AU cache contains duplicate or empty sample_id")
            if row["status"] not in {"success", "failed"}:
                raise ValueError("Existing AU cache contains an invalid status")
            records[row["sample_id"]] = row
    return records


def _apply_libreface_compat():
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

    detector = importlib.import_module("libreface.detect_mediapipe_image")
    face_mesh_module = detector.mp.solutions.face_mesh
    original = face_mesh_module.FaceMesh
    if getattr(original, "_cs_aualign_single_face", False):
        return

    def single_face_mesh(*args, **kwargs):
        # LibreFace converts its landmark lists to ndarrays inside the face loop.
        kwargs["max_num_faces"] = 1
        return original(*args, **kwargs)

    single_face_mesh._cs_aualign_single_face = True
    face_mesh_module.FaceMesh = single_face_mesh


def _should_process(record, retry_failed, retry_error_types):
    if record is None:
        return True
    if record["status"] != "failed" or not retry_failed:
        return False
    error_type = record.get("reason", "").partition(":")[0]
    return not retry_error_types or error_type in retry_error_types


def _mean(records, sample_ids, column):
    values = [
        float(records[sid][column])
        for sid in sample_ids
        if records[sid]["status"] == "success"
    ]
    return sum(values) / len(values) if values else None


def _save(records, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = ["sample_id", "status", "reason", *AU_COLUMNS]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records[sample_id] for sample_id in sorted(records))
    os.replace(temporary, output)


def run(args):
    import libreface

    _apply_libreface_compat()
    flush_every = getattr(args, "flush_every", 100)
    if flush_every < 1:
        raise ValueError("flush_every must be positive")
    retry_failed = getattr(args, "retry_failed", False)
    retry_error_types = set(getattr(args, "retry_error_type", ()))
    if retry_error_types and not retry_failed:
        raise ValueError("--retry-error-type requires --retry-failed")
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    records = _existing(output)
    rows = [
        row
        for row in read_jsonl(manifest)
        if row.get("split") == "train" and row.get("usable")
    ]
    pending = 0
    retried = retry_success = 0
    for row in rows:
        sample_id = row["sample_id"]
        previous = records.get(sample_id)
        if not _should_process(previous, retry_failed, retry_error_types):
            continue
        is_retry = previous is not None
        path = (manifest.parent / row["processed_path"]).resolve()
        try:
            result = libreface.get_facial_attributes(str(path), device=args.device)
            intensities = result["au_intensities"]
            records[sample_id] = {
                "sample_id": sample_id,
                "status": "success",
                "reason": "",
                **{column: intensities[column] for column in AU_COLUMNS},
            }
            retry_success += int(is_retry)
        except Exception as error:  # noqa: BLE001 - detector failures are cache data
            records[sample_id] = {
                "sample_id": sample_id,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                **dict.fromkeys(AU_COLUMNS, ""),
            }
        retried += int(is_retry)
        pending += 1
        if pending >= flush_every:
            _save(records, output)
            print(f"Cached {len(records)}/{len(rows)} samples in {output}", flush=True)
            pending = 0
    if pending or not output.exists():
        _save(records, output)
    sample_ids = [row["sample_id"] for row in rows]
    successful = [sid for sid in sample_ids if records[sid]["status"] == "success"]
    happiness = [row["sample_id"] for row in rows if row.get("class_name") == "happiness"]
    anger = [row["sample_id"] for row in rows if row.get("class_name") == "anger"]
    return {
        "output": str(output),
        "samples": len(sample_ids),
        "success": len(successful),
        "failed": len(sample_ids) - len(successful),
        "retried": retried,
        "retry_success": retry_success,
        "retry_failed": retried - retry_success,
        "au12_happiness_mean": _mean(records, happiness, "au_12_intensity"),
        "au12_other_mean": _mean(records, set(sample_ids) - set(happiness), "au_12_intensity"),
        "au4_anger_mean": _mean(records, anger, "au_4_intensity"),
        "au4_other_mean": _mean(records, set(sample_ids) - set(anger), "au_4_intensity"),
    }


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
