"""Extract a resumable sample-id keyed LibreFace AU cache outside training."""

import argparse
import csv
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
            records[row["sample_id"]] = row
    return records


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

    flush_every = getattr(args, "flush_every", 100)
    if flush_every < 1:
        raise ValueError("flush_every must be positive")
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    records = _existing(output)
    rows = [
        row
        for row in read_jsonl(manifest)
        if row.get("split") == "train" and row.get("usable")
    ]
    pending = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in records:
            continue
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
        except Exception as error:
            records[sample_id] = {
                "sample_id": sample_id,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                **dict.fromkeys(AU_COLUMNS, ""),
            }
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
        "au12_happiness_mean": _mean(records, happiness, "au_12_intensity"),
        "au12_other_mean": _mean(records, set(sample_ids) - set(happiness), "au_12_intensity"),
        "au4_anger_mean": _mean(records, anger, "au_4_intensity"),
        "au4_other_mean": _mean(records, set(sample_ids) - set(anger), "au_4_intensity"),
    }


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
