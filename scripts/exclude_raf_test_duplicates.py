"""Build a new RAF preparation excluding test copies of train, reusing crops."""

import argparse
import json
import shutil
from pathlib import Path

from cs_aualign_vit.data.deduplication import exclude_raf_test_duplicates
from cs_aualign_vit.data.manifests import read_jsonl, sha256, validate_rows, write_json, write_jsonl
from cs_aualign_vit.data.preprocessing import summarize


def exclude_existing(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    rows = validate_rows(read_jsonl(manifest))
    if not rows or {r["dataset"] for r in rows} != {"raf_db"}:
        raise ValueError("Expected a RAF-DB manifest")
    preparation = json.loads((manifest.parent / "preparation.json").read_text())
    rows, excluded = exclude_raf_test_duplicates(rows)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "policy": "exclude-raf-test-duplicates-of-train-v1",
        "parent_manifest": str(manifest),
        "parent_manifest_sha256": sha256(manifest),
        "excluded_count": len(excluded),
    }
    write_json(output / "status.json", {"state": "running", **provenance})
    try:
        for row in rows:
            if not row["usable"]:
                continue
            source = (manifest.parent / row["processed_path"]).resolve()
            destination = (output / row["processed_path"]).resolve()
            if not source.is_relative_to(manifest.parent) or not destination.is_relative_to(output):
                raise ValueError("Crop path escapes preparation output")
            if sha256(source) != row["processed_sha256"]:
                raise ValueError("Parent crop checksum mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(manifest.parent / "splits.jsonl", output / "splits.jsonl")
        preparation["postprocessing"] = provenance
        write_json(output / "preparation.json", preparation)
        write_json(output / "excluded_duplicates.json", excluded)
        write_jsonl(output / "processed.jsonl", rows)
        write_json(output / "summary.json", summarize(rows, preparation["source_info"]))
        write_json(
            output / "status.json",
            {
                "state": "completed",
                **provenance,
                "manifest_sha256": sha256(output / "processed.jsonl"),
            },
        )
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), **provenance})
        raise
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(exclude_existing(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
