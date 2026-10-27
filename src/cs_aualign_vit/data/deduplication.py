"""Exclude RAF test copies of training images without changing official roles."""

from collections import defaultdict


def exclude_raf_test_duplicates(rows):
    train = defaultdict(list)
    for row in rows:
        if row["dataset"] == "raf_db" and row["split"] == "train" and row.get("raw_sha256"):
            train[row["raw_sha256"]].append(row["sample_id"])
    result, excluded = [], []
    for original in rows:
        row = dict(original)
        matches = train.get(row.get("raw_sha256"), [])
        if row["dataset"] == "raf_db" and row["split"] == "test" and row["usable"] and matches:
            excluded.append(
                {
                    "sample_id": row["sample_id"],
                    "raw_sha256": row["raw_sha256"],
                    "matching_train_ids": sorted(matches),
                }
            )
            row.update(usable=False, processed_path=None, reject_reason="duplicate_of_raf_train")
        result.append(row)
    return result, excluded
