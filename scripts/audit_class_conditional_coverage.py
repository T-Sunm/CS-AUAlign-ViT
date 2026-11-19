"""Estimate batch-level AU/class centroid coverage under the E2 sampling policy.

This uses audit labels for target coverage; target_adaptation training data
intentionally hides those labels, so the result is an oracle upper bound.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cs_aualign_vit.data.manifests import CLASSES, read_jsonl


def audit(source_manifest, target_manifest, batch_size, n_eff_min, batches, seed):
    rng = random.Random(seed)

    def rows(path, dataset):
        records = [
            row for row in read_jsonl(path)
            if row["usable"] and row["split"] == "train" and row["dataset"] == dataset
        ]
        if not records:
            raise ValueError(f"No usable {dataset} training rows in {path}")
        return records

    domains = {
        "source": rows(source_manifest, "raf_db"),
        "target": rows(target_manifest, "expw"),
    }
    counts = {domain: Counter(row["label"] for row in records) for domain, records in domains.items()}
    # Approximate each domain's per-step class draw using its empirical class prior.
    # AU validity/intensity and spatial masks are not in the manifests; therefore
    # this is a label-coverage upper bound, not the final AU-valid coverage.
    pair_coverage = Counter()
    for _ in range(batches):
        present = {}
        for domain, values in domains.items():
            sample = [rng.choice(values) for _ in range(batch_size)]
            present[domain] = Counter(row["label"] for row in sample)
        for cls in range(len(CLASSES)):
            valid = all(present[d][cls] >= n_eff_min for d in domains)
            if valid:
                pair_coverage[cls] += 12
    total = batches * len(CLASSES) * 12
    return {
        "batch_size": batch_size,
        "n_eff_min": n_eff_min,
        "batches": batches,
        "target_label_source": "manifest ground truth (oracle upper bound)",
        "class_counts": {
            domain: {CLASSES[c]: counts[domain][c] for c in range(len(CLASSES))}
            for domain in domains
        },
        "pairs_per_class": {
            CLASSES[c]: pair_coverage[c] for c in range(len(CLASSES))
        },
        "mean_covered_AU_class_pairs_per_batch": sum(pair_coverage.values()) / batches,
        "coverage_fraction": sum(pair_coverage.values()) / total if total else 0.0,
        "limitations": [
            "Manifest-only audit: AU validity, intensity weights, spatial mask mass and effective sample size are unavailable.",
            "Source sampling is approximated as empirical-prior random sampling; verify against the configured loader sampler.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-eff-min", type=float, default=8)
    parser.add_argument("--batches", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_eff_min < 0 or args.batches < 1:
        parser.error("batch size/batches must be positive and n-eff-min non-negative")
    print(json.dumps(audit(args.source_manifest, args.target_manifest, args.batch_size,
                           args.n_eff_min, args.batches, args.seed), indent=2))


if __name__ == "__main__":
    main()
