"""Independent dataset preparation. Run with the installed project package."""

import argparse
import json
from pathlib import Path

import yaml

from cs_aualign_vit.data.manifests import CLASSES, sha256
from cs_aualign_vit.data.preprocessing import MTCNNDetector, preprocess
from cs_aualign_vit.data.readers import read_expw, read_raf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["raf_db", "expw"], required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/raf_cross_domain.yaml"))
    parser.add_argument("--benchmark-split", type=Path)
    parser.add_argument(
        "--face-boxes", type=Path, help="RAF boundingbox directory for annotated-face matching"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Subset smoke check; always use separate output")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["canonical_classes"] != list(CLASSES):
        parser.error("Config canonical classes differ from package mapping")
    if args.dataset == "raf_db":
        if "aligned" in str(args.image_root).lower():
            parser.error("Task 1 requires original RAF inputs, not aligned crops")
        rows, info = read_raf(args.annotation, args.image_root, args.face_boxes)
    else:
        if args.face_boxes:
            parser.error("ExpW reference boxes already come from its annotation")
        split = config["expw_split"]
        rows, info = read_expw(
            args.annotation, args.image_root, split["seed"], split["ratios"], args.benchmark_split
        )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
        info["smoke_limit"] = args.limit
    info.update(annotation_sha256=sha256(args.annotation), config_sha256=sha256(args.config))
    if args.benchmark_split:
        info["benchmark_split_sha256"] = sha256(args.benchmark_split)
    prep = config["preprocessing"]
    detector = MTCNNDetector(args.device, prep["min_face_size"], prep["thresholds"])
    preprocess(
        rows,
        args.image_root,
        args.output,
        detector,
        info,
        prep["min_face_size"],
        prep["min_match_iou"],
    )
    print(json.dumps({"output": str(args.output.resolve()), "state": "completed"}))


if __name__ == "__main__":
    main()
