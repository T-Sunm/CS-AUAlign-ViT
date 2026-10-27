"""Audit every processed row and crop; exercise loaders without importing MTCNN."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cs_aualign_vit.data.manifests import read_jsonl, sha256, validate_rows, write_json
from cs_aualign_vit.data.preprocessing import summarize
from cs_aualign_vit.data.readers import image_path, read_raf


def audit(manifest, verify_originals=False, raf_annotation=None):
    manifest = Path(manifest).resolve()
    rows = validate_rows(read_jsonl(manifest))
    preparation = json.loads((manifest.parent / "preparation.json").read_text())
    split_rows = read_jsonl(manifest.parent / "splits.jsonl")
    if len(rows) != len(split_rows) or any(
        any(row[key] != source[key] for key in source)
        for row, source in zip(rows, split_rows, strict=True)
    ):
        raise ValueError("Processed manifest changed input split/annotation fields")
    official_raf_verified = False
    if raf_annotation is not None:
        if {r["dataset"] for r in rows} != {"raf_db"}:
            raise ValueError("--raf-annotation applies only to RAF-DB")
        if sha256(raf_annotation) != preparation["source_info"].get("annotation_sha256"):
            raise ValueError("Official RAF annotation checksum differs from preparation")
        expected, _ = read_raf(raf_annotation, preparation["image_root"])
        if "smoke_limit" in preparation["source_info"]:
            expected = expected[: preparation["source_info"]["smoke_limit"]]
        fields = ("sample_id", "image_path", "dataset", "split", "label", "class_name")
        if len(expected) != len(rows) or any(
            any(actual[key] != native[key] for key in fields)
            for actual, native in zip(rows, expected, strict=True)
        ):
            raise ValueError("Manifest differs from official RAF labels/partition")
        official_raf_verified = True
    hashes = defaultdict(set)
    crop_shapes = set()
    originals_checked = 0
    last_raw_path, raw_image, raw_digest = None, None, None
    for row in rows:
        if type(row["usable"]) is not bool:
            raise ValueError("Usable status must be a boolean")
        if not row["usable"]:
            if not row["reject_reason"] or row["processed_path"] is not None:
                raise ValueError("Invalid rejected row")
            continue
        path = (manifest.parent / row["processed_path"]).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError("Crop path escapes preparation output")
        if sha256(path) != row["processed_sha256"]:
            raise ValueError("Crop checksum mismatch")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if image.mode != "RGB" or [width, height] != row["processed_size"]:
                raise ValueError("Crop metadata mismatch")
            if verify_originals:
                source = image_path(preparation["image_root"], row["image_path"])
                if source != last_raw_path:
                    raw_digest = sha256(source)
                    with Image.open(source) as original_image:
                        raw_image = original_image.convert("RGB")
                    last_raw_path = source
                if raw_digest != row["raw_sha256"] or list(raw_image.size) != row["original_size"]:
                    raise ValueError("Original image identity/size mismatch")
                expected = np.asarray(raw_image.crop(row["bbox"]))
                if not np.array_equal(np.asarray(image), expected):
                    raise ValueError("Crop pixels differ from the original image bounding box")
                originals_checked += 1
        box = row["bbox"]
        original_width, original_height = row["original_size"]
        if not (0 <= box[0] < box[2] <= original_width and 0 <= box[1] < box[3] <= original_height):
            raise ValueError("Crop box escapes original image")
        if [width, height] != [box[2] - box[0], box[3] - box[1]] or min(width, height) < 20:
            raise ValueError("Crop resized offline or smaller than 20px")
        original, local = np.asarray(row["landmarks_original"]), np.asarray(row["landmarks_crop"])
        if original.shape != (5, 2) or not np.isfinite(original).all():
            raise ValueError("Invalid original landmarks")
        if local.shape != (5, 2) or not np.allclose(local, original - np.array(box[:2])):
            raise ValueError("Invalid crop-coordinate landmarks")
        if row["reject_reason"] is not None:
            raise ValueError("Usable row has reject reason")
        if row["dataset"] == "expw" and row["raw_sha256"] != row["group_id"]:
            raise ValueError("ExpW original content changed after split assignment")
        hashes[row["raw_sha256"]].add(row["split"])
        crop_shapes.add((width, height))
    crossing = sum(len(splits) > 1 for splits in hashes.values())
    limits = [
        "Byte checks do not rule out near duplicates or shared identities",
        "No model efficacy or training has been evaluated",
    ]
    if crossing:
        if not official_raf_verified:
            raise ValueError(
                f"Raw-image bytes cross splits: {crossing}; RAF requires checksum-bound "
                "--raf-annotation to verify its mandated official partition"
            )
        limits.append(
            f"RAF official partition contains {crossing} identical-byte groups across splits; "
            "official roles retained, so source test is not byte-disjoint"
        )
    return {
        "state": "validated",
        "manifest_sha256": sha256(manifest),
        "rows_checked": len(rows),
        "distinct_crop_shapes": len(crop_shapes),
        "raw_hash_split_overlap": crossing,
        "original_pixel_crops_checked": originals_checked,
        "official_raf_partition_verified": official_raf_verified,
        "summary": summarize(rows, preparation["source_info"]),
        "limits": limits,
    }


def check_loaders(manifest, batch_size, config):
    import torch

    from cs_aualign_vit.data.datasets import build_loader

    rows = read_jsonl(manifest)
    dataset = rows[0]["dataset"]
    role = "source_train" if dataset == "raf_db" else "target_adaptation"
    evidence = {}
    kwargs = config["transforms"]
    for split, current_role in [
        ("train", role),
        ("test", "evaluation"),
        ("validation", "evaluation"),
    ]:
        if not any(r["split"] == split for r in rows):
            continue
        if not any(r["usable"] and r["split"] == split for r in rows):
            raise ValueError(f"No usable samples remain in input split {split}")
        loader = build_loader(
            manifest, current_role, batch_size=batch_size, split=split, seed=42, **kwargs
        )
        images, values = next(iter(loader))
        assert tuple(images.shape[1:]) == (3, kwargs["image_size"], kwargs["image_size"])
        assert torch.isfinite(images).all()
        metadata = loader.dataset.metadata(loader.dataset.samples[0]["sample_id"])
        if current_role == "target_adaptation":
            assert all(isinstance(x, str) for x in values)
            assert all("label" not in r and "class_name" not in r for r in loader.dataset.samples)
            assert "label" not in metadata and "class_name" not in metadata
        if current_role == "evaluation":
            assert torch.equal(loader.dataset[0][0], loader.dataset[0][0])
            assert values.dtype == torch.int64 and ((values >= 0) & (values < 7)).all()
        evidence[split] = {
            "role": current_role,
            "samples": len(loader.dataset),
            "batch_shape": list(images.shape),
            "landmarks_retrievable": True,
        }
    import sys

    assert "facenet_pytorch" not in sys.modules, "Loading must not invoke/import MTCNN"
    return evidence


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-loaders", action="store_true")
    parser.add_argument(
        "--raf-annotation",
        type=Path,
        help="Original checksum-bound RAF annotation to verify official roles",
    )
    parser.add_argument(
        "--verify-originals",
        action="store_true",
        help="Check each usable crop pixel-for-pixel against its original image",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/data/raf_cross_domain.yaml"))
    args = parser.parse_args()
    result = audit(
        args.manifest, verify_originals=args.verify_originals, raf_annotation=args.raf_annotation
    )
    if args.check_loaders:
        result["loaders"] = check_loaders(
            args.manifest, args.batch_size, yaml.safe_load(args.config.read_text())
        )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
