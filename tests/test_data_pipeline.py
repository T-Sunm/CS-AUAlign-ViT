import json

import numpy as np
import pytest
from PIL import Image

from cs_aualign_vit.data.manifests import group_split, read_jsonl, write_jsonl
from cs_aualign_vit.data.preprocessing import preprocess, select_detection
from cs_aualign_vit.data.readers import EXPW_LABELS, RAF_LABELS, read_expw, read_raf


class FixtureDetector:
    def __init__(self):
        self.identity = {"implementation": "synthetic-fixture"}
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return ([[2, 3, 30, 35]], [0.99], [[[8, 12], [22, 12], [15, 20], [10, 28], [20, 28]]])


def test_native_label_mapping():
    assert [RAF_LABELS[i] for i in range(1, 8)] == [6, 2, 1, 3, 5, 0, 4]
    assert [EXPW_LABELS[i] for i in range(7)] == [0, 1, 2, 3, 5, 6, 4]


def test_raf_official_roles(tmp_path):
    annotation = tmp_path / "labels.txt"
    annotation.write_text("train_00001.jpg 7\ntest_00001.jpg 5\n")
    rows, _ = read_raf(annotation, tmp_path)
    assert [(r["split"], r["class_name"]) for r in rows] == [
        ("train", "neutral"),
        ("test", "sadness"),
    ]
    annotation.write_text("validation_00001.jpg 1\n")
    with pytest.raises(ValueError, match="official"):
        read_raf(annotation, tmp_path)


def test_expw_dedup_grouping_and_order_independence(tmp_path):
    Image.new("RGB", (48, 48), (30, 10, 5)).save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    annotation = tmp_path / "label.lst"
    lines = ["a.png 0 0 0 30 30 99 6", "a.png 1 2 2 35 35 90 4", "b.png 0 0 0 30 30 99 6"]
    annotation.write_text("\n".join(lines + [lines[0]]))
    rows, info = read_expw(annotation, tmp_path, 7, [0.8, 0.1, 0.1])
    assert len(rows) == 3 and info["duplicates_removed"] == 1
    assert len({r["split"] for r in rows}) == 1
    annotation.write_text("\n".join(reversed(lines)))
    reordered, _ = read_expw(annotation, tmp_path, 7, [0.8, 0.1, 0.1])
    assert rows == reordered
    annotation.write_text(lines[0] + "\na.png 0 0 0 30 30 99 3")
    with pytest.raises(ValueError, match="Conflicting"):
        read_expw(annotation, tmp_path, 7, [0.8, 0.1, 0.1])


def test_raf_official_face_box_and_provenance(tmp_path):
    annotation = tmp_path / "labels.txt"
    annotation.write_text("train_00001.jpg 7\n")
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    box = boxes / "train_00001_boundingbox.txt"
    box.write_text("5.5 6.0 30.0 35.25")
    rows, info = read_raf(annotation, tmp_path, boxes)
    assert rows[0]["reference_bbox"] == [5.5, 6.0, 30.0, 35.25]
    digest = info["bounding_boxes_sha256"]
    box.write_text("6.5 6.0 30.0 35.25")
    assert read_raf(annotation, tmp_path, boxes)[1]["bounding_boxes_sha256"] != digest
    box.write_text("nan 6 30 35")
    with pytest.raises(ValueError, match="Invalid RAF reference box"):
        read_raf(annotation, tmp_path, boxes)


def test_split_validation_and_benchmark_overlap(tmp_path):
    with pytest.raises(ValueError):
        group_split("a", 1, [0.8, 0.3, 0.1])
    annotation = tmp_path / "label.lst"
    annotation.write_text("a.jpg 0 0 0 30 30 99 0\na.jpg 1 0 0 30 30 99 1")
    supplied = tmp_path / "split.json"
    supplied.write_text(json.dumps({"expw:a.jpg:0": "train", "expw:a.jpg:1": "test"}))
    with pytest.raises(ValueError, match="crosses splits"):
        read_expw(annotation, tmp_path, 1, [0.8, 0.1, 0.1], supplied)


def test_select_annotated_identity_and_reject_small():
    points = [[[5, 5]] * 5, [[60, 60]] * 5]
    selected, reason = select_detection(
        [[0, 0, 30, 30], [50, 50, 100, 100]], [0.9, 0.99], points, (120, 120), [0, 0, 30, 30]
    )
    assert reason is None and selected["bbox"] == [0, 0, 30, 30]
    assert (
        select_detection([[0, 0, 19, 30]], [1], [points[0]], (100, 100))[1]
        == "face_below_minimum_size"
    )
    assert select_detection(None, None, None, (100, 100))[1] == "no_face"
    assert (
        select_detection([[50, 50, 100, 100]], [1], [points[0]], (120, 120), [0, 0, 30, 30])[1]
        == "annotated_face_unmatched"
    )


def prepared_fixture(tmp_path, dataset="raf_db"):
    annotation = tmp_path / "annotation"
    for name in ("train_00001.png", "test_00001.png"):
        # Asymmetric pixels let the test observe horizontal flipping.
        pixels = np.zeros((48, 48, 3), dtype=np.uint8)
        pixels[:, :16, 0] = 200
        pixels[:, 16:, 1] = 100 if name.startswith("train") else 150
        Image.fromarray(pixels).save(tmp_path / name)
    annotation.write_text("train_00001.png 1\ntest_00001.png 7")
    rows, info = read_raf(annotation, tmp_path)
    if dataset == "expw":
        for row in rows:
            row["dataset"] = "expw"
            row["sample_id"] = row["sample_id"].replace("raf_db", "expw")
    output = tmp_path / "prepared"
    detector = FixtureDetector()
    preprocess(rows, tmp_path, output, detector, info)
    return output, detector, rows, info


def test_native_crops_landmarks_and_resume(tmp_path):
    output, detector, rows, info = prepared_fixture(tmp_path)
    processed = read_jsonl(output / "processed.jsonl")
    assert len(processed) == 2 and all(r["usable"] for r in processed)
    assert processed[0]["processed_size"] == [28, 32]
    assert processed[0]["landmarks_crop"][0] == [6, 9]
    calls = detector.calls
    preprocess(rows, tmp_path, output, detector, info)
    assert detector.calls == calls
    with pytest.raises(ValueError, match="different preparation"):
        preprocess(rows, tmp_path, output, detector, info, min_iou=0.4)


def test_missing_and_no_detection_are_recorded(tmp_path):
    _output, detector, rows, info = prepared_fixture(tmp_path)
    rows[0]["image_path"] = "missing.jpg"
    detector.detect = lambda image: (None, None, None)
    other = tmp_path / "rejections"
    preprocess(rows, tmp_path, other, detector, info)
    assert {r["reject_reason"] for r in read_jsonl(other / "processed.jsonl")} == {
        "missing_image",
        "no_face",
    }


def test_role_interfaces_and_online_transforms(tmp_path):
    import torch

    from cs_aualign_vit.data.datasets import FaceDataset, build_loader
    from cs_aualign_vit.data.transforms import build_transform

    output, *_ = prepared_fixture(tmp_path, "expw")
    manifest = output / "processed.jsonl"
    target = FaceDataset(manifest, "target_adaptation")
    target_features = FaceDataset(manifest, "target_features")
    image, sid = target[0]
    assert image.shape == (3, 224, 224) and torch.isfinite(image).all()
    assert isinstance(sid, str) and "label" not in target.metadata(sid)
    assert all("label" not in sample for sample in target.samples)
    assert [sample["sample_id"] for sample in target_features.samples] == [
        sample["sample_id"] for sample in target.samples
    ]
    assert target_features[0][1] == sid
    assert torch.equal(target_features[0][0], target_features[0][0])
    assert all("label" not in sample for sample in target_features.samples)
    target.metadata(sid)["landmarks_crop"][0][0] = 999
    assert target.metadata(sid)["landmarks_crop"][0][0] != 999
    evaluation = FaceDataset(manifest, "evaluation")
    assert isinstance(evaluation[0][1], int)
    assert torch.equal(evaluation[0][0], evaluation[0][0])
    with pytest.raises(ValueError):
        FaceDataset(manifest, "target_adaptation", split="test")
    with pytest.raises(ValueError):
        FaceDataset(manifest, "source_train")
    loader = build_loader(manifest, "target_adaptation", batch_size=1)
    assert next(iter(loader))[0].shape == (1, 3, 224, 224)
    feature_loader = build_loader(manifest, "target_features", batch_size=1)
    assert next(iter(feature_loader))[1] == (sid,)
    with Image.open(target.samples[0]["processed_path"]) as crop:
        transform = build_transform(True, 32, (0, 0, 0), (1, 1, 1))
        outputs = [transform(crop) for _ in range(20)]
        assert any(not torch.equal(outputs[0], x) for x in outputs[1:])
    with pytest.raises(ValueError):
        build_transform(False, std=(0, 1, 1))


def test_source_interface(tmp_path):
    from cs_aualign_vit.data.datasets import FaceDataset

    output, *_ = prepared_fixture(tmp_path)
    source = FaceDataset(output / "processed.jsonl", "source_train")
    assert len(source) == 1 and source[0][1] == 6


def test_loader_keeps_global_rng_state_and_rejects_escaping_crop_path(tmp_path):
    import random

    import torch

    from cs_aualign_vit.data.datasets import FaceDataset, build_loader

    output, *_ = prepared_fixture(tmp_path)
    manifest = output / "processed.jsonl"
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    build_loader(manifest, "source_train", batch_size=1, seed=42)
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)

    rows = read_jsonl(manifest)
    rows[0]["processed_path"] = "../outside.png"
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="escapes preparation output"):
        FaceDataset(manifest, "source_train")


def test_corrupt_image_after_good_image_is_rejected(tmp_path):
    _output, detector, rows, info = prepared_fixture(tmp_path)
    (tmp_path / "test_00001.png").write_bytes(b"not an image")
    output = tmp_path / "corrupt-input"
    preprocess(rows, tmp_path, output, detector, info)
    results = read_jsonl(output / "processed.jsonl")
    assert results[0]["usable"]
    assert results[1]["reject_reason"] == "image_decode_error"


def test_resume_rejects_corrupt_crop_and_recovers_partial_journal(tmp_path):
    output, detector, rows, info = prepared_fixture(tmp_path)
    journal = output / "journal.jsonl"
    with journal.open("ab") as stream:
        stream.write(b'{"interrupted":')
    preprocess(rows, tmp_path, output, detector, info)
    assert len(read_jsonl(journal)) == 2
    first = read_jsonl(journal)[0]
    (output / first["processed_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing/corrupt"):
        preprocess(rows, tmp_path, output, detector, info)


def test_manifest_rejects_tampered_labels(tmp_path):
    from cs_aualign_vit.data.datasets import FaceDataset

    output, *_ = prepared_fixture(tmp_path)
    path = output / "processed.jsonl"
    rows = read_jsonl(path)
    rows[0]["label"] = 0
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="mismatch"):
        FaceDataset(path, "source_train")


def test_audit_cli_checks_every_crop_and_loads_without_detector(tmp_path):
    import subprocess
    import sys

    output, *_ = prepared_fixture(tmp_path)
    report = tmp_path / "audit.json"
    command = [
        sys.executable,
        "scripts/audit_data.py",
        "--manifest",
        str(output / "processed.jsonl"),
        "--output",
        str(report),
        "--check-loaders",
        "--verify-originals",
        "--batch-size",
        "1",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(report.read_text())
    assert evidence["rows_checked"] == 2
    assert evidence["original_pixel_crops_checked"] == 2
    assert evidence["loaders"]["train"]["batch_shape"] == [1, 3, 224, 224]
    row = read_jsonl(output / "processed.jsonl")[0]
    (output / row["processed_path"]).write_bytes(b"corrupt")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0 and "checksum mismatch" in result.stderr


def test_audit_detects_pixel_change_even_with_updated_crop_checksum(tmp_path):
    import runpy

    from cs_aualign_vit.data.manifests import sha256

    output, *_ = prepared_fixture(tmp_path)
    manifest = output / "processed.jsonl"
    rows = read_jsonl(manifest)
    crop_path = output / rows[0]["processed_path"]
    with Image.open(crop_path) as image:
        altered = image.copy()
    altered.putpixel((0, 0), (1, 2, 3))
    altered.save(crop_path)
    rows[0]["processed_sha256"] = sha256(crop_path)
    write_jsonl(manifest, rows)
    audit = runpy.run_path("scripts/audit_data.py")["audit"]
    assert audit(manifest)["rows_checked"] == 2
    with pytest.raises(ValueError, match="Crop pixels differ"):
        audit(manifest, verify_originals=True)


def test_loader_audit_rejects_a_split_with_no_usable_samples(tmp_path):
    import runpy

    _output, detector, rows, info = prepared_fixture(tmp_path)
    detector.detect = lambda image: (None, None, None)
    output = tmp_path / "all-rejected"
    preprocess(rows, tmp_path, output, detector, info)
    check_loaders = runpy.run_path("scripts/audit_data.py")["check_loaders"]
    with pytest.raises(ValueError, match="No usable samples"):
        check_loaders(output / "processed.jsonl", 1, {"transforms": {"image_size": 224}})


def test_raf_test_duplicates_are_excluded_and_auditor_rejects_overlap(tmp_path):
    import runpy

    from cs_aualign_vit.data.datasets import FaceDataset
    from cs_aualign_vit.data.manifests import sha256

    _output, detector, rows, info = prepared_fixture(tmp_path)
    annotation = tmp_path / "annotation"
    info["annotation_sha256"] = sha256(annotation)
    (tmp_path / "test_00001.png").write_bytes((tmp_path / "train_00001.png").read_bytes())
    # Retain an independent test image to exercise the evaluation loader.
    Image.new("RGB", (48, 48), (5, 15, 25)).save(tmp_path / "test_00002.png")
    annotation.write_text(annotation.read_text() + "\ntest_00002.png 7\n")
    rows, _ = read_raf(annotation, tmp_path)
    info["annotation_sha256"] = sha256(annotation)
    output = tmp_path / "native-overlap"
    preprocess(rows, tmp_path, output, detector, info)
    manifest = output / "processed.jsonl"
    records = read_jsonl(manifest)
    assert records[0]["usable"]
    assert not records[1]["usable"]
    assert records[1]["reject_reason"] == "duplicate_of_raf_train"
    assert records[1]["split"] == "test" and records[1]["processed_path"] is None
    assert len(FaceDataset(manifest, "source_train")) == 1
    evaluation = FaceDataset(manifest, "evaluation")
    assert [r["sample_id"] for r in evaluation.samples] == ["raf_db:test_00002.png"]
    audit = runpy.run_path("scripts/audit_data.py")["audit"]
    evidence = audit(manifest, verify_originals=True, raf_annotation=annotation)
    assert evidence["raw_hash_split_overlap"] == 0
    assert evidence["official_raf_partition_verified"]
    # Restore the pre-filter journal: native annotations must not waive overlap.
    write_jsonl(manifest, read_jsonl(output / "journal.jsonl"))
    with pytest.raises(ValueError, match="duplicates train bytes"):
        audit(manifest, raf_annotation=annotation)
    # The migration command also filters old manifests, without changing them.
    parent_hash = sha256(manifest)
    migrate = runpy.run_path("scripts/exclude_raf_test_duplicates.py")["exclude_existing"]
    derived = tmp_path / "deduplicated"
    assert migrate(manifest, derived)["excluded_count"] == 1
    assert sha256(manifest) == parent_hash
    assert audit(derived / "processed.jsonl", verify_originals=True)["raw_hash_split_overlap"] == 0
    with pytest.raises(FileExistsError):
        migrate(manifest, derived)
    # Editing both split manifests cannot bypass native label/role checks.
    for filename in ("processed.jsonl", "splits.jsonl"):
        records = read_jsonl(derived / filename)
        records[0].update(label=0, class_name="anger")
        write_jsonl(derived / filename, records)
    with pytest.raises(ValueError, match="official RAF labels/partition"):
        audit(derived / "processed.jsonl", raf_annotation=annotation)


def test_raf_duplicate_filter_includes_rejected_train_and_preserves_other_rows():
    from cs_aualign_vit.data.deduplication import exclude_raf_test_duplicates

    rows = [
        {
            "dataset": "raf_db",
            "split": "train",
            "sample_id": "a",
            "raw_sha256": "same",
            "usable": False,
        },
        {
            "dataset": "raf_db",
            "split": "train",
            "sample_id": "b",
            "raw_sha256": "same",
            "usable": True,
        },
        {
            "dataset": "raf_db",
            "split": "test",
            "sample_id": "c",
            "raw_sha256": "same",
            "usable": True,
        },
        {
            "dataset": "raf_db",
            "split": "test",
            "sample_id": "d",
            "raw_sha256": "same",
            "usable": True,
        },
        {
            "dataset": "expw",
            "split": "test",
            "sample_id": "e",
            "raw_sha256": "same",
            "usable": True,
        },
    ]
    filtered, excluded = exclude_raf_test_duplicates(rows)
    assert filtered[:2] == rows[:2] and filtered[4] == rows[4]
    assert rows[2]["usable"]  # Do not mutate the parent records.
    assert [r["sample_id"] for r in excluded] == ["c", "d"]
    assert excluded[0]["matching_train_ids"] == ["a", "b"]
    assert not filtered[2]["usable"] and not filtered[3]["usable"]
    assert exclude_raf_test_duplicates(filtered)[0] == filtered


def test_raf_annotation_cannot_allow_expw_overlap(tmp_path):
    import runpy

    output, *_ = prepared_fixture(tmp_path, "expw")
    audit = runpy.run_path("scripts/audit_data.py")["audit"]
    with pytest.raises(ValueError, match="only to RAF-DB"):
        audit(output / "processed.jsonl", raf_annotation=tmp_path / "annotation")
