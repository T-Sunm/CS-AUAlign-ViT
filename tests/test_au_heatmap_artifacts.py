import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from str_helpers import make_source

from cs_aualign_vit.data.au_guidance import AU_COLUMNS
from cs_aualign_vit.data.manifests import read_jsonl, write_jsonl


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    calls: ClassVar[list[int]] = []

    def __init__(self, _checkpoint, device="cpu"):
        self.device = device

    def __call__(self, images):
        self.calls.append(len(images))
        heatmaps = np.full((len(images), 12, 8, 8), 0.7, dtype=np.float32)
        heatmaps[:, 0] = 0.9
        return heatmaps, heatmaps.max(axis=(2, 3))


def extractor_args(manifest, output, checkpoint):
    return SimpleNamespace(
        manifest=manifest,
        output=output,
        checkpoint=checkpoint,
        split="train",
        peak_threshold=0.8,
        device="cpu",
        batch_size=2,
        checkpoint_every=2,
    )


def test_batched_heatmap_export_is_resumable_and_keeps_rows(tmp_path, monkeypatch):
    script = load_script("extract_au_heatmap_cache")
    manifest = make_source(tmp_path / "images", {0: 3})
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "heatmaps"
    FakeDetector.calls = []
    monkeypatch.setitem(sys.modules, "au_heatmap", SimpleNamespace(AUHeatmapDetector=FakeDetector))

    script.run(extractor_args(manifest, output, checkpoint))

    assert FakeDetector.calls == [2, 1]
    masks = np.load(output / "spatial_masks.npy")
    assert masks.shape == (3, 49, 12)
    assert np.allclose(masks[:, :, 0].max(axis=1), 1)
    assert not np.count_nonzero(masks[:, :, 1:])
    with (output / "index.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    stable_rows = {record["sample_id"]: record["row"] for record in records}

    records[1]["status"] = "failed"
    with (output / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "row", "status"])
        writer.writeheader()
        writer.writerows(records)
    FakeDetector.calls = []
    script.run(extractor_args(manifest, output, checkpoint))

    assert FakeDetector.calls == [1]
    with (output / "index.csv").open(newline="") as stream:
        retried = list(csv.DictReader(stream))
    assert {record["sample_id"]: record["row"] for record in retried} == stable_rows
    assert all(record["status"] == "success" for record in retried)

    backup = output.with_name(output.name + ".previous")
    os.replace(output, backup)
    script.load_existing(output)
    assert output.is_dir() and not backup.exists()


def test_heatmap_audit_reports_gate_coverage_and_distinguishability(tmp_path, monkeypatch):
    extractor = load_script("extract_au_heatmap_cache")
    audit = load_script("audit_au_heatmap_cache")
    manifest = make_source(tmp_path / "images", {0: 2})
    rows = read_jsonl(manifest)
    for row in rows:
        row["processed_size"] = [100, 100]
    write_jsonl(manifest, rows)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "heatmaps"
    monkeypatch.setitem(sys.modules, "au_heatmap", SimpleNamespace(AUHeatmapDetector=FakeDetector))
    extractor.run(extractor_args(manifest, output, checkpoint))
    au_cache = tmp_path / "au.csv"
    with au_cache.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "status", *AU_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"sample_id": row["sample_id"], "status": "success", **dict.fromkeys(AU_COLUMNS, 3)}
            )
    report_path = tmp_path / "audit.json"

    report = audit.run(
        SimpleNamespace(
            source_manifest=manifest,
            source_au_cache=au_cache,
            source_heatmap_cache=output,
            target_manifest=manifest,
            target_au_cache=au_cache,
            target_heatmap_cache=output,
            output=report_path,
        )
    )

    assert report_path.is_file()
    assert report["source"]["g0_valid_rate"] == 1
    assert report["source"]["g1_valid_rate"] == 1
    assert report["source"]["channels"][0]["zero_mask_rate"] == 0
    assert report["source"]["channels"][1]["zero_mask_rate"] == 1
    assert report["source"]["cosine_similarity"]["AU1_AU2"]["count"] == 0
