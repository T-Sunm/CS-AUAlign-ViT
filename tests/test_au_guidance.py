import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from cs_aualign_vit.data.au_guidance import (
    AU_COLUMNS,
    AU_IDS,
    AUCache,
    AUSpatialCache,
    build_au_mask,
    compute_guidance,
    heatmaps_to_masks,
    shuffle_valid_au,
    spatial_cache_identity,
)
from cs_aualign_vit.data.transforms import GuidanceTransform


def write_cache(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "status", *AU_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def test_au_cache_validates_schema_range_and_missing_samples(tmp_path):
    path = tmp_path / "au.csv"
    write_cache(
        path,
        [
            {"sample_id": "ok", "status": "success", **dict.fromkeys(AU_COLUMNS, 2.5)},
            {"sample_id": "failed", "status": "failed", **dict.fromkeys(AU_COLUMNS, "")},
        ],
    )
    cache = AUCache(path)
    values, valid = cache.get("ok")
    assert valid and torch.equal(values, torch.full((12,), 0.5))
    assert cache.get("failed")[1] is False
    assert cache.get("missing")[1] is False

    write_cache(
        path,
        [{"sample_id": "bad", "status": "success", **dict.fromkeys(AU_COLUMNS, 6)}],
    )
    with pytest.raises(ValueError, match=r"outside \[0, 5\]"):
        AUCache(path)


def test_mask_guidance_direction_missing_fallback_and_shuffle():
    landmarks = [[30, 30], [70, 30], [50, 50], [35, 75], [65, 75]]
    mask, valid = build_au_mask(landmarks, (100, 100))
    assert valid and mask.shape == (49, 12)
    invalid, valid = build_au_mask(None, (100, 100))
    assert not valid and not torch.count_nonzero(invalid)

    target_p = torch.stack((torch.ones(12), torch.ones(12)))
    source_p = torch.stack((torch.ones(12), torch.ones(12) * 0.5))
    target_m = torch.stack((mask, mask))
    source_m = torch.stack((mask.flip(0), mask))
    guidance = compute_guidance(
        target_p, source_p, target_m, source_m, torch.tensor([True, False])
    )
    assert guidance.shape == (2, 49, 49)
    assert torch.allclose(guidance[0], mask @ mask.flip(0).transpose(0, 1) / 12)
    assert not torch.count_nonzero(guidance[1])

    values = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    shuffled, changed = shuffle_valid_au(
        values, torch.tensor([True, False, True]), generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(shuffled[1], values[1])
    assert changed == int((shuffled != values).any(dim=1).sum())


def test_guidance_transform_flips_image_and_mask_together(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: torch.tensor(0.0))
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    mask = torch.arange(49, dtype=torch.float32).reshape(49, 1)
    transform = GuidanceTransform(image_size=2, mean=(0, 0, 0), std=(1, 1, 1))

    transformed, transformed_mask = transform(image, mask)

    assert transformed[2, 0, 0] == 1 and transformed[0, 0, 1] == 1
    assert torch.equal(transformed_mask.reshape(7, 7), mask.reshape(7, 7).flip(1))


def test_heatmap_conversion_applies_relu_pool_normalization_and_peak_gate():
    heatmaps = torch.zeros(12, 64, 64)
    heatmaps[0, :32] = 0.8
    heatmaps[1] = 0.79
    heatmaps[2] = -3

    masks, peaks, valid = heatmaps_to_masks(heatmaps, peak_threshold=0.8)

    assert valid and masks.shape == (49, 12)
    assert peaks[:3].tolist() == pytest.approx([0.8, 0.79, 0.0])
    assert masks[:, 0].max() == pytest.approx(1.0)
    assert not torch.count_nonzero(masks[:, 1:])
    invalid, invalid_peaks, valid = heatmaps_to_masks(
        torch.full((12, 64, 64), float("nan"))
    )
    assert not valid and not torch.count_nonzero(invalid)
    assert not torch.count_nonzero(invalid_peaks)


def test_spatial_cache_is_strict_and_failed_rows_are_zero(tmp_path):
    cache_dir = tmp_path / "spatial"
    cache_dir.mkdir()
    masks = np.ones((2, 49, 12), dtype=np.float32)
    np.save(cache_dir / "spatial_masks.npy", masks)
    np.save(cache_dir / "peaks.npy", np.ones((2, 12), dtype=np.float32))
    (cache_dir / "metadata.json").write_text(
        json.dumps({"au_order": list(AU_IDS), "peak_threshold": 0.8})
    )
    (cache_dir / "index.csv").write_text(
        "sample_id,row,status\nok,0,success\nfailed,1,failed\n"
    )

    cache = AUSpatialCache(cache_dir, peak_threshold=0.8)

    assert cache.get("ok")[1] and torch.equal(cache.get("ok")[0], torch.ones(49, 12))
    assert not cache.get("failed")[1] and not torch.count_nonzero(cache.get("failed")[0])
    with pytest.raises(KeyError, match="missing sample_id"):
        cache.get("absent")
    with pytest.raises(ValueError, match="peak threshold"):
        AUSpatialCache(cache_dir, peak_threshold=0.7)
    assert spatial_cache_identity(cache_dir) == spatial_cache_identity(cache_dir)
