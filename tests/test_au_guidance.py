import csv

import pytest
import torch
from PIL import Image

from cs_aualign_vit.data.au_guidance import (
    AU_COLUMNS,
    AUCache,
    build_au_mask,
    compute_guidance,
    shuffle_valid_au,
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
