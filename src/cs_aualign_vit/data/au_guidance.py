"""Offline AU cache loading and landmark-guidance construction."""

import csv
from pathlib import Path

import torch

AU_IDS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
AU_COLUMNS = tuple(f"au_{au}_intensity" for au in AU_IDS)
AU_REGIONS = {
    1: "brow",
    2: "brow",
    4: "brow",
    5: "eye",
    6: "eye",
    9: "nose",
    12: "mouth",
    15: "mouth",
    17: "chin",
    20: "mouth",
    25: "mouth",
    26: "mouth",
}


class AUCache:
    """Strict sample-id keyed view of a LibreFace CSV export."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        with self.path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"sample_id", "status", *AU_COLUMNS}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"AU cache is missing columns: {sorted(missing)}")
            self._records = {}
            for line, row in enumerate(reader, start=2):
                sample_id = row["sample_id"]
                if not sample_id or sample_id in self._records:
                    raise ValueError(f"Invalid or duplicate AU sample_id at line {line}")
                if row["status"] == "success":
                    try:
                        values = torch.tensor([float(row[c]) for c in AU_COLUMNS])
                    except ValueError as error:
                        raise ValueError(f"Non-numeric AU value at line {line}") from error
                    if not torch.isfinite(values).all() or not (
                        (0 <= values) & (values <= 5)
                    ).all():
                        raise ValueError(f"AU intensity outside [0, 5] at line {line}")
                    self._records[sample_id] = (values / 5.0, True)
                elif row["status"] == "failed":
                    self._records[sample_id] = (torch.zeros(len(AU_IDS)), False)
                else:
                    raise ValueError(f"Invalid AU status at line {line}: {row['status']!r}")

    def get(self, sample_id):
        values, valid = self._records.get(sample_id, (torch.zeros(len(AU_IDS)), False))
        return values.clone(), valid


def build_au_mask(landmarks, image_size, *, grid=7):
    """Return ``[grid*grid, 12]`` and whether five landmarks were usable."""
    try:
        points = torch.as_tensor(landmarks, dtype=torch.float32)
        width, height = map(float, image_size)
    except (TypeError, ValueError):
        return torch.zeros(grid * grid, len(AU_IDS)), False
    if (
        points.shape != (5, 2)
        or width <= 0
        or height <= 0
        or not torch.isfinite(points).all()
    ):
        return torch.zeros(grid * grid, len(AU_IDS)), False
    points = points / torch.tensor((width, height))
    if not ((0 <= points) & (points <= 1)).all():
        return torch.zeros(grid * grid, len(AU_IDS)), False
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye = (left_eye + right_eye) / 2
    mouth = (left_mouth + right_mouth) / 2
    eye_distance = max(float(torch.linalg.vector_norm(left_eye - right_eye)), 1 / grid)
    regions = {
        "brow": (eye[1] - 0.30 * eye_distance, eye[0], 0.18, 0.65),
        "eye": (eye[1], eye[0], 0.18, 0.65),
        "nose": (nose[1], nose[0], 0.25, 0.30),
        "mouth": (mouth[1], mouth[0], 0.25, 0.55),
        "chin": (mouth[1] + 0.35 * eye_distance, mouth[0], 0.22, 0.40),
    }
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    masks = []
    for au in AU_IDS:
        cy, cx, sy_factor, sx_factor = regions[AU_REGIONS[au]]
        sy = sy_factor * eye_distance
        sx = sx_factor * eye_distance
        masks.append(torch.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)))
    mask = torch.stack(masks, dim=-1).reshape(grid * grid, len(AU_IDS))
    return mask, bool(torch.isfinite(mask).all())


def compute_guidance(target_p, source_p, target_m, source_m, valid_pairs):
    """Return guidance with target queries in rows and source keys in columns."""
    target_a = target_m * target_p[:, None, :]
    source_a = source_m * source_p[:, None, :]
    guidance = target_a @ source_a.transpose(-2, -1) / len(AU_IDS)
    return guidance * valid_pairs[:, None, None].to(guidance.dtype)


def shuffle_valid_au(values, valid, *, generator=None):
    """Shuffle only valid rows and report how many rows moved."""
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if len(indices) < 2:
        return values, 0
    permutation = indices[torch.randperm(len(indices), generator=generator)]
    shuffled = values.clone()
    shuffled[indices] = values[permutation]
    changed = (shuffled[indices] != values[indices]).any(dim=1)
    return shuffled, int(changed.sum())
