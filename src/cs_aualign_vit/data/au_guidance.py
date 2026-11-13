"""Offline AU cache loading and landmark-guidance construction."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

SPATIAL_FILES = ("spatial_masks.npy", "peaks.npy", "index.csv", "metadata.json")


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


def heatmaps_to_masks(heatmaps, *, peak_threshold=0.8, grid=7, epsilon=1.0e-8):
    """Convert ``[12, H, W]`` AU heatmaps to ``[grid*grid, 12]`` masks."""
    values = torch.as_tensor(heatmaps, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != len(AU_IDS):
        raise ValueError("AU heatmaps must have shape [12, H, W]")
    if grid <= 0 or peak_threshold < 0 or epsilon <= 0:
        raise ValueError("Invalid heatmap conversion parameters")
    finite_channels = torch.isfinite(values).all(dim=(1, 2))
    positive = torch.where(torch.isfinite(values), values, 0).clamp_min(0)
    peaks = positive.amax(dim=(1, 2))
    pooled = F.adaptive_avg_pool2d(positive[:, None], (grid, grid))[:, 0]
    maxima = pooled.amax(dim=(1, 2))
    keep = finite_channels & (peaks >= peak_threshold) & (maxima > epsilon)
    normalized = pooled / (maxima[:, None, None] + epsilon)
    normalized = torch.where(keep[:, None, None], normalized, 0)
    masks = normalized.permute(1, 2, 0).reshape(grid * grid, len(AU_IDS))
    return masks, peaks, bool(finite_channels.any() and torch.isfinite(masks).all())


def spatial_cache_identity(path):
    """Hash the four-file spatial artifact without depending on directory metadata."""
    root = Path(path).resolve()
    digest = hashlib.sha256()
    for name in SPATIAL_FILES:
        file = root / name
        if not file.is_file():
            raise FileNotFoundError(f"Spatial cache is missing {file}")
        digest.update(name.encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


class AUSpatialCache:
    """Strict sample-id view over a memory-mapped learned AU mask artifact."""

    def __init__(self, path, *, manifest=None, peak_threshold=None):
        self.path = Path(path).resolve()
        metadata_path = self.path / "metadata.json"
        try:
            self.metadata = json.loads(metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid spatial cache metadata: {metadata_path}") from error
        if tuple(self.metadata.get("au_order", ())) != AU_IDS:
            raise ValueError("Spatial cache AU order differs from the training contract")
        if peak_threshold is not None and self.metadata.get("peak_threshold") != peak_threshold:
            raise ValueError("Spatial cache peak threshold differs from the configuration")
        if manifest is not None:
            self.validate_manifest(manifest)
        self._masks = np.load(self.path / "spatial_masks.npy", mmap_mode="r")
        self._peaks = np.load(self.path / "peaks.npy", mmap_mode="r")
        if self._masks.ndim != 3 or self._masks.shape[1:] != (49, len(AU_IDS)):
            raise ValueError("Spatial masks must have shape [N, 49, 12]")
        if self._peaks.shape != (len(self._masks), len(AU_IDS)):
            raise ValueError("Spatial peaks must have shape [N, 12]")
        self._records = {}
        used_rows = set()
        with (self.path / "index.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != {"sample_id", "row", "status"}:
                raise ValueError("Spatial index must contain sample_id,row,status")
            for line, record in enumerate(reader, start=2):
                sample_id = record["sample_id"]
                if not sample_id or sample_id in self._records:
                    raise ValueError(f"Invalid or duplicate spatial sample_id at line {line}")
                try:
                    row = int(record["row"])
                except ValueError as error:
                    raise ValueError(f"Invalid spatial row at line {line}") from error
                if (
                    not 0 <= row < len(self._masks)
                    or row in used_rows
                    or record["status"] not in {"success", "failed"}
                ):
                    raise ValueError(f"Invalid spatial record at line {line}")
                used_rows.add(row)
                self._records[sample_id] = (row, record["status"] == "success")
        if len(self._records) != len(self._masks):
            raise ValueError("Spatial index and arrays have different lengths")

    def get(self, sample_id):
        record = self._records.get(sample_id)
        if record is None:
            raise KeyError(f"Spatial cache is missing sample_id: {sample_id}")
        row, valid = record
        mask = torch.from_numpy(np.array(self._masks[row], dtype=np.float32, copy=True))
        if not valid:
            mask.zero_()
        if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError(f"Invalid spatial mask for sample_id: {sample_id}")
        return mask, valid

    def validate_manifest(self, manifest):
        identities = self.metadata.get(
            "manifest_sha256s", [self.metadata.get("manifest_sha256")]
        )
        if _sha256_file(Path(manifest).resolve()) not in identities:
            raise ValueError("Spatial cache manifest identity differs from the dataset")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
