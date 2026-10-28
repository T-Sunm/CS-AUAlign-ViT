"""Role-specific interfaces; adaptation never retains ground-truth labels."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .manifests import read_jsonl, validate_rows
from .transforms import build_transform


class FaceDataset(Dataset):
    def __init__(self, manifest, role, split=None, transform=None):
        allowed = {"source_train", "target_adaptation", "evaluation"}
        if role not in allowed:
            raise ValueError(f"role must be one of {sorted(allowed)}")
        manifest = Path(manifest).resolve()
        rows = validate_rows(read_jsonl(manifest))
        if not rows:
            raise ValueError("Empty manifest")
        datasets = {r["dataset"] for r in rows}
        if len(datasets) != 1:
            raise ValueError("Expected a single dataset per manifest")
        dataset = next(iter(datasets))
        if role == "source_train" and dataset != "raf_db":
            raise ValueError("Source training requires RAF-DB")
        if role == "target_adaptation" and dataset != "expw":
            raise ValueError("Target adaptation requires ExpW")
        if role != "evaluation":
            if split not in (None, "train"):
                raise ValueError("Training roles must use train split")
            split = "train"
        else:
            split = split or "test"
            if split not in {"validation", "test"}:
                raise ValueError("Evaluation requires validation/test")
        self.role = role
        self.transform = transform or build_transform(role != "evaluation")
        self.samples = []
        for row in rows:
            if row["split"] != split or not row["usable"]:
                continue
            record = {
                "sample_id": row["sample_id"],
                "processed_path": str((manifest.parent / row["processed_path"]).resolve()),
                "landmarks_crop": row["landmarks_crop"],
                "landmarks_original": row["landmarks_original"],
                "bbox": row["bbox"],
            }
            if role != "target_adaptation":
                record["label"] = row["label"]
            self.samples.append(record)
        if not self.samples:
            raise ValueError(f"No usable samples for {role}/{split}")
        self._index = {r["sample_id"]: r for r in self.samples}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        with Image.open(record["processed_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, (
            record["sample_id"] if self.role == "target_adaptation" else record["label"]
        )

    def metadata(self, sample_id):
        """STR accesses stored coordinates; adaptation metadata contains no labels."""
        import copy

        return copy.deepcopy(self._index[sample_id])


def build_loader(
    manifest,
    role,
    *,
    batch_size,
    split=None,
    num_workers=0,
    seed=0,
    image_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    import random

    import numpy as np
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = FaceDataset(
        manifest, role, split, build_transform(role != "evaluation", image_size, mean, std)
    )
    generator = torch.Generator().manual_seed(seed)
    # torch seeds each worker; seed Python/NumPy as well for future transforms.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=role != "evaluation",
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _seed_worker(worker_id):
    import random

    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
