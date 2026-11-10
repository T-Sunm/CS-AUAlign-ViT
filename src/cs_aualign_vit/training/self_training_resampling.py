"""Self-Training Resampling (Task 5): add high-confidence target patches to minority source images."""

import copy
import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from cs_aualign_vit.data.datasets import FaceDataset, build_source_loader
from cs_aualign_vit.data.manifests import CLASSES, read_jsonl, write_json, write_jsonl
from cs_aualign_vit.data.str_patches import compose, compute_rois, validate_rois

LOGGER = logging.getLogger(__name__)
REJECT_REASONS = ("invalid_landmark", "out_of_bounds", "overlap", "no_valid_donor")


def str_sample_id(round_id, target_id):
    return hashlib.sha256(f"str:{round_id}:{target_id}".encode()).hexdigest()


def _donor_order(donors, global_seed, round_id, target_id):
    digest = hashlib.sha256(f"{global_seed}:{round_id}:{target_id}".encode()).digest()
    order = list(donors)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


class SelfTrainingResampler:
    def __init__(
        self,
        *,
        source_manifest,
        target_feature_loader,
        prototype_banks,
        output_dir,
        global_seed,
        batch_size,
        device,
        num_workers=0,
        num_classes=7,
        confidence_threshold=0.95,
        au_cache=None,
        au_python=None,
    ):
        dataset = target_feature_loader.dataset
        if not isinstance(dataset, FaceDataset) or dataset.role != "target_features":
            raise ValueError("target_feature_loader must wrap a target_features FaceDataset")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")
        self.source_manifest = Path(source_manifest).resolve()
        self.target_feature_loader = target_feature_loader
        self.target_dataset = dataset
        self.prototype_banks = prototype_banks
        self.output_dir = Path(output_dir).resolve()
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.images_dir = self.output_dir / "images"
        self.transactions_dir = self.output_dir / "transactions"
        self.global_seed = int(global_seed)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.au_cache = Path(au_cache).resolve() if au_cache is not None else None
        self.au_python = Path(au_python).absolute() if au_python is not None else None
        if (self.au_cache is None) != (self.au_python is None):
            raise ValueError("au_cache and au_python must be configured together")
        if self.au_python is not None and not self.au_python.is_file():
            raise FileNotFoundError(f"LibreFace Python executable not found: {self.au_python}")
        source = FaceDataset(self.source_manifest, "source_train")
        usable = {s["sample_id"] for s in source.samples}
        self._donors = {c: [] for c in range(num_classes)}
        for row in read_jsonl(self.source_manifest):
            if row["sample_id"] in usable:
                row["_path"] = (self.source_manifest.parent / row["processed_path"]).resolve()
                self._donors[row["label"]].append(row)
        for rows in self._donors.values():
            rows.sort(key=lambda r: r["sample_id"])

    # -- persistence -----------------------------------------------------------------
    def _transaction_config(self):
        return {
            "global_seed": self.global_seed,
            "confidence_threshold": self.confidence_threshold,
            "num_classes": self.num_classes,
        }

    def _validate_transaction_config(self, transaction):
        for key, expected in self._transaction_config().items():
            if transaction.get(key) != expected:
                raise ValueError(
                    f"STR transaction configuration mismatch for {key}: "
                    f"saved={transaction.get(key)!r}, current={expected!r}"
                )

    def _transaction_path(self, round_id):
        return self.transactions_dir / f"round-{round_id:02d}.json"

    def _read_manifest(self):
        return read_jsonl(self.manifest_path) if self.manifest_path.exists() else []

    def get_transaction(self, round_id):
        path = self._transaction_path(round_id)
        if not path.exists():
            return None
        transaction = json.loads(path.read_text())
        self._validate_transaction_config(transaction)
        return transaction

    def _counts(self, rows):
        counts = [len(self._donors[c]) for c in range(self.num_classes)]
        for row in rows:
            counts[row["label"]] += 1
        return counts

    def create_transaction(self, *, round_id, epoch, checkpoint_path):
        existing = self.get_transaction(round_id)
        if existing is not None:
            return existing
        counts = self._counts(self._read_manifest())
        selected = min(range(self.num_classes), key=lambda c: (counts[c], c))
        transaction = {
            "round_id": round_id,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "status": "in_progress",
            "selected_class": selected,
            **self._transaction_config(),
            "counts_before": counts,
        }
        write_json(self._transaction_path(round_id), transaction)
        return transaction

    def _validate_manifest(self):
        rows = self._read_manifest()
        used_target_ids = set()
        for row in rows:
            target_id = row.get("target_id")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("STR record has invalid target_id")
            if target_id in used_target_ids:
                raise ValueError(f"STR manifest has duplicate target_id: {target_id}")
            used_target_ids.add(target_id)
            if type(row.get("round_id")) is not int or row["round_id"] < 1:
                raise ValueError(f"STR record has invalid round_id: {row.get('round_id')}")
            if row.get("sample_id") != str_sample_id(row.get("round_id"), row.get("target_id")):
                raise ValueError(f"STR record has inconsistent sample_id: {row.get('sample_id')}")
            path = (self.output_dir / row["processed_path"]).resolve()
            if not path.is_relative_to(self.output_dir):
                raise ValueError("STR image path escapes the output directory")
            try:
                with Image.open(path) as image:
                    image.load()
            except Exception as error:
                raise ValueError(f"STR image is missing or corrupt: {path}") from error
        return rows

    # -- generation ------------------------------------------------------------------
    def _predict(self, model, banks):
        predictions = {}
        observed_ids = []
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                for images, sample_ids in self.target_feature_loader:
                    observed_ids.extend(sample_ids)
                    features = model.forward_target_features(images.to(self.device)).float()
                    probabilities = F.softmax(banks.target_logits(features), dim=1)
                    confidence, labels = probabilities.max(dim=1)
                    for sid, c, label in zip(
                        sample_ids, confidence.tolist(), labels.tolist(), strict=True
                    ):
                        predictions[sid] = (label, c)
        finally:
            model.train(was_training)
        expected_ids = [sample["sample_id"] for sample in self.target_dataset.samples]
        if observed_ids != expected_ids:
            raise ValueError(
                "Target feature scan sample_ids differ from the full target dataset "
                "(missing, duplicated, or out-of-order samples)"
            )
        return predictions

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    def run(self, transaction, model):
        self._validate_transaction_config(transaction)
        round_id, epoch = transaction["round_id"], transaction["epoch"]
        selected = transaction["selected_class"]
        state = torch.load(transaction["checkpoint_path"], map_location=self.device, weights_only=True)
        if state.get("completed_epoch") != epoch:
            raise ValueError("STR snapshot completed_epoch differs from the transaction epoch")
        snapshot_model = copy.deepcopy(model)
        snapshot_banks = copy.deepcopy(self.prototype_banks).to(self.device)
        snapshot_model.load_state_dict(state["model_state_dict"])
        snapshot_banks.load_state_dict(state["prototype_state_dict"])

        rows = self._validate_manifest()
        used = {row["target_id"] for row in rows}
        predictions = self._predict(snapshot_model, snapshot_banks)
        del snapshot_model, snapshot_banks
        gated = sorted(
            sid
            for sid, (label, confidence) in predictions.items()
            if label == selected and confidence > transaction["confidence_threshold"]
        )
        candidates = [sid for sid in gated if sid not in used]
        rejects = dict.fromkeys(REJECT_REASONS, 0)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        for target_id in candidates:
            meta = self.target_dataset.metadata(target_id)
            target_image = self._load_rgb(meta["processed_path"])
            rois_t = compute_rois(meta["landmarks_crop"], target_image.size)
            reason = validate_rois(rois_t, target_image.size)
            if reason is not None:
                rejects[reason] += 1
                continue
            row = None
            for donor in _donor_order(self._donors[selected], transaction["global_seed"], round_id, target_id):
                donor_image = self._load_rgb(donor["_path"])
                rois_s = compute_rois(donor["landmarks_crop"], donor_image.size)
                if validate_rois(rois_s, donor_image.size) is not None:
                    continue
                row = self._commit(
                    compose(donor_image, target_image, rois_s, rois_t), donor, round_id, target_id, rows
                )
                break
            if row is None:
                rejects["no_valid_donor"] += 1
            else:
                rows.append(row)
        created = sum(row["round_id"] == round_id for row in rows)
        if self.au_cache is not None and created:
            extractor = Path(__file__).resolve().parents[3] / "scripts" / "extract_au_cache.py"
            subprocess.run(
                [
                    str(self.au_python),
                    str(extractor),
                    "--manifest",
                    str(self.manifest_path),
                    "--output",
                    str(self.au_cache),
                ],
                check=True,
            )
        counts_after = self._counts(rows)
        completed = {
            **transaction,
            "status": "completed",
            "gate_count": len(gated),
            "created": created,
            "rejects": rejects,
            "counts_after": counts_after,
        }
        write_json(self._transaction_path(round_id), completed)
        LOGGER.info(
            "STR round %d: class=%s gate=%d created=%d counts_before=%s counts_after=%s rejects=%s",
            round_id,
            CLASSES[selected],
            len(gated),
            created,
            transaction.get("counts_before"),
            counts_after,
            rejects,
        )
        return completed

    def _commit(self, image, donor, round_id, target_id, rows):
        sample_id = str_sample_id(round_id, target_id)
        final = self.images_dir / f"{sample_id}.png"
        temporary = final.with_name(final.name + ".tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, final)
        row = {
            "sample_id": sample_id,
            "dataset": "raf_db",
            "split": "train",
            "label": donor["label"],
            "class_name": donor["class_name"],
            "group_id": donor["group_id"],
            "usable": True,
            "processed_path": f"images/{sample_id}.png",
            "bbox": donor["bbox"],
            "landmarks_crop": donor["landmarks_crop"],
            "landmarks_original": donor["landmarks_original"],
            "round_id": round_id,
            "donor_id": donor["sample_id"],
            "target_id": target_id,
            "landmark_source": "donor_inherited",
        }
        write_jsonl(self.manifest_path, [*rows, row])
        return row

    def rebuild_source_loader(self, transaction, *, source_transform=None):
        round_id = transaction["round_id"]
        rows = self._validate_manifest()
        if any(row["round_id"] > round_id for row in rows):
            raise ValueError("STR manifest contains records from a later round")
        if (
            transaction.get("status") == "completed"
            and "counts_after" in transaction
            and self._counts(rows) != transaction["counts_after"]
        ):
            raise ValueError("STR manifest class counts differ from completed transaction")
        return build_source_loader(
            [self.source_manifest, self.manifest_path],
            batch_size=self.batch_size,
            drop_last=True,
            seed=self.global_seed + round_id,
            num_workers=self.num_workers,
            transform=source_transform,
            au_cache=self.au_cache,
        )
