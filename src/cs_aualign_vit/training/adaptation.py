"""End-to-end orchestration for POST adaptation."""

import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.training.prototype_learning import compute_prototype_losses
from cs_aualign_vit.training.target_supervision import (
    TargetSupervision,
    refresh_target_supervision,
)

ADAPTATION_CHECKPOINT_VERSION = 3
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptationConfig:
    max_epochs: int = 256
    kmeans_interval: int = 10
    str_epochs: tuple[int, ...] = tuple(range(25, 251, 25))
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    amp: bool = False
    num_classes: int = 7

    def __post_init__(self):
        if self.max_epochs < 1 or self.kmeans_interval < 1 or self.num_classes < 1:
            raise ValueError("Epoch counts, kmeans_interval and num_classes must be positive")
        if any(epoch < 1 or epoch > self.max_epochs for epoch in self.str_epochs):
            raise ValueError("STR epochs must be inside the adaptation horizon")
        if tuple(sorted(set(self.str_epochs))) != self.str_epochs:
            raise ValueError("STR epochs must be unique and sorted")


class AdaptationTrainer:
    """Own the POST update loop while delegating module internals to Tasks 1-5."""

    def __init__(
        self,
        *,
        model,
        prototype_banks,
        adversarial_module,
        optimizer,
        scheduler,
        source_loader,
        target_loader,
        target_feature_loader,
        target_sample_ids,
        device,
        checkpoint_path,
        config=None,
        scaler=None,
        str_service=None,
        refresh_fn=refresh_target_supervision,
        source_checkpoint=None,
    ):
        self.model = model
        self.prototype_banks = prototype_banks
        self.adversarial_module = adversarial_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.source_loader = source_loader
        source_dataset = source_loader.dataset
        if isinstance(source_dataset, ConcatDataset):
            source_dataset = source_dataset.datasets[0]
        self.source_transform = getattr(source_dataset, "transform", None)
        self.target_loader = target_loader
        self.target_feature_loader = target_feature_loader
        self.target_sample_ids = list(target_sample_ids)
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config or AdaptationConfig()
        self.scaler = scaler
        self.str_service = str_service
        self.refresh_fn = refresh_fn
        self.source_checkpoint = str(source_checkpoint) if source_checkpoint is not None else None
        self.supervision = None
        self.completed_epoch = 0
        self.global_step = 0
        self.boundary_completed = True
        self._source_restored = False
        self._source_generator_state = None

        config = self.config
        if config.amp and (self.device.type != "cuda" or scaler is None):
            raise ValueError("AMP requires a CUDA device and GradScaler")
        self._validate_loader(source_loader, "source")
        self._validate_loader(target_loader, "target")
        self._validate_optimizer()
        if not self.target_sample_ids or len(set(self.target_sample_ids)) != len(
            self.target_sample_ids
        ):
            raise ValueError("target_sample_ids must be a non-empty unique sequence")
        if config.str_epochs and str_service is None:
            raise ValueError("Scheduled STR epochs require a str_service")

        self.model.to(self.device)
        self.prototype_banks.to(self.device)
        self.adversarial_module.to(self.device)

    @staticmethod
    def _validate_loader(loader, domain):
        if not len(loader):
            dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else "unknown"
            batch_size = getattr(loader, "batch_size", "unknown")
            drop_last = getattr(loader, "drop_last", "unknown")
            raise ValueError(
                f"{domain} training loader is empty: samples={dataset_size}, "
                f"batch_size={batch_size}, drop_last={drop_last}"
            )
        if getattr(loader, "drop_last", None) is not True:
            raise ValueError(f"{domain} paired training loader must use drop_last=True")

    def _validate_optimizer(self):
        expected = {
            id(parameter)
            for module in (self.model, self.prototype_banks, self.adversarial_module)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        actual = [id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len(set(actual)):
            raise ValueError("A trainable parameter appears more than once in the optimizer")
        if set(actual) != expected:
            raise ValueError("Optimizer must contain every and only adaptation trainable parameter")

    def refresh_supervision(self, epoch):
        self.supervision = self.refresh_fn(
            self.model,
            self.prototype_banks,
            self.target_feature_loader,
            self.target_sample_ids,
            epoch=epoch,
            num_classes=self.config.num_classes,
            previous=self.supervision,
        )
        return self.supervision

    def train_step(self, source_batch, target_batch):
        if self.supervision is None:
            raise RuntimeError("Target supervision must be refreshed before training")
        source_images, source_labels = source_batch
        target_images, sample_ids = target_batch
        if len(source_images) != len(target_images):
            raise ValueError("Paired source and target batches must have the same size")
        pseudo_labels, trusted_mask = self.supervision.lookup(list(sample_ids))
        source_images = source_images.to(self.device)
        source_labels = source_labels.to(self.device)
        target_images = target_images.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.config.amp
        ):
            features = self.model.forward_features(source_images, target_images)
            losses = compute_prototype_losses(
                features,
                self.prototype_banks,
                source_labels,
                pseudo_labels,
                trusted_mask,
                num_classes=self.config.num_classes,
            )
            adversarial_loss = self.adversarial_module(
                features.source_features, features.target_features
            )
            if not isinstance(adversarial_loss, torch.Tensor) or adversarial_loss.ndim:
                raise TypeError("Adversarial module must return one scalar loss tensor")
            loss_in = losses.l_s + losses.l_t
            loss_cross = (
                self.config.alpha * losses.l_t2s
                + self.config.beta * losses.l_s2t
                + self.config.gamma * adversarial_loss
            )
            total = loss_in + loss_cross

        if self.config.amp:
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            self.optimizer.step()
        self.global_step += 1
        return {
            "loss_total": float(total.detach()),
            "loss_in": float(loss_in.detach()),
            "loss_cross": float(loss_cross.detach()),
            "loss_s": float(losses.l_s.detach()),
            "loss_t": float(losses.l_t.detach()),
            "loss_t2s": float(losses.l_t2s.detach()),
            "loss_s2t": float(losses.l_s2t.detach()),
            "loss_ad": float(adversarial_loss.detach()),
            "num_trusted": losses.num_trusted,
            "trusted_ratio": losses.num_trusted / len(source_images),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **getattr(self.adversarial_module, "last_metrics", {}),
        }

    def train_epoch(self, *, progress=False):
        self.model.train()
        self.prototype_banks.train()
        self.adversarial_module.train()
        source_iterator = iter(self.source_loader)
        totals = {}
        steps = 0
        for target_batch in self.target_loader:
            try:
                source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = iter(self.source_loader)
                source_batch = next(source_iterator)
            metrics = self.train_step(source_batch, target_batch)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            steps += 1
            if progress:
                completed = int(30 * steps / len(self.target_loader))
                bar = "=" * completed + "." * (30 - completed)
                sys.stdout.write(
                    f"\rEpoch {self.completed_epoch + 1}/{self.config.max_epochs} "
                    f"[{bar}] {steps}/{len(self.target_loader)} "
                    f"loss={totals['loss_total'] / steps:.4f}"
                )
                sys.stdout.flush()
        if progress:
            sys.stdout.write("\n")
        return {name: value / steps for name, value in totals.items()} | {"steps": steps}

    def run(self, on_epoch_end=None, *, progress=False):
        if self.supervision is None:
            LOGGER.info("Refreshing target supervision at epoch 0")
            self.refresh_supervision(0)
        self._finish_pending_boundary()
        history = []
        for epoch in range(self.completed_epoch + 1, self.config.max_epochs + 1):
            metrics = self.train_epoch(progress=progress)
            self.scheduler.step()
            self.completed_epoch = epoch
            if epoch % self.config.kmeans_interval == 0 and epoch < self.config.max_epochs:
                self.refresh_supervision(epoch)
            self.boundary_completed = epoch not in self.config.str_epochs
            self.save_checkpoint(boundary_snapshot=epoch in self.config.str_epochs)
            if epoch in self.config.str_epochs:
                self._finish_pending_boundary()
            row = {"epoch": epoch, **metrics}
            history.append(row)
            if on_epoch_end is not None:
                on_epoch_end(row)
        return history

    def _round_id(self, epoch):
        return self.config.str_epochs.index(epoch) + 1

    @staticmethod
    def _transaction_status(transaction):
        if transaction is None:
            return "absent"
        return transaction["status"] if isinstance(transaction, dict) else transaction.status

    @staticmethod
    def _loader_generator_state(loader):
        generator = getattr(loader, "generator", None)
        return generator.get_state() if generator is not None else None

    @staticmethod
    def _restore_loader_generator(loader, state):
        generator = getattr(loader, "generator", None)
        if generator is not None and state is not None:
            generator.set_state(state.cpu())

    def _finish_pending_boundary(self):
        if self.completed_epoch == 0 or self.completed_epoch not in self.config.str_epochs:
            self.boundary_completed = True
            self._restore_source_loader()
            return
        round_id = self._round_id(self.completed_epoch)
        transaction = self.str_service.get_transaction(round_id)
        status = self._transaction_status(transaction)
        if self.boundary_completed:
            if status != "completed":
                raise RuntimeError(
                    "Inconsistent boundary/STR state: "
                    f"epoch={self.completed_epoch}, expected_round_id={round_id}, "
                    f"boundary=completed, transaction={status}"
                )
            self._restore_source_loader()
            return
        if status == "absent":
            transaction = self.str_service.create_transaction(
                round_id=round_id,
                epoch=self.completed_epoch,
                checkpoint_path=str(self._boundary_checkpoint_path()),
            )
            status = self._transaction_status(transaction)
        if status == "in_progress":
            transaction = self.str_service.run(transaction, self.model)
            status = self._transaction_status(transaction)
        if status != "completed":
            raise RuntimeError(
                f"STR round {round_id} for epoch {self.completed_epoch} did not complete"
            )
        self.source_loader = self.str_service.rebuild_source_loader(
            transaction, source_transform=self.source_transform
        )
        self._validate_loader(self.source_loader, "source")
        self.boundary_completed = True
        self.save_checkpoint()

    def _restore_source_loader(self):
        """After resume, rebuild the source loader from the latest completed STR round."""
        if self._source_restored:
            return
        self._source_restored = True
        if self.str_service is not None:
            passed = [e for e in self.config.str_epochs if e <= self.completed_epoch]
            for epoch in reversed(passed):
                transaction = self.str_service.get_transaction(self._round_id(epoch))
                if self._transaction_status(transaction) == "completed":
                    self.source_loader = self.str_service.rebuild_source_loader(
                        transaction, source_transform=self.source_transform
                    )
                    self._validate_loader(self.source_loader, "source")
                    break
        self._restore_loader_generator(self.source_loader, self._source_generator_state)
        self._source_generator_state = None

    def state_dict(self):
        if self.supervision is None:
            raise RuntimeError("Cannot checkpoint before target supervision exists")
        return {
            "schema_version": ADAPTATION_CHECKPOINT_VERSION,
            "completed_epoch": self.completed_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "prototype_state_dict": self.prototype_banks.state_dict(),
            "adversarial_state_dict": self.adversarial_module.state_dict(),
            "adversarial_config": self.adversarial_module.config(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "target_supervision": self.supervision.state_dict(),
            "target_sample_ids": self.target_sample_ids,
            "config": asdict(self.config),
            "class_mapping": list(CLASSES[: self.config.num_classes]),
            "feature_dim": self.prototype_banks.source_prototypes.shape[1],
            "source_checkpoint": self.source_checkpoint,
            "boundary_completed": self.boundary_completed,
            "expected_str_round_id": (
                self._round_id(self.completed_epoch)
                if self.completed_epoch in self.config.str_epochs
                else None
            ),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "source_loader_generator_state": self._loader_generator_state(self.source_loader),
            "target_loader_generator_state": self._loader_generator_state(self.target_loader),
        }

    def _boundary_checkpoint_path(self):
        return self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}.epoch-{self.completed_epoch}.boundary"
            f"{self.checkpoint_path.suffix}"
        )

    @staticmethod
    def _atomic_save(state, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def save_checkpoint(self, *, boundary_snapshot=False):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        if boundary_snapshot:
            # Snapshot first: a crash in between must never leave a transaction without it.
            self._atomic_save(state, self._boundary_checkpoint_path())
        self._atomic_save(state, self.checkpoint_path)

    def load_checkpoint(self):
        paths = [self.checkpoint_path] if self.checkpoint_path.exists() else []
        paths.extend(
            self.checkpoint_path.parent.glob(
                f"{self.checkpoint_path.stem}.epoch-*.boundary{self.checkpoint_path.suffix}"
            )
        )
        if not paths:
            raise FileNotFoundError(f"No adaptation checkpoint found for {self.checkpoint_path}")
        states = [torch.load(path, map_location=self.device, weights_only=True) for path in paths]
        state = max(states, key=lambda candidate: candidate.get("completed_epoch", -1))
        if state.get("schema_version") != ADAPTATION_CHECKPOINT_VERSION:
            raise ValueError("Unsupported adaptation checkpoint schema")
        if state.get("target_sample_ids") != self.target_sample_ids:
            raise ValueError("Saved target sample_ids differ from the adaptation dataset")
        if state.get("config") != asdict(self.config):
            raise ValueError("Saved adaptation configuration differs from the current run")
        if state.get("class_mapping") != list(CLASSES[: self.config.num_classes]):
            raise ValueError("Saved class mapping differs from the current run")
        if state.get("feature_dim") != self.prototype_banks.source_prototypes.shape[1]:
            raise ValueError("Saved feature dimension differs from the current model")
        if state.get("adversarial_config") != self.adversarial_module.config():
            raise ValueError("Saved adversarial configuration differs from the current run")
        self.model.load_state_dict(state["model_state_dict"])
        self.prototype_banks.load_state_dict(state["prototype_state_dict"])
        self.adversarial_module.load_state_dict(state["adversarial_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.scaler is not None and state["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.supervision = TargetSupervision.from_state_dict(
            state["target_supervision"], self.target_sample_ids
        )
        self.completed_epoch = state["completed_epoch"]
        self.global_step = state["global_step"]
        self.boundary_completed = state["boundary_completed"]
        self._source_restored = False
        self._source_generator_state = (
            state["source_loader_generator_state"] if self.boundary_completed else None
        )
        self._restore_loader_generator(self.source_loader, self._source_generator_state)
        self._restore_loader_generator(
            self.target_loader, state["target_loader_generator_state"]
        )
        random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng_state"]])
        return state
