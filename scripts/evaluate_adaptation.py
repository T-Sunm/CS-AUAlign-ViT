"""Evaluate a completed POST adaptation epoch checkpoint on ExpW validation or test."""

import argparse
import json
from pathlib import Path

import torch

from cs_aualign_vit.data.datasets import build_loader
from cs_aualign_vit.data.manifests import CLASSES, write_json
from cs_aualign_vit.models import POSTBCSModel, PrototypeBanks
from cs_aualign_vit.training import evaluate_target
from cs_aualign_vit.training.adaptation import ADAPTATION_CHECKPOINT_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get("schema_version") != ADAPTATION_CHECKPOINT_VERSION:
        raise ValueError("Unsupported adaptation checkpoint schema")
    if not state.get("boundary_completed"):
        raise ValueError("Evaluation requires a completed adaptation boundary")
    if state.get("class_mapping") != list(CLASSES):
        raise ValueError("Adaptation checkpoint class mapping differs from the data pipeline")
    source_checkpoint = args.source_checkpoint or state.get("source_checkpoint")
    if source_checkpoint is None:
        raise ValueError("A source checkpoint is required to rebuild the adaptation model")
    source_state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    preprocessing = source_state.get("preprocessing_config")
    if preprocessing is None:
        raise ValueError("Source checkpoint is missing preprocessing configuration")
    if not isinstance(preprocessing, dict):
        raise TypeError("Source checkpoint preprocessing configuration must be a dictionary")
    model = POSTBCSModel.from_source_checkpoint(source_checkpoint).to(device)
    prototype_banks = PrototypeBanks.from_source_checkpoint(
        source_checkpoint, feature_dim=model.feature_dim
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    prototype_banks.load_state_dict(state["prototype_state_dict"])
    loader = build_loader(
        args.manifest,
        "evaluation",
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **preprocessing,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "completed_epoch": state["completed_epoch"],
        "split": args.split,
        **evaluate_target(model, prototype_banks, loader, device, num_classes=len(CLASSES)),
    }
    write_json(args.output, result)
    return result


def main():
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
