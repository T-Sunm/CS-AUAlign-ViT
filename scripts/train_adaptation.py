"""Run POST adaptation from a source-pretraining checkpoint."""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from cs_aualign_vit.data.datasets import FaceDataset, build_loader, build_source_loader
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import AdversarialAlignment, POSTBCSModel, PrototypeBanks
from cs_aualign_vit.training import AdaptationConfig, AdaptationTrainer, evaluate_target
from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/adaptation.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-au-cache", type=Path)
    parser.add_argument("--target-au-cache", type=Path)
    parser.add_argument("--au-python", type=Path, default=Path(".venv-au/bin/python"))
    return parser.parse_args()


def validate_config(config):
    config.setdefault(
        "au_guidance",
        {"mode": "off", "g_scale": 1.0, "lambda_learning_rate": 0.01},
    )
    config.setdefault(
        "au_alignment",
        {
            "enabled": False,
            "mode": "normal",
            "tau_mask": None,
            "n_eff_min": None,
            "q_c": None,
            "epsilon": 1.0e-8,
            "epsilon_weight": 1.0e-8,
            "r_target": 0.1,
            "eta": None,
        },
    )
    sections = (
        "training",
        "loss",
        "adversarial",
        "str",
        "preprocessing",
        "au_guidance",
        "au_alignment",
    )
    if any(not isinstance(config.get(section), dict) for section in sections):
        raise ValueError("Incomplete adaptation configuration")
    config["au_alignment"].setdefault("mode", "normal")
    required = {
        "training": {
            "max_epochs",
            "batch_size",
            "learning_rate",
            "momentum",
            "weight_decay",
            "num_workers",
            "seed",
            "device",
            "amp",
            "kmeans_interval",
            "str_epochs",
        },
        "loss": {"alpha", "beta", "gamma"},
        "adversarial": {"hidden_dim", "dropout", "lambda_grl"},
        "str": {"confidence_threshold"},
        "preprocessing": {"image_size", "mean", "std"},
        "au_guidance": {"mode", "g_scale", "lambda_learning_rate"},
        "au_alignment": {
            "enabled",
            "mode",
            "tau_mask",
            "n_eff_min",
            "q_c",
            "epsilon",
            "epsilon_weight",
            "r_target",
            "eta",
        },
    }
    if any(required[name] - config[name].keys() for name in required):
        raise ValueError("Incomplete adaptation configuration")
    training = config["training"]
    if training["batch_size"] < 1 or training["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if type(training["amp"]) is not bool:
        raise ValueError("training.amp must be true or false")
    guidance = config["au_guidance"]
    if guidance["mode"] not in {"off", "dynamic", "dynamic_shuffled"}:
        raise ValueError("Invalid au_guidance.mode")
    if guidance["g_scale"] <= 0 or guidance["lambda_learning_rate"] <= 0:
        raise ValueError("AU scale and lambda learning rate must be positive")
    alignment = config["au_alignment"]
    if type(alignment["enabled"]) is not bool:
        raise ValueError("au_alignment.enabled must be true or false")
    if alignment["mode"] not in {"normal", "shuffled"}:
        raise ValueError("au_alignment.mode must be normal or shuffled")
    if alignment["epsilon"] <= 0 or alignment["epsilon_weight"] < 0:
        raise ValueError("Invalid AU alignment epsilons")
    if alignment["r_target"] <= 0:
        raise ValueError("au_alignment.r_target must be positive")
    if alignment["enabled"]:
        calibrated = (alignment["tau_mask"], alignment["n_eff_min"], alignment["eta"])
        if any(value is None or not np.isfinite(value) or value < 0 for value in calibrated):
            raise ValueError("Enabled AU alignment requires calibrated thresholds and eta")
        q_c = alignment["q_c"]
        if (
            not isinstance(q_c, list)
            or len(q_c) != 12
            or any(
                not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                for value in q_c
            )
        ):
            raise ValueError("Enabled AU alignment requires 12 non-negative q_c values")
    return config


def resolve_output(output, resume, runs_dir=Path("runs/adaptation")):
    if resume is not None:
        if output is not None and output.resolve() != resume.parent.resolve():
            raise ValueError("--output must be the directory containing --resume")
        return resume.parent
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        return output
    runs_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = runs_dir / f"run-{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def prepare_run_files(output, config, source_au_cache=None, *, resume=False):
    """Persist resolved config and isolate the mutable source AU cache."""
    resolved_config = output / "resolved_config.yaml"
    if resolved_config.exists():
        if yaml.safe_load(resolved_config.read_text()) != config:
            raise ValueError("Resolved adaptation configuration differs from the current run")
    else:
        resolved_config.write_text(yaml.safe_dump(config, sort_keys=False))

    if source_au_cache is None:
        return None
    working_cache = output / "source_au_cache.csv"
    if resume:
        if not working_cache.is_file():
            raise FileNotFoundError(f"Run source AU cache is missing: {working_cache}")
    else:
        if working_cache.exists():
            raise FileExistsError(f"Run source AU cache already exists: {working_cache}")
        shutil.copy2(source_au_cache, working_cache)
    return working_cache


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = validate_config(yaml.safe_load(args.config.read_text()))
    training = config["training"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if training["amp"] and device.type != "cuda":
        raise ValueError("AMP requires a CUDA training device")

    preprocessing = config["preprocessing"]
    guidance = config["au_guidance"]
    alignment = config["au_alignment"]
    guidance_enabled = guidance["mode"] != "off"
    au_enabled = guidance_enabled or alignment["enabled"]
    original_source_au_cache = getattr(args, "source_au_cache", None)
    target_au_cache = getattr(args, "target_au_cache", None)
    au_python = getattr(args, "au_python", Path(".venv-au/bin/python"))
    if au_enabled and (original_source_au_cache is None or target_au_cache is None):
        raise ValueError("AU modes require --source-au-cache and --target-au-cache")
    output = resolve_output(args.output, args.resume)
    source_au_cache = prepare_run_files(
        output,
        config,
        original_source_au_cache if au_enabled else None,
        resume=args.resume is not None,
    )
    LOGGER.info("adaptation output: %s", output)
    source_state = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    if source_state.get("preprocessing_config") != preprocessing:
        raise ValueError("Adaptation preprocessing differs from the source checkpoint")
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "seed": seed,
        **preprocessing,
    }
    source_loader = build_source_loader(
        [args.source_manifest],
        drop_last=True,
        au_cache=source_au_cache if au_enabled else None,
        **loader_options,
    )
    target_loader = build_loader(
        args.target_manifest,
        "target_adaptation",
        drop_last=True,
        au_cache=target_au_cache if au_enabled else None,
        **loader_options,
    )
    target_feature_loader = build_loader(args.target_manifest, "target_features", **loader_options)
    validation_loader = build_loader(
        args.target_manifest, "evaluation", split="validation", **loader_options
    )
    target_sample_ids = [
        sample["sample_id"]
        for sample in FaceDataset(args.target_manifest, "target_features").samples
    ]

    model = (
        POSTBCSModel.from_source_checkpoint(args.source_checkpoint)
        if not au_enabled
        else POSTBCSModel.from_source_checkpoint(
            args.source_checkpoint, au_mode=guidance["mode"], g_scale=guidance["g_scale"]
        )
    )
    prototype_banks = PrototypeBanks.from_source_checkpoint(
        args.source_checkpoint, feature_dim=model.feature_dim
    )
    adversarial = AdversarialAlignment(input_dim=model.feature_dim, **config["adversarial"])
    lambda_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("lambda_target_to_source", "lambda_source_to_target"))
    ]
    lambda_ids = {id(parameter) for parameter in lambda_parameters}
    base_parameters = [
        parameter
        for module in (model, prototype_banks, adversarial)
        for parameter in module.parameters()
        if id(parameter) not in lambda_ids
    ]
    parameter_groups = [{"params": base_parameters}]
    if lambda_parameters:
        parameter_groups.append(
            {
                "params": lambda_parameters,
                "lr": guidance["lambda_learning_rate"],
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.SGD(
        parameter_groups,
        lr=training["learning_rate"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=training["amp"])
    adaptation_config = AdaptationConfig(
        seed=seed,
        max_epochs=training["max_epochs"],
        kmeans_interval=training["kmeans_interval"],
        str_epochs=tuple(training["str_epochs"]),
        amp=training["amp"],
        num_classes=len(CLASSES),
        au_mode=guidance["mode"],
        au_g_scale=guidance["g_scale"],
        au_lambda_learning_rate=guidance["lambda_learning_rate"],
        au_alignment_enabled=alignment["enabled"],
        au_alignment_mode=alignment["mode"],
        au_alignment_tau_mask=alignment["tau_mask"],
        au_alignment_n_eff_min=alignment["n_eff_min"],
        au_alignment_q_c=(
            tuple(alignment["q_c"]) if alignment["q_c"] is not None else None
        ),
        au_alignment_epsilon=alignment["epsilon"],
        au_alignment_epsilon_weight=alignment["epsilon_weight"],
        au_alignment_r_target=alignment["r_target"],
        au_alignment_eta=alignment["eta"],
        **config["loss"],
    )
    resampler = SelfTrainingResampler(
        source_manifest=args.source_manifest,
        target_feature_loader=target_feature_loader,
        prototype_banks=prototype_banks,
        output_dir=output / "str",
        global_seed=seed,
        batch_size=training["batch_size"],
        device=device,
        num_workers=training["num_workers"],
        num_classes=len(CLASSES),
        confidence_threshold=config["str"]["confidence_threshold"],
        au_cache=source_au_cache if au_enabled else None,
        au_python=au_python if au_enabled else None,
    )
    trainer = AdaptationTrainer(
        model=model,
        prototype_banks=prototype_banks,
        adversarial_module=adversarial,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=target_feature_loader,
        target_sample_ids=target_sample_ids,
        device=device,
        checkpoint_path=args.resume or output / "adaptation.pt",
        config=adaptation_config,
        str_service=resampler,
        source_checkpoint=args.source_checkpoint,
        source_au_cache=source_au_cache if au_enabled else None,
        target_au_cache=target_au_cache if au_enabled else None,
    )
    if args.resume:
        trainer.load_checkpoint()

    metrics_path = output / "metrics.jsonl"
    best_validation_accuracy = float("-inf")
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            previous = json.loads(line)
            if previous["epoch"] <= trainer.completed_epoch:
                best_validation_accuracy = max(
                    best_validation_accuracy, previous.get("validation_accuracy", float("-inf"))
                )

    def report_epoch(metrics):
        nonlocal best_validation_accuracy
        validation = evaluate_target(
            model, prototype_banks, validation_loader, device, num_classes=len(CLASSES)
        )
        metrics.update(
            validation_accuracy=validation["accuracy"],
            validation_per_class_accuracy=validation["per_class_accuracy"],
        )
        if guidance_enabled:
            metrics["lambda_target_to_source"] = (
                model.bcs_block1.lambda_target_to_source.detach().cpu().tolist()
            )
            metrics["lambda_source_to_target"] = (
                model.bcs_block1.lambda_source_to_target.detach().cpu().tolist()
            )
        improved = validation["accuracy"] > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation["accuracy"]
            shutil.copy2(trainer.checkpoint_path, output / "best.pt")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        LOGGER.info(
            "completed epoch %d/%d: loss=%.4f val_accuracy=%.4f val_per_class=%s best=%.4f",
            metrics["epoch"],
            training["max_epochs"],
            metrics["loss_total"],
            validation["accuracy"],
            validation["per_class_accuracy"],
            best_validation_accuracy,
        )

    trainer.run(on_epoch_end=report_epoch, progress=True)
    return trainer.checkpoint_path


def main():
    print(run(parse_args()))


if __name__ == "__main__":
    main()
