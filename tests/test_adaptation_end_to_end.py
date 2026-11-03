import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from str_helpers import TinyImageModel, make_source, make_target

from cs_aualign_vit.data.manifests import read_jsonl, write_jsonl
from cs_aualign_vit.models import PrototypeBanks


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prototypes():
    return torch.tensor(
        [[1, 0], [0, 1], [1, 1], [-1, 1], [1, -1], [-1, -1], [0.5, 1]],
        dtype=torch.float32,
    )


def test_source_checkpoint_to_adaptation_resume_and_evaluation(tmp_path, monkeypatch, capsys):
    train_adaptation = load_script("train_adaptation")
    evaluate_adaptation = load_script("evaluate_adaptation")
    TinyImageModel.feature_dim = 2
    monkeypatch.setattr(
        train_adaptation.POSTBCSModel,
        "from_source_checkpoint",
        staticmethod(lambda _path: TinyImageModel()),
    )
    monkeypatch.setattr(
        train_adaptation.PrototypeBanks,
        "from_source_checkpoint",
        staticmethod(lambda _path, *, feature_dim: PrototypeBanks(prototypes())),
    )

    source_manifest = make_source(tmp_path / "source", {0: 2, 1: 2})
    target_manifest = make_target(tmp_path / "target", [0, 1, 0, 1])
    evaluation_manifest = make_target(tmp_path / "evaluation", [0, 1])
    evaluation_rows = read_jsonl(evaluation_manifest)
    for row in evaluation_rows:
        row["split"] = "test"
    write_jsonl(evaluation_manifest, evaluation_rows)
    source_checkpoint = tmp_path / "source.pt"
    preprocessing = {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    torch.save({"preprocessing_config": preprocessing}, source_checkpoint)
    config_path = tmp_path / "adaptation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "max_epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.001,
                    "num_workers": 0,
                    "seed": 3,
                    "device": "cpu",
                    "amp": False,
                    "kmeans_interval": 1,
                    "str_epochs": [],
                },
                "loss": {"alpha": 1.0, "beta": 0.5, "gamma": 0.1},
                "adversarial": {"hidden_dim": 4, "dropout": 0.0, "lambda_grl": 1.0},
                "str": {"confidence_threshold": 0.95},
                "preprocessing": preprocessing,
            }
        )
    )
    output = tmp_path / "run"
    train_args = SimpleNamespace(
        source_manifest=source_manifest,
        target_manifest=target_manifest,
        source_checkpoint=source_checkpoint,
        config=config_path,
        output=output,
        resume=None,
    )

    checkpoint = train_adaptation.run(train_args)
    assert "Epoch 1/1 [" in capsys.readouterr().out
    train_args.resume = checkpoint
    assert train_adaptation.run(train_args) == checkpoint
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 1

    result = evaluate_adaptation.run(
        SimpleNamespace(
            manifest=evaluation_manifest,
            checkpoint=checkpoint,
            source_checkpoint=source_checkpoint,
            split="test",
            batch_size=2,
            num_workers=0,
            device="cpu",
            output=output / "evaluation.json",
        )
    )

    assert result["completed_epoch"] == 1
    assert result["samples"] == 2
    assert len(result["confusion_matrix"]) == 7
    assert (output / "evaluation.json").exists()
