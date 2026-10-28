import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs_aualign_vit.data.manifests import CLASSES, write_jsonl
from cs_aualign_vit.models import SourcePrototypeClassifier, copy_target_prototypes
from cs_aualign_vit.training import evaluate, load_checkpoint, save_checkpoint, train_epoch
from cs_aualign_vit.training.source_pretraining import stratified_split_indices


def tiny_model():
    return SourcePrototypeClassifier(nn.Sequential(nn.Flatten(), nn.Linear(12, 5)), 5)


def test_model_shapes_and_backward_reach_backbone_and_prototypes():
    model = tiny_model()
    features, logits = model(torch.randn(3, 3, 2, 2))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()

    assert features.shape == (3, 5)
    assert model.source_prototypes.shape == (7, 5) and logits.shape == (3, 7)
    assert model.backbone[1].weight.grad is not None
    assert model.source_prototypes.grad is not None


def test_training_step_updates_backbone_and_prototypes():
    model = tiny_model()
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 2, 2), torch.tensor([0, 1, 2, 3])), batch_size=4
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backbone_before = model.backbone[1].weight.detach().clone()
    prototypes_before = model.source_prototypes.detach().clone()

    metrics = train_epoch(model, loader, optimizer, torch.device("cpu"))
    validation_metrics = evaluate(model, loader, torch.device("cpu"))

    assert metrics.keys() == {"loss", "accuracy"}
    assert not torch.equal(backbone_before, model.backbone[1].weight)
    assert not torch.equal(prototypes_before, model.source_prototypes)
    assert validation_metrics.keys() == {"loss", "accuracy"}

    samples = [
        {"sample_id": f"{label}:{index}", "label": label}
        for label in range(7)
        for index in range(10)
    ]
    train_indices, validation_indices = stratified_split_indices(samples, 0.1, 42)
    assert len(train_indices) == 63 and len(validation_indices) == 7
    assert set(train_indices).isdisjoint(validation_indices)


def test_checkpoint_round_trip_and_independent_target_bank(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    expected_backbone = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    expected_prototypes = model.source_prototypes.detach().clone()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=0,
        model_config={"backbone": "fixture"},
        preprocessing={"image_size": 2},
        training_config={"epochs": 2},
        best_validation_accuracy=0.5,
        epochs_without_improvement=1,
    )
    with torch.no_grad():
        model.source_prototypes.zero_()
        model.backbone[1].weight.zero_()

    checkpoint = load_checkpoint(path, model, optimizer, scheduler, scaler)
    target = copy_target_prototypes(model.source_prototypes)

    assert checkpoint["class_mapping"] == list(CLASSES)
    assert checkpoint["backbone_config"] == {"backbone": "fixture"}
    assert checkpoint["preprocessing_config"] == {"image_size": 2}
    assert checkpoint["training_config"] == {"epochs": 2}
    assert checkpoint["scaler_state_dict"] == {}
    assert checkpoint["best_validation_accuracy"] == 0.5
    assert checkpoint["epochs_without_improvement"] == 1
    assert all(
        torch.equal(value, expected_backbone[name])
        for name, value in model.backbone.state_dict().items()
    )
    assert torch.equal(model.source_prototypes, expected_prototypes)
    assert torch.equal(target, model.source_prototypes)
    assert target.data_ptr() != model.source_prototypes.data_ptr()


def test_raf_source_loader_trains_without_target_data(tmp_path):
    from cs_aualign_vit.data.datasets import build_loader

    crop = tmp_path / "face.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(crop)
    manifest = tmp_path / "processed.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "sample_id": "raf_db:train_1.png",
                "dataset": "raf_db",
                "split": "train",
                "label": 0,
                "class_name": CLASSES[0],
                "group_id": "fixture",
                "usable": True,
                "processed_path": crop.name,
                "landmarks_crop": [[1, 1]] * 5,
                "landmarks_original": [[1, 1]] * 5,
                "bbox": [0, 0, 8, 8],
            }
        ],
    )
    loader = build_loader(
        manifest,
        "source_train",
        batch_size=1,
        image_size=2,
        mean=(0, 0, 0),
        std=(1, 1, 1),
    )
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_epoch(model, loader, optimizer, torch.device("cpu"))

    assert metrics["loss"] > 0
