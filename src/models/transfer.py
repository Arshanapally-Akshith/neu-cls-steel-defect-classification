"""ResNet18 transfer learning (Phase 3): frozen ImageNet backbone + a
fine-tuned linear classification head.

Model selection strategy: k-fold CV on the pooled train+val ("dev") manifest
picks the number of training epochs (the epoch with the best mean val
f1_macro across folds); the final model is then retrained from scratch on
the full dev set for that many epochs and evaluated exactly once on the
frozen test manifest. See scripts/run_phase3_transfer.py for the orchestration.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.data.torch_dataset import NEUClsDataset
from src.eval.metrics import compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    input_size = tuple(cfg["input_size"])
    mean = cfg["imagenet_mean"]
    std = cfg["imagenet_std"]
    aug = cfg["augmentation"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=tuple(aug["random_resized_crop_scale"])),
        transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if aug["vertical_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(aug["random_rotation_degrees"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def build_model(num_classes: int, freeze_backbone: bool, pretrained: bool, seed: int) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replacing fc after freezing: new layer's params default requires_grad=True.
    set_seed(seed)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_dataloader(manifest: pd.DataFrame, raw_dir: Path, class_to_idx: dict, transform, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    dataset = NEUClsDataset(manifest, raw_dir, class_to_idx, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, generator=generator if shuffle else None,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, idx_to_class: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(idx_to_class[p] for p in preds)
        all_labels.extend(idx_to_class[l] for l in labels.numpy())
    return np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def predict_with_confidence(model: nn.Module, loader: DataLoader, device: torch.device, idx_to_class: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `predict`, but also returns each prediction's softmax confidence
    (the predicted class's probability)."""
    model.eval()
    all_preds = []
    all_labels = []
    all_conf = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = probs.max(dim=1)
        all_preds.extend(idx_to_class[p] for p in preds.cpu().numpy())
        all_labels.extend(idx_to_class[l] for l in labels.numpy())
        all_conf.extend(conf.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_conf)


def load_trained_model(checkpoint_path: Path) -> tuple[nn.Module, list[str], dict]:
    """Load the Phase 3 checkpoint saved by scripts/run_phase3_transfer.py.
    Pure inference — never retrains or otherwise modifies the model.
    `pretrained=False` here skips the ImageNet-weights download entirely,
    since load_state_dict immediately overwrites every parameter (backbone
    included) with the trained checkpoint anyway."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(len(classes), freeze_backbone=True, pretrained=False, seed=0)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes, checkpoint


@dataclass
class FoldResult:
    fold: int
    per_epoch_val_metrics: list = field(default_factory=list)  # list of compute_metrics() dicts, one per epoch


@dataclass
class CVResult:
    folds: list = field(default_factory=list)  # list of FoldResult
    selected_epoch: int = 0  # 1-indexed
    mean_std_by_epoch: dict = field(default_factory=dict)  # {epoch: {metric: {mean, std}}}


def _mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def run_cross_validation(
    dev_manifest: pd.DataFrame,
    raw_dir: Path,
    classes: list[str],
    tl_cfg: dict,
) -> CVResult:
    """Stratified k-fold CV over dev_manifest ONLY (never touches test).

    For each fold: trains a fresh frozen-backbone ResNet18 for
    tl_cfg['training']['max_epochs'] epochs, recording val metrics after
    every epoch. Returns per-fold per-epoch metrics plus the epoch (1..max)
    whose mean val f1_macro across folds is highest — the epoch count to
    use when retraining the final model on the full dev set.
    """
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    cv_cfg = tl_cfg["cv"]
    train_cfg = tl_cfg["training"]
    device = torch.device("cpu")

    dev_manifest = dev_manifest.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=True, random_state=cv_cfg["seed"])

    train_transform, eval_transform = build_transforms(tl_cfg)

    folds: list[FoldResult] = []
    for fold_idx, (train_pos, val_pos) in enumerate(skf.split(dev_manifest["filename"], dev_manifest["class"])):
        fold_seed = tl_cfg["seed"] + fold_idx
        set_seed(fold_seed)

        train_fold_manifest = dev_manifest.iloc[train_pos].reset_index(drop=True)
        val_fold_manifest = dev_manifest.iloc[val_pos].reset_index(drop=True)

        train_loader = make_dataloader(
            train_fold_manifest, raw_dir, class_to_idx, train_transform,
            batch_size=train_cfg["batch_size"], shuffle=True,
            num_workers=train_cfg["num_workers"], seed=fold_seed,
        )
        val_loader = make_dataloader(
            val_fold_manifest, raw_dir, class_to_idx, eval_transform,
            batch_size=train_cfg["batch_size"], shuffle=False,
            num_workers=train_cfg["num_workers"], seed=fold_seed,
        )

        model = build_model(len(classes), tl_cfg["freeze_backbone"], tl_cfg["pretrained"], seed=fold_seed).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        fold_result = FoldResult(fold=fold_idx)
        for epoch in range(train_cfg["max_epochs"]):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            y_true, y_pred = predict(model, val_loader, device, idx_to_class)
            epoch_metrics = compute_metrics(y_true, y_pred, classes)
            fold_result.per_epoch_val_metrics.append(epoch_metrics)

        folds.append(fold_result)

    selection_metric = cv_cfg["selection_metric"]
    max_epochs = train_cfg["max_epochs"]
    mean_std_by_epoch = {}
    for epoch in range(max_epochs):
        epoch_scores = [f.per_epoch_val_metrics[epoch][selection_metric] for f in folds]
        mean_std_by_epoch[epoch + 1] = {
            selection_metric: _mean_std(epoch_scores),
            "accuracy": _mean_std([f.per_epoch_val_metrics[epoch]["accuracy"] for f in folds]),
        }

    best_epoch = max(mean_std_by_epoch, key=lambda e: mean_std_by_epoch[e][selection_metric]["mean"])

    return CVResult(folds=folds, selected_epoch=best_epoch, mean_std_by_epoch=mean_std_by_epoch)


def train_final_model(
    dev_manifest: pd.DataFrame,
    raw_dir: Path,
    classes: list[str],
    tl_cfg: dict,
    epochs: int,
) -> nn.Module:
    """Train a fresh model on the FULL dev set (train + val pooled) for a
    fixed number of epochs (selected via run_cross_validation)."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    train_cfg = tl_cfg["training"]
    device = torch.device("cpu")
    seed = tl_cfg["seed"]
    set_seed(seed)

    train_transform, _eval_transform = build_transforms(tl_cfg)
    train_loader = make_dataloader(
        dev_manifest.reset_index(drop=True), raw_dir, class_to_idx, train_transform,
        batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], seed=seed,
    )

    model = build_model(len(classes), tl_cfg["freeze_backbone"], tl_cfg["pretrained"], seed=seed).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    return model


def evaluate_on_manifest(
    model: nn.Module,
    manifest: pd.DataFrame,
    raw_dir: Path,
    classes: list[str],
    tl_cfg: dict,
) -> dict:
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    _train_transform, eval_transform = build_transforms(tl_cfg)
    device = torch.device("cpu")

    loader = make_dataloader(
        manifest.reset_index(drop=True), raw_dir, class_to_idx, eval_transform,
        batch_size=tl_cfg["training"]["batch_size"], shuffle=False,
        num_workers=tl_cfg["training"]["num_workers"], seed=tl_cfg["seed"],
    )
    y_true, y_pred = predict(model, loader, device, idx_to_class)
    return compute_metrics(y_true, y_pred, classes)
