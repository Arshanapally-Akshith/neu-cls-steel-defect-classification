from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import StratifiedKFold

from src.data import loader as loader_module
from src.models.transfer import (
    build_model,
    build_transforms,
    evaluate_on_manifest,
    run_cross_validation,
    set_seed,
    train_final_model,
    train_one_epoch,
)

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


# ---------------------------------------------------------------------------
# Model construction: backbone frozen, only fc trainable, seed-reproducible.
# ---------------------------------------------------------------------------

def test_build_model_freezes_backbone_only():
    model = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=0)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_build_model_output_shape():
    model = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=0)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6)


def test_build_model_seed_reproducibility():
    model_a = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=123)
    model_b = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=123)
    torch.testing.assert_close(model_a.fc.weight, model_b.fc.weight)

    model_c = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=456)
    assert not torch.allclose(model_a.fc.weight, model_c.fc.weight)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_build_transforms_output_shape(config):
    tl_cfg = config["transfer_learning"]
    train_t, eval_t = build_transforms(tl_cfg)

    from PIL import Image
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))

    train_out = train_t(img)
    eval_out = eval_t(img)
    assert train_out.shape == (3, *tl_cfg["input_size"])
    assert eval_out.shape == (3, *tl_cfg["input_size"])


# ---------------------------------------------------------------------------
# Training step: loss is finite and the fc weights actually move.
# ---------------------------------------------------------------------------

def test_train_one_epoch_updates_fc_weights():
    model = build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=0)
    weights_before = model.fc.weight.clone().detach()

    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 6, (8,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    assert np.isfinite(avg_loss)
    assert not torch.allclose(weights_before, model.fc.weight)


# ---------------------------------------------------------------------------
# k-fold CV: fold disjointness (same guarantee sklearn's StratifiedKFold
# gives, verified directly the way Phase 1 verifies split disjointness),
# correct shapes, and NO access to the test manifest at any point.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dev_manifest(split_manifests):
    return split_manifests["train"].groupby("class", group_keys=False).head(3).reset_index(drop=True)


@pytest.fixture
def tiny_tl_cfg(config):
    import copy
    cfg = copy.deepcopy(config["transfer_learning"])
    cfg["cv"]["n_splits"] = 2
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["batch_size"] = 4
    return cfg


def test_cv_fold_indices_are_disjoint(tiny_dev_manifest, tiny_tl_cfg):
    skf = StratifiedKFold(n_splits=tiny_tl_cfg["cv"]["n_splits"], shuffle=True, random_state=tiny_tl_cfg["cv"]["seed"])
    all_val_indices = []
    for train_pos, val_pos in skf.split(tiny_dev_manifest["filename"], tiny_dev_manifest["class"]):
        assert set(train_pos).isdisjoint(set(val_pos))
        all_val_indices.extend(val_pos.tolist())
    assert sorted(all_val_indices) == list(range(len(tiny_dev_manifest)))


def test_run_cross_validation_shapes_and_epoch_selection(tiny_dev_manifest, tiny_tl_cfg, raw_dir):
    result = run_cross_validation(tiny_dev_manifest, raw_dir, CLASSES, tiny_tl_cfg)

    assert len(result.folds) == tiny_tl_cfg["cv"]["n_splits"]
    for fold in result.folds:
        assert len(fold.per_epoch_val_metrics) == tiny_tl_cfg["training"]["max_epochs"]
    assert 1 <= result.selected_epoch <= tiny_tl_cfg["training"]["max_epochs"]
    assert set(result.mean_std_by_epoch.keys()) == set(range(1, tiny_tl_cfg["training"]["max_epochs"] + 1))


def test_cross_validation_never_loads_test_manifest(tiny_dev_manifest, tiny_tl_cfg, raw_dir, split_manifests):
    test_filenames = set(split_manifests["test"]["filename"])
    original_load = loader_module.load_split_images
    loaded_filenames = []

    def spy(manifest, raw_dir_arg, grayscale=True):
        loaded_filenames.extend(manifest["filename"].tolist())
        return original_load(manifest, raw_dir_arg, grayscale=grayscale)

    with patch("src.data.torch_dataset.load_split_images", side_effect=spy):
        run_cross_validation(tiny_dev_manifest, raw_dir, CLASSES, tiny_tl_cfg)

    assert set(loaded_filenames).isdisjoint(test_filenames)
    assert set(loaded_filenames) == set(tiny_dev_manifest["filename"])


# ---------------------------------------------------------------------------
# End-to-end smoke test on a small real sample: train_final_model + evaluate.
# ---------------------------------------------------------------------------

def test_train_final_model_and_evaluate_real_sample(tiny_dev_manifest, tiny_tl_cfg, raw_dir, split_manifests):
    test_sample = split_manifests["test"].groupby("class", group_keys=False).head(2).reset_index(drop=True)

    model = train_final_model(tiny_dev_manifest, raw_dir, CLASSES, tiny_tl_cfg, epochs=1)
    metrics = evaluate_on_manifest(model, test_sample, raw_dir, CLASSES, tiny_tl_cfg)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_samples"] == len(test_sample)
    assert set(metrics["per_class"].keys()) == set(CLASSES)


def test_set_seed_makes_torch_rand_reproducible():
    set_seed(7)
    a = torch.rand(5)
    set_seed(7)
    b = torch.rand(5)
    torch.testing.assert_close(a, b)
