import hashlib
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
    load_trained_model,
    make_dataloader,
    predict,
    predict_image,
    predict_with_confidence,
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


# ---------------------------------------------------------------------------
# predict_with_confidence: used by Phase 4 error analysis for per-sample
# confidences that Phase 3 itself never persisted.
# ---------------------------------------------------------------------------

def test_predict_with_confidence_matches_predict_and_is_a_probability(tiny_dev_manifest, tiny_tl_cfg, raw_dir):
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    model = build_model(len(CLASSES), tiny_tl_cfg["freeze_backbone"], tiny_tl_cfg["pretrained"], seed=0)

    _train_t, eval_t = build_transforms(tiny_tl_cfg)
    loader = make_dataloader(
        tiny_dev_manifest, raw_dir, class_to_idx, eval_t,
        batch_size=tiny_tl_cfg["training"]["batch_size"], shuffle=False,
        num_workers=tiny_tl_cfg["training"]["num_workers"], seed=0,
    )

    y_true_a, preds_a = predict(model, loader, torch.device("cpu"), idx_to_class)
    y_true_b, preds_b, confidences = predict_with_confidence(model, loader, torch.device("cpu"), idx_to_class)

    np.testing.assert_array_equal(preds_a, preds_b)
    np.testing.assert_array_equal(y_true_a, y_true_b)
    assert ((confidences >= 0.0) & (confidences <= 1.0)).all()
    # 6-class softmax confidence can never be below 1/6 for the argmax class
    assert (confidences >= 1.0 / len(CLASSES) - 1e-6).all()


# ---------------------------------------------------------------------------
# load_trained_model: pure inference on the real Phase 3 checkpoint — must
# not modify the checkpoint file on disk.
# ---------------------------------------------------------------------------

def test_load_trained_model_is_read_only(resnet_checkpoint_path):
    hash_before = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()
    model, classes, checkpoint = load_trained_model(resnet_checkpoint_path)
    hash_after = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()

    assert hash_before == hash_after
    assert classes == CLASSES
    assert not model.training  # loaded in eval() mode


def test_load_trained_model_predicts_on_real_test_sample(resnet_checkpoint_path, split_manifests, raw_dir, config):
    model, classes, _ckpt = load_trained_model(resnet_checkpoint_path)
    tl_cfg = config["transfer_learning"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    sample = split_manifests["test"].head(4)
    _train_t, eval_t = build_transforms(tl_cfg)
    loader = make_dataloader(
        sample, raw_dir, class_to_idx, eval_t,
        batch_size=tl_cfg["training"]["batch_size"], shuffle=False,
        num_workers=tl_cfg["training"]["num_workers"], seed=tl_cfg["seed"],
    )

    y_true, preds, confidences = predict_with_confidence(model, loader, torch.device("cpu"), idx_to_class)
    assert len(preds) == len(sample)
    assert set(preds) <= set(classes)
    assert ((confidences > 0.0) & (confidences <= 1.0)).all()


# ---------------------------------------------------------------------------
# predict_image: single-image inference used by the Streamlit demo (Phase 6).
# ---------------------------------------------------------------------------

def test_predict_image_output_structure(config):
    tl_cfg = config["transfer_learning"]
    model = build_model(len(CLASSES), tl_cfg["freeze_backbone"], tl_cfg["pretrained"], seed=0)
    _train_t, eval_t = build_transforms(tl_cfg)

    from PIL import Image
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))

    result = predict_image(model, img, eval_t, CLASSES)

    assert result["predicted_class"] in CLASSES
    assert result["predicted_idx"] == CLASSES.index(result["predicted_class"])
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["class_probabilities"].keys()) == set(CLASSES)
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["class_probabilities"][result["predicted_class"]] == pytest.approx(result["confidence"])
    assert result["input_tensor"].shape == (1, 3, *tl_cfg["input_size"])


def test_predict_image_handles_non_rgb_input(config):
    """Grayscale (mode 'L') and RGBA uploads must not crash — predict_image
    converts to RGB internally, same as training did (grayscale=False load)."""
    tl_cfg = config["transfer_learning"]
    model = build_model(len(CLASSES), tl_cfg["freeze_backbone"], tl_cfg["pretrained"], seed=0)
    _train_t, eval_t = build_transforms(tl_cfg)

    from PIL import Image
    gray_img = Image.fromarray(np.zeros((200, 200), dtype=np.uint8), mode="L")
    rgba_img = Image.fromarray(np.zeros((200, 200, 4), dtype=np.uint8), mode="RGBA")

    for img in (gray_img, rgba_img):
        result = predict_image(model, img, eval_t, CLASSES)
        assert result["predicted_class"] in CLASSES


def test_predict_image_tensor_reusable_for_gradcam(config):
    """The returned input_tensor must be usable directly by GradCAM without
    re-running the transform — that's the whole point of returning it."""
    from PIL import Image

    from src.gradcam.gradcam import GradCAM

    tl_cfg = config["transfer_learning"]
    model = build_model(len(CLASSES), tl_cfg["freeze_backbone"], tl_cfg["pretrained"], seed=0)
    _train_t, eval_t = build_transforms(tl_cfg)
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))

    result = predict_image(model, img, eval_t, CLASSES)
    with GradCAM(model, model.layer4[-1]) as gc:
        cam_result = gc.generate(result["input_tensor"], class_idx=result["predicted_idx"])

    assert cam_result["class_idx"] == result["predicted_idx"]
    assert cam_result["cam"].shape == (7, 7)


def test_predict_image_on_real_checkpoint_is_read_only(resnet_checkpoint_path, split_manifests, raw_dir, config):
    hash_before = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()

    model, classes, _ckpt = load_trained_model(resnet_checkpoint_path)
    tl_cfg = config["transfer_learning"]
    _train_t, eval_t = build_transforms(tl_cfg)

    sample_row = split_manifests["test"].iloc[0]
    from PIL import Image
    img = Image.open(raw_dir / sample_row["filename"])
    result = predict_image(model, img, eval_t, classes)

    hash_after = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()
    assert hash_before == hash_after
    assert result["predicted_class"] in classes
