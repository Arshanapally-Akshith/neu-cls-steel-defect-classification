import hashlib

import numpy as np
import pytest
import torch

from src.gradcam.gradcam import GradCAM, border_energy_fraction, overlay_heatmap, resize_cam
from src.models.transfer import build_model, build_transforms, load_trained_model


@pytest.fixture(scope="module")
def frozen_model():
    return build_model(num_classes=6, freeze_backbone=True, pretrained=True, seed=0)


# ---------------------------------------------------------------------------
# GradCAM class
# ---------------------------------------------------------------------------

def test_gradcam_generate_shapes_and_range(frozen_model):
    with GradCAM(frozen_model, frozen_model.layer4[-1]) as gc:
        x = torch.randn(1, 3, 224, 224)
        result = gc.generate(x, class_idx=2)

    assert result["class_idx"] == 2
    assert result["cam"].shape == (7, 7)
    assert result["cam"].min() >= 0.0
    assert result["cam"].max() <= 1.0 + 1e-6
    assert 0.0 <= result["probability"] <= 1.0


def test_gradcam_defaults_to_argmax_class(frozen_model):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected_class = frozen_model(x).argmax(dim=1).item()

    with GradCAM(frozen_model, frozen_model.layer4[-1]) as gc:
        result = gc.generate(x, class_idx=None)

    assert result["class_idx"] == expected_class


def test_gradcam_does_not_unfreeze_backbone_params(frozen_model):
    assert not frozen_model.conv1.weight.requires_grad  # sanity: frozen before
    with GradCAM(frozen_model, frozen_model.layer4[-1]) as gc:
        gc.generate(torch.randn(1, 3, 224, 224), class_idx=0)
    assert not frozen_model.conv1.weight.requires_grad  # still frozen after
    assert not frozen_model.layer4[-1].conv2.weight.requires_grad


def test_gradcam_different_target_classes_give_different_cams(frozen_model):
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(frozen_model, frozen_model.layer4[-1]) as gc:
        cam_a = gc.generate(x, class_idx=0)["cam"]
        cam_b = gc.generate(x, class_idx=1)["cam"]
    assert not np.allclose(cam_a, cam_b)


def test_gradcam_close_removes_hooks(frozen_model):
    target = frozen_model.layer4[-1]
    n_fwd_before = len(target._forward_hooks)

    gc = GradCAM(frozen_model, target)
    assert len(target._forward_hooks) == n_fwd_before + 1
    gc.close()
    assert len(target._forward_hooks) == n_fwd_before


# ---------------------------------------------------------------------------
# resize_cam / overlay_heatmap
# ---------------------------------------------------------------------------

def test_resize_cam_shape_and_range():
    cam = np.random.rand(7, 7).astype(np.float32)
    resized = resize_cam(cam, (224, 224))
    assert resized.shape == (224, 224)
    assert resized.min() >= 0.0
    assert resized.max() <= 1.0 + 1e-6


def test_overlay_heatmap_alpha_zero_returns_original():
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    cam = np.random.rand(10, 10).astype(np.float32)
    blended = overlay_heatmap(image, cam, alpha=0.0)
    np.testing.assert_array_equal(blended, image)


def test_overlay_heatmap_output_shape_and_dtype():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    cam = np.ones((16, 16), dtype=np.float32)
    blended = overlay_heatmap(image, cam, alpha=0.5)
    assert blended.shape == (16, 16, 3)
    assert blended.dtype == np.uint8


# ---------------------------------------------------------------------------
# border_energy_fraction
# ---------------------------------------------------------------------------

def test_border_energy_fraction_all_energy_in_center():
    cam = np.zeros((7, 7))
    cam[3, 3] = 1.0  # dead center, 1-cell border ring at 15% -> excludes center
    frac = border_energy_fraction(cam, border_frac=0.15)
    assert frac == 0.0


def test_border_energy_fraction_all_energy_on_edge():
    cam = np.zeros((7, 7))
    cam[0, 0] = 1.0  # corner, inside the border ring
    frac = border_energy_fraction(cam, border_frac=0.15)
    assert frac == 1.0


def test_border_energy_fraction_uniform_cam_matches_ring_area():
    cam = np.ones((10, 10))
    frac = border_energy_fraction(cam, border_frac=0.2)
    # border ring width = round(10*0.2) = 2 cells each side -> interior is 6x6=36 of 100
    expected = 1 - (6 * 6) / (10 * 10)
    assert frac == pytest.approx(expected, abs=1e-9)


def test_border_energy_fraction_all_zero_cam_is_zero():
    cam = np.zeros((7, 7))
    assert border_energy_fraction(cam) == 0.0


# ---------------------------------------------------------------------------
# Integration: real trained checkpoint, real test image. Must be read-only.
# ---------------------------------------------------------------------------

def test_gradcam_on_real_checkpoint_is_read_only(resnet_checkpoint_path, split_manifests, raw_dir, config):
    hash_before = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()

    model, classes, _ckpt = load_trained_model(resnet_checkpoint_path)
    tl_cfg = config["transfer_learning"]
    _train_t, eval_t = build_transforms(tl_cfg)

    from PIL import Image
    from src.data.loader import load_split_images

    sample = split_manifests["test"].head(1)
    images, labels = load_split_images(sample, raw_dir, grayscale=False)
    pil_img = Image.fromarray(images[0])
    input_tensor = eval_t(pil_img).unsqueeze(0)

    with GradCAM(model, model.layer4[-1]) as gc:
        result = gc.generate(input_tensor)

    hash_after = hashlib.sha256(resnet_checkpoint_path.read_bytes()).hexdigest()
    assert hash_before == hash_after
    assert result["cam"].shape == (7, 7)
    assert result["class_idx"] in range(len(classes))
    assert 0.0 <= result["probability"] <= 1.0
