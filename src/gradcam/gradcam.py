"""Grad-CAM (Selvaraju et al., 2017) for the Phase 3 ResNet18 model.

Grad-CAM highlights where the gradient of a class score is concentrated in
the final conv feature map — it is evidence about what the model's decision
was locally sensitive to, NOT proof of causal reasoning, and it says
nothing beyond that gradient signal about the (frozen, generic-ImageNet)
backbone's internal representations. See reports/gradcam_analysis.md for
the full honesty framing this project applies to every finding here.

Works correctly even though the backbone is frozen (requires_grad=False on
every conv/bn weight): `generate()` sets requires_grad=True on the INPUT
tensor, so every intermediate activation still requires grad (autograd's
requires_grad propagates via OR across an op's inputs), and gradients
w.r.t. the target layer's output are captured via a backward hook — no
parameter gradients are needed or computed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> dict:
        """input_tensor: (1, C, H, W), already preprocessed (resized + normalized).

        Returns {"cam": (h', w') array in [0, 1] at the target layer's
        spatial resolution (not yet resized to the input size), "class_idx":
        the class the CAM was computed for, "probability": the model's
        softmax probability for that class}.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self._gradients[0]      # (C, h, w)
        activations = self._activations[0]  # (C, h, w)
        weights = gradients.mean(dim=(1, 2))  # (C,) global-average-pooled gradient (Grad-CAM paper, Eq. 1)

        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return {"cam": cam, "class_idx": class_idx, "probability": float(probs[0, class_idx].item())}

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a [0,1] CAM to `size` = (width, height), e.g. the model's input size."""
    im = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image_rgb: (H,W,3) uint8. cam: (H,W) float in [0,1], already resized to match image_rgb."""
    heatmap = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    blended = image_rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def border_energy_fraction(cam: np.ndarray, border_frac: float = 0.15) -> float:
    """Fraction of a CAM's total activation energy lying in a border ring of
    width `border_frac` (relative to each spatial dimension) around the
    edge of the map.

    A single quantitative signal to accompany — never replace — visual
    inspection: a high fraction means the model's gradient-weighted
    attention concentrates near the image edge rather than the interior,
    which is CONSISTENT WITH (not proof of) a border/background shortcut
    feature. Computed on the CAM at its native (low) resolution, before any
    upsampling, since upsampling doesn't add information and only smooths
    the border boundary.
    """
    h, w = cam.shape
    border_h = max(1, int(round(h * border_frac)))
    border_w = max(1, int(round(w * border_frac)))
    mask = np.zeros_like(cam, dtype=bool)
    mask[:border_h, :] = True
    mask[h - border_h:, :] = True
    mask[:, :border_w] = True
    mask[:, w - border_w:] = True

    total = float(cam.sum())
    if total <= 0:
        return 0.0
    return float(cam[mask].sum() / total)
