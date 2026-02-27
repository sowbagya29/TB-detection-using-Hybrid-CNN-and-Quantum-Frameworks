"""
Grad-CAM for highlighting TB-affected regions (impressive feature).
Works with HybridQuantumTBClassifier (EfficientNet backbone).
"""
from __future__ import annotations

import numpy as np
import torch
import cv2


@torch.no_grad()
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def generate_gradcam_overlay(
    model,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    class_index: int | None = None,
    alpha: float = 0.45,
):
    """
    Returns (heatmap_uint8, overlay_uint8).

    - model: HybridQuantumTBClassifier
    - input_tensor: shape (1,3,H,W), normalized
    - original_rgb: numpy RGB image (H,W,3)
    """
    model.eval()
    device = next(model.parameters()).device
    x = input_tensor.to(device)

    # Pick target layer: last feature block output
    target_layer = model.backbone_features[-1]

    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    logits = model(x)
    if class_index is None:
        class_index = int(torch.argmax(logits, dim=1).item())

    # Backward w.r.t. chosen class
    model.zero_grad(set_to_none=True)
    score = logits[0, class_index]
    score.backward(retain_graph=False)

    # Cleanup hooks
    h1.remove()
    h2.remove()

    acts = activations["value"]  # (1,C,h,w)
    grads = gradients["value"]   # (1,C,h,w)

    # Global average pooling on gradients => weights
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = torch.sum(weights * acts, dim=1)  # (1,h,w)
    cam = torch.relu(cam)
    cam = cam[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize to original image size
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = _to_uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * original_rgb).astype(np.float32)
    overlay = _to_uint8(overlay)
    return heatmap_color, overlay

