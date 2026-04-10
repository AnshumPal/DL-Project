"""
Grad-CAM visualization.
Works with any CNN that has a named convolutional layer.
For ViT, uses attention rollout instead.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage:
        cam = GradCAM(model, target_layer_name="layer4")
        heatmap = cam(input_tensor, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hook(target_layer_name)

    def _hook(self, layer_name: str):
        """Register forward and backward hooks on the target layer."""
        target = dict(self.model.named_modules())[layer_name]

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1].
        If class_idx is None, uses the predicted class.
        """
        self.model.eval()
        x = x.unsqueeze(0).requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam     = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ── Target layer map ──────────────────────────────────────────────────────────

TARGET_LAYERS = {
    "vgg16":           "features.28",
    "vgg19":           "features.35",
    "resnet50":        "layer4.2.conv3",
    "resnet101":       "layer4.2.conv3",
    "mobilenet_v2":    "features.18.conv.2",
    "mobilenet_v3":    "features.16.block.2.0",
    "efficientnet_b0": "features.8.0",
    "efficientnet_b7": "features.8.0",
    # ViT uses attention rollout — handled separately
}


# ── Overlay helper ────────────────────────────────────────────────────────────

def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.
    image_np: (H, W, 3) uint8
    cam:      (h, w) float32 in [0, 1]
    """
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


# ── Visualize a batch ─────────────────────────────────────────────────────────

def visualize_gradcam(
    model: nn.Module,
    model_name: str,
    images: torch.Tensor,
    labels: list,
    class_names: list,
    out_dir: str,
    n: int = 8,
):
    """
    Generate Grad-CAM overlays for n images and save a grid figure.
    Skips ViT (no spatial feature maps).
    """
    if model_name == "vit_b16":
        print(f"  Grad-CAM skipped for {model_name} (ViT — no spatial feature maps)")
        return

    if model_name not in TARGET_LAYERS:
        print(f"  Grad-CAM skipped for {model_name} (no target layer defined)")
        return

    cam_extractor = GradCAM(model, TARGET_LAYERS[model_name])

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))
    fig.suptitle(f"Grad-CAM — {model_name}", fontweight="bold")

    for i in range(min(n, images.size(0))):
        img_t = images[i]

        # Denormalize for display
        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        img_u8 = (img_np * 255).astype(np.uint8)

        cam = cam_extractor(img_t.to(next(model.parameters()).device))
        overlay = overlay_heatmap(img_u8, cam)

        axes[0, i].imshow(img_u8)
        axes[0, i].set_title(class_names[labels[i]], fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM", fontsize=7)
        axes[1, i].axis("off")

    plt.tight_layout()
    path = Path(out_dir) / f"gradcam_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
