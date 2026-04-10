"""
All 10 model architectures with transfer learning.
Strategy:
  - Freeze all backbone layers
  - Replace the classifier head with a new head for num_classes
  - Fine-tune: unfreeze last N backbone layers after initial warm-up
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


# ── Head builder ──────────────────────────────────────────────────────────────

def _make_head(in_features: int, num_classes: int, dropout: float = 0.4) -> nn.Sequential:
    """Standard classification head: BN → Dropout → Linear."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(512, num_classes),
    )


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except those in the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False


def unfreeze_last_n_layers(model: nn.Module, n: int) -> None:
    """
    Unfreeze the last n layers of the backbone for fine-tuning.
    Call this after the warm-up phase (e.g. epoch 3).
    """
    layers = list(model.named_parameters())
    for name, param in layers[-n:]:
        param.requires_grad = True


# ── VGG16 ─────────────────────────────────────────────────────────────────────

def build_vgg16(num_classes: int) -> nn.Module:
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    _freeze_backbone(m)
    in_f = m.classifier[0].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── VGG19 ─────────────────────────────────────────────────────────────────────

def build_vgg19(num_classes: int) -> nn.Module:
    m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    _freeze_backbone(m)
    in_f = m.classifier[0].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── ResNet50 ──────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    _freeze_backbone(m)
    in_f = m.fc.in_features
    m.fc = _make_head(in_f, num_classes)
    return m


# ── ResNet101 ─────────────────────────────────────────────────────────────────

def build_resnet101(num_classes: int) -> nn.Module:
    m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    _freeze_backbone(m)
    in_f = m.fc.in_features
    m.fc = _make_head(in_f, num_classes)
    return m


# ── MobileNetV2 ───────────────────────────────────────────────────────────────

def build_mobilenet_v2(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    _freeze_backbone(m)
    in_f = m.classifier[1].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── MobileNetV3 ───────────────────────────────────────────────────────────────

def build_mobilenet_v3(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    _freeze_backbone(m)
    in_f = m.classifier[0].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── EfficientNet-B0 ───────────────────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    _freeze_backbone(m)
    in_f = m.classifier[1].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── EfficientNet-B7 ───────────────────────────────────────────────────────────

def build_efficientnet_b7(num_classes: int) -> nn.Module:
    m = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    _freeze_backbone(m)
    in_f = m.classifier[1].in_features
    m.classifier = _make_head(in_f, num_classes)
    return m


# ── Vision Transformer (ViT-B/16) ─────────────────────────────────────────────

def build_vit_b16(num_classes: int) -> nn.Module:
    # timm gives the cleanest ViT API
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    _freeze_backbone(m)
    in_f = m.num_features
    m.head = _make_head(in_f, num_classes)
    return m


# ── YOLOv8 (classification mode) ─────────────────────────────────────────────

def build_yolov8(num_classes: int) -> nn.Module:
    """
    YOLOv8 in classification mode via ultralytics.
    Returns a wrapper so it fits the same train loop interface.
    """
    from ultralytics import YOLO

    class YOLOv8Classifier(nn.Module):
        def __init__(self, nc: int):
            super().__init__()
            self.yolo = YOLO("yolov8n-cls.pt")   # nano — swap to m/l for more capacity
            # Replace the final linear layer
            self.yolo.model.model[-1].linear = nn.Linear(
                self.yolo.model.model[-1].linear.in_features, nc
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.yolo.model(x)

    return YOLOv8Classifier(num_classes)


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "vgg16":           build_vgg16,
    "vgg19":           build_vgg19,
    "resnet50":        build_resnet50,
    "resnet101":       build_resnet101,
    "mobilenet_v2":    build_mobilenet_v2,
    "mobilenet_v3":    build_mobilenet_v3,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b7": build_efficientnet_b7,
    "vit_b16":         build_vit_b16,
    "yolov8":          build_yolov8,
}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
