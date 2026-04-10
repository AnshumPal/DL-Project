"""
Evaluation: accuracy, top-5 accuracy, macro F1, per-class F1, confusion matrix.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from tqdm import tqdm


# ── Collect predictions ───────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model: nn.Module, loader, device: torch.device):
    """Return (all_labels, all_preds, all_probs) numpy arrays."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="evaluating", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs, class_names: list) -> dict:
    """
    Returns dict with:
      - top1_accuracy
      - top5_accuracy
      - macro_f1
      - weighted_f1
      - per_class_f1  (list)
      - report_str    (full sklearn classification report)
    """
    top1 = (labels == preds).mean()
    top5 = top_k_accuracy_score(labels, probs, k=min(5, probs.shape[1]))
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class   = f1_score(labels, preds, average=None,       zero_division=0)

    report = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0
    )

    return {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "macro_f1":      float(macro_f1),
        "weighted_f1":   float(weighted_f1),
        "per_class_f1":  per_class.tolist(),
        "report_str":    report,
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names: list, model_name: str, out_dir: str):
    cm = confusion_matrix(labels, preds)
    # Normalize to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(10, len(class_names) - 2)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.3, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name} (%)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    path = Path(out_dir) / f"confusion_matrix_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── Learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(history: dict, model_name: str, out_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Learning Curves — {model_name}", fontweight="bold")

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2980b9")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#e74c3c", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#27ae60")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#e67e22", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    path = Path(out_dir) / f"learning_curves_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
