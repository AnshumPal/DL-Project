"""
Training loop with:
- Mixed precision (AMP) for GPU efficiency
- Early stopping
- LR scheduling (CosineAnnealingLR)
- Warm-up phase: backbone frozen for first 3 epochs, then unfreeze last 30 layers
"""

import time
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .models import unfreeze_last_n_layers, count_parameters


WARMUP_EPOCHS   = 3    # epochs with frozen backbone
UNFREEZE_LAYERS = 30   # layers to unfreeze after warm-up


# ── Single epoch ──────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """Run one train or eval epoch. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(enabled=scaler is not None):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Full training loop ────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Train one model. Returns a results dict with history and timing.

    Fine-tuning strategy
    --------------------
    Epochs 1–WARMUP_EPOCHS : only the new head is trained (backbone frozen).
    Epoch WARMUP_EPOCHS+1  : unfreeze last UNFREEZE_LAYERS backbone layers.
    This prevents destroying pretrained features in early epochs when the
    head weights are still random and gradients are large.
    """
    t_cfg   = cfg["training"]
    out_dir = Path(cfg["paths"]["checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)

    params  = count_parameters(model)
    print(f"\n  {model_name}: {params['trainable']:,} trainable / {params['total']:,} total params")

    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg["label_smoothing"])
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"], eta_min=1e-6)
    scaler    = GradScaler() if t_cfg["mixed_precision"] and device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_ctr  = 0
    t_start       = time.time()

    for epoch in range(1, t_cfg["epochs"] + 1):

        # Unfreeze backbone after warm-up
        if epoch == WARMUP_EPOCHS + 1:
            unfreeze_last_n_layers(model, UNFREEZE_LAYERS)
            # Re-create optimizer to include newly unfrozen params
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=t_cfg["learning_rate"] / 10,   # lower LR for backbone layers
                weight_decay=t_cfg["weight_decay"],
            )
            print(f"  Epoch {epoch}: backbone partially unfrozen")

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:02d}/{t_cfg['epochs']}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= t_cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    duration = time.time() - t_start
    print(f"  Done in {duration:.1f}s  |  best val_acc={best_val_acc:.4f}")

    return {
        "model":        model,
        "model_name":   model_name,
        "history":      history,
        "best_val_acc": best_val_acc,
        "train_time":   duration,
        "params":       params,
    }
