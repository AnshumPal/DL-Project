"""
Benchmark: compare all models on accuracy, F1, params, training time, GPU memory.
Generates a summary table and comparison plots.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(results: list, out_dir: str) -> pd.DataFrame:
    """
    results: list of dicts from train_model() + evaluate metrics merged in.
    Each dict must have keys:
      model_name, best_val_acc, top1_accuracy, top5_accuracy,
      macro_f1, train_time, params (dict with total/trainable)
    """
    rows = []
    for r in results:
        rows.append({
            "Model":            r["model_name"],
            "Val Acc":          f"{r['best_val_acc'] * 100:.2f}%",
            "Test Acc (Top-1)": f"{r['top1_accuracy'] * 100:.2f}%",
            "Test Acc (Top-5)": f"{r['top5_accuracy'] * 100:.2f}%",
            "Macro F1":         f"{r['macro_f1']:.4f}",
            "Total Params (M)": f"{r['params']['total'] / 1e6:.1f}",
            "Trainable (M)":    f"{r['params']['trainable'] / 1e6:.1f}",
            "Train Time (s)":   f"{r['train_time']:.0f}",
            "GPU Mem (MB)":     r.get("gpu_mem_mb", "N/A"),
        })

    df = pd.DataFrame(rows)
    path = Path(out_dir) / "benchmark_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  [saved] {path}")
    print(f"\n{df.to_string(index=False)}")
    return df


# ── Comparison plots ──────────────────────────────────────────────────────────

def plot_accuracy_comparison(results: list, out_dir: str):
    names  = [r["model_name"] for r in results]
    top1   = [r["top1_accuracy"] * 100 for r in results]
    top5   = [r["top5_accuracy"] * 100 for r in results]
    f1     = [r["macro_f1"] * 100 for r in results]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))

    w = 0.25
    ax.bar([i - w for i in x], top1, width=w, label="Top-1 Acc (%)", color="#2980b9")
    ax.bar([i     for i in x], top5, width=w, label="Top-5 Acc (%)", color="#27ae60")
    ax.bar([i + w for i in x], f1,   width=w, label="Macro F1 (%)",  color="#e67e22")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy & F1", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    for i, (t1, t5, f) in enumerate(zip(top1, top5, f1)):
        ax.text(i - w, t1 + 0.5, f"{t1:.1f}", ha="center", fontsize=7)
        ax.text(i,     t5 + 0.5, f"{t5:.1f}", ha="center", fontsize=7)
        ax.text(i + w, f  + 0.5, f"{f:.1f}",  ha="center", fontsize=7)

    path = Path(out_dir) / "accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_efficiency_scatter(results: list, out_dir: str):
    """
    Scatter: x = total params (M), y = top-1 accuracy (%).
    Bubble size = training time. Color = model family.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", len(results))

    for i, r in enumerate(results):
        params_m = r["params"]["total"] / 1e6
        acc      = r["top1_accuracy"] * 100
        time_s   = r["train_time"]

        ax.scatter(params_m, acc, s=time_s / 10, color=palette[i],
                   alpha=0.8, edgecolors="white", linewidths=0.8)
        ax.annotate(r["model_name"], (params_m, acc),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.set_xlabel("Total Parameters (M)")
    ax.set_ylabel("Top-1 Test Accuracy (%)")
    ax.set_title("Efficiency Trade-off: Accuracy vs Parameters\n(bubble size = training time)",
                 fontweight="bold")
    ax.grid(alpha=0.3)

    path = Path(out_dir) / "efficiency_scatter.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_training_time_bar(results: list, out_dir: str):
    names = [r["model_name"] for r in results]
    times = [r["train_time"] / 60 for r in results]   # minutes

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(names, times, color=sns.color_palette("tab10", len(names)))
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{t:.1f}m", ha="center", fontsize=8)

    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison", fontweight="bold")
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    path = Path(out_dir) / "training_time.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ── GPU memory measurement ────────────────────────────────────────────────────

def measure_gpu_memory(model, loader, device) -> float:
    """
    Run one forward pass and return peak GPU memory in MB.
    Returns 0 if not on CUDA.
    """
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    imgs, _ = next(iter(loader))
    with torch.no_grad():
        model(imgs.to(device))

    return torch.cuda.max_memory_allocated(device) / 1024 ** 2
