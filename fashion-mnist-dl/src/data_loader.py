import os
# phase1_data_profiling.py — Data Understanding & Profiling

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist

from src.config import CLASS_NAMES, PLOT_SAVE_PATH, REPORT_SAVE_PATH
from src.utils import save_fig, ensure_dirs, print_section



def load_data():
    """Load Fashion-MNIST, print shapes/dtypes, return all four arrays."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(f"  x_train : shape={x_train.shape}  dtype={x_train.dtype}")
    print(f"  y_train : shape={y_train.shape}  dtype={y_train.dtype}")
    print(f"  x_test  : shape={x_test.shape}   dtype={x_test.dtype}")
    print(f"  y_test  : shape={y_test.shape}   dtype={y_test.dtype}")
    print(f"  Pixel range : [{x_train.min()}, {x_train.max()}]")

    return x_train, y_train, x_test, y_test


# ── Step 2 ────────────────────────────────────────────────────────────────────

def visualize_samples(x_train, y_train):
    """
    5×10 grid: 5 random images per class.
    Rows = random picks, Columns = classes.
    Saved to PLOT_SAVE_PATH/sample_grid.png.
    """
    n_rows, n_cols = 5, 10
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9))
    fig.suptitle("5 Random Samples per Class", fontsize=14, fontweight='bold')

    for col, cls_idx in enumerate(range(n_cols)):
        class_indices = np.where(y_train == cls_idx)[0]
        chosen = rng.choice(class_indices, size=n_rows, replace=False)
        for row, img_idx in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(x_train[img_idx], cmap='gray', interpolation='nearest')
            ax.axis('off')
            if row == 0:
                ax.set_title(CLASS_NAMES[cls_idx], fontsize=8, pad=3)

    save_fig(os.path.join(PLOT_SAVE_PATH, "sample_grid.png"))


# ── Step 3 ────────────────────────────────────────────────────────────────────

def plot_class_distribution(y_train):
    """
    Bar chart of class frequency in y_train.
    Each bar has a distinct colour; count labels sit on top.
    Saved to PLOT_SAVE_PATH/class_distribution.png.
    """
    counts  = np.bincount(y_train)
    palette = sns.color_palette("tab10", len(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=palette, width=0.6, edgecolor='white')

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            str(cnt),
            ha='center', va='bottom', fontsize=9
        )

    ax.set_title("Class Distribution — Training Set", fontweight='bold')
    ax.set_ylabel("Sample Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    save_fig(os.path.join(PLOT_SAVE_PATH, "class_distribution.png"))
    print("  All 10 classes are perfectly balanced (6,000 samples each).")


# ── Step 4 ────────────────────────────────────────────────────────────────────

def plot_pixel_histogram(x_train):
    """
    Histogram of all pixel values across the training set (50 bins).
    Vertical dashed lines mark the mean and median.
    Saved to PLOT_SAVE_PATH/pixel_histogram.png.
    """
    pixels = x_train.flatten().astype(np.float32)
    mean_val   = pixels.mean()
    median_val = np.median(pixels)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pixels, bins=50, color='#3498db', alpha=0.85, edgecolor='white', density=True)

    ax.axvline(mean_val,   color='#e74c3c', linestyle='--', linewidth=1.8,
               label=f'Mean   = {mean_val:.1f}')
    ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=1.8,
               label=f'Median = {median_val:.1f}')

    ax.set_title("Pixel Intensity Distribution — All Training Images", fontweight='bold')
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    save_fig(os.path.join(PLOT_SAVE_PATH, "pixel_histogram.png"))
    print(f"  Pixel mean={mean_val:.2f}  median={median_val:.2f}")


# ── Step 5 ────────────────────────────────────────────────────────────────────

def plot_mean_images(x_train, y_train):
    """
    Pixel-wise mean image for each class displayed in a 2×5 grid.
    Saved to PLOT_SAVE_PATH/mean_images.png.
    """
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.suptitle("Mean Image per Class", fontsize=14, fontweight='bold')

    for cls_idx, ax in enumerate(axes.flat):
        mean_img = x_train[y_train == cls_idx].mean(axis=0)
        im = ax.imshow(mean_img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(CLASS_NAMES[cls_idx], fontsize=9)
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Mean Pixel Value')
    save_fig(os.path.join(PLOT_SAVE_PATH, "mean_images.png"))


# ── Step 6 ────────────────────────────────────────────────────────────────────

def generate_profiling_report(x_train, y_train):
    """
    Flatten x_train → (60000, 784), sample 2000 rows, build a DataFrame,
    add a 'label' column, run ydata-profiling (minimal=True), save HTML.
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("  ydata-profiling not installed — run: pip install ydata-profiling")
        return

    ensure_dirs(REPORT_SAVE_PATH)

    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(x_train), size=2000, replace=False)
    flat    = x_train[idx].reshape(2000, -1).astype(np.float32)
    columns = [f"px_{i}" for i in range(flat.shape[1])]

    df = pd.DataFrame(flat, columns=columns)
    df['label'] = [CLASS_NAMES[l] for l in y_train[idx]]

    print("  Generating ydata-profiling report (minimal=True) — please wait…")
    report = ProfileReport(
        df,
        title="Fashion-MNIST Data Profile",
        minimal=True,
        progress_bar=False,
    )

    out_path = os.path.join(REPORT_SAVE_PATH, "fashion_mnist_profile.html")
    report.to_file(out_path)
    print(f"  [saved] {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_phase1():
    ensure_dirs(PLOT_SAVE_PATH, REPORT_SAVE_PATH)

    x_train, y_train, x_test, y_test = load_data()
    visualize_samples(x_train, y_train)
    plot_class_distribution(y_train)
    plot_pixel_histogram(x_train)
    plot_mean_images(x_train, y_train)
    generate_profiling_report(x_train, y_train)

    print("\n  Phase 1 complete. All plots saved to:", PLOT_SAVE_PATH)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    run_phase1()
