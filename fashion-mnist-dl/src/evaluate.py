# phase5_evaluation.py — Evaluation & Analysis

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.config import CLASS_NAMES, PLOT_SAVE_PATH, REPORT_SAVE_PATH
from src.utils import save_fig, ensure_dirs, print_section


# ── Helper: get integer predictions from a model ─────────────────────────────

def _predict_classes(model, x_test):
    """Return argmax predictions as a 1-D integer array."""
    probs = model.predict(x_test, verbose=0)
    return np.argmax(probs, axis=1)


# ── 1. Model comparison bar chart ────────────────────────────────────────────

def plot_model_comparison_bar(results_df):
    """
    Seaborn barplot of test accuracy for all four models.

    - Each bar gets a distinct colour from the tab10 palette.
    - Value labels formatted as percentages sit above each bar.
    - A horizontal dashed line marks the 90 % threshold.

    Saved to PLOT_SAVE_PATH/model_comparison.png
    """
    ensure_dirs(PLOT_SAVE_PATH)

    models   = results_df['Model'].tolist()
    accs     = results_df['Test Accuracy'].tolist()
    palette  = sns.color_palette('tab10', len(models))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(models, accs, color=palette, width=0.5, edgecolor='white', zorder=3)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{acc * 100:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    # 90 % threshold line
    ax.axhline(y=0.90, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2)
    ax.text(
        len(models) - 0.5, 0.902,
        '90% threshold',
        color='#e74c3c', fontsize=9, va='bottom', ha='right',
    )

    ax.set_ylim(0.80, min(max(accs) + 0.06, 1.0))
    ax.set_title('Test Accuracy Comparison — All Models', fontweight='bold', fontsize=13)
    ax.set_ylabel('Test Accuracy')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.35, zorder=1)

    out_path = os.path.join(PLOT_SAVE_PATH, 'model_comparison.png')
    save_fig(out_path)


# ── 2. Learning curves — 2×2 grid ────────────────────────────────────────────

def plot_learning_curves(results_dict):
    """
    2×2 subplot grid — one panel per model.
    Each panel shows train accuracy (blue solid) and val accuracy (orange dashed).
    All panels share the same y-axis limits [0.7, 1.0] for direct comparison.

    Saved to PLOT_SAVE_PATH/learning_curves.png
    """
    ensure_dirs(PLOT_SAVE_PATH)

    names = list(results_dict.keys())          # preserves insertion order
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Learning Curves — Train vs Validation Accuracy',
                 fontsize=14, fontweight='bold')

    for ax, name in zip(axes.flat, names):
        h  = results_dict[name]['history'].history
        ep = range(1, len(h['accuracy']) + 1)

        ax.plot(ep, h['accuracy'],     color='#2980b9', linestyle='-',
                linewidth=1.8, label='Train Accuracy')
        ax.plot(ep, h['val_accuracy'], color='#e67e22', linestyle='--',
                linewidth=1.8, label='Val Accuracy')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.70, 1.00)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

    out_path = os.path.join(PLOT_SAVE_PATH, 'learning_curves.png')
    save_fig(out_path)


# ── 3. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(model, x_test, y_test_raw):
    """
    Compute and plot the confusion matrix for the given model.

    Parameters
    ----------
    model      : trained Keras model
    x_test     : normalised + reshaped test images  (N, 28, 28, 1)
    y_test_raw : integer class labels               (N,)

    Saved to PLOT_SAVE_PATH/confusion_matrix.png
    """
    ensure_dirs(PLOT_SAVE_PATH)

    y_pred = _predict_classes(model, x_test)
    cm     = confusion_matrix(y_test_raw, y_pred)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.4,
        linecolor='#cccccc',
        ax=ax,
    )

    ax.set_title('Confusion Matrix — Regularized CNN', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)

    out_path = os.path.join(PLOT_SAVE_PATH, 'confusion_matrix.png')
    save_fig(out_path)


# ── 4. Classification report ──────────────────────────────────────────────────

def print_classification_report(model, x_test, y_test_raw):
    """
    Print sklearn classification_report to stdout and save as a text file.

    Columns: precision, recall, F1-score, support — one row per class.

    Saved to REPORT_SAVE_PATH/classification_report.txt
    """
    ensure_dirs(REPORT_SAVE_PATH)

    y_pred  = _predict_classes(model, x_test)
    report  = classification_report(
        y_test_raw, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    header = (
        "Classification Report — Regularized CNN\n"
        + "=" * 60 + "\n"
    )
    full_report = header + report

    print("\n" + full_report)

    out_path = os.path.join(REPORT_SAVE_PATH, 'classification_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"  [saved] {out_path}")


# ── 5. Misclassified image grid ───────────────────────────────────────────────

def visualize_misclassified(model, x_test, y_test_raw):
    """
    Randomly sample 15 misclassified images and display them in a 3×5 grid.
    Each cell title shows 'True: X\\nPred: Y' in red.

    Parameters
    ----------
    model      : trained Keras model
    x_test     : normalised + reshaped test images  (N, 28, 28, 1)
    y_test_raw : integer class labels               (N,)

    Saved to PLOT_SAVE_PATH/misclassified.png
    """
    ensure_dirs(PLOT_SAVE_PATH)

    y_pred = _predict_classes(model, x_test)
    wrong  = np.where(y_test_raw != y_pred)[0]

    rng     = np.random.default_rng(42)
    sample  = rng.choice(wrong, size=min(15, len(wrong)), replace=False)

    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 8))
    fig.suptitle(
        f'Misclassified Samples (15 of {len(wrong):,} errors)',
        fontsize=13, fontweight='bold',
    )

    for ax, idx in zip(axes.flat, sample):
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray', interpolation='nearest')
        ax.set_title(
            f"True:  {CLASS_NAMES[y_test_raw[idx]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[idx]]}",
            fontsize=8, color='#e74c3c', pad=4,
        )
        ax.axis('off')

    # Hide any unused axes if fewer than 15 wrong samples exist
    for ax in axes.flat[len(sample):]:
        ax.axis('off')

    out_path = os.path.join(PLOT_SAVE_PATH, 'misclassified.png')
    save_fig(out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_phase5(results_dict, results_df, x_test, y_test, y_test_raw):
    """
    Run all Phase 5 evaluation steps using the best model.

    Parameters
    ----------
    results_dict : dict  {name: {'model', 'history', 'time'}}
    results_df   : pandas DataFrame from save_training_results()
                   must contain columns 'Model' and 'Test Accuracy'
    x_test       : normalised + reshaped test images  (N, 28, 28, 1)
    y_test       : one-hot encoded test labels        (N, 10)
    y_test_raw   : integer class labels               (N,)
    """
    print_section("PHASE 5 — Evaluation & Analysis")

    # Identify best model by test accuracy
    best_row   = results_df.loc[results_df['Test Accuracy'].idxmax()]
    best_name  = best_row['Model']
    best_acc   = best_row['Test Accuracy']
    best_model = results_dict[best_name]['model']

    print(f"\n  Best model : {best_name}  (test accuracy = {best_acc * 100:.2f}%)")

    plot_model_comparison_bar(results_df)
    plot_learning_curves(results_dict)
    plot_confusion_matrix(best_model, x_test, y_test_raw)
    print_classification_report(best_model, x_test, y_test_raw)
    visualize_misclassified(best_model, x_test, y_test_raw)

    print("\n  Phase 5 complete. All outputs saved.")
    return best_model, best_name


if __name__ == "__main__":
    # Standalone smoke-test — runs the full pipeline up to Phase 5
    import pandas as pd
    from tensorflow.keras.datasets import fashion_mnist
    from src.preprocessing import run_phase2
    from src.models import get_all_models
    from src.train import train_all_models, save_training_results

    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = run_phase2(x_tr, y_tr, x_te, y_te)

    models      = get_all_models()
    results     = train_all_models(models, x_train, y_train, x_val, y_val)
    results_df  = save_training_results(results, x_test, y_test)

    run_phase5(results, results_df, x_test, y_test, y_te)
