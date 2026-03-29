# phase4_training.py — Training Loop, Callbacks & Regularization Deep Dive

import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH, PLOT_SAVE_PATH, REPORT_SAVE_PATH
from src.utils import save_fig, ensure_dirs, print_section


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(model_name):
    """
    Return three standard training callbacks.

    EarlyStopping
        Monitors val_loss. Stops training if it does not improve for 5
        consecutive epochs and restores the weights from the best epoch.
        Prevents wasting compute on epochs that only overfit.

    ModelCheckpoint
        Saves the model weights to disk whenever val_loss improves.
        save_best_only=True means only the single best checkpoint is kept,
        not one file per epoch.

    ReduceLROnPlateau
        Halves the learning rate when val_loss plateaus for 3 epochs.
        Allows the optimizer to take finer steps near a local minimum
        without manually scheduling the LR. min_lr=1e-6 prevents the LR
        from collapsing to zero.
    """
    ensure_dirs(MODEL_SAVE_PATH)
    ckpt_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_best.keras")

    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ── Single model training ─────────────────────────────────────────────────────

def train_model(model, model_name, x_train, y_train, x_val, y_val):
    """
    Train one model and return its history and wall-clock duration.

    Parameters
    ----------
    model       : compiled Keras model
    model_name  : str — used for checkpoint filename and progress output
    x_train, y_train : training data (already normalised + one-hot)
    x_val,   y_val   : validation data

    Returns
    -------
    history  : keras History object  (history.history is a dict of lists)
    duration : float — training time in seconds
    """
    print(f"\n  Training: {model_name}")
    print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  "
          f"Train={len(x_train):,}  Val={len(x_val):,}")

    t_start = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    duration = time.time() - t_start

    best_val_acc  = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    epochs_run    = len(history.history['loss'])

    print(f"\n  ✓ {model_name} done in {duration:.1f}s  |  "
          f"epochs={epochs_run}  |  "
          f"best val_acc={best_val_acc:.4f}  |  "
          f"best val_loss={best_val_loss:.4f}")

    return history, duration


# ── Train all four models ─────────────────────────────────────────────────────

def train_all_models(models_dict, x_train, y_train, x_val, y_val):
    """
    Iterate over models_dict, train each model, collect results.

    Parameters
    ----------
    models_dict : dict  {name: compiled Keras model}
    x_train, y_train, x_val, y_val : preprocessed numpy arrays

    Returns
    -------
    results_dict : dict
        {
          name: {
            'model':   Keras model (weights restored to best epoch),
            'history': keras History object,
            'time':    float (seconds),
          }
        }
    """
    print_section("PHASE 4 — Training All Models")
    results_dict = {}

    for name, model in models_dict.items():
        history, duration = train_model(
            model, name, x_train, y_train, x_val, y_val
        )
        results_dict[name] = {
            'model':   model,
            'history': history,
            'time':    duration,
        }

    print_section("All models trained")
    return results_dict


# ── Persist results to CSV ────────────────────────────────────────────────────

def save_training_results(results_dict, x_test, y_test):
    """
    Evaluate every model on the held-out test set, build a summary
    DataFrame, save it as CSV, and print it.

    Columns
    -------
    Model | Test Accuracy | Test Loss | Parameters | Train Time (s)

    Returns
    -------
    df : pandas DataFrame
    """
    print_section("PHASE 4 — Test-Set Evaluation Summary")
    ensure_dirs(REPORT_SAVE_PATH)

    rows = []
    for name, res in results_dict.items():
        test_loss, test_acc = res['model'].evaluate(x_test, y_test, verbose=0)
        rows.append({
            'Model':          name,
            'Test Accuracy':  round(test_acc,  4),
            'Test Loss':      round(test_loss, 4),
            'Parameters':     res['model'].count_params(),
            'Train Time (s)': round(res['time'], 1),
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(REPORT_SAVE_PATH, 'training_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  [saved] {csv_path}\n")
    print(df.to_string(index=False))

    return df


# ── Regularization comparison plot ───────────────────────────────────────────

def plot_regularization_comparison(results_dict):
    """
    Four-line plot comparing train/val loss for Deeper CNN vs Regularized CNN.

    Lines
    -----
    Deeper CNN      — train loss  (solid blue)
    Deeper CNN      — val loss    (dashed blue)
    Regularized CNN — train loss  (solid green)
    Regularized CNN — val loss    (dashed green)

    A vertical red dashed line marks the epoch where the Deeper CNN's
    val_loss first exceeds its minimum by more than 1 % — the point at
    which overfitting visibly begins.

    Saved to PLOT_SAVE_PATH/regularization_comparison.png
    """
    ensure_dirs(PLOT_SAVE_PATH)

    deeper_hist = results_dict['Deeper CNN']['history'].history
    reg_hist    = results_dict['Regularized CNN']['history'].history

    ep_deeper = range(1, len(deeper_hist['loss']) + 1)
    ep_reg    = range(1, len(reg_hist['loss'])    + 1)

    # ── Detect overfitting epoch ──────────────────────────────────────────────
    val_losses   = np.array(deeper_hist['val_loss'])
    min_val_loss = val_losses.min()
    # First epoch where val_loss rises more than 1 % above its minimum
    overfit_candidates = np.where(val_losses > min_val_loss * 1.01)[0]
    # Only flag it if it occurs after the minimum, not before
    min_epoch = int(np.argmin(val_losses))
    post_min  = overfit_candidates[overfit_candidates > min_epoch]
    overfit_epoch = int(post_min[0]) + 1 if len(post_min) > 0 else None

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    # Deeper CNN
    ax.plot(ep_deeper, deeper_hist['loss'],
            color='#2980b9', linestyle='-',  linewidth=1.8,
            label='Deeper CNN — Train Loss')
    ax.plot(ep_deeper, deeper_hist['val_loss'],
            color='#2980b9', linestyle='--', linewidth=1.8,
            label='Deeper CNN — Val Loss')

    # Regularized CNN
    ax.plot(ep_reg, reg_hist['loss'],
            color='#27ae60', linestyle='-',  linewidth=1.8,
            label='Regularized CNN — Train Loss')
    ax.plot(ep_reg, reg_hist['val_loss'],
            color='#27ae60', linestyle='--', linewidth=1.8,
            label='Regularized CNN — Val Loss')

    # Overfitting marker
    if overfit_epoch is not None:
        ax.axvline(
            x=overfit_epoch,
            color='#e74c3c', linestyle='--', linewidth=1.6,
            label=f'Overfitting begins (epoch {overfit_epoch})',
        )
        y_annot = ax.get_ylim()[1] * 0.92
        ax.annotate(
            'Overfitting\nbegins',
            xy=(overfit_epoch, y_annot),
            xytext=(overfit_epoch + 0.4, y_annot),
            color='#e74c3c',
            fontsize=9,
            fontweight='bold',
            va='top',
        )

    ax.set_title(
        'Regularization Deep Dive: Train vs Val Loss\n'
        'Deeper CNN (no reg) vs Regularized CNN (Dropout + L2)',
        fontweight='bold',
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    out_path = os.path.join(PLOT_SAVE_PATH, 'regularization_comparison.png')
    save_fig(out_path)


if __name__ == "__main__":
    # Standalone smoke-test: train all models on real data and save results
    from tensorflow.keras.datasets import fashion_mnist
    from src.preprocessing import run_phase2
    from src.models import get_all_models

    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = run_phase2(x_tr, y_tr, x_te, y_te)

    models  = get_all_models()
    results = train_all_models(models, x_train, y_train, x_val, y_val)
    save_training_results(results, x_test, y_test)
    plot_regularization_comparison(results)
