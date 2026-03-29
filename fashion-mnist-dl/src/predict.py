# phase6_prediction.py — Live Prediction from a Local Image File

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.config import CLASS_NAMES, PLOT_SAVE_PATH, MODEL_SAVE_PATH
from src.utils import save_fig, ensure_dirs, print_section


# ── 1. Image preprocessing ────────────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Load, convert, resize, normalise, and reshape a single image for inference.

    Pipeline
    --------
    1. PIL.Image.open()       — supports JPEG, PNG, BMP, WEBP, etc.
    2. .convert('L')          — collapse RGB/RGBA to single-channel grayscale.
                                Fashion-MNIST was trained on grayscale; feeding
                                a colour image without this step would produce
                                wrong channel counts and garbage predictions.
    3. .resize((28, 28), LANCZOS) — downsample to the model's expected input
                                resolution. LANCZOS (Lanczos3) is the highest-
                                quality PIL resampling filter; it anti-aliases
                                edges rather than just dropping pixels.
    4. np.array / 255.0       — normalise to [0, 1] float32, matching the
                                preprocessing applied to the training set.
    5. reshape(1, 28, 28, 1)  — add batch dim (1) and channel dim (1) so the
                                array matches the model's expected input shape.

    Parameters
    ----------
    image_path : str — path to any PIL-readable image file

    Returns
    -------
    model_input : np.ndarray, shape (1, 28, 28, 1), dtype float32
    pil_28x28   : PIL.Image (grayscale, 28×28) — the exact pixels the model sees
    """
    img      = Image.open(image_path)
    gray     = img.convert('L')
    resized  = gray.resize((28, 28), Image.LANCZOS)
    arr      = np.array(resized, dtype='float32') / 255.0
    model_input = arr.reshape(1, 28, 28, 1)
    return model_input, resized


# ── 2. Run inference ──────────────────────────────────────────────────────────

def predict_image(model, image_path):
    """
    Preprocess an image and return the top-3 class predictions.

    Uses np.argsort on the probability vector (ascending), then reverses
    with [::-1] to get descending order, and slices [:3] for the top three.
    This is O(n log n) but n=10 so it is negligible; it also avoids the
    ambiguity of np.argmax which only returns the single top index.

    Parameters
    ----------
    model      : trained Keras model
    image_path : str — path to image file

    Returns
    -------
    predictions : list of 3 tuples  [(class_name, confidence_pct), ...]
                  confidence_pct is a float in [0, 100]
    pil_28x28   : PIL.Image — the preprocessed image (passed to visualize)
    """
    model_input, pil_28x28 = preprocess_image(image_path)

    probs    = model.predict(model_input, verbose=0)[0]   # shape (10,)
    top3_idx = np.argsort(probs)[::-1][:3]

    predictions = [
        (CLASS_NAMES[i], float(probs[i]) * 100)
        for i in top3_idx
    ]
    return predictions, pil_28x28


# ── 3. Visualise result ───────────────────────────────────────────────────────

def visualize_prediction(image_path, predictions, pil_28x28):
    """
    Three-panel figure:
      Panel 1 — original image as loaded (full colour / resolution)
      Panel 2 — the 28×28 grayscale image actually fed to the model
      Panel 3 — horizontal bar chart of top-3 confidence scores

    Bar chart design:
      - Top-1 bar: orange  (#e67e22) — draws the eye to the winner
      - Top-2/3 bars: steelblue (#2980b9) — secondary information
      - Percentage labels at the right end of each bar
      - x-axis fixed at [0, 100] so the chart is always to scale

    Prints the top-1 prediction to stdout.
    Saves to PLOT_SAVE_PATH/prediction_result.png.

    Parameters
    ----------
    image_path  : str — used to reload the original image for panel 1
    predictions : list of 3 tuples [(class_name, confidence_pct), ...]
    pil_28x28   : PIL.Image — the preprocessed 28×28 grayscale image
    """
    ensure_dirs(PLOT_SAVE_PATH)

    top1_class, top1_conf = predictions[0]
    print(f"\n  Prediction: {top1_class} ({top1_conf:.1f}%)")

    # Unpack for bar chart (reverse so highest bar is at the top)
    names  = [p[0] for p in predictions][::-1]
    confs  = [p[1] for p in predictions][::-1]
    colors = ['#2980b9', '#2980b9', '#e67e22']   # reversed: top-1 is last → top of chart

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f'Prediction: {top1_class}  ({top1_conf:.1f}% confidence)',
        fontsize=13, fontweight='bold',
    )

    # Panel 1 — original image
    original = Image.open(image_path)
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')

    # Panel 2 — 28×28 model input
    axes[1].imshow(pil_28x28, cmap='gray', interpolation='nearest',
                   vmin=0, vmax=255)
    axes[1].set_title('Model Input (28×28)', fontsize=10)
    axes[1].axis('off')

    # Panel 3 — confidence bar chart
    bars = axes[2].barh(names, confs, color=colors, edgecolor='white', height=0.5)
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel('Confidence (%)')
    axes[2].set_title('Top-3 Predictions', fontsize=10)
    axes[2].grid(axis='x', alpha=0.3)

    for bar, conf in zip(bars, confs):
        axes[2].text(
            min(conf + 1.5, 98),
            bar.get_y() + bar.get_height() / 2,
            f'{conf:.1f}%',
            va='center', ha='left', fontsize=9, fontweight='bold',
        )

    out_path = os.path.join(PLOT_SAVE_PATH, 'prediction_result.png')
    save_fig(out_path)


# ── 4. Demo entry point ───────────────────────────────────────────────────────

def run_prediction_demo(model, image_path):
    """
    Validate the image path, run inference, and visualise the result.

    If the file does not exist, prints a clear error with a suggestion
    rather than raising an unhandled FileNotFoundError.

    Parameters
    ----------
    model      : trained Keras model
    image_path : str — path to a clothing image (JPEG, PNG, etc.)
    """
    print_section("PHASE 6 — Live Prediction Demo")

    if not os.path.isfile(image_path):
        print(f"\n  [ERROR] File not found: '{image_path}'")
        print(
            "\n  Suggestion: Place a clothing photo (JPEG or PNG) in the project\n"
            "  folder and pass its filename as the argument, e.g.:\n\n"
            "      python -m scripts.run_pipeline my_shirt.jpg\n\n"
            "  Tips for best results:\n"
            "    • Use a plain white or black background\n"
            "    • Centre the garment in the frame\n"
            "    • Avoid shadows and strong lighting gradients\n"
            "    • The model was trained on Fashion-MNIST (studio-style images),\n"
            "      so real-world photos with cluttered backgrounds may confuse it."
        )
        return

    predictions, pil_28x28 = predict_image(model, image_path)
    visualize_prediction(image_path, predictions, pil_28x28)

    print("\n  Top-3 predictions:")
    for rank, (cls, conf) in enumerate(predictions, start=1):
        bar = '█' * int(conf / 5)
        print(f"    {rank}. {cls:<15}  {conf:5.1f}%  {bar}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tensorflow as tf

    model_path = os.path.join(MODEL_SAVE_PATH, 'Regularized_CNN_best.keras')

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        print(
            f"\n  [ERROR] Saved model not found at '{model_path}'.\n"
            "  Run main.py first to train and save the model:\n\n"
            "      python main.py\n"
        )
        sys.exit(1)

    print(f"  Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("  Model loaded successfully.")

    # ── Resolve image path ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print(
            "\n  Usage:\n"
            "      python -m scripts.run_pipeline <path_to_image>\n\n"
            "  Examples:\n"
            "      python -m scripts.run_pipeline my_shirt.jpg\n"
            "      python -m scripts.run_pipeline C:/photos/sneaker.png\n\n"
            "  No image path provided — exiting."
        )
        sys.exit(0)

    run_prediction_demo(model, image_path)
