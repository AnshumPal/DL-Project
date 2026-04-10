"""
Entry point — runs the full benchmark pipeline.

Usage:
    python scripts/run_benchmark.py                    # all models
    python scripts/run_benchmark.py --models resnet50 vit_b16
    python scripts/run_benchmark.py --models resnet50 --epochs 5  # quick test
"""

import sys
import os
import argparse
import yaml
import torch

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset  import get_dataloaders
from src.models   import build_model
from src.train    import train_model
from src.evaluate import get_predictions, compute_metrics, plot_confusion_matrix, plot_learning_curves
from src.gradcam  import visualize_gradcam
from src.benchmark import (
    build_summary_table,
    plot_accuracy_comparison,
    plot_efficiency_scatter,
    plot_training_time_bar,
    measure_gpu_memory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--models",  nargs="+", default=None,
                   help="Subset of models to run. Default: all from config.")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override epochs from config (useful for quick tests).")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate Grad-CAM visualizations after evaluation.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    model_names = args.models or cfg["models"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n  Loading dataset…")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    num_classes = len(class_names)
    print(f"  Classes: {num_classes}  |  "
          f"Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    all_results = []

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  MODEL: {name}")
        print(f"{'='*60}")

        # Build
        model = build_model(name, num_classes).to(device)

        # Measure GPU memory before training
        gpu_mem = measure_gpu_memory(model, train_loader, device)

        # Train
        result = train_model(model, name, train_loader, val_loader, cfg, device)
        result["gpu_mem_mb"] = f"{gpu_mem:.0f}" if gpu_mem else "N/A"

        # Load best checkpoint for evaluation
        ckpt = os.path.join(cfg["paths"]["checkpoints"], f"{name}_best.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))

        # Evaluate on test set
        labels, preds, probs = get_predictions(model, test_loader, device)
        metrics = compute_metrics(labels, preds, probs, class_names)
        result.update(metrics)

        # Save classification report
        report_dir = cfg["paths"]["reports"]
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, f"report_{name}.txt"), "w") as f:
            f.write(metrics["report_str"])

        # Plots
        plot_dir = cfg["paths"]["plots"]
        plot_learning_curves(result["history"], name, plot_dir)
        plot_confusion_matrix(labels, preds, class_names, name, plot_dir)

        # Grad-CAM
        if args.gradcam:
            sample_imgs, sample_labels = next(iter(test_loader))
            visualize_gradcam(model, name, sample_imgs, sample_labels.tolist(),
                              class_names, plot_dir)

        all_results.append(result)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Final benchmark summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")

    report_dir = cfg["paths"]["reports"]
    plot_dir   = cfg["paths"]["plots"]

    build_summary_table(all_results, report_dir)
    plot_accuracy_comparison(all_results, plot_dir)
    plot_efficiency_scatter(all_results, plot_dir)
    plot_training_time_bar(all_results, plot_dir)

    print("\n  All outputs saved to outputs/")


if __name__ == "__main__":
    main()
