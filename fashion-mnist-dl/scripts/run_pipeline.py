# scripts/run_pipeline.py — Fashion-MNIST End-to-End Pipeline entry point
# Run: python -m scripts.run_pipeline   OR   make train

import os
import sys
import time
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from src.config import RANDOM_SEED, MODEL_SAVE_PATH, PLOT_SAVE_PATH, REPORT_SAVE_PATH,EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE, L2_LAMBDA, VAL_SPLIT, NUM_CLASSES, CLASS_NAMES
from src.utils  import seed_everything, ensure_dirs, print_section

from src.data_loader   import run_phase1
from src.preprocessing import run_phase2
from src.models        import get_all_models, print_model_summaries
from src.train         import train_all_models, save_training_results, plot_regularization_comparison
from src.evaluate      import run_phase5
from src.predict       import run_prediction_demo
from src.conclusion    import run_phase7

print("=== Config Parameters Loaded ===")
print(f"Random Seed   : {RANDOM_SEED}")
print(f"Epochs        : {EPOCHS}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Model Path    : {MODEL_SAVE_PATH}")
print(f"Plots Path    : {PLOT_SAVE_PATH}")
print(f"Reports Path  : {REPORT_SAVE_PATH}")
print("================================")



def _phase_banner(number: int, title: str) -> None:
    bar = '─' * 60
    print(f"\n{bar}\n  === Phase {number}: {title} ===\n{bar}")


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main() -> None:
    pipeline_start = time.time()

    print_section("Fashion-MNIST — End-to-End Deep Learning Pipeline")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  NumPy      : {np.__version__}")
    
    seed_everything(RANDOM_SEED)
    ensure_dirs(MODEL_SAVE_PATH, PLOT_SAVE_PATH, REPORT_SAVE_PATH)

    _phase_banner(1, "Data Understanding & Profiling")
    t0 = time.time()
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = run_phase1()
    print(f"  Phase 1 done in {_elapsed(time.time() - t0)}")

    _phase_banner(2, "Preprocessing")
    t0 = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = run_phase2(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )
    print(f"  Phase 2 done in {_elapsed(time.time() - t0)}")

    _phase_banner(3, "Building Models")
    t0 = time.time()
    models_dict = get_all_models()
    print_model_summaries(models_dict)
    print(f"  Phase 3 done in {_elapsed(time.time() - t0)}")

    _phase_banner(4, "Training")
    t0 = time.time()
    results_dict = train_all_models(models_dict, x_train, y_train, x_val, y_val)
    results_df   = save_training_results(results_dict, x_test, y_test)
    plot_regularization_comparison(results_dict)
    print(f"  Phase 4 done in {_elapsed(time.time() - t0)}")

    _phase_banner(5, "Evaluation")
    t0 = time.time()
    best_model, best_name = run_phase5(
        results_dict, results_df, x_test, y_test, y_test_raw,
    )
    print(f"  Phase 5 done in {_elapsed(time.time() - t0)}")

    _phase_banner(6, "Live Prediction Demo")
    try:
        image_path = input(
            "\n  Enter path to a clothing image (or press Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        image_path = ''

    if image_path and os.path.isfile(image_path):
        t0 = time.time()
        run_prediction_demo(best_model, image_path)
        print(f"  Phase 6 done in {_elapsed(time.time() - t0)}")
    elif image_path:
        print(f"  [WARNING] File not found: '{image_path}' — skipping.")
    else:
        print("  Skipping live demo.")

    _phase_banner(7, "Conclusion & Reflection")
    t0 = time.time()
    for _, row in results_df.iterrows():
        name = row['Model']
        if name in results_dict:
            h = results_dict[name]['history'].history
            results_df.loc[results_df['Model'] == name, 'Best Train Acc'] = max(h['accuracy'])
            results_df.loc[results_df['Model'] == name, 'Best Val Acc']   = max(h['val_accuracy'])
    run_phase7(results_df)
    print(f"  Phase 7 done in {_elapsed(time.time() - t0)}")

    total = time.time() - pipeline_start
    print(f"\n{'═' * 60}")
    print(f"  Full pipeline complete in {_elapsed(total)}  ({total:.1f}s)")
    print(f"{'═' * 60}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Pipeline stopped by user.")
        sys.exit(0)
    except Exception:
        print("\n" + "═" * 60)
        print("  [ERROR] Pipeline failed:")
        print("═" * 60 + "\n")
        traceback.print_exc()
        sys.exit(1)
