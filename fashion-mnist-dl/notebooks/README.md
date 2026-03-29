# Notebooks

Each notebook maps to one phase of the pipeline. Read them in order —
each one picks up where the previous left off conceptually.

| # | Notebook | Phase | Key output |
|---|----------|-------|------------|
| 01 | `01_data_profiling.ipynb` | Data Understanding | sample_grid, class_distribution, pixel_histogram, mean_images, HTML profile |
| 02 | `02_preprocessing.ipynb` | Preprocessing | Normalised + reshaped arrays, split summary table |
| 03 | `03_model_comparison.ipynb` | Model Definitions | 4 model summaries, parameter counts |
| 04 | `04_regularization_deepdive.ipynb` | Training + Reg. Deep Dive | training_results.csv, regularization_comparison.png |
| 05 | `05_evaluation.ipynb` | Evaluation & Analysis | model_comparison, learning_curves, confusion_matrix, classification_report |
| 06 | `06_live_prediction.ipynb` | Live Prediction | prediction_result.png, top-3 confidence scores |

## Running a single notebook

```bash
# Activate environment first
conda activate fashion-mnist-dl   # or: source venv/bin/activate

jupyter notebook notebooks/01_data_profiling.ipynb
```

## Running the full pipeline without notebooks

```bash
make train
# or directly:
python -m scripts.run_pipeline
```

## First cell pattern (copy into every notebook)

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))  # project root

from src.utils import seed_everything
from src.config import RANDOM_SEED

seed_everything(RANDOM_SEED)   # always first — before any other import
```

## Concept focus

These notebooks are designed to explain *why*, not just *what*.
Every preprocessing step, architecture choice, and regularization
decision is justified in markdown cells. The goal is understanding
the ML flow, not just chasing accuracy numbers.
