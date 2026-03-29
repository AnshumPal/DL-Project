# Fashion-MNIST: End-to-End Deep Learning Pipeline

A concept-driven comparative study of deep learning architectures on Fashion-MNIST —
from raw data profiling to live clothing prediction. The emphasis throughout is on
understanding **why** each decision is made, not just hitting accuracy numbers.

---

## Project structure

```
fashion-mnist-dl/
├── notebooks/          # Phase-by-phase exploratory notebooks
│   └── README.md       # Guide to reading notebooks in order
├── src/                # Importable Python package
│   ├── config.py       # All hyperparameters & paths
│   ├── utils.py        # seed_everything(), save_fig(), print_section()
│   ├── data_loader.py  # Phase 1 — data loading & profiling
│   ├── preprocessing.py# Phase 2 — normalize, reshape, encode, split
│   ├── models.py       # Phase 3 — 4 model architectures
│   ├── train.py        # Phase 4 — training loop & callbacks
│   ├── evaluate.py     # Phase 5 — metrics, confusion matrix, misclassified
│   ├── predict.py      # Phase 6 — live image inference
│   └── conclusion.py   # Phase 7 — summary table & report
├── scripts/
│   └── run_pipeline.py # CLI entry point — runs all 7 phases end-to-end
├── tests/              # Unit tests (pytest)
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_utils.py
├── configs/
│   └── config.yaml     # Single source of truth for all hyperparameters
├── models/             # Saved .keras checkpoints (gitignored)
├── outputs/
│   ├── figures/        # All generated plots (gitignored)
│   ├── reports/        # HTML profile, CSV results, text reports (gitignored)
│   └── logs/           # Training history CSVs (gitignored)
├── data/
│   ├── raw/            # Auto-downloaded by keras (gitignored)
│   └── processed/      # Saved numpy splits if needed (gitignored)
├── Makefile
├── setup.py
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
└── LICENSE
```

---

## Quickstart

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate fashion-mnist-dl
pip install -e .
make train
```

### Option B — pip + venv

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make train
```

---

## Pipeline phases

| Phase | Description | Key outputs |
|-------|-------------|-------------|
| 1 | Data Understanding & Profiling | sample grid, class distribution, pixel histogram, mean images, HTML profile |
| 2 | Preprocessing | normalise → reshape → one-hot encode → train/val/test split |
| 3 | Model Definitions | Baseline MLP, Simple CNN, Deeper CNN, Regularized CNN |
| 4 | Training | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, regularization comparison plot |
| 5 | Evaluation | model comparison bar chart, learning curves, confusion matrix, classification report, misclassified grid |
| 6 | Live Prediction | top-3 predictions with confidence scores from any image file |
| 7 | Conclusion | ranked summary table, written reflection report |

---

## Common commands

```bash
make install      # install dependencies
make test         # run all unit tests
make train        # run full pipeline
make lint         # flake8 on src/ and tests/
make format       # black auto-format
make clean        # remove __pycache__, .pyc
make clean-outputs# clear generated figures/reports/models
```

---

## Key design decisions

**`seed_everything(42)`** is called before any other import in every entry point and
notebook. This fixes Python, NumPy, and TensorFlow random states for reproducible runs.

**`configs/config.yaml`** is the single source of truth. Hyperparameters are never
hardcoded in notebooks or training scripts.

**`src/` is a proper Python package** — `pip install -e .` makes imports like
`from src.models import build_regularized_cnn` work from any notebook without path hacking.

---

## Tech stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | model building & training |
| NumPy, Pandas | data handling |
| Matplotlib, Seaborn | visualization |
| scikit-learn | confusion matrix, classification report |
| ydata-profiling | automated data profiling |
| Pillow | image loading for live prediction |
| pytest | unit testing |

---

## Results summary

*(populated after first run)*

| Model | Test Accuracy | Parameters | Train Time |
|-------|--------------|------------|------------|
| Baseline MLP | — | — | — |
| Simple CNN | — | — | — |
| Deeper CNN | — | — | — |
| Regularized CNN | — | — | — |

---

## License

MIT — see [LICENSE](LICENSE)
