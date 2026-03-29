# Changelog

All notable changes to this project are documented here.
Format: `[Phase] — description`

---

## [Unreleased]

## [0.1.0] — Initial build

### Phase 1 — Data Understanding
- Loaded Fashion-MNIST via `keras.datasets`
- Visualised 5 random samples per class (sample_grid.png)
- Plotted class distribution — confirmed perfect balance (6,000/class)
- Plotted pixel intensity histogram with mean/median markers
- Computed mean image per class (mean_images.png)
- Generated ydata-profiling HTML report (minimal=True, 2,000 sample rows)

### Phase 2 — Preprocessing
- Normalised pixels [0,255] → [0.0,1.0] float32
- Reshaped (N,28,28) → (N,28,28,1) for CNN channel dim
- One-hot encoded labels via `to_categorical`
- Split: 80% train / 10% val / 10% test (tail-slice, no shuffle needed)

### Phase 3 — Model Definitions
- Baseline MLP: Flatten → Dense(256) → Dense(128) → Softmax
- Simple CNN: Conv2D(32) → MaxPool → Dense(128) → Softmax
- Deeper CNN: 2× Conv blocks + BatchNorm → Dense(256) → Softmax
- Regularized CNN: Deeper CNN + Dropout(0.3/0.5) + L2(0.001)

### Phase 4 — Training
- Trained all 4 models with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Saved best weights per model to `models/`
- Generated regularization comparison plot (train vs val loss)
- Saved training_results.csv to `outputs/reports/`

### Phase 5 — Evaluation
- Bar chart comparing all 4 models' test accuracy
- 2×2 learning curves grid (train vs val accuracy)
- Confusion matrix heatmap for best model
- Classification report (precision, recall, F1 per class)
- 15-image misclassification grid with true/pred labels

### Phase 6 — Live Prediction
- Preprocesses any PIL-readable image: grayscale → 28×28 → [0,1]
- Returns top-3 predictions with confidence scores
- 3-panel figure: original image, 28×28 model input, confidence bar chart

### Phase 7 — Conclusion
- Final ranked summary table with Overfits? column
- Structured conclusion report saved to `outputs/reports/conclusion.txt`
- Key findings: Shirt/T-shirt/Pullover hardest to separate at 28×28
- Next steps: augmentation, transfer learning, attention mechanisms, ensemble

### Infrastructure
- Migrated flat scripts into `src/` package structure
- Added `seed_everything()` to `src/utils.py`
- Added `configs/config.yaml` as single source of truth
- Added `tests/` with unit tests for preprocessing, models, utils
- Added `Makefile` with install, train, test, lint, format, clean targets
- Added `setup.py` for `pip install -e .`
- Added `environment.yml` for conda reproducibility
