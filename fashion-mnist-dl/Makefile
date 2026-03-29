# Makefile — Fashion-MNIST Deep Learning Project
# Usage: make <target>

PYTHON   = python
PIP      = pip
VENV     = venv
SRC      = src
TESTS    = tests
SCRIPTS  = scripts

.PHONY: help install install-dev train test clean lint format dirs

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Fashion-MNIST DL — available commands"
	@echo "  ──────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make install-dev  Install + dev tools (pytest, black, flake8)"
	@echo "  make dirs         Create output directories"
	@echo "  make train        Run the full pipeline  (scripts/run_pipeline.py)"
	@echo "  make test         Run all unit tests      (pytest tests/ -v)"
	@echo "  make lint         Lint src/ and tests/    (flake8)"
	@echo "  make format       Auto-format             (black)"
	@echo "  make clean        Remove __pycache__, .pyc files"
	@echo "  make clean-outputs Remove outputs/ and models/ (keeps dirs)"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest black flake8

# ── Directories ───────────────────────────────────────────────────────────────
dirs:
	mkdir -p models
	mkdir -p outputs/figures
	mkdir -p outputs/reports
	mkdir -p outputs/logs
	mkdir -p data/raw
	mkdir -p data/processed
	@echo "  Output directories ready."

# ── Pipeline ──────────────────────────────────────────────────────────────────
train: dirs
	$(PYTHON) -m scripts.run_pipeline

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest $(TESTS)/ -v --tb=short

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	flake8 $(SRC)/ $(TESTS)/ --max-line-length=100 --ignore=E501,W503

format:
	black $(SRC)/ $(TESTS)/ scripts/ --line-length=100

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "  Cache files removed."

clean-outputs:
	rm -f outputs/figures/*.png
	rm -f outputs/reports/*.html outputs/reports/*.csv outputs/reports/*.txt
	rm -f outputs/logs/*.csv
	rm -f models/*.h5 models/*.keras
	@echo "  Generated outputs cleared (directories preserved)."
