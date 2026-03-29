# src/config.py — All hyperparameters and project-wide constants
# Single source of truth. Import everywhere via:
#   from src.config import EPOCHS, CLASS_NAMES, ...

# import os

# # ── Reproducibility ───────────────────────────────────────────────────────────
# RANDOM_SEED   = 42

# # ── Training ──────────────────────────────────────────────────────────────────
# EPOCHS        = 20
# BATCH_SIZE    = 64
# LEARNING_RATE = 0.001
# DROPOUT_RATE  = 0.3
# L2_LAMBDA     = 0.001
# VAL_SPLIT     = 0.1
# NUM_CLASSES   = 10

# # ── Dataset ───────────────────────────────────────────────────────────────────
# CLASS_NAMES = [
#     'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#     'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot',
# ]

# # ── Paths (relative to project root) ─────────────────────────────────────────
# MODEL_SAVE_PATH  = os.path.join('models')
# PLOT_SAVE_PATH   = os.path.join('outputs', 'figures')
# REPORT_SAVE_PATH = os.path.join('outputs', 'reports')
# LOG_SAVE_PATH    = os.path.join('outputs', 'logs')



import os 
import yaml 

with open(config_path := os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f
                            )

# Pull Value from yaml 
RANDOM_SEED = config['RANDOM_SEED']
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
LEARNING_RATE = config['LEARNING_RATE']
DROPOUT_RATE = config['DROPOUT_RATE']
L2_LAMBDA = config['L2_LAMBDA']
VAL_SPLIT = config['VAL_SPLIT']
NUM_CLASSES = config['NUM_CLASSES']
CLASS_NAMES = config['CLASS_NAMES']


# paths  
MODEL_SAVE_PATH  = cfg["paths"]["models"]
PLOT_SAVE_PATH   = cfg["paths"]["figures"]
REPORT_SAVE_PATH = cfg["paths"]["reports"]
LOG_SAVE_PATH    = cfg["paths"]["logs"]