# src/utils.py — Shared helper functions

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dirs(*paths: str) -> None:
    """Create one or more directories (and parents) if they don't exist."""
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def save_fig(path: str, tight: bool = True) -> None:
    """Save the current matplotlib figure to *path*, then close it."""
    ensure_dirs(os.path.dirname(path))
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")


def print_section(title: str) -> None:
    """Print a consistent phase/section banner to stdout."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
