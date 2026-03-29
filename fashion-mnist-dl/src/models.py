# phase3_models.py — Four Model Architectures

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Flatten, Dense,
    Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.config import NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE, L2_LAMBDA
from src.utils import print_section


# ── Shared compile helper ─────────────────────────────────────────────────────

def _compile(model):
    """Compile with Adam + categorical crossentropy + accuracy metric."""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Model 1: Baseline MLP ─────────────────────────────────────────────────────

def build_baseline_mlp():
    """
    Flatten → Dense(256, relu) → Dense(128, relu) → Dense(10, softmax)

    Treats every pixel as an independent feature — no spatial awareness.
    Serves as the performance floor: any CNN should beat this.
    """
    model = Sequential([
        Input(shape=(28, 28, 1), name='input'),
        Flatten(name='flatten'),
        Dense(256, activation='relu',    name='dense_1'),
        Dense(128, activation='relu',    name='dense_2'),
        Dense(NUM_CLASSES, activation='softmax', name='output'),
    ], name='Baseline_MLP')
    return _compile(model)


# ── Model 2: Simple CNN ───────────────────────────────────────────────────────

def build_simple_cnn():
    """
    Conv2D(32,(3,3),relu,same) → MaxPooling2D(2,2)
    → Flatten → Dense(128,relu) → Dense(10,softmax)

    One conv block learns local spatial features (edges, textures).
    MaxPooling halves spatial dimensions, reducing parameters downstream.
    """
    model = Sequential([
        Input(shape=(28, 28, 1), name='input'),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1'),
        MaxPooling2D((2, 2), name='pool_1'),
        Flatten(name='flatten'),
        Dense(128, activation='relu',    name='dense_1'),
        Dense(NUM_CLASSES, activation='softmax', name='output'),
    ], name='Simple_CNN')
    return _compile(model)


# ── Model 3: Deeper CNN (no regularization) ───────────────────────────────────

def build_deeper_cnn():
    """
    Block 1: Conv2D(32) → BN → MaxPool
    Block 2: Conv2D(64) → BN → MaxPool
    → Flatten → Dense(256,relu) → Dense(10,softmax)

    Two conv blocks build a hierarchy: block 1 detects low-level features
    (edges, corners); block 2 combines them into higher-level patterns
    (sleeves, collars). BatchNormalization normalizes activations after
    each conv, reducing internal covariate shift and allowing higher LR.
    No regularization — intentionally prone to overfitting for Phase 4
    comparison.
    """
    model = Sequential([
        Input(shape=(28, 28, 1), name='input'),
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1a'),
        BatchNormalization(name='bn_1a'),
        MaxPooling2D((2, 2), name='pool_1'),
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2a'),
        BatchNormalization(name='bn_2a'),
        MaxPooling2D((2, 2), name='pool_2'),
        # Head
        Flatten(name='flatten'),
        Dense(256, activation='relu',    name='dense_1'),
        Dense(NUM_CLASSES, activation='softmax', name='output'),
    ], name='Deeper_CNN')
    return _compile(model)


# ── Model 4: Regularized CNN ──────────────────────────────────────────────────

def build_regularized_cnn():
    """
    Same topology as Deeper CNN plus:
      - kernel_regularizer=l2(L2_LAMBDA) on both Conv2D layers and Dense(256)
      - Dropout(DROPOUT_RATE) after each MaxPooling2D
      - Dropout(0.5) after Dense(256)

    L2 penalizes large weights in the loss function, discouraging the
    network from relying on any single feature too heavily.
    Dropout randomly zeros DROPOUT_RATE fraction of activations each
    training step, forcing redundant representations and preventing
    co-adaptation of neurons. At inference Dropout is disabled and
    activations are scaled automatically by Keras.
    """
    reg = l2(L2_LAMBDA)

    model = Sequential([
        Input(shape=(28, 28, 1), name='input'),
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=reg, name='conv_1a'),
        BatchNormalization(name='bn_1a'),
        MaxPooling2D((2, 2), name='pool_1'),
        Dropout(DROPOUT_RATE, name='drop_1'),
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=reg, name='conv_2a'),
        BatchNormalization(name='bn_2a'),
        MaxPooling2D((2, 2), name='pool_2'),
        Dropout(DROPOUT_RATE, name='drop_2'),
        # Head
        Flatten(name='flatten'),
        Dense(256, activation='relu', kernel_regularizer=reg, name='dense_1'),
        Dropout(0.5, name='drop_3'),
        Dense(NUM_CLASSES, activation='softmax', name='output'),
    ], name='Regularized_CNN')
    return _compile(model)


# ── Model registry ────────────────────────────────────────────────────────────

def get_all_models():
    """
    Build and return all four models as an ordered dict.
    Keys match the display names used in Phase 5 plots.
    """
    return {
        'Baseline MLP':    build_baseline_mlp(),
        'Simple CNN':      build_simple_cnn(),
        'Deeper CNN':      build_deeper_cnn(),
        'Regularized CNN': build_regularized_cnn(),
    }


# ── Summary printer ───────────────────────────────────────────────────────────

def print_model_summaries(models_dict):
    """
    Print Keras model.summary() for every model in models_dict.
    Also prints total trainable parameter count per model.
    """
    print_section("PHASE 3 — Model Summaries")
    sep = "-" * 60

    for name, model in models_dict.items():
        print(f"\n{sep}")
        print(f"  Model : {name}")
        print(sep)
        model.summary()
        trainable = model.count_params()
        print(f"  Total trainable parameters: {trainable:,}")

    print(f"\n{sep}")
    print("  Parameter count comparison:")
    print(sep)
    for name, model in models_dict.items():
        print(f"  {name:<20}  {model.count_params():>10,} params")


if __name__ == "__main__":
    print_section("PHASE 3 — Model Definitions (standalone check)")
    models = get_all_models()
    print_model_summaries(models)
