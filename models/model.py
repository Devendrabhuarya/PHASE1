"""
Model architecture matching the base paper (Rahman et al., 2025):
4 Dense layers (512 -> 256 -> 128 -> 64), BatchNorm, LeakyReLU(0.1),
Dropout, sigmoid output for binary access-decision classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout


def build_model(input_dim: int):
    """
    Build the FL client model.

    Args:
        input_dim: number of input features (depends on preprocessing/encoding
                    of the dataset — computed dynamically in preprocess.py)

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Dense(512, input_dim=input_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.1),

        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.1),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # quick smoke test: build a model with a dummy input dimension
    m = build_model(input_dim=9)
    m.summary()
