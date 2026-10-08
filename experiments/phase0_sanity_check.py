"""
Phase 0, Day 5: Sanity-check training.

Trains the model architecture (models/model.py) on the FULL dataset
(centralized, no FL, no ABAC yet) just to confirm:
  1. Data loads and encodes correctly
  2. Model builds and trains without errors
  3. Accuracy is in a reasonable range (paper 2 reports ~0.90+)

This is throwaway — its only job is to prove the foundation works
before Phase 1 adds Flower + ABAC on top of it.

Run from the project root:
    python experiments/phase0_sanity_check.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split  # noqa: E402
from data.preprocess import load_raw, encode_features  # noqa: E402
from models.model import build_model  # noqa: E402


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    print(f"Loading dataset from: {data_path}")
    df = load_raw(data_path)

    X, y, input_dim, _ = encode_features(df)
    print(f"Encoded shape: X={X.shape}, y={y.shape}, input_dim={input_dim}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=input_dim)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {acc:.4f}")
    print(f"Final test loss: {loss:.4f}")

    if acc < 0.6:
        print(
            "\nWARNING: accuracy is low. If you're using the synthetic "
            "fallback dataset this is expected (weak signal by design). "
            "Once you swap in the real Kaggle dataset, accuracy should "
            "rise to paper-2-comparable levels."
        )


if __name__ == "__main__":
    main()
