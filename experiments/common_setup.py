"""
Shared configuration, seeding and data preparation for Phase 1+ experiments.

Used by every phase so differences in results come from the defense being
tested, not from setup drift or random-number luck.

Key points:
  - set_seed(): makes batch order / dropout reproducible. Call at the START
    of every scenario so all scenarios see the same randomness.
  - load_dataset_once(): scans the (possibly huge) CICIoT2023 files ONE time;
    build_clients() then makes per-seed client splits cheaply.
  - compute_global_class_weight(): a FIXED class weight from the clean pool,
    so a poisoned client's flipped labels cannot cancel the attack through
    its own locally-computed class weights.
"""

import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

from data.preprocess_ciciot import (
    detect_label_column_from_df, load_and_sample, map_binary_label,
    get_feature_columns, fit_scaler, transform_with_scaler,
)
from clients.client_utils import partition_random, inspect_partitions
from attacks.poisoning import label_flip

N_CLIENTS = 4
N_ROUNDS = 5
EPOCHS_PER_ROUND = 3
POISON_CLIENT_INDEX = 2   # client_2 - valid identity, poisoned data
POISON_FRACTION = 0.2      # matches paper 2's 20% label-flip
SEED = 42

MAX_ROWS_PER_RAW_LABEL = 1500
CHUNKSIZE = 50_000
EXPECTED_CLASSES = 34     # CICIoT2023 has 34 labels; scan stops early once all are full (harmless otherwise)


def set_seed(seed: int):
    """Seeds python, numpy and TensorFlow (via keras) in one call."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _cached_sample(data_dir: str):
    """
    Scanning 63 big files takes a long time and every script (diagnostic, phase2, phase3)
    would repeat it. The sampled rows are cached on disk; the cache key covers the file
    names, sizes and the sampling settings, so changed data re-scans automatically.
    """
    import glob, hashlib, pandas as pd
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    key = hashlib.md5(repr([(os.path.basename(f), os.path.getsize(f)) for f in files]
                           + [MAX_ROWS_PER_RAW_LABEL]).encode()).hexdigest()[:12]
    cache = os.path.join(os.path.dirname(data_dir.rstrip("/\\")), f"_sample_cache_{key}.pkl")
    if os.path.exists(cache):
        print(f"Using cached sample: {cache}  (delete this file to force a re-scan)")
        return pd.read_pickle(cache)
    df = load_and_sample(data_dir, max_rows_per_class=MAX_ROWS_PER_RAW_LABEL, chunksize=CHUNKSIZE,
                         expected_classes=EXPECTED_CLASSES)
    df.to_pickle(cache)
    print(f"Saved sample cache: {cache}")
    return df


def load_dataset_once(data_dir: str = None, verbose: bool = True):
    """
    Scans the dataset files once. Returns a dict with the train pool
    (DataFrame), held-out test set (DataFrame), feature columns and a
    scaler fit on the pool only.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "ciciot_raw")

    df = _cached_sample(data_dir)
    label_col = detect_label_column_from_df(df)
    df = map_binary_label(df, label_col=label_col)
    feature_cols = get_feature_columns(df, label_col=label_col)
    if verbose:
        print(f"\nUsing {len(feature_cols)} numeric feature columns.")

    df_pool, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["ACTION"])
    scaler = fit_scaler(df_pool, feature_cols)
    return {"pool": df_pool, "test": df_test, "feature_cols": feature_cols, "scaler": scaler}


def compute_global_class_weight(df_pool) -> dict:
    """Balanced class weights from the CLEAN pool labels (before poisoning)."""
    y = df_pool["ACTION"].values
    n = len(y)
    out = {}
    for c in (0, 1):
        n_c = int((y == c).sum())
        out[c] = n / (2 * n_c) if n_c else 1.0
    return out


def build_clients(dataset: dict, seed: int = SEED, poison_fraction: float = POISON_FRACTION,
                  poison: bool = True, verbose: bool = True):
    """
    Splits the pool into N_CLIENTS partitions (split depends on `seed`),
    optionally label-flips client POISON_CLIENT_INDEX, and returns
    (client_data, test_data, input_dim). client_data = [(X, y), ...].
    """
    feature_cols, scaler = dataset["feature_cols"], dataset["scaler"]
    partitions = partition_random(dataset["pool"], n_clients=N_CLIENTS, seed=seed)
    if verbose:
        print("\nBefore poisoning:")
        inspect_partitions(partitions, target_col="ACTION")

    if poison and poison_fraction > 0:
        partitions[POISON_CLIENT_INDEX] = label_flip(
            partitions[POISON_CLIENT_INDEX], frac=poison_fraction, target_col="ACTION", seed=seed
        )
        if verbose:
            print(f"\nAfter poisoning client_{POISON_CLIENT_INDEX} ({poison_fraction*100:.0f}% label-flip):")
            inspect_partitions(partitions, target_col="ACTION")

    X_test, y_test = transform_with_scaler(dataset["test"], feature_cols, scaler)
    client_data = [transform_with_scaler(p, feature_cols, scaler) for p in partitions]
    return client_data, (X_test, y_test), X_test.shape[1]


def prepare_data():
    """Backward-compatible helper used by Phase 1: (client_data, test_data, input_dim)."""
    dataset = load_dataset_once()
    return build_clients(dataset, seed=SEED, poison_fraction=POISON_FRACTION)
