"""
Shared configuration and data preparation for Phase 1+ experiments.

Factored out so phase1_run.py and phase2_run.py use IDENTICAL data,
clients, and poisoning setup — any difference in results between
phases is then guaranteed to come from the actual defense mechanism
being tested, not from an accidental setup drift between scripts.
"""

import os

from sklearn.model_selection import train_test_split

from data.preprocess_ciciot import (
    detect_label_column_from_df, load_and_sample, map_binary_label,
    get_feature_columns, fit_scaler, transform_with_scaler,
)
from clients.client_utils import partition_random, inspect_partitions
from attacks.poisoning import label_flip

N_CLIENTS = 4
N_ROUNDS = 5
EPOCHS_PER_ROUND = 3       # too few steps (e.g. 1) risks majority-class collapse
POISON_CLIENT_INDEX = 2   # client_2 — valid identity, poisoned data
POISON_FRACTION = 0.2      # matches paper 2's 20% label-flip
SEED = 42

# CICIoT2023 settings
MAX_ROWS_PER_RAW_LABEL = 1500   # per unique attack-type/benign label, during sampling
CHUNKSIZE = 200_000               # rows per chunk while scanning the (possibly huge) CSVs


def prepare_data():
    """
    Returns (client_data, test_data, input_dim) where client_data is a
    list of (X, y) tuples, one per simulated client (client_2 already
    poisoned), and test_data is the held-out (X_test, y_test) global
    evaluation set.
    """
    # Points at the FOLDER containing Merged01.csv...Merged63.csv (or
    # however many you have / the synthetic fallback's 3 files).
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ciciot_raw")

    df = load_and_sample(
        data_path,
        max_rows_per_class=MAX_ROWS_PER_RAW_LABEL, chunksize=CHUNKSIZE,
    )
    label_col = detect_label_column_from_df(df)
    df = map_binary_label(df, label_col=label_col)
    feature_cols = get_feature_columns(df, label_col=label_col)
    print(f"\nUsing {len(feature_cols)} numeric feature columns.")

    # held-out global test set, used to evaluate the aggregated model
    df_pool, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["ACTION"]
    )

    # fit the scaler on the TRAIN POOL ONLY (avoids test-set leakage) —
    # important since CICIoT2023's raw features span wildly different
    # numeric scales and anomaly scoring (Phase 2+) is scale-sensitive
    scaler = fit_scaler(df_pool, feature_cols)

    partitions = partition_random(df_pool, n_clients=N_CLIENTS, seed=SEED)
    print("\nBefore poisoning:")
    inspect_partitions(partitions, target_col="ACTION")

    # poison ONLY the designated client's local partition
    partitions[POISON_CLIENT_INDEX] = label_flip(
        partitions[POISON_CLIENT_INDEX], frac=POISON_FRACTION, target_col="ACTION", seed=SEED
    )
    print(f"\nAfter poisoning client_{POISON_CLIENT_INDEX} ({POISON_FRACTION*100:.0f}% label-flip):")
    inspect_partitions(partitions, target_col="ACTION")

    X_test, y_test = transform_with_scaler(df_test, feature_cols, scaler)

    client_data = []
    for part in partitions:
        X, y = transform_with_scaler(part, feature_cols, scaler)
        client_data.append((X, y))

    input_dim = X_test.shape[1]
    return client_data, (X_test, y_test), input_dim
