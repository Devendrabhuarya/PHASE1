"""
Partitions the (encoded) dataset across N simulated smart-home
clients/devices, following the same style as the base paper
(partition by ROLE_DEPTNAME to mimic distinct devices/households).
"""

import numpy as np
import pandas as pd


def partition_by_department(df: pd.DataFrame, n_clients: int = 3, seed: int = 42):
    """
    Groups rows by ROLE_DEPTNAME and distributes department-groups
    round-robin across n_clients, so each client gets a mix of
    departments rather than exactly one (keeps class balance saner).

    Returns:
        List[pd.DataFrame] of length n_clients.
    """
    rng = np.random.default_rng(seed)
    depts = df["ROLE_DEPTNAME"].unique()
    rng.shuffle(depts)

    client_dept_map = {i: [] for i in range(n_clients)}
    for i, dept in enumerate(depts):
        client_dept_map[i % n_clients].append(dept)

    partitions = []
    for i in range(n_clients):
        mask = df["ROLE_DEPTNAME"].isin(client_dept_map[i])
        partitions.append(df[mask].reset_index(drop=True))

    return partitions


def partition_random(df: pd.DataFrame, n_clients: int = 3, seed: int = 42):
    """
    Simple random split across n_clients — use this if department-based
    partitioning gives too-imbalanced client sizes. Returns a list of
    DataFrames (uses manual index-chunking rather than np.array_split,
    which does not reliably preserve DataFrame type).
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunk_indices = np.array_split(np.arange(len(df_shuffled)), n_clients)
    return [df_shuffled.iloc[idx].reset_index(drop=True) for idx in chunk_indices]


def inspect_partitions(partitions, target_col: str = "ACTION"):
    """Sanity check: print row count and class balance per client."""
    for i, part in enumerate(partitions):
        n = len(part)
        pos_rate = part[target_col].mean() if n > 0 else float("nan")
        print(f"Client {i}: {n} rows, {target_col}=1 rate: {pos_rate:.3f}")


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.preprocess import load_raw  # noqa: E402

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    df = load_raw(data_path)
    parts = partition_by_department(df, n_clients=3)
    inspect_partitions(parts)
