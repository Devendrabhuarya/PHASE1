"""
Poisoning attack implementations.

Phase 1 uses label-flip poisoning only (matches the base paper's
20% label-flip experiment). Additional attack types (Sybil is handled
via ABAC identity in abac/policy.py; subtle poisoning and repeated
poisoning come in Phase 5's full attack suite).
"""

import numpy as np
import pandas as pd


def label_flip(df: pd.DataFrame, frac: float = 0.2, target_col: str = "ACTION", seed: int = 42) -> pd.DataFrame:
    """
    Flips the binary label on a random `frac` fraction of rows.

    Args:
        df: a single client's partition (NOT the full dataset — poison
            only the malicious client's local data).
        frac: fraction of rows to flip (paper 2 used 0.2 / 20%).
        target_col: name of the binary label column.
        seed: for reproducibility.

    Returns:
        A copy of df with flipped labels on the sampled rows.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_flip = int(len(df) * frac)
    if n_flip == 0:
        return df
    idx = rng.choice(df.index, size=n_flip, replace=False)
    df.loc[idx, target_col] = 1 - df.loc[idx, target_col]
    return df
