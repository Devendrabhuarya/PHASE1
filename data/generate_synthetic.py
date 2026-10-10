"""
Generates a synthetic stand-in dataset that MATCHES the Amazon Employee
Access Challenge's column schema, so you can test your entire pipeline
(partitioning, model training, Flower rounds) BEFORE downloading the
real Kaggle dataset.

This is NOT a substitute for the real dataset in your final experiments —
use it only to unblock Phase 0 development if the Kaggle download is
delayed. Swap in the real train.csv as soon as you have it; no other
code needs to change since the schema matches.

Usage:
    python data/generate_synthetic.py
    -> writes data/train.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 5000
N_DEPTS = 6          # gives you room to partition into 3-5 clients
N_ROLES = 30
N_MANAGERS = 50
N_RESOURCES = 100


def generate(n_rows=N_ROWS) -> pd.DataFrame:
    dept = np.random.randint(0, N_DEPTS, n_rows)
    role_title = np.random.randint(0, N_ROLES, n_rows)
    mgr_id = np.random.randint(0, N_MANAGERS, n_rows)
    resource = np.random.randint(0, N_RESOURCES, n_rows)
    rollup1 = np.random.randint(0, 20, n_rows)
    rollup2 = np.random.randint(0, 20, n_rows)
    family_desc = np.random.randint(0, 15, n_rows)
    family = np.random.randint(0, 15, n_rows)
    code = np.random.randint(0, 40, n_rows)

    # make ACTION weakly dependent on department + role so the model
    # has something real to learn, plus noise. Base rate matches the
    # REAL Amazon Employee Access dataset's known ~94% approval rate —
    # deliberately heavily imbalanced, so Phase 1+ testing catches the
    # "model collapses to majority class" failure mode early, instead
    # of only surfacing it once the real dataset is plugged in.
    base_prob = 0.94 + 0.02 * (dept % 3) - 0.015 * (role_title % 4)
    base_prob = np.clip(base_prob, 0.85, 0.98)
    action = (np.random.rand(n_rows) < base_prob).astype(int)

    df = pd.DataFrame({
        "ACTION": action,
        "RESOURCE": resource,
        "MGR_ID": mgr_id,
        "ROLE_ROLLUP_1": rollup1,
        "ROLE_ROLLUP_2": rollup2,
        "ROLE_DEPTNAME": dept,
        "ROLE_TITLE": role_title,
        "ROLE_FAMILY_DESC": family_desc,
        "ROLE_FAMILY": family,
        "ROLE_CODE": code,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/train.csv", index=False)
    print(f"Synthetic dataset written to data/train.csv, shape={df.shape}")
    print(df["ACTION"].value_counts(normalize=True))
