"""
Preprocessing for the Amazon Employee Access Challenge dataset.

Expected columns (standard Kaggle 'train.csv'):
    ACTION, RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2,
    ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE

ACTION is the binary target (1 = approved, 0 = denied).
All other columns are high-cardinality categorical IDs -> label-encoded.

Download the dataset from:
    https://www.kaggle.com/c/amazon-employee-access-challenge/data
and place train.csv in this data/ folder.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "ACTION"
FEATURE_COLS = [
    "RESOURCE",
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Make sure you downloaded the correct Kaggle 'train.csv'."
        )
    return df


def encode_features(df: pd.DataFrame):
    """
    Label-encode every categorical feature column.
    Returns (X, y, input_dim, encoders_dict) so the same encoders
    can later be reused consistently across client partitions.
    """
    df = df.copy()
    encoders = {}
    for col in FEATURE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURE_COLS].values.astype("float32")
    y = df[TARGET_COL].values.astype("float32")
    input_dim = X.shape[1]
    return X, y, input_dim, encoders


def fit_encoders(df: pd.DataFrame):
    """
    Fit one LabelEncoder per feature column on the FULL dataset's
    vocabulary. Use this once up front, then reuse the same encoders
    on every client partition and the held-out test set so everything
    shares consistent integer encodings (required before partitioning
    into FL clients).
    """
    encoders = {}
    for col in FEATURE_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def transform_with_encoders(df: pd.DataFrame, encoders: dict):
    """
    Transform a (sub)dataframe using already-fitted encoders. Since any
    client partition or test split is just a subset of the dataframe
    the encoders were fit on, no unseen-label issues occur.
    """
    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = encoders[col].transform(df[col].astype(str))
    X = df[FEATURE_COLS].values.astype("float32")
    y = df[TARGET_COL].values.astype("float32")
    return X, y


def inspect(df: pd.DataFrame):
    """Quick sanity print — run this first on any new dataset copy."""
    print("Shape:", df.shape)
    print("\nClass balance (ACTION):")
    print(df[TARGET_COL].value_counts(normalize=True))
    print("\nUnique ROLE_DEPTNAME values (candidate client-partition key):")
    print(df["ROLE_DEPTNAME"].nunique())
    print("\nTop department sizes:")
    print(df["ROLE_DEPTNAME"].value_counts().head(10))


if __name__ == "__main__":
    df = load_raw("data/train.csv")
    inspect(df)
    X, y, input_dim, _ = encode_features(df)
    print(f"\nEncoded feature matrix shape: {X.shape}, input_dim={input_dim}")
