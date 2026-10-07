"""
Preprocessing for the CICIoT2023 dataset.

The real dataset (from https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/)
is NOT one file — it's 63 separate files (Merged01.csv ... Merged63.csv),
each a chunk of network-flow records. Attack types are NOT evenly spread
across files, so building a sample from only one file risks missing
entire attack categories (or benign traffic) entirely. Everything below
samples ACROSS all 63 files, not just one.

Real column schema (confirmed from the actual dataset — this project's
loader auto-detects columns at runtime rather than hardcoding a
feature list, so it adapts to whichever exact column set your files
have, tab- or comma-separated):

    Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number,
    syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number,
    ece_flag_number, cwr_flag_number, ack_count, syn_count, fin_count,
    rst_count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP,
    ARP, ICMP, IGMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size,
    IAT, Number, Variance, Label

All feature columns are continuous network-flow statistics. Label has
~34 fine-grained attack-type values (DDoS-*, DoS-*, Mirai-*, Recon-*,
Spoofing, BruteForce, Web-based attacks) plus 'BenignTraffic'.

To keep the rest of the pipeline (client_utils.py, poisoning.py,
phase1_run.py, abac/policy.py) UNCHANGED, this module produces a
dataframe with a column literally named "ACTION" (1 = benign/allow,
0 = attack/deny) — exactly like the project's earlier dataset did —
so everything downstream that already expects "ACTION" keeps working.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# If auto-detection below picks the wrong column/separator, override here.
LABEL_COL_OVERRIDE = None     # e.g. "Label"
SEP_OVERRIDE = None             # e.g. "\t" or ","
BENIGN_KEYWORD = "benign"        # case-insensitive substring match
# ---------------------------------------------------------------------


def resolve_file_list(csv_path_or_glob) -> list:
    """
    Accepts a single file path, a glob pattern (e.g.
    'data/ciciot_raw/Merged*.csv'), or a directory (in which case all
    *.csv files inside it are used). Always returns a sorted list of
    file paths.
    """
    if isinstance(csv_path_or_glob, (list, tuple)):
        files = list(csv_path_or_glob)
    elif os.path.isdir(csv_path_or_glob):
        files = glob.glob(os.path.join(csv_path_or_glob, "*.csv"))
    else:
        files = glob.glob(csv_path_or_glob)
        if not files and os.path.isfile(csv_path_or_glob):
            files = [csv_path_or_glob]

    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for '{csv_path_or_glob}'. Check the path/glob."
        )
    return files


def detect_separator(file_path: str) -> str:
    """Peeks at the first line to tell tab-separated from comma-separated."""
    if SEP_OVERRIDE is not None:
        return SEP_OVERRIDE
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    return ","


def detect_label_column(file_path: str, sep: str = None) -> str:
    """Reads only the header row (fast, even on a large file)."""
    if sep is None:
        sep = detect_separator(file_path)
    header = pd.read_csv(file_path, sep=sep, nrows=0)
    cols = [c.strip() for c in header.columns]

    if LABEL_COL_OVERRIDE is not None:
        if LABEL_COL_OVERRIDE not in cols:
            raise ValueError(
                f"LABEL_COL_OVERRIDE='{LABEL_COL_OVERRIDE}' not found in "
                f"columns: {cols}"
            )
        return LABEL_COL_OVERRIDE

    candidates = [c for c in cols if "label" in c.lower()]
    if candidates:
        return candidates[0]

    print(
        f"WARNING: no column with 'label' in its name found. Falling back "
        f"to the LAST column ('{cols[-1]}') as the label. If that's wrong, "
        f"set LABEL_COL_OVERRIDE at the top of this file."
    )
    return cols[-1]


def load_and_sample(
    csv_path_or_glob,
    label_col: str = None,
    max_rows_per_class: int = 2000,
    chunksize: int = 200_000,
    max_files: int = None,
) -> pd.DataFrame:
    """
    Scans across ALL matched files (e.g. Merged01.csv ... Merged63.csv),
    reading each in chunks, keeping up to `max_rows_per_class` rows for
    EACH unique raw label value encountered — counted CUMULATIVELY
    across every file, not per-file. This guarantees every attack
    subtype (and benign traffic) is represented in the final sample,
    regardless of how unevenly attack types are distributed across the
    63 files.

    `max_files` caps how many of the (sorted) files get scanned — use
    this for a quick first pass during development; leave it None
    (scan everything) for your actual reported results, since rare
    classes may only appear in a handful of the 63 files.
    """
    files = resolve_file_list(csv_path_or_glob)
    if max_files is not None:
        files = files[:max_files]
    print(f"Found {len(files)} file(s) to scan.")

    sep = detect_separator(files[0])
    print(f"Detected separator: {'TAB' if sep == chr(9) else repr(sep)}")

    if label_col is None:
        label_col = detect_label_column(files[0], sep=sep)
    print(f"Using label column: '{label_col}'")

    collected = {}  # label_value -> list of row-DataFrames
    counts = {}

    for file_idx, file_path in enumerate(files, start=1):
        reader = pd.read_csv(file_path, sep=sep, chunksize=chunksize, low_memory=False)
        file_rows_kept = 0
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.dropna()
            for label_value, group in chunk.groupby(label_col):
                have = counts.get(label_value, 0)
                if have >= max_rows_per_class:
                    continue
                need = max_rows_per_class - have
                take = group.iloc[:need]
                collected.setdefault(label_value, []).append(take)
                counts[label_value] = have + len(take)
                file_rows_kept += len(take)

        print(f"  [{file_idx}/{len(files)}] {os.path.basename(file_path)}: "
              f"kept {file_rows_kept} rows this file "
              f"(running classes seen: {len(counts)})")

    if not collected:
        raise ValueError(
            f"No data collected from {files}. Check the file path(s), "
            f"separator, and label column detection."
        )

    df = pd.concat([pd.concat(parts, ignore_index=True) for parts in collected.values()],
                    ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\nFinal sample: {len(df)} rows across {len(counts)} unique raw label values:")
    print(df[label_col].value_counts())

    return df


def map_binary_label(df: pd.DataFrame, label_col: str = None, benign_keyword: str = BENIGN_KEYWORD) -> pd.DataFrame:
    """
    Collapses the many fine-grained attack-type labels down to a
    single binary ACTION column: 1 = benign (allow), 0 = attack
    (deny) — matching the rest of the pipeline's ACTION semantics.
    """
    if label_col is None:
        label_col = detect_label_column_from_df(df)

    df = df.copy()
    is_benign = df[label_col].astype(str).str.lower().str.contains(benign_keyword)
    df["ACTION"] = is_benign.astype(int)

    n_benign = int(df["ACTION"].sum())
    n_attack = len(df) - n_benign
    print(
        f"\nBinary mapping: {n_benign} benign (ACTION=1) / {n_attack} attack "
        f"(ACTION=0)  -  {n_benign/len(df)*100:.2f}% benign"
    )
    if n_benign == 0 or n_attack == 0:
        raise ValueError(
            f"Binary mapping produced only one class! Check BENIGN_KEYWORD "
            f"('{benign_keyword}') against actual label values: "
            f"{df[label_col].unique()[:10]}"
        )
    return df


def detect_label_column_from_df(df: pd.DataFrame) -> str:
    if LABEL_COL_OVERRIDE is not None and LABEL_COL_OVERRIDE in df.columns:
        return LABEL_COL_OVERRIDE
    candidates = [c for c in df.columns if "label" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError("Could not detect label column from dataframe columns.")


def get_feature_columns(df: pd.DataFrame, label_col: str) -> list:
    """
    Every numeric column except the raw label and the derived ACTION
    column is treated as a feature. This auto-adapts to whichever
    exact column set your copy of the dataset has.
    """
    exclude = {label_col, "ACTION"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found after excluding the label. "
            "Check that the CSV(s) loaded correctly (right separator?)."
        )
    return feature_cols


def fit_scaler(df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    """Fit on the TRAIN POOL only (call this after your test holdout split)."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    return scaler


def transform_with_scaler(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Mirrors the earlier dataset's transform function signature/shape."""
    X = scaler.transform(df[feature_cols].values).astype("float32")
    y = df["ACTION"].values.astype("float32")
    return X, y


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ciciot_raw/"

    files = resolve_file_list(path)
    print(f"Would scan {len(files)} files, e.g.: {files[:3]}")

    df = load_and_sample(path, max_rows_per_class=500, max_files=3)
    label_col = detect_label_column(files[0])
    df = map_binary_label(df, label_col=label_col)
    feature_cols = get_feature_columns(df, label_col=label_col)
    print(f"\nDetected {len(feature_cols)} feature columns:")
    print(feature_cols)
