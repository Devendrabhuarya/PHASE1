"""
Generates synthetic stand-in data matching the REAL CICIoT2023 column
schema you confirmed from your actual files, split across MULTIPLE
files (Merged01.csv, Merged02.csv, Merged03.csv) to exercise the
multi-file loading path in preprocess_ciciot.py before your real
63-file download is ready to use.

This is NOT a substitute for the real dataset in your final reported
experiments — swap in the real Merged01.csv...Merged63.csv files as
soon as you have them (point load_and_sample() at the folder they're
in; nothing else needs to change since columns/labels are auto-detected).

Usage:
    python data/generate_synthetic_ciciot.py
    -> writes data/ciciot_raw/Merged01.csv, Merged02.csv, Merged03.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_FILES = 3
ROWS_PER_FILE = 7000
OUT_DIR = "data/ciciot_raw"

# Real column schema, confirmed from the actual dataset files
FEATURE_COLUMNS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
    "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]

# Real CICIoT2023 label values, in realistic (attack-heavy) proportions.
# Deliberately NOT uniform across the simulated files below — mimics
# the real dataset's uneven per-file class distribution, so the
# multi-file sampler's "accumulate across files" logic actually gets
# exercised rather than trivially passing on a single balanced file.
LABEL_POOL = [
    "BenignTraffic", "DDoS-SYN_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood",
    "DDoS-PSHACK_Flood", "DoS-UDP_Flood", "DoS-SYN_Flood",
    "Mirai-greeth_flood", "Mirai-udpplain", "Recon-PortScan",
    "Recon-OSScan", "MITM-ArpSpoofing", "DNS_Spoofing",
    "DictionaryBruteForce", "SqlInjection",
]


def generate_file(file_idx: int, n_rows: int, seed_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(42 + seed_offset)

    # each simulated file over/under-represents different labels,
    # mimicking how the real 63-file split is not evenly shuffled
    weights = rng.dirichlet(np.ones(len(LABEL_POOL)) * 0.6)
    label_col = rng.choice(LABEL_POOL, size=n_rows, p=weights)
    is_benign = (label_col == "BenignTraffic").astype(int)

    data = {}
    for col in FEATURE_COLUMNS:
        base = rng.exponential(scale=1.0, size=n_rows)
        attack_boost = (1 - is_benign) * rng.exponential(scale=0.6, size=n_rows)
        noise = rng.normal(loc=0, scale=0.8, size=n_rows)
        data[col] = np.clip(base + attack_boost + noise, 0, None).astype("float32")

    data["Label"] = label_col
    return pd.DataFrame(data)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(1, N_FILES + 1):
        df = generate_file(i, ROWS_PER_FILE, seed_offset=i)
        out_path = os.path.join(OUT_DIR, f"Merged{i:02d}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}, shape={df.shape}")
        print(df["Label"].value_counts().head(5))
        print()

    print(f"Synthetic CICIoT2023-schema data written to {OUT_DIR}/ "
          f"({N_FILES} files, matches the Merged01.csv...Merged63.csv "
          f"naming pattern of the real dataset).")
