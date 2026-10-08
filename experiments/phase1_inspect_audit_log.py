"""
Quick helper to summarize the Phase 1 audit log — useful for your
report's "audit trail" section (mirrors paper 2's CouchDB/chaincode
log screenshots, just from the local CSV log for now).

Run from the project root:
    python experiments/phase1_inspect_audit_log.py
"""

import os
import pandas as pd


def main():
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "phase1_audit_log.csv")
    df = pd.read_csv(log_path)

    print("Decision counts by scenario:")
    print(df.groupby(["scenario", "decision"]).size().unstack(fill_value=0))

    print("\nPer-client decision breakdown (With_ABAC scenario):")
    with_abac = df[df["scenario"] == "With_ABAC"]
    print(with_abac.groupby(["client_id", "decision"]).size().unstack(fill_value=0))

    denied = with_abac[with_abac["decision"] == "DENY"]
    if not denied.empty:
        print(f"\nDenied clients and reasons (With_ABAC):")
        print(denied[["client_id", "reason"]].drop_duplicates())


if __name__ == "__main__":
    main()
