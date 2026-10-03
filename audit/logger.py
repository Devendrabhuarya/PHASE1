"""
Local audit logger.

This plays the role the diagram's "Blockchain Audit Log" boxes play —
every ABAC decision (allow/deny) and round-level event gets recorded
here. It is intentionally simple (CSV) per the project's scoping
decision to defer blockchain to an optional later stretch goal; the
schema below is exactly what would map onto on-chain fields if that
stretch goal is picked up.
"""

import csv
import os
from datetime import datetime, timezone


class AuditLogger:
    FIELDNAMES = ["timestamp", "round", "client_id", "scenario", "decision", "reason"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self, round_num: int, client_id: str, scenario: str, decision: str, reason: str = ""):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "round": round_num,
                "client_id": client_id,
                "scenario": scenario,
                "decision": decision,
                "reason": reason,
            })
