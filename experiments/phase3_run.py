"""
Phase 3: risk-aware aggregation vs hard reject.

Scenarios per (seed, poison fraction), all from the same seed and initial weights:
  With_ABAC_only            identity gate only
  Hard_reject               ABAC + anomaly gate, reject a client when score > 1
  Risk_aware_weighting      ABAC + anomaly score, weight in [0,1] (ramp 1.0 -> 1.4)
Plus once per seed: Clean_ABAC_only (no attack) as the reference.

Why: the hard rule caught only 60-68% of poisoned rounds on real data, and every
missed round got FULL weight. Soft weighting reduces the influence of borderline
clients without needing a perfect threshold.

    python experiments/phase3_run.py --seeds 5 --poison 0.2 0.5
    python experiments/phase3_run.py --seeds 1 --poison 0.5        # quick check
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from common_setup import (  # noqa: E402
    N_ROUNDS, set_seed, load_dataset_once, build_clients, compute_global_class_weight,
)
from phase2_run import run_scenario, calibrate, detection_stats  # noqa: E402
from anomaly.scoring import SCOPES  # noqa: E402
from audit.logger import AuditLogger, AnomalyAuditLogger  # noqa: E402
from models.model import build_model  # noqa: E402
from anomaly.scoring import thresholds_to_dict  # noqa: E402

SCENARIOS = [
    ("With_ABAC_only", True, False, "hard"),
    ("Hard_reject", True, True, "hard"),
    ("Risk_aware_weighting", True, True, "soft"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--poison", type=float, nargs="+", default=[0.2, 0.5])
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--soft-low", type=float, default=1.0)
    ap.add_argument("--soft-high", type=float, default=1.4)
    ap.add_argument("--scope", choices=list(SCOPES), default="output")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    results_dir, logs_dir = os.path.join(root, "results"), os.path.join(root, "logs")
    os.makedirs(results_dir, exist_ok=True)
    for f in ("phase3_audit_log.csv", "phase3_anomaly_log.csv"):
        p = os.path.join(logs_dir, f)
        if os.path.exists(p):
            os.remove(p)
    audit = AuditLogger(os.path.join(logs_dir, "phase3_audit_log.csv"))
    alog = AnomalyAuditLogger(os.path.join(logs_dir, "phase3_anomaly_log.csv"))

    dataset = load_dataset_once()
    cw = compute_global_class_weight(dataset["pool"])
    th = calibrate(dataset, args.margin, cw, args.scope)
    with open(os.path.join(results_dir, "phase3_thresholds.json"), "w") as f:
        json.dump(thresholds_to_dict(th), f, indent=2)

    hist, feats = [], {}
    for seed in range(args.seeds):
        print(f"\n########## SEED {seed} ##########")
        cd, td, dim = build_clients(dataset, seed=seed, poison=False, verbose=False)
        set_seed(seed)
        init = build_model(dim).get_weights()
        h, _ = run_scenario("Clean_ABAC_only", True, False, th, cd, td, dim, init, seed, 0.0, cw)
        hist += h
        for pf in args.poison:
            cd, td, dim = build_clients(dataset, seed=seed, poison_fraction=pf, verbose=False)
            set_seed(seed)
            init = build_model(dim).get_weights()
            for name, abac, anom, mode in SCENARIOS:
                h, fr = run_scenario(name, abac, anom, th, cd, td, dim, init, seed, pf, cw,
                                     audit, alog, weighting=mode,
                                     soft_low=args.soft_low, soft_high=args.soft_high,
                                     scope=args.scope)
                hist += h
                if anom:
                    feats.setdefault((name, pf), []).extend(fr)

    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(results_dir, "phase3_results.csv"), index=False)

    final = df[df["round"] == N_ROUNDS]
    cols = ("auc", "ap", "balanced_acc", "f1")
    rows = []
    for (scen, pf), g in final.groupby(["scenario", "poison_fraction"]):
        rows.append({"scenario": scen, "poison_fraction": pf, "n_seeds": g["seed"].nunique(),
                     **{f"{m}_mean": g[m].mean() for m in cols}, **{f"{m}_std": g[m].std(ddof=0) for m in cols}})
    summary = pd.DataFrame(rows).sort_values(["poison_fraction", "scenario"])
    summary.to_csv(os.path.join(results_dir, "phase3_summary.csv"), index=False)

    drows = []
    for (name, pf), fr in feats.items():
        a, b = detection_stats(fr), detection_stats(fr, skip_round1=True)
        drows.append({"scenario": name, "poison_fraction": pf,
                      **{f"all_{k}": v for k, v in a.items()}, **{f"after_round1_{k}": v for k, v in b.items()}})
    det = pd.DataFrame(drows)
    det.to_csv(os.path.join(results_dir, "phase3_detection.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 70 + "\nFINAL-ROUND METRICS (mean over seeds)\n" + "=" * 70)
    print(summary.round(4).to_string(index=False))
    print("\n" + "=" * 70 + "\nDETECTION (a client counts as 'flagged' only when weight = 0)\n" + "=" * 70)
    print(det.round(3).to_string(index=False))
    print("\nNext: python experiments/phase_analyze.py --prefix phase3   (paired statistics + threshold sweep)")


if __name__ == "__main__":
    main()
