"""
Post-hoc analysis of a finished run (works for phase2 or phase3 outputs).

  python experiments/phase_analyze.py --prefix phase2
  python experiments/phase_analyze.py --prefix phase3

Part A  PAIRED comparison across seeds (same seed = same split and same initial weights).
        For each attack strength: difference in AUC / AP / balanced accuracy between
        scenarios, mean difference, bootstrap 95% interval, and how many seeds the first
        scenario wins. Also "damage recovered" = (defended - attacked) / (clean - attacked).
Part B  THRESHOLD SWEEP on the logged anomaly scores: detection rate and false-positive
        rate (with Wilson 95% interval) if a client were flagged when score > t.
        Scores are the logged ones from the run (the run itself used t = 1), so this is an
        approximation: changing t would also change which clients enter later rounds.
Needs results/<prefix>_results.csv and logs/<prefix>_anomaly_log.csv.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def paired(df, metric, scen_a, scen_b, pf, rnd):
    sub = df[(df["round"] == rnd) & (df["poison_fraction"] == pf)]
    a = sub[sub["scenario"] == scen_a].set_index("seed")[metric]
    b = sub[sub["scenario"] == scen_b].set_index("seed")[metric]
    both = a.index.intersection(b.index)
    return (a.loc[both] - b.loc[both]).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="phase3")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    res = pd.read_csv(os.path.join(ROOT, "results", f"{args.prefix}_results.csv"))
    scenarios = list(res["scenario"].unique())
    attacked = [s for s in scenarios if s != "Clean_ABAC_only"]
    base = "With_ABAC_only"
    pd.set_option("display.width", 200)

    print("=" * 78 + "\nPART A - paired differences vs With_ABAC_only (final round)\n" + "=" * 78)
    for pf in sorted(res[res["poison_fraction"] > 0]["poison_fraction"].unique()):
        print(f"\n--- label-flip {pf} ---")
        for scen in attacked:
            if scen == base:
                continue
            for metric in ("auc", "ap", "balanced_acc"):
                d = paired(res, metric, scen, base, pf, args.rounds)
                if len(d) == 0:
                    continue
                lo, hi = boot_ci(d)
                print(f"{scen:24s} {metric:13s} mean diff {d.mean():+.4f}  95% boot [{lo:+.4f}, {hi:+.4f}]  "
                      f"wins {(d > 0).sum()}/{len(d)} seeds")
        clean = res[(res["scenario"] == "Clean_ABAC_only") & (res["round"] == args.rounds)].set_index("seed")["auc"]
        att = res[(res["scenario"] == base) & (res["round"] == args.rounds) & (res["poison_fraction"] == pf)].set_index("seed")["auc"]
        for scen in attacked:
            if scen == base:
                continue
            dfn = res[(res["scenario"] == scen) & (res["round"] == args.rounds) & (res["poison_fraction"] == pf)].set_index("seed")["auc"]
            idx = clean.index.intersection(att.index).intersection(dfn.index)
            damage = (clean.loc[idx] - att.loc[idx]).mean()
            rec = (dfn.loc[idx] - att.loc[idx]).mean()
            if damage > 1e-6:
                print(f"{scen:24s} AUC damage from attack {damage:.4f}, recovered {rec:.4f} ({rec/damage*100:.0f}%)")
            else:
                print(f"{scen:24s} attack did not hurt AUC here (damage {damage:+.4f}); nothing to recover")

    logp = os.path.join(ROOT, "logs", f"{args.prefix}_anomaly_log.csv")
    if not os.path.exists(logp):
        print("\n(no anomaly log found, skipping threshold sweep)")
        return
    lg = pd.read_csv(logp)
    lg["poisoned"] = lg["client_id"] == "client_2"
    print("\n" + "=" * 78 + "\nPART B - threshold sweep on logged anomaly scores\n" + "=" * 78)
    for (scen, pf), g in lg.groupby(["scenario", "poison_fraction"]):
        print(f"\n--- {scen}, label-flip {pf} (rows judged: {len(g)}) ---")
        print(f"{'t':>5} {'detect':>8} {'(after r1)':>11} {'false pos':>10} {'FP 95% upper':>13}")
        for t in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20):
            fl = g["anomaly_score"] > t
            p, h = g[g["poisoned"]], g[~g["poisoned"]]
            p2 = p[p["round"] > 1]
            det = (p["anomaly_score"] > t).mean() if len(p) else float("nan")
            det2 = (p2["anomaly_score"] > t).mean() if len(p2) else float("nan")
            fpk = int((h["anomaly_score"] > t).sum())
            fpr = fpk / len(h) if len(h) else float("nan")
            up = wilson(fpk, len(h))[1]
            print(f"{t:5.2f} {det:8.2f} {det2:11.2f} {fpr:10.3f} {up:13.3f}")
    print("\nA threshold is only trustworthy if the false-positive upper bound is small; with few "
          "honest observations even 0 false positives allows a fairly high true rate.")


if __name__ == "__main__":
    main()
