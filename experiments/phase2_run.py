"""
Phase 2 (v2): anomaly detection on model updates, evaluated properly.

What changed versus the first Phase 2 script (all were real problems):
  1. SEEDING   every scenario calls set_seed(seed) and starts from the SAME initial
               weights, so scenario differences are not random-init noise.
  2. MULTI-SEED results are averaged over several seeds (different client splits and
               randomness) and reported as mean +/- std.
  3. DETECTOR  scale-free scoring (cos / ratios), no forced outlier, no judgement
               with < 3 clients (see anomaly/scoring.py).
  4. THRESHOLDS are calibrated on clean attack-free runs (separate seeds), not guessed.
  5. CLASS WEIGHT is FIXED from the clean pool, so a poisoned client cannot cancel its
               own attack through locally computed weights.
  6. ATTACK STRENGTH several label-flip fractions are tested (default 20% and 50%).
  7. METRICS   AUC and balanced accuracy are reported next to F1@0.5 (F1@0.5 alone is
               misleading on this imbalanced data).
  8. CLEAN REFERENCE a no-attack run shows whether poisoning hurts the model at all.

Scenarios per (seed, poison fraction):
  With_ABAC_only          identity gate only (client_3 blocked, client_2 admitted)
  With_ABAC_plus_Anomaly  identity gate + update anomaly gate
  (optional --with-no-abac adds the no-gate baseline)
Plus once per seed: Clean_ABAC_only (no poisoning) as the upper reference.

Run from the project root:
    python experiments/phase2_run.py                      # 3 seeds, poison 0.2 and 0.5
    python experiments/phase2_run.py --seeds 5 --poison 0.2 0.5 0.8
    python experiments/phase2_run.py --seeds 1 --poison 0.5   # quick check
"""

import argparse
import copy
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score,
)

from common_setup import (  # noqa: E402
    N_ROUNDS, EPOCHS_PER_ROUND, POISON_CLIENT_INDEX,
    set_seed, load_dataset_once, build_clients, compute_global_class_weight,
)
from abac.policy import default_phase1_clients, pre_training_gate  # noqa: E402
from fl.fl_client import FLClient  # noqa: E402
from fl.aggregation import fedavg  # noqa: E402
from audit.logger import AuditLogger, AnomalyAuditLogger  # noqa: E402
from models.model import build_model  # noqa: E402
from anomaly.scoring import (  # noqa: E402
    Thresholds, compute_delta, compute_round_features, score_features,
    calibrate_thresholds, thresholds_to_dict, get_trainable_mask, filter_by_mask, risk_weight,
    select_scope, SCOPES,
)

POISONED_ID = f"client_{POISON_CLIENT_INDEX}"
CAL_SEEDS = [1000, 1001]     # calibration seeds are kept SEPARATE from evaluation seeds
NO_FLAG = Thresholds(cos_min=-1.0, dist_max=1e9, norm_max=1e9)   # judge nobody


def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    proba = model.predict(X_test, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    ap = average_precision_score(y_test, proba)   # threshold-free, suited to imbalanced data
    return {"accuracy": acc, "balanced_acc": balanced_accuracy_score(y_test, pred),
            "precision": p, "recall": r, "f1": f1, "auc": auc, "ap": ap, "loss": loss}


def run_scenario(name, use_abac, use_anomaly, th, client_data, test_data, input_dim,
                 init_weights, seed, poison_fraction, class_weight,
                 audit_logger=None, anomaly_logger=None, verbose=True,
                 weighting="hard", soft_low=1.0, soft_high=1.4, scope="output"):
    # weighting: "hard" = reject a client when flagged (score > 1);
    #            "soft" = risk-aware weight in [0,1] from the score (see risk_weight)
    set_seed(seed)                                  # identical randomness for every scenario
    attrs_list = default_phase1_clients()
    clients = [FLClient(a.client_id, a, X, y, input_dim) for a, (X, y) in zip(attrs_list, client_data)]
    model = build_model(input_dim)
    mask = get_trainable_mask(model)
    global_weights = copy.deepcopy(init_weights)    # identical starting point for every scenario
    model.set_weights(global_weights)
    X_test, y_test = test_data

    history, feature_rows = [], []
    for rnd in range(1, N_ROUNDS + 1):
        admitted = []
        for c in clients:
            allowed = pre_training_gate(c.attrs) if use_abac else True
            if audit_logger is not None:
                audit_logger.log(rnd, c.client_id, name, "ALLOW" if allowed else "DENY",
                                 "" if allowed else f"role '{c.attrs.role}' not permitted")
            if allowed:
                admitted.append(c)

        before = global_weights
        trained = {}
        for c in admitted:
            w, n = c.train(before, epochs=EPOCHS_PER_ROUND, batch_size=32, class_weight=class_weight)
            trained[c.client_id] = (w, n, compute_delta(w, before))

        accepted = list(trained.keys())
        agg_weight = {}
        if use_anomaly or th is NO_FLAG:
            feats = compute_round_features({cid: select_scope(filter_by_mask(t[2], mask), scope)
                                            for cid, t in trained.items()})
            scored = score_features(feats, th)
            accepted = []
            agg_weight = {}
            for cid, s in scored.items():
                if use_anomaly and weighting == "soft":
                    w_risk = risk_weight(s["anomaly_score"], soft_low, soft_high)
                    flagged = w_risk == 0.0
                else:
                    w_risk = 0.0 if (use_anomaly and s["flagged"]) else 1.0
                    flagged = use_anomaly and s["flagged"]
                agg_weight[cid] = w_risk
                feature_rows.append({"seed": seed, "poison_fraction": poison_fraction, "round": rnd,
                                     "client_id": cid, "is_poisoned": cid == POISONED_ID,
                                     "flagged": flagged, **{k: s[k] for k in
                                     ("cos_sim", "dist_ratio", "norm_dev", "judged", "anomaly_score")}})
                if anomaly_logger is not None and use_anomaly:
                    anomaly_logger.log(seed=seed, poison_fraction=poison_fraction, round=rnd,
                                       client_id=cid, scenario=name, norm=s["norm"], cos_sim=s["cos_sim"],
                                       distance=s["distance"], dist_ratio=s["dist_ratio"],
                                       norm_dev=s["norm_dev"], anomaly_score=s["anomaly_score"], weight=w_risk,
                                       decision="REJECT" if w_risk == 0.0 else ("DOWNWEIGHT" if w_risk < 1.0 else "ACCEPT"))
                if w_risk > 0.0:
                    accepted.append(cid)

        if accepted:
            wts = [trained[c][1] * (agg_weight.get(c, 1.0) if (use_anomaly and len(trained) >= 3) else 1.0)
                   for c in accepted]
            global_weights = fedavg([trained[c][0] for c in accepted], wts)
        model.set_weights(global_weights)
        m = evaluate_model(model, X_test, y_test)
        if verbose:
            print(f"  [{name} seed={seed} pf={poison_fraction}] round {rnd}: accepted={accepted} "
                  f"auc={m['auc']:.4f} bal_acc={m['balanced_acc']:.4f} f1={m['f1']:.4f}")
        history.append({"seed": seed, "poison_fraction": poison_fraction, "scenario": name,
                        "round": rnd, "accepted_clients": ";".join(accepted), **m})
    return history, feature_rows


def calibrate(dataset, margin, class_weight, scope="output"):
    print("\n=== Calibrating thresholds on CLEAN (attack-free) runs ===")
    rows = []
    for s in CAL_SEEDS:
        client_data, test_data, input_dim = build_clients(dataset, seed=s, poison=False, verbose=False)
        set_seed(s)
        init = build_model(input_dim).get_weights()
        _, fr = run_scenario("Calibration", True, False, NO_FLAG, client_data, test_data, input_dim,
                             init, s, 0.0, class_weight, verbose=False, scope=scope)
        rows += fr
    th = calibrate_thresholds(rows, margin=margin)
    print(f"Honest observations: {sum(r['judged'] for r in rows)}  |  margin={margin}")
    print(f"Calibrated thresholds: {thresholds_to_dict(th)}")
    return th


def detection_stats(feature_rows, skip_round1=False):
    df = pd.DataFrame(feature_rows)
    if df.empty:
        return {}
    df = df[df["judged"]]
    if skip_round1:
        df = df[df["round"] > 1]
    pois, hon = df[df["is_poisoned"]], df[~df["is_poisoned"]]
    return {"detection_rate": float(pois["flagged"].mean()) if len(pois) else float("nan"),
            "false_positive_rate": float(hon["flagged"].mean()) if len(hon) else float("nan"),
            "n_poisoned_obs": int(len(pois)), "n_honest_obs": int(len(hon))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--poison", type=float, nargs="+", default=[0.2, 0.5])
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--scope", choices=list(SCOPES), default="output",
                    help="which part of the update is compared (default: output layer)")
    ap.add_argument("--with-no-abac", action="store_true")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    results_dir, logs_dir = os.path.join(root, "results"), os.path.join(root, "logs")
    os.makedirs(results_dir, exist_ok=True)
    for f in ("phase2_audit_log.csv", "phase2_anomaly_log.csv"):
        p = os.path.join(logs_dir, f)
        if os.path.exists(p):
            os.remove(p)
    audit_logger = AuditLogger(os.path.join(logs_dir, "phase2_audit_log.csv"))
    anomaly_logger = AnomalyAuditLogger(os.path.join(logs_dir, "phase2_anomaly_log.csv"))

    dataset = load_dataset_once()
    class_weight = compute_global_class_weight(dataset["pool"])
    print(f"Fixed global class weight (from clean pool): {class_weight}")

    th = calibrate(dataset, args.margin, class_weight, args.scope)
    with open(os.path.join(results_dir, "phase2_thresholds.json"), "w") as f:
        json.dump(thresholds_to_dict(th), f, indent=2)

    all_hist, all_feats = [], []
    for seed in range(args.seeds):
        print(f"\n########## SEED {seed} ##########")
        # clean reference: how good is the model with NO attack?
        cd, td, dim = build_clients(dataset, seed=seed, poison=False, verbose=False)
        set_seed(seed)
        init = build_model(dim).get_weights()
        h, _ = run_scenario("Clean_ABAC_only", True, False, th, cd, td, dim, init, seed, 0.0, class_weight)
        all_hist += h

        for pf in args.poison:
            cd, td, dim = build_clients(dataset, seed=seed, poison_fraction=pf, verbose=(seed == 0))
            set_seed(seed)
            init = build_model(dim).get_weights()
            scen = []
            if args.with_no_abac:
                scen.append(("No_ABAC", False, False))
            scen += [("With_ABAC_only", True, False), ("With_ABAC_plus_Anomaly", True, True)]
            for name, abac, anom in scen:
                h, fr = run_scenario(name, abac, anom, th, cd, td, dim, init, seed, pf, class_weight,
                                     audit_logger, anomaly_logger, scope=args.scope)
                all_hist += h
                if anom:
                    all_feats += fr

    hist = pd.DataFrame(all_hist)
    hist.to_csv(os.path.join(results_dir, "phase2_results.csv"), index=False)

    # ---------- summary tables ----------
    final = hist[hist["round"] == N_ROUNDS]
    rows = []
    for (scen, pf), g in final.groupby(["scenario", "poison_fraction"]):
        rows.append({"scenario": scen, "poison_fraction": pf, "n_seeds": g["seed"].nunique(),
                     **{f"{m}_mean": g[m].mean() for m in ("auc", "ap", "balanced_acc", "f1")},
                     **{f"{m}_std": g[m].std(ddof=0) for m in ("auc", "ap", "balanced_acc", "f1")}})
    summary = pd.DataFrame(rows).sort_values(["poison_fraction", "scenario"])
    summary.to_csv(os.path.join(results_dir, "phase2_summary.csv"), index=False)

    det_rows = []
    for pf in args.poison:
        fr = [r for r in all_feats if r["poison_fraction"] == pf]
        a, b = detection_stats(fr), detection_stats(fr, skip_round1=True)
        det_rows.append({"poison_fraction": pf, **{f"all_{k}": v for k, v in a.items()},
                         **{f"after_round1_{k}": v for k, v in b.items()}})
    det = pd.DataFrame(det_rows)
    det.to_csv(os.path.join(results_dir, "phase2_detection.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70 + "\nFINAL-ROUND METRICS (mean over seeds)\n" + "=" * 70)
    print(summary.round(4).to_string(index=False))
    print("\n" + "=" * 70 + "\nDETECTOR EVALUATION (pooled over seeds)\n" + "=" * 70)
    print(det.round(3).to_string(index=False))
    print("\nRead this honestly: if the Clean_ABAC_only AUC is about the same as With_ABAC_only, "
          "the poisoning is not hurting the model and there is nothing for the detector to "
          "improve; compare ABAC_only vs ABAC_plus_Anomaly only where the attack really hurts.")

    try:
        import matplotlib.pyplot as plt
        fdf = pd.DataFrame(all_feats)
        n = len(args.poison)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
        for ax, pf in zip(axes[0], args.poison):
            sub = fdf[(fdf["poison_fraction"] == pf) & (fdf["seed"] == 0)]
            for cid in sorted(sub["client_id"].unique()):
                s = sub[sub["client_id"] == cid]
                ax.plot(s["round"], s["anomaly_score"], "o--" if cid == POISONED_ID else "o-", label=cid)
            ax.axhline(1.0, color="red", linestyle=":", label="flag threshold (=1)")
            ax.set_title(f"Anomaly score, label-flip {int(pf*100)}% (seed 0)")
            ax.set_xlabel("FL round"); ax.set_ylabel("anomaly score"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "phase2_anomaly_scores.png"), dpi=150)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pivot = summary.pivot(index="poison_fraction", columns="scenario", values="auc_mean")
        err = summary.pivot(index="poison_fraction", columns="scenario", values="auc_std")
        pivot.plot(kind="bar", yerr=err, ax=ax2, capsize=3)
        ax2.set_ylabel("final-round AUC (mean +/- std over seeds)")
        ax2.set_title("Model quality by scenario and attack strength")
        ax2.set_ylim(max(0.5, float(np.nanmin(pivot.values)) - 0.05), 1.0)
        fig2.tight_layout()
        fig2.savefig(os.path.join(results_dir, "phase2_auc_comparison.png"), dpi=150)
    except ImportError:
        print("matplotlib not available, skipping plots.")

    print("\nSaved: results/phase2_results.csv, phase2_summary.csv, phase2_detection.csv, "
          "phase2_thresholds.json, plots; logs/phase2_*.csv")


if __name__ == "__main__":
    main()
