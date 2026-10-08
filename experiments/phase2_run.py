"""
Phase 2: adds anomaly detection on top of Phase 1's ABAC gate.

Runs THREE scenarios over the IDENTICAL data/attack setup as Phase 1
(same prepare_data(), same poisoned client_2, same Sybil client_3):

  Scenario A — "No_ABAC": no filtering at all (same as Phase 1).

  Scenario B — "With_ABAC_only": identity-only gate (same as Phase 1's
      "With_ABAC"). Catches client_3 (Sybil), NOT client_2 (poisoned
      but valid identity).

  Scenario C — "With_ABAC_plus_Anomaly" (NEW): identity gate runs
      first (still catches client_3), THEN each ABAC-admitted
      client's model update is scored for anomaly (norm / cosine
      similarity / distance relative to its peers, leave-one-out —
      see anomaly/scoring.py) and excluded from aggregation if its
      anomaly score crosses ANOMALY_THRESHOLD. This is the scenario
      that should finally catch client_2.

Run from the project root:
    python experiments/phase2_run.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # noqa: E402

from common_setup import (  # noqa: E402
    N_ROUNDS, EPOCHS_PER_ROUND, POISON_CLIENT_INDEX, prepare_data,
)
from abac.policy import default_phase1_clients, pre_training_gate  # noqa: E402
from fl.fl_client import FLClient  # noqa: E402
from fl.aggregation import fedavg  # noqa: E402
from audit.logger import AuditLogger, AnomalyAuditLogger  # noqa: E402
from models.model import build_model  # noqa: E402
from anomaly.scoring import (  # noqa: E402
    compute_delta, compute_round_anomaly_scores, get_trainable_mask, filter_by_mask,
)

# Calibrated empirically (see README's Phase 2 section for how this
# was chosen): anomaly_score combines two z-scored features (so
# "normal" clients cluster near 0 by construction) plus a bounded
# [0,2] cosine-dissimilarity term. A clearly-poisoned client tends to
# land well above 1.0; this is a starting point, NOT a universal
# constant — recalibrate against your own real-data results.
ANOMALY_THRESHOLD = 0.8


def run_scenario(scenario_name, use_abac, use_anomaly_detection,
                  client_data, test_data, input_dim, audit_logger, anomaly_logger):
    print(f"\n{'='*60}\nRunning scenario: {scenario_name}\n{'='*60}")

    attrs_list = default_phase1_clients()
    clients = [
        FLClient(client_id=attrs.client_id, attrs=attrs, X_train=X, y_train=y, input_dim=input_dim)
        for attrs, (X, y) in zip(attrs_list, client_data)
    ]

    global_model = build_model(input_dim=input_dim)
    global_weights = global_model.get_weights()
    # anomaly scoring uses TRAINABLE weights only (excludes BatchNorm
    # moving_mean/moving_variance, which add client-specific noise)
    trainable_mask = get_trainable_mask(global_model)
    X_test, y_test = test_data

    history = []
    # tracks, for the known-poisoned client, how many rounds it was
    # correctly rejected vs incorrectly admitted (and the reverse for
    # honest clients) — this is the key Phase 2 evaluation metric
    detection_log = []

    for round_num in range(1, N_ROUNDS + 1):
        # --- Step 1: ABAC identity gate (same as Phase 1) ---
        admitted = []
        for client in clients:
            if use_abac:
                allowed = pre_training_gate(client.attrs)
                decision = "ALLOW" if allowed else "DENY"
                reason = "" if allowed else f"role '{client.attrs.role}' not permitted"
            else:
                allowed = True
                decision = "ALLOW"
                reason = "ABAC disabled (baseline)"

            audit_logger.log(round_num, client.client_id, scenario_name, decision, reason)
            if allowed:
                admitted.append(client)

        # --- Step 2: local training, keeping each client's DELTA ---
        weights_before_round = global_weights
        trained = {}  # client_id -> (client, new_weights, num_examples, delta)
        for client in admitted:
            new_weights, n = client.train(weights_before_round, epochs=EPOCHS_PER_ROUND, batch_size=32)
            delta = compute_delta(new_weights, weights_before_round)
            trained[client.client_id] = (client, new_weights, n, delta)

        # --- Step 3: anomaly detection (Phase 2's new gate) ---
        accepted_ids = list(trained.keys())
        if use_anomaly_detection and len(trained) >= 2:
            deltas_for_scoring = {
                cid: filter_by_mask(trained[cid][3], trainable_mask) for cid in trained
            }
            scores = compute_round_anomaly_scores(deltas_for_scoring)

            accepted_ids = []
            for cid, s in scores.items():
                is_anomalous = s["anomaly_score"] > ANOMALY_THRESHOLD
                decision = "REJECT" if is_anomalous else "ACCEPT"
                anomaly_logger.log(
                    round_num, cid, scenario_name,
                    norm=s["norm"], cos_sim=s["cos_sim"], distance=s["distance"],
                    anomaly_score=s["anomaly_score"], decision=decision,
                )
                is_known_poisoned = (cid == f"client_{POISON_CLIENT_INDEX}")
                detection_log.append({
                    "round": round_num, "client_id": cid,
                    "is_known_poisoned": is_known_poisoned,
                    "decision": decision, "anomaly_score": s["anomaly_score"],
                })
                if not is_anomalous:
                    accepted_ids.append(cid)

        # --- Step 4: aggregate only ACCEPTED clients ---
        weight_updates = [trained[cid][1] for cid in accepted_ids]
        num_examples = [trained[cid][2] for cid in accepted_ids]

        if not weight_updates:
            print(f"Round {round_num}: WARNING — every admitted client was rejected; "
                  f"keeping previous global model unchanged this round.")
        else:
            global_weights = fedavg(weight_updates, num_examples)

        global_model.set_weights(global_weights)
        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = global_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = float("nan")

        print(
            f"Round {round_num}: admitted(ABAC)={list(trained.keys())} "
            f"accepted(post-anomaly)={accepted_ids} -> "
            f"acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}"
        )

        history.append({
            "scenario": scenario_name, "round": round_num,
            "accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "auc": auc, "loss": loss,
            "admitted_clients": ";".join(trained.keys()),
            "accepted_clients": ";".join(accepted_ids),
        })

    return history, detection_log


def summarize_detection(detection_log, scenario_name):
    """
    Computes the detector's own precision/recall: of all rounds the
    KNOWN-poisoned client participated in, how many did it correctly
    get REJECTED (detection rate)? Of all rounds HONEST clients
    participated in, how many were incorrectly REJECTED (false
    positive rate)?
    """
    if not detection_log:
        return None
    df = pd.DataFrame(detection_log)

    poisoned_rows = df[df["is_known_poisoned"]]
    honest_rows = df[~df["is_known_poisoned"]]

    detection_rate = (poisoned_rows["decision"] == "REJECT").mean() if len(poisoned_rows) else float("nan")
    false_positive_rate = (honest_rows["decision"] == "REJECT").mean() if len(honest_rows) else float("nan")

    print(f"\n--- Anomaly detector evaluation ({scenario_name}) ---")
    print(f"Poisoned client correctly REJECTED: {detection_rate*100:.1f}% of its rounds "
          f"({(poisoned_rows['decision']=='REJECT').sum()}/{len(poisoned_rows)})")
    print(f"Honest clients incorrectly REJECTED (false positives): {false_positive_rate*100:.1f}% of their rounds "
          f"({(honest_rows['decision']=='REJECT').sum()}/{len(honest_rows)})")
    return {"detection_rate": detection_rate, "false_positive_rate": false_positive_rate}


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(results_dir, exist_ok=True)

    audit_logger = AuditLogger(os.path.join(logs_dir, "phase2_audit_log.csv"))
    anomaly_logger = AnomalyAuditLogger(os.path.join(logs_dir, "phase2_anomaly_log.csv"))

    client_data, test_data, input_dim = prepare_data()

    all_history = []

    h_no_abac, _ = run_scenario(
        "No_ABAC", use_abac=False, use_anomaly_detection=False,
        client_data=client_data, test_data=test_data, input_dim=input_dim,
        audit_logger=audit_logger, anomaly_logger=anomaly_logger,
    )
    all_history += h_no_abac

    h_abac_only, _ = run_scenario(
        "With_ABAC_only", use_abac=True, use_anomaly_detection=False,
        client_data=client_data, test_data=test_data, input_dim=input_dim,
        audit_logger=audit_logger, anomaly_logger=anomaly_logger,
    )
    all_history += h_abac_only

    h_abac_anomaly, detection_log = run_scenario(
        "With_ABAC_plus_Anomaly", use_abac=True, use_anomaly_detection=True,
        client_data=client_data, test_data=test_data, input_dim=input_dim,
        audit_logger=audit_logger, anomaly_logger=anomaly_logger,
    )
    all_history += h_abac_anomaly

    detection_summary = summarize_detection(detection_log, "With_ABAC_plus_Anomaly")

    results_df = pd.DataFrame(all_history)
    results_path = os.path.join(results_dir, "phase2_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # --- Plot 1: accuracy/F1 comparison across all 3 scenarios ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for scenario in results_df["scenario"].unique():
            sub = results_df[results_df["scenario"] == scenario]
            axes[0].plot(sub["round"], sub["accuracy"], marker="o", label=scenario)
            axes[1].plot(sub["round"], sub["f1"], marker="o", label=scenario)
        axes[0].set_xlabel("FL Round"); axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy across 3 scenarios")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("FL Round"); axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 across 3 scenarios")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle("Phase 2: No ABAC vs ABAC-only vs ABAC+Anomaly Detection")
        fig.tight_layout()
        plot_path = os.path.join(results_dir, "phase2_accuracy_comparison.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")

        # --- Plot 2: per-client anomaly score over rounds (the KEY
        # Phase 2 plot — shows client_2 standing out from the rest) ---
        anomaly_df = pd.read_csv(os.path.join(logs_dir, "phase2_anomaly_log.csv"))
        anomaly_df = anomaly_df[anomaly_df["scenario"] == "With_ABAC_plus_Anomaly"]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for cid in sorted(anomaly_df["client_id"].unique()):
            sub = anomaly_df[anomaly_df["client_id"] == cid]
            style = "o--" if cid == f"client_{POISON_CLIENT_INDEX}" else "o-"
            ax2.plot(sub["round"], sub["anomaly_score"], style, label=cid)
        ax2.axhline(ANOMALY_THRESHOLD, color="red", linestyle=":", label=f"threshold={ANOMALY_THRESHOLD}")
        ax2.set_xlabel("FL Round")
        ax2.set_ylabel("Anomaly Score")
        ax2.set_title(f"Per-client anomaly score each round\n"
                      f"(client_{POISON_CLIENT_INDEX} is the poisoned-but-credentialed client)")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        plot2_path = os.path.join(results_dir, "phase2_anomaly_scores.png")
        fig2.savefig(plot2_path, dpi=150)
        print(f"Plot saved to: {plot2_path}")
    except ImportError:
        print("matplotlib not available, skipping plots.")

    print("\n" + "=" * 60)
    print("SUMMARY — F1 score across all 3 scenarios")
    print("=" * 60)
    print(results_df.pivot(index="round", columns="scenario", values="f1"))

    print(
        "\nKey takeaway: compare 'With_ABAC_only' (client_2 still "
        "admitted every round) against 'With_ABAC_plus_Anomaly' (check "
        "the detection rate printed above, and phase2_anomaly_scores.png "
        "— client_2 should sit visibly above the threshold line while "
        "honest clients stay below it). If detection_rate is high and "
        "false_positive_rate is low, the anomaly layer is doing its job "
        "independently of — and in addition to — ABAC's identity check."
    )
    print(f"\nAudit logs written to: logs/phase2_audit_log.csv, logs/phase2_anomaly_log.csv")


if __name__ == "__main__":
    main()
