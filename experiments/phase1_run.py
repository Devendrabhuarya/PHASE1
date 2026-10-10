"""
Phase 1: ABAC-gated Federated Learning baseline, with a poisoning
attack and a Sybil (invalid-identity) client.

Runs TWO scenarios over the same data/attack setup:

  Scenario A — "No ABAC": every client (honest, poisoned, AND the
      invalid-identity Sybil client) is admitted into every round,
      no gating at all.

  Scenario B — "With ABAC" (identity-only, matching the base paper):
      the pre-training ABAC gate runs each round. The Sybil client
      (invalid role) is correctly denied. BUT the poisoned client
      (client_2) has perfectly valid credentials, so it is still
      admitted — ABAC alone cannot catch it.

This produces the key Phase 1 result that motivates Phase 2: ABAC
fixes the Sybil/identity problem but NOT the "valid identity, bad
behavior" problem.

Run from the project root:
    python experiments/phase1_run.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # noqa: E402

from common_setup import (  # noqa: E402
    N_ROUNDS, EPOCHS_PER_ROUND, SEED, prepare_data, set_seed,
)
import copy  # noqa: E402
from abac.policy import default_phase1_clients, pre_training_gate  # noqa: E402
from fl.fl_client import FLClient  # noqa: E402
from fl.aggregation import fedavg  # noqa: E402
from audit.logger import AuditLogger  # noqa: E402
from models.model import build_model  # noqa: E402


def run_scenario(scenario_name, use_abac, client_data, test_data, input_dim, logger, init_weights):
    print(f"\n{'='*60}\nRunning scenario: {scenario_name}\n{'='*60}")

    set_seed(SEED)   # same randomness for every scenario
    attrs_list = default_phase1_clients()
    clients = [
        FLClient(client_id=attrs.client_id, attrs=attrs, X_train=X, y_train=y, input_dim=input_dim)
        for attrs, (X, y) in zip(attrs_list, client_data)
    ]

    global_model = build_model(input_dim=input_dim)
    global_weights = copy.deepcopy(init_weights)   # same starting weights for every scenario
    global_model.set_weights(global_weights)
    X_test, y_test = test_data

    history = []

    for round_num in range(1, N_ROUNDS + 1):
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

            logger.log(round_num, client.client_id, scenario_name, decision, reason)
            if allowed:
                admitted.append(client)

        weight_updates, num_examples = [], []
        for client in admitted:
            w, n = client.train(global_weights, epochs=EPOCHS_PER_ROUND, batch_size=32)
            weight_updates.append(w)
            num_examples.append(n)

        global_weights = fedavg(weight_updates, num_examples)
        global_model.set_weights(global_weights)
        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)

        # Accuracy alone is misleading under class imbalance (the real
        # dataset is ~94%/6%) — a model that always predicts the
        # majority class scores ~94% without learning anything. Report
        # precision/recall/F1/AUC too so a collapsed model is visible.
        y_pred_proba = global_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = float("nan")  # happens if y_pred_proba has no variance at all

        admitted_ids = [c.client_id for c in admitted]
        print(
            f"Round {round_num}: admitted={admitted_ids} -> "
            f"acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"f1={f1:.4f} auc={auc:.4f}"
        )

        history.append({
            "scenario": scenario_name,
            "round": round_num,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "loss": loss,
            "admitted_clients": ";".join(admitted_ids),
        })

    return history


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    logger = AuditLogger(os.path.join(os.path.dirname(__file__), "..", "logs", "phase1_audit_log.csv"))

    client_data, test_data, input_dim = prepare_data()

    set_seed(SEED)
    init_weights = build_model(input_dim=input_dim).get_weights()

    history_no_abac = run_scenario("No_ABAC", use_abac=False,
                                    client_data=client_data, test_data=test_data,
                                    input_dim=input_dim, logger=logger, init_weights=init_weights)
    history_with_abac = run_scenario("With_ABAC", use_abac=True,
                                      client_data=client_data, test_data=test_data,
                                      input_dim=input_dim, logger=logger, init_weights=init_weights)

    results_df = pd.DataFrame(history_no_abac + history_with_abac)
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for scenario in results_df["scenario"].unique():
            sub = results_df[results_df["scenario"] == scenario]
            axes[0].plot(sub["round"], sub["accuracy"], marker="o", label=scenario)
            axes[1].plot(sub["round"], sub["f1"], marker="o", label=scenario)

        axes[0].set_xlabel("FL Round")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy (can be misleading under class\nimbalance — see F1 plot alongside)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("FL Round")
        axes[1].set_ylabel("F1 Score (minority class)")
        axes[1].set_title("F1 — the metric that actually reveals\nwhether the model learned anything")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.suptitle("Phase 1: No ABAC vs With ABAC (identity-only)")
        fig.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_accuracy_comparison.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\n" + "=" * 60)
    print("SUMMARY — Accuracy (can be misleading under imbalance)")
    print("=" * 60)
    print(results_df.pivot(index="round", columns="scenario", values="accuracy"))

    print("\n" + "=" * 60)
    print("SUMMARY — F1 score (more trustworthy under class imbalance)")
    print("=" * 60)
    print(results_df.pivot(index="round", columns="scenario", values="f1"))

    print(
        "\nIf accuracy is ~constant across every round/scenario but F1 "
        "is also flat near 0, the model has collapsed to predicting the "
        "majority class only — i.e. it has NOT actually learned "
        "anything, and accuracy is just reporting the dataset's class "
        "imbalance back at you. If that happens even after class "
        "weighting + more epochs, try: more epochs per round, a lower "
        "learning rate, or checking that EVERY client partition "
        "actually contains both classes."
    )
    print(
        "\nKey takeaway (once the model is actually learning): "
        "With_ABAC correctly excludes client_3 (invalid identity) every "
        "round, but client_2 (poisoned, VALID identity) is still "
        "admitted in both scenarios -> degradation from poisoning "
        "persists even WITH identity-only ABAC. This is exactly the "
        "gap Phase 2's anomaly detection is built to close."
    )
    print(f"\nFull audit log written to: logs/phase1_audit_log.csv")


if __name__ == "__main__":
    main()
