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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data.preprocess import load_raw, fit_encoders, transform_with_encoders  # noqa: E402
from clients.client_utils import partition_random, inspect_partitions  # noqa: E402
from attacks.poisoning import label_flip  # noqa: E402
from abac.policy import default_phase1_clients, pre_training_gate  # noqa: E402
from fl.fl_client import FLClient  # noqa: E402
from fl.aggregation import fedavg  # noqa: E402
from audit.logger import AuditLogger  # noqa: E402
from models.model import build_model  # noqa: E402

N_CLIENTS = 4
N_ROUNDS = 5
POISON_CLIENT_INDEX = 2   # client_2 — valid identity, poisoned data
POISON_FRACTION = 0.2      # matches paper 2's 20% label-flip
SEED = 42


def prepare_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    df = load_raw(data_path)

    # fit encoders on the FULL dataset so every partition/test split
    # shares the same consistent integer encoding
    encoders = fit_encoders(df)

    # held-out global test set, used to evaluate the aggregated model
    df_pool, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["ACTION"]
    )

    partitions = partition_random(df_pool, n_clients=N_CLIENTS, seed=SEED)
    print("\nBefore poisoning:")
    inspect_partitions(partitions)

    # poison ONLY the designated client's local partition
    partitions[POISON_CLIENT_INDEX] = label_flip(
        partitions[POISON_CLIENT_INDEX], frac=POISON_FRACTION, seed=SEED
    )
    print(f"\nAfter poisoning client_{POISON_CLIENT_INDEX} ({POISON_FRACTION*100:.0f}% label-flip):")
    inspect_partitions(partitions)

    X_test, y_test = transform_with_encoders(df_test, encoders)

    client_data = []
    for part in partitions:
        X, y = transform_with_encoders(part, encoders)
        client_data.append((X, y))

    input_dim = X_test.shape[1]
    return client_data, (X_test, y_test), input_dim


def run_scenario(scenario_name, use_abac, client_data, test_data, input_dim, logger):
    print(f"\n{'='*60}\nRunning scenario: {scenario_name}\n{'='*60}")

    attrs_list = default_phase1_clients()
    clients = [
        FLClient(client_id=attrs.client_id, attrs=attrs, X_train=X, y_train=y, input_dim=input_dim)
        for attrs, (X, y) in zip(attrs_list, client_data)
    ]

    global_model = build_model(input_dim=input_dim)
    global_weights = global_model.get_weights()
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
            w, n = client.train(global_weights, epochs=1, batch_size=32)
            weight_updates.append(w)
            num_examples.append(n)

        global_weights = fedavg(weight_updates, num_examples)
        global_model.set_weights(global_weights)
        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)

        admitted_ids = [c.client_id for c in admitted]
        print(f"Round {round_num}: admitted={admitted_ids} -> test_accuracy={acc:.4f}")

        history.append({
            "scenario": scenario_name,
            "round": round_num,
            "accuracy": acc,
            "loss": loss,
            "admitted_clients": ";".join(admitted_ids),
        })

    return history


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    logger = AuditLogger(os.path.join(os.path.dirname(__file__), "..", "logs", "phase1_audit_log.csv"))

    client_data, test_data, input_dim = prepare_data()

    history_no_abac = run_scenario("No_ABAC", use_abac=False,
                                    client_data=client_data, test_data=test_data,
                                    input_dim=input_dim, logger=logger)
    history_with_abac = run_scenario("With_ABAC", use_abac=True,
                                      client_data=client_data, test_data=test_data,
                                      input_dim=input_dim, logger=logger)

    results_df = pd.DataFrame(history_no_abac + history_with_abac)
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for scenario in results_df["scenario"].unique():
            sub = results_df[results_df["scenario"] == scenario]
            ax.plot(sub["round"], sub["accuracy"], marker="o", label=scenario)
        ax.set_xlabel("FL Round")
        ax.set_ylabel("Global Model Test Accuracy")
        ax.set_title("Phase 1: No ABAC vs With ABAC (identity-only)\n"
                      "Sybil client is blocked either way, but accuracy still\n"
                      "degrades from the poisoned-but-credentialed client in BOTH")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_accuracy_comparison.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results_df.pivot(index="round", columns="scenario", values="accuracy"))
    print(
        "\nKey takeaway: With_ABAC correctly excludes client_3 (invalid "
        "identity) every round, but client_2 (poisoned, VALID identity) "
        "is still admitted in both scenarios -> accuracy degradation "
        "from poisoning persists even WITH identity-only ABAC. This is "
        "exactly the gap Phase 2's anomaly detection is built to close."
    )
    print(f"\nFull audit log written to: logs/phase1_audit_log.csv")


if __name__ == "__main__":
    main()
