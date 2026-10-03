# ABAC-Secured Federated Learning with Anomaly Detection — Phase 0

This is the Phase 0 (Week 1) starter kit: environment setup, dataset
handling, model architecture, data partitioning, and a Flower
"hello world" test. Nothing here does ABAC gating, anomaly detection,
or risk-aware aggregation yet — those come in later phases.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This project is designed around the **Amazon Employee Access
Challenge** dataset (matches the base paper's setup):

1. Download `train.csv` from:
   https://www.kaggle.com/c/amazon-employee-access-challenge/data
2. Place it at `data/train.csv`

### No Kaggle access yet? Use the synthetic fallback

To unblock development immediately, generate a synthetic dataset with
the *same column schema* as the real one:

```bash
python data/generate_synthetic.py
```

This writes `data/train.csv` with realistic-looking (but fake) data.
**Swap in the real dataset as soon as you have it** — no other code
needs to change, since the schema matches exactly. Do NOT use the
synthetic data for your final reported experiments/results.

## Day-by-day checklist

- [ ] **Day 1** — Install everything above, verify:
  ```bash
  python -c "import flwr, tensorflow, sklearn; print('OK')"
  ```
- [ ] **Day 2** — Get the dataset (real or synthetic), inspect it:
  ```bash
  python data/preprocess.py
  ```
  Note down: row count, class balance, number of unique departments.

- [ ] **Day 3** — Repo structure (already set up in this zip).

- [ ] **Day 4** — Model architecture (already in `models/model.py`,
  matches the base paper: 512→256→128→64 Dense layers, BatchNorm,
  LeakyReLU, Dropout, sigmoid output). Smoke-test it:
  ```bash
  python models/model.py
  ```

- [ ] **Day 5** — Sanity-check centralized training (no FL yet):
  ```bash
  python experiments/phase0_sanity_check.py
  ```
  Confirms the model actually learns something on your data before
  you add FL/ABAC complexity on top.

- [ ] **Day 6** — Partition the dataset across simulated clients:
  ```bash
  python clients/client_utils.py
  ```
  Check the printed row counts / class balance per client — no
  client should end up with near-zero rows or all-one-class.

- [ ] **Day 7** — Flower hello-world (confirms FL plumbing works):

  Terminal 1:
  ```bash
  python experiments/phase0_flower_server.py
  ```
  Terminal 2 and 3 (run the same command in each, simulating 2 clients):
  ```bash
  python experiments/phase0_flower_client.py
  ```
  You should see 2 training rounds complete with no errors.

## Project structure

```
project/
├── data/
│   ├── train.csv                    <- put real or synthetic data here
│   ├── preprocess.py                 <- loading + encoding
│   └── generate_synthetic.py         <- fallback dataset generator
├── clients/
│   └── client_utils.py               <- partitioning across N clients
├── models/
│   └── model.py                       <- Keras architecture (matches paper 2)
├── abac/
│   └── policy.py                       <- stub ABAC gate (expanded in Phase 1+)
├── logs/                                <- (empty — audit logs land here later)
├── experiments/
│   ├── phase0_sanity_check.py            <- Day 5
│   ├── phase0_flower_server.py            <- Day 7
│   └── phase0_flower_client.py            <- Day 7
├── requirements.txt
└── README.md
```

## What's NOT in Phase 0 (comes later)

- Real ABAC-gated FL rounds using your actual partitioned client data
  (Phase 1)
- Poisoning attack injection (Phase 1)
- Anomaly detection: update norm / cosine similarity / distance
  (Phase 2)
- Risk-aware weighted aggregation (Phase 3)
- Reputation tracking + revocation loop (Phase 4)
- Attack evaluation suite (Phase 5)

Once every box above is checked, you're ready for Phase 1.

---

# Phase 1 — ABAC-Gated FL Baseline + Poisoning (Weeks 2-4)

Builds the real experiment: 4 simulated smart-home clients, one
poisoned (valid identity, bad data), one Sybil (invalid identity).
Runs two scenarios — No ABAC vs With ABAC (identity-only) — and shows
that ABAC correctly blocks the Sybil client but CANNOT catch the
poisoned-but-credentialed one. This gap is exactly what Phase 2's
anomaly detection is built to close.

## Design note: why this isn't using Flower's simulation API

Phase 0 already proved real Flower client/server networking works
(`phase0_flower_*.py`). From Phase 1 onward, the project uses a
lightweight, self-contained round-driver instead of Flower's
simulation/Ray backend. This is a deliberate scoping choice: later
phases need fine-grained per-round control (ABAC gating, anomaly
scoring, risk-aware weighting, reputation updates) that is much
simpler to implement directly than to bend Flower's `Strategy` class
around. The `FLClient` class in `fl/fl_client.py` mirrors Flower's
`NumPyClient` interface on purpose, so it stays conceptually
consistent and could be ported back to real Flower later if desired.
State this explicitly in your report as a scoping decision, not an
oversight.

## Run it

```bash
python experiments/phase1_run.py
```

This will:
1. Load `data/train.csv`, hold out 20% as a global test set
2. Partition the rest across 4 clients
3. Poison `client_2`'s labels (20% label-flip) — identity stays valid
4. Run 5 rounds of "No_ABAC" (all 4 clients admitted every round)
5. Run 5 rounds of "With_ABAC" (client_3 — invalid role — denied
   every round; client_2 still admitted since its credentials are
   fine)
6. Save results to `results/phase1_results.csv` and
   `results/phase1_accuracy_comparison.png`
7. Log every ABAC decision to `logs/phase1_audit_log.csv`

Then inspect the audit trail:
```bash
python experiments/phase1_inspect_audit_log.py
```

## What to look for in your results

- **client_3 should be DENY'd every round** in the With_ABAC scenario,
  ALLOW'd every round in No_ABAC — confirms the ABAC gate works
  correctly for identity-based attacks (Sybil)
- **client_2 should be ALLOW'd every round in BOTH scenarios** — this
  is the expected limitation: its credentials are valid, so identity
  checking alone cannot flag it as malicious
- Accuracy in the With_ABAC scenario should still be affected by
  client_2's poisoned contributions, similar to No_ABAC — proving
  ABAC alone doesn't solve poisoning

## Phase 1 checklist

- [ ] `phase1_run.py` runs cleanly end to end
- [ ] Audit log clearly shows client_3 denied, client_2 admitted
      (With_ABAC scenario)
- [ ] Accuracy comparison plot saved and makes visual sense
- [ ] **Swap in the real Kaggle dataset** before treating any numbers
      as final results — synthetic data only proves the pipeline
      works, not the actual security findings
- [ ] Re-run with the real dataset and more rounds (e.g. 10-15) once
      available, to get a cleaner accuracy-degradation signal

Once this is solid, you're ready for **Phase 2** (anomaly detection:
update norm, cosine similarity, distance — the piece that finally
catches client_2).
