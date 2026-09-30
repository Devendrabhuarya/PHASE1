# ABAC-Secured Federated Learning with Anomaly Detection — Phase 0

This is the Phase 0 (Week 1) starter kit: environment setup, dataset
handling, model architecture, data partitioning, and a Flower
"hello world" test. Nothing here does ABAC gating, anomaly detection,
or risk-aware aggregation yet — those come in later phases.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
.\venv\Scripts\Activate.ps1
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
