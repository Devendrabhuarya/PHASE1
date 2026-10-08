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

## Dataset switch: CICIoT2023 (replaces Amazon Employee Access)

As of Phase 1, the project uses **CICIoT2023** instead of the Amazon
Employee Access dataset. CICIoT2023 is real IoT network-traffic data
with genuine attack labels (DDoS, DoS, Mirai, Recon, Spoofing,
BruteForce, Web-based attacks) plus benign traffic — which maps
naturally onto an access-control decision: benign traffic = allow,
attack traffic = deny. This is a better fit for the project's framing
than the original Amazon dataset's corporate role/department IDs.

**The old Amazon-based files (`data/preprocess.py`,
`data/generate_synthetic.py`) are left in place for reference but are
no longer used by the experiments** — `experiments/phase1_run.py` now
imports from `data/preprocess_ciciot.py` instead.

### The real dataset is 63 separate files, not one CSV

Downloaded from https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/,
the dataset comes as `Merged01.csv` ... `Merged63.csv`. **Attack types
are NOT evenly spread across these files** — some attack subtypes may
be concentrated in just a handful of the 63 files. `load_and_sample()`
scans EVERY file in the folder and accumulates up to
`MAX_ROWS_PER_RAW_LABEL` rows per unique raw label **cumulatively
across all files**, so rare classes still get picked up even if they
only appear in file #47.

Confirmed real column schema (39 numeric features + `Label`):
```
Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number,
syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number,
ece_flag_number, cwr_flag_number, ack_count, syn_count, fin_count,
rst_count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP,
ARP, ICMP, IGMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size,
IAT, Number, Variance, Label
```
The loader auto-detects columns at runtime rather than hardcoding
this list, so it will keep working even if your exact files have a
few extra/different columns.

### What's different about CICIoT2023's data handling

1. **Continuous numeric features, not categorical IDs.** No
   label-encoding needed — instead, features are **standardized**
   (`StandardScaler`, fit on the train pool only, after the test
   holdout split) since raw feature scales vary wildly, which matters
   a lot once Phase 2's anomaly detection (distance/cosine-similarity
   based) gets added.

2. **~34 fine-grained attack labels, collapsed to binary.** The raw
   `Label` column has many values (`DDoS-SYN_Flood`,
   `Mirai-greeth_flood`, `BenignTraffic`, ...). `map_binary_label()`
   collapses these to a single `ACTION` column — 1 = benign (allow),
   0 = any attack (deny) — by matching the substring "benign"
   case-insensitively. The rest of the pipeline (partitioning,
   poisoning, ABAC, FedAvg, metrics) is UNCHANGED, since it just
   expects an `ACTION` column, exactly like before.

3. **Separator auto-detection.** Some CICIoT2023 exports are
   tab-separated despite the `.csv` extension — `detect_separator()`
   peeks at the first line of the first file and picks tab or comma
   automatically. Override with `SEP_OVERRIDE` at the top of
   `data/preprocess_ciciot.py` if it ever guesses wrong.

### Using your real files

1. Download/extract all `Merged01.csv` ... `Merged63.csv` files into
   `data/ciciot_raw/` (create this folder)
2. Run `python data/preprocess_ciciot.py data/ciciot_raw/` on its own
   first, to sanity-check: confirm the detected separator and label
   column are correct, and the printed benign/attack percentage looks
   sane. **If auto-detection picks the wrong label column**, set
   `LABEL_COL_OVERRIDE` at the top of `data/preprocess_ciciot.py`.
3. For a quick first pass during development (scanning all 63 files
   fully can be slow), pass `max_files=N` to `load_and_sample()` in
   `prepare_data()` — e.g. `max_files=10` to only scan the first 10
   files. **Use the full scan (no `max_files` limit) for your actual
   reported results**, since rare attack classes may only live in
   later files.

### No real files yet? Use the synthetic fallback

```bash
python data/generate_synthetic_ciciot.py
```

Writes 3 files (`data/ciciot_raw/Merged01.csv`, `02.csv`, `03.csv`)
matching the real confirmed column schema, each with a DIFFERENT
uneven label distribution (mimicking how the real 63 files aren't
evenly shuffled) — this exercises the multi-file accumulation logic
properly, not just a trivial single-file case. Still NOT a substitute
for the real data in your final results.

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

## Important: class imbalance (read this before trusting any accuracy number)

The real Amazon Employee Access dataset is heavily imbalanced —
roughly **94% approved / 6% denied**. Without correction, a model can
trivially get ~94% "accuracy" by predicting the majority class for
*every* input, without learning anything at all. This produces a
symptom that's easy to misdiagnose as "my pipeline is broken": the
exact same accuracy value repeated across every round and every
scenario, since the model never actually changes its (trivial)
behavior.

This starter kit already applies two fixes for you:
- **Class-weighted training** (`fl/fl_client.py`) — each client
  balances its own local loss using its own class distribution
- **3 epochs per round instead of 1** — gives the model enough steps
  per round to move past the trivial solution
- **Precision / recall / F1 / AUC reported alongside accuracy** — F1
  in particular won't lie to you the way raw accuracy can

**If you still see a flat, repeated accuracy/F1 across every round**,
something is still collapsing — check that `python data/preprocess.py`
shows a reasonable class balance for your actual data, and that every
client partition contains both classes (`inspect_partitions` prints
this). Don't trust a single accuracy number under imbalance — always
look at the F1 plot alongside it before drawing conclusions.

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

---

# Phase 2 — Anomaly Detection on Model Updates

Adds a second gate after ABAC. Each ABAC-admitted client's update
(delta = weights after training minus global weights before) is scored
against its peers, and excluded from aggregation if its score is
above `ANOMALY_THRESHOLD` (0.8, in `experiments/phase2_run.py`).

Features per client per round (`anomaly/scoring.py`):
- norm of the delta
- cosine similarity to the mean of the OTHER clients' deltas (leave-one-out)
- distance from that mean

`anomaly_score = (z(norm) + (1 - cos_sim) + z(distance)) / 3`

## Run it
```bash
python experiments/phase2_run.py
```
Outputs: `results/phase2_results.csv`, `results/phase2_accuracy_comparison.png`,
`results/phase2_anomaly_scores.png` (the key plot), `logs/phase2_anomaly_log.csv`.

## Findings during development (worth reporting honestly)

1. **Scoring all 26 weight arrays failed** (0% detection). BatchNorm
   moving_mean/moving_variance update from each client's local batch
   statistics, not gradients, and added noise that hid the poisoning
   signal. Fix: score TRAINABLE weights only (`get_trainable_mask`).
2. **Round 1 is not detected.** All clients start from the same random
   weights, so updates look alike. Expect a "burn-in" round.
3. **Small N is fragile.** After ABAC only 3 clients remain, so z-scores
   are computed over 3 points. Test with more clients (Phase 5).
4. **Threshold 0.8 was calibrated on synthetic data.** Recalibrate on
   your real CICIoT2023 run.
5. **Model quality did not clearly improve** in the synthetic run
   (F1 of ABAC-only vs ABAC+Anomaly is similar). Detection works, but a
   20% flip on a ~6% minority class barely hurts the global model here.
   Try a stronger attack (higher flip fraction) before claiming an
   accuracy benefit.
