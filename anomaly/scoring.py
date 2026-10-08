"""
Anomaly scoring on FEDERATED LEARNING MODEL UPDATES (not on network
traffic — that's the global model's job).

Each round, every admitted client trains locally starting from the
same global weights, producing a weight DELTA (ΔW = weights_after -
weights_before). Honest clients' deltas tend to point in roughly the
same direction (they're all learning from similarly-distributed
traffic); a poisoned client's delta — even with perfectly VALID ABAC
credentials — tends to diverge, because it's training on data where
some attacks are mislabeled as benign, pushing its local weights in a
direction the honest clients are NOT pushing toward.

Three features per client, each round:
  - norm:      ||ΔW_k||         — how hard this client is pushing
  - cos_sim:   cos(ΔW_k, mean of OTHER clients' ΔW)  — same direction
               as the rest of the group, or not?
  - distance:  ||ΔW_k - mean of OTHER clients' ΔW||  — how far from
               the group's consensus direction

"Mean of OTHER clients" (leave-one-out) rather than "mean of everyone
including this client" is used deliberately — with only 3-4 admitted
clients per round, including a client in its own reference average
dilutes exactly the signal we're trying to detect (a single outlier
pulls the mean toward itself). Excluding it gives a cleaner
"how different is this client from everyone else" comparison, and
matches the spirit of established FL defenses (Krum, trimmed mean)
that also compare each client against the rest of the group.

These combine into one anomaly_score per client per round (z-scored
across whichever clients are being compared THIS round, so no
hardcoded absolute scale is needed).
"""

import numpy as np


def flatten_weights(weights_list) -> np.ndarray:
    """weights_list: what model.get_weights() returns — a list of
    numpy arrays, one per layer (including BatchNorm's gamma/beta/
    moving_mean/moving_variance). Flattened into one 1D vector."""
    return np.concatenate([w.flatten() for w in weights_list])


def get_trainable_mask(model) -> list:
    """
    Returns a list of booleans, same length and order as
    model.get_weights(), marking which arrays are TRAINABLE (kernel,
    bias, BatchNorm gamma/beta) vs NON-TRAINABLE (BatchNorm
    moving_mean/moving_variance).

    This matters a lot for anomaly scoring: moving_mean/moving_variance
    update via an exponential moving average of each client's own
    LOCAL batch statistics — not via gradient descent — so they drift
    based on each client's data distribution somewhat independently of
    whether that client is poisoned. Empirically (see README), this
    adds enough noise to the full flattened vector that it can mask
    the poisoning signal, especially in early rounds. Filtering down
    to trainable-only weights gives cleaner separation.

    Derived from the model itself (via model.weights[i].trainable)
    rather than hardcoded indices, so it stays correct even if the
    architecture in models/model.py changes.
    """
    return [w.trainable for w in model.weights]


def filter_by_mask(weights_list, mask: list):
    """Keeps only the entries where mask[i] is True, preserving order."""
    return [w for w, keep in zip(weights_list, mask) if keep]


def compute_delta(weights_after, weights_before):
    """Per-layer subtraction, returned in the same list-of-arrays
    shape as model.get_weights() (NOT flattened yet)."""
    return [w_after - w_before for w_after, w_before in zip(weights_after, weights_before)]


def _zscore_safe(values: np.ndarray) -> np.ndarray:
    """Standard z-score, but returns zeros instead of dividing by zero
    when every value is identical (std == 0) — this happens e.g. with
    only one client, or in degenerate test cases."""
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_round_anomaly_scores(client_deltas: dict) -> dict:
    """
    Args:
        client_deltas: {client_id: delta_weights_list} for every
            client ADMITTED this round (i.e. already passed ABAC).

    Returns:
        {client_id: {"norm": float, "cos_sim": float, "distance": float,
                      "anomaly_score": float}}

        anomaly_score is NOT bounded to a fixed range — it's a
        combination of two z-scored features (norm, distance; z-scored
        relative to the other clients admitted THIS round) and one
        bounded feature (1 - cos_sim, range [0, 2]). Higher = more
        anomalous relative to its peers this round. There is no
        universal "safe" threshold — Phase 2's experiment calibrates
        one empirically against the known-poisoned client (see
        ANOMALY_THRESHOLD in experiments/phase2_run.py).
    """
    ids = list(client_deltas.keys())
    flat = {cid: flatten_weights(client_deltas[cid]) for cid in ids}

    raw = {}
    for cid in ids:
        others = [flat[other] for other in ids if other != cid]
        if not others:
            # only one client admitted this round — no peers to
            # compare against, so there's no meaningful "relative"
            # anomaly signal. Treat as neutral (not anomalous).
            raw[cid] = {"norm": float(np.linalg.norm(flat[cid])), "cos_sim": 1.0, "distance": 0.0}
            continue

        mean_others = np.mean(others, axis=0)
        norm_k = float(np.linalg.norm(flat[cid]))
        denom = norm_k * np.linalg.norm(mean_others) + 1e-8
        cos_sim = float(np.dot(flat[cid], mean_others) / denom)
        distance = float(np.linalg.norm(flat[cid] - mean_others))
        raw[cid] = {"norm": norm_k, "cos_sim": cos_sim, "distance": distance}

    norms = [raw[cid]["norm"] for cid in ids]
    distances = [raw[cid]["distance"] for cid in ids]
    norm_z = _zscore_safe(norms)
    dist_z = _zscore_safe(distances)

    scores = {}
    for idx, cid in enumerate(ids):
        anomaly = float((norm_z[idx] + (1.0 - raw[cid]["cos_sim"]) + dist_z[idx]) / 3.0)
        scores[cid] = {**raw[cid], "anomaly_score": anomaly}
    return scores
