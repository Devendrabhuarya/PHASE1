"""
Anomaly scoring on FEDERATED LEARNING MODEL UPDATES (not on network traffic).

Each round every admitted client trains from the same global weights, giving a
delta  dW_k = weights_after - weights_before. We compare deltas between clients.

WHY THIS VERSION (v2) REPLACED THE FIRST ONE
The first version z-scored norm/distance across the clients present in the same
round. With ~3 clients a z-score is bounded (max ~1.41) and somebody is ALWAYS
the top value, so the detector flagged "whoever is most different today" even
when everyone was honest (30-40% false positives on real data). v2 uses
scale-free measures that do NOT force an outlier to exist:

  cos_sim     best cosine between dW_k and any peer's dW      absolute scale [-1,1]
  dist_ratio  nearest-peer distance / median of all nearest-peer distances
  norm_dev    max(r, 1/r), r = ||dW_k|| / median of all norms
(best/nearest peer, not the mean of the others: the mean is contaminated by the very
 outlier we want to find, which made honest clients look anomalous.)

If all clients are alike, dist_ratio ~ 1, norm_dev ~ 1 and cos_sim is high, so
nobody is flagged. Thresholds are CALIBRATED on a clean (attack-free) run of the
same setup instead of being guessed (see calibrate_thresholds).

anomaly_score = max( dist_ratio / DIST_MAX, (1 - cos_sim) / (1 - COS_MIN),
                     norm_dev / NORM_MAX );  flagged when anomaly_score > 1.

Limits: needs >= MIN_PEERS (3) clients in the round, otherwise no judgement is
made (with 2 clients you cannot tell which one is odd). Updates are compared on
TRAINABLE weights only (BatchNorm moving statistics are excluded, they come from
local batch statistics rather than gradients and add noise).
"""

from dataclasses import dataclass, asdict

import numpy as np

MIN_PEERS = 3

# Which trainable arrays of the update are compared. Label flipping mainly changes the
# decision boundary, which lives in the LAST layers; hidden layers add noise. Measured
# separation of the poisoned client (see experiments/phase2_scope_diagnostic.py):
# 'all' ~1.0x (none), 'last_block' ~1.2x, 'output' ~1.7x. Verify on your own data.
SCOPES = {"all": None, "last_block": 6, "output": 2}


def select_scope(trainable_delta: list, scope: str) -> list:
    """trainable_delta: delta arrays after filter_by_mask (trainable only)."""
    k = SCOPES[scope]
    return trainable_delta if k is None else trainable_delta[-k:]


@dataclass
class Thresholds:
    cos_min: float = 0.0      # flag if cos_sim falls below this
    dist_max: float = 1.5     # flag if dist_ratio exceeds this
    norm_max: float = 2.0     # flag if norm_dev exceeds this


def flatten_weights(weights_list) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights_list])


def compute_delta(weights_after, weights_before):
    return [a - b for a, b in zip(weights_after, weights_before)]


def get_trainable_mask(model) -> list:
    """Booleans aligned with model.get_weights(): True for trainable arrays
    (kernel, bias, BN gamma/beta), False for BN moving mean/variance."""
    return [w.trainable for w in model.weights]


def filter_by_mask(weights_list, mask: list):
    return [w for w, keep in zip(weights_list, mask) if keep]


def compute_round_features(client_deltas: dict) -> dict:
    """
    client_deltas: {client_id: list-of-arrays delta} for clients admitted this round.
    Returns {client_id: {norm, cos_sim, distance, dist_ratio, norm_dev, judged}}.

    Peer comparison uses the BEST-MATCHING peer, not the mean of the others. With ~3 clients
    the mean of "the others" contains the poisoned update for every honest client, which
    drags the reference away and makes honest clients look anomalous (80% false positives
    in testing). Honest clients have an honest peer to match; a poisoned one does not.
      cos_sim     max over peers of cosine(dW_k, dW_j)           (high = has a similar peer)
      distance    distance to the NEAREST peer
      dist_ratio  distance / median of all clients' nearest-peer distances
      norm_dev    max(r, 1/r), r = ||dW_k|| / median of all norms
    Assumes an honest majority (with 3 clients, at most 1 attacker).
    With fewer than MIN_PEERS clients nobody is judged (neutral values).
    """
    ids = list(client_deltas.keys())
    flat = {c: flatten_weights(client_deltas[c]) for c in ids}
    norms = {c: float(np.linalg.norm(flat[c])) for c in ids}

    if len(ids) < MIN_PEERS:
        return {c: {"norm": norms[c], "cos_sim": 1.0, "distance": 0.0,
                    "dist_ratio": 1.0, "norm_dev": 1.0, "judged": False} for c in ids}

    cos_best, nn = {}, {}
    for c in ids:
        cs, ds = [], []
        for o in ids:
            if o == c:
                continue
            cs.append(float(np.dot(flat[c], flat[o]) / (norms[c] * norms[o] + 1e-12)))
            ds.append(float(np.linalg.norm(flat[c] - flat[o])))
        cos_best[c], nn[c] = max(cs), min(ds)

    med_nn = float(np.median(list(nn.values()))) + 1e-12
    med_norm = float(np.median(list(norms.values()))) + 1e-12
    out = {}
    for c in ids:
        r = norms[c] / med_norm
        out[c] = {"norm": norms[c], "cos_sim": cos_best[c], "distance": nn[c],
                  "dist_ratio": nn[c] / med_nn, "norm_dev": float(max(r, 1.0 / max(r, 1e-12))),
                  "judged": True}
    return out


def score_features(features: dict, th: Thresholds) -> dict:
    """Adds anomaly_score and flagged to each client's feature dict."""
    scored = {}
    for c, f in features.items():
        if not f["judged"]:
            score = 0.0
        else:
            score = max(
                f["dist_ratio"] / th.dist_max,
                (1.0 - f["cos_sim"]) / max(1.0 - th.cos_min, 1e-6),
                f["norm_dev"] / th.norm_max,
            )
        scored[c] = {**f, "anomaly_score": float(score), "flagged": bool(f["judged"] and score > 1.0)}
    return scored


def compute_round_anomaly_scores(client_deltas: dict, th: Thresholds = None) -> dict:
    return score_features(compute_round_features(client_deltas), th or Thresholds())


def calibrate_thresholds(honest_feature_rows: list, margin: float = 1.25) -> Thresholds:
    """
    Sets thresholds from features observed in an attack-free run.
    honest_feature_rows: list of feature dicts (judged ones) from honest clients.
      dist_max = margin * max honest dist_ratio
      norm_max = margin * max honest norm_dev
      cos_min  = min honest cos_sim minus a (margin-1)*|min| allowance, floored at -1
    Larger `margin` -> fewer false positives, more missed attacks.
    """
    rows = [r for r in honest_feature_rows if r.get("judged")]
    if not rows:
        return Thresholds()
    max_dist = max(r["dist_ratio"] for r in rows)
    max_norm = max(r["norm_dev"] for r in rows)
    min_cos = min(r["cos_sim"] for r in rows)
    cos_min = min_cos - (margin - 1.0) * max(abs(min_cos), 0.1)
    return Thresholds(cos_min=float(max(cos_min, -1.0)),
                      dist_max=float(max_dist * margin),
                      norm_max=float(max_norm * margin))


def thresholds_to_dict(th: Thresholds) -> dict:
    return asdict(th)


def risk_weight(anomaly_score: float, low: float = 1.0, high: float = 1.4) -> float:
    """
    Risk-aware aggregation weight in [0, 1] (Phase 3).
      score <= low  -> 1.0  (full weight; low=1.0 is the CALIBRATED honest limit, so honest
                       clients keep full weight. A ramp starting below the honest range
                       down-weights everyone equally and, because FedAvg normalizes, does nothing.)
      score >= high -> 0.0  (excluded)
      in between    -> linear ramp down
    Compared with the hard rule (reject only when score > 1), this lowers the
    influence of a suspicious client smoothly, so a missed detection near the
    threshold still gets reduced weight instead of full weight.
    """
    if anomaly_score <= low:
        return 1.0
    if anomaly_score >= high:
        return 0.0
    return float((high - anomaly_score) / (high - low))
