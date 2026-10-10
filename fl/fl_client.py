"""
FL client wrapper.

Phase 0 proved real Flower client/server plumbing works end-to-end
(experiments/phase0_flower_*.py). Phase 1 onward uses a lightweight,
self-contained round-driver instead of Flower's simulation/Ray
backend, because later phases need fine-grained control per round
(ABAC gating, anomaly scoring, risk-aware weighting, reputation
updates) that is easiest to implement directly. The client interface
below mirrors Flower's NumPyClient shape (get/set weights, train,
evaluate) so it stays conceptually consistent and easy to port back
to real Flower later if desired.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from models.model import build_model  # noqa: E402


def compute_class_weight(y):
    """
    Standard balanced class-weighting: w_c = n_total / (n_classes * n_c).
    Without this, a client training on a heavily imbalanced local split
    (the real Amazon Employee Access dataset is ~94%/6%) will trivially
    minimize binary cross-entropy by predicting the majority class for
    everything — accuracy then looks high but the model has learned
    nothing, and that collapsed accuracy stops reflecting anything
    (poisoning, ABAC, risk weighting...) you're trying to measure.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    n_classes = len(classes)
    weights = {}
    for c, n_c in zip(classes, counts):
        weights[int(c)] = n_total / (n_classes * n_c) if n_c > 0 else 1.0
    return weights


class FLClient:
    def __init__(self, client_id: str, attrs, X_train, y_train, input_dim: int):
        self.client_id = client_id
        self.attrs = attrs  # abac.policy.ClientAttributes
        self.X_train = X_train
        self.y_train = y_train
        self.model = build_model(input_dim=input_dim)

    def train(self, global_weights, epochs: int = 3, batch_size: int = 32,
              use_class_weight: bool = True, class_weight: dict = None):
        """
        Loads the current global weights, trains locally for one or
        more epochs, and returns (updated_weights, num_examples).

        use_class_weight=True (default) applies balanced class
        weighting computed from this client's OWN local label
        distribution — this is local, per-client information a real
        device would naturally have, so it's a legitimate thing for
        the client to apply before submitting its update.
        """
        self.model.set_weights(global_weights)
        # class_weight (if given) is a FIXED, externally-supplied weight, e.g. computed
        # from the clean global label distribution. Needed for attack experiments:
        # if a poisoned client derives weights from its OWN flipped labels, the
        # weighting partly cancels the poisoning and hides the attack.
        if class_weight is None and use_class_weight:
            class_weight = compute_class_weight(self.y_train)
        if not use_class_weight and class_weight is None:
            class_weight = None
        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size, verbose=0,
            class_weight=class_weight,
        )
        return self.model.get_weights(), len(self.X_train)
