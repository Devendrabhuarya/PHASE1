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

from models.model import build_model  # noqa: E402


class FLClient:
    def __init__(self, client_id: str, attrs, X_train, y_train, input_dim: int):
        self.client_id = client_id
        self.attrs = attrs  # abac.policy.ClientAttributes
        self.X_train = X_train
        self.y_train = y_train
        self.model = build_model(input_dim=input_dim)

    def train(self, global_weights, epochs: int = 1, batch_size: int = 32):
        """
        Loads the current global weights, trains locally for one or
        more epochs, and returns (updated_weights, num_examples).
        """
        self.model.set_weights(global_weights)
        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size, verbose=0,
        )
        return self.model.get_weights(), len(self.X_train)
