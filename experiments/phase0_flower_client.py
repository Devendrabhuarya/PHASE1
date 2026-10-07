"""
Phase 0, Day 7: Flower "hello world" — client side.

Minimal FL client using dummy random data + the real model architecture
from models/model.py, just to prove the Flower client/server round-trip
works end-to-end before wiring in the real dataset (Phase 1).

Run the server first (phase0_flower_server.py), then run this script
in 2 or more separate terminals to simulate multiple clients.

    python experiments/phase0_flower_client.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import flwr as fl  # noqa: E402
from models.model import build_model  # noqa: E402

INPUT_DIM = 9  # matches the 9 encoded features in preprocess.py


def make_dummy_data(n=200, input_dim=INPUT_DIM):
    X = np.random.rand(n, input_dim).astype("float32")
    y = (np.random.rand(n) > 0.5).astype("float32")
    return X, y


class DummyClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model(input_dim=INPUT_DIM)
        self.X_train, self.y_train = make_dummy_data()
        self.X_test, self.y_test = make_dummy_data(n=50)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return loss, len(self.X_test), {"accuracy": acc}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=DummyClient(),
    )
