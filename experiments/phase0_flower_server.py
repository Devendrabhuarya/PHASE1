"""
Phase 0, Day 7: Flower "hello world" — server side.

Minimal FL server just to confirm Flower is installed correctly and
a basic federated round completes. Uses dummy random data, NOT the
real dataset yet — that integration happens in Phase 1.

Run this FIRST, then run phase0_flower_client.py in 2-3 separate
terminals (simulating 2-3 clients).

    python experiments/phase0_flower_server.py
"""

import flwr as fl


def main():
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )

    print("Starting Flower server on 127.0.0.1:8080 ...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
