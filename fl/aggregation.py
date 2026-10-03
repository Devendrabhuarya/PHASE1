"""
Aggregation functions.

Phase 1: plain FedAvg (weighted by number of local samples), applied
over whichever clients were ADMITTED for that round (all of them in
the "No ABAC" scenario, only ABAC-approved ones in the "With ABAC"
scenario). Risk-aware weighted aggregation (low/medium/high) is added
in Phase 3 — this function stays as the simple baseline to compare
against.
"""

import numpy as np


def fedavg(weight_updates: list, num_examples: list):
    """
    Args:
        weight_updates: list of client weight lists (each is a list of
            numpy arrays, one per model layer — i.e. what
            model.get_weights() returns).
        num_examples: list of ints, number of training samples each
            client used (same order as weight_updates).

    Returns:
        A single list of numpy arrays: the weighted-average global
        model weights.
    """
    total = sum(num_examples)
    n_layers = len(weight_updates[0])
    avg_weights = []
    for layer_idx in range(n_layers):
        layer_sum = sum(
            w[layer_idx] * n for w, n in zip(weight_updates, num_examples)
        )
        avg_weights.append(layer_sum / total)
    return avg_weights
