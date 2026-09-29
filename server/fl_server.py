"""
server/fl_server.py
-------------------
Flower FL Server using FedAvg aggregation.

The server:
  - Initialises the global model
  - Coordinates rounds: sends weights → collects updates → aggregates → repeat
  - Logs metrics per round
  - Saves the final global model
"""

import os
import json
import numpy as np
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict

from models.mlp import get_model, get_parameters


# ─────────────────────────────────────────────
# METRICS AGGREGATION (called after each round)
# ─────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate per-client evaluation metrics weighted by number of samples.
    This is called automatically by Flower after each evaluation round.
    """
    total_samples = sum(n for n, _ in metrics)

    aggregated = {}
    for key in ["auc", "f1", "precision", "recall"]:
        weighted_sum = sum(n * m.get(key, 0.0) for n, m in metrics)
        aggregated[key] = weighted_sum / total_samples

    print(f"\n  [Server] Aggregated metrics → "
          f"AUC: {aggregated['auc']:.4f} | "
          f"F1: {aggregated['f1']:.4f} | "
          f"Precision: {aggregated['precision']:.4f} | "
          f"Recall: {aggregated['recall']:.4f}")

    return aggregated


# ─────────────────────────────────────────────
# METRICS HISTORY LOGGER
# ─────────────────────────────────────────────

class MetricsLogger:
    """Saves per-round metrics to a JSON file for later plotting."""

    def __init__(self, path: str = "results/metrics_history.json"):
        self.path    = path
        self.history = []
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, round_num: int, metrics: dict):
        entry = {"round": round_num, **metrics}
        self.history.append(entry)
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)

    def summary(self):
        print(f"\n[Server] Metrics history saved to '{self.path}'")


# ─────────────────────────────────────────────
# CUSTOM STRATEGY (FedAvg + logging)
# ─────────────────────────────────────────────

class LoggingFedAvg(FedAvg):
    """FedAvg with per-round metric logging."""

    def __init__(self, logger: MetricsLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        if metrics_aggregated:
            self.logger.log(server_round, {"loss": loss_aggregated, **metrics_aggregated})
        return loss_aggregated, metrics_aggregated


# ─────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────

def run_server(
    n_clients: int   = 5,
    n_rounds: int    = 10,
    min_fit: int     = 3,      # Minimum clients per training round
    min_eval: int    = 3,      # Minimum clients per evaluation round
    local_epochs: int = 3,
    lr: float         = 1e-3,
    server_address: str = "0.0.0.0:8080",
):
    print(f"\n{'='*55}")
    print(f"  Federated Fraud Detection — FL Server")
    print(f"  Clients: {n_clients} | Rounds: {n_rounds} | Local epochs: {local_epochs}")
    print(f"{'='*55}\n")

    # Initialise global model weights
    initial_model  = get_model()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    logger = MetricsLogger()

    strategy = LoggingFedAvg(
        logger                         = logger,
        fraction_fit                   = min_fit / n_clients,
        fraction_evaluate              = min_eval / n_clients,
        min_fit_clients                = min_fit,
        min_evaluate_clients           = min_eval,
        min_available_clients          = n_clients,
        initial_parameters             = initial_params,
        evaluate_metrics_aggregation_fn= weighted_average,
        on_fit_config_fn               = lambda rnd: {
            "local_epochs": local_epochs,
            "lr": lr,
            "server_round": rnd,
        },
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )

    logger.summary()


if __name__ == "__main__":
    run_server(
        n_clients    = 5,
        n_rounds     = 10,
        local_epochs = 3,
    )
