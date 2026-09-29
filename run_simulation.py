"""
run_simulation.py
-----------------
Run the ENTIRE federated learning experiment in a SINGLE process.
This uses Flower's built-in simulation mode (no separate server/client terminals needed).

Fair evaluation:
  - The federated global model AND the centralised baseline are both evaluated
    on the SAME global test set (real fraud ratio, never oversampled, never trained on).
  - Client-side validation (real ratio, local) is logged separately.

Usage:
    python run_simulation.py
    python run_simulation.py --n_clients 5 --n_rounds 10 --strategy non_iid
"""

import argparse
import os
import random
import sys
import json
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
import flwr as fl
from flwr.common import Metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mlp import get_model, get_parameters, set_parameters
from utils.data_loader import (load_raw_data, preprocess, partition_data,
                               partitions_match, load_global_test)
from imblearn.over_sampling import SMOTE

METRIC_KEYS = ["auc", "pr_auc", "f1", "precision", "recall"]


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Federated Fraud Detection Simulation")
    p.add_argument("--n_clients",     type=int,   default=5,        help="Number of FL clients (banks)")
    p.add_argument("--n_rounds",      type=int,   default=10,       help="Number of FL rounds")
    p.add_argument("--local_epochs",  type=int,   default=3,        help="Local training epochs per round")
    p.add_argument("--lr",            type=float, default=1e-3,     help="Learning rate")
    p.add_argument("--strategy",      type=str,   default="non_iid", choices=["iid", "non_iid"])
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--no_smote",      action="store_true",          help="Disable SMOTE")
    p.add_argument("--threshold",     type=float, default=0.5,      help="Decision threshold for fraud")
    p.add_argument("--data_csv",      type=str,   default="data/creditcard.csv")
    p.add_argument("--partitions_dir", type=str,  default="data/partitions")
    p.add_argument("--results_dir",   type=str,   default="results")
    p.add_argument("--seed",          type=int,   default=42,       help="Random seed for reproducible results")
    p.add_argument("--backend",       type=str,   default="auto", choices=["auto", "ray", "local"],
                   help="'ray' = Flower start_simulation (needs flwr[simulation]); "
                        "'local' = same Flower clients + FedAvg in a simple loop (no Ray); "
                        "'auto' = Ray if installed, otherwise local")
    return p.parse_args()


def set_seed(seed: int):
    """Fix all random generators so every run gives the same results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ─────────────────────────────────────────────
# SHARED EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5, batch_size: int = 2048):
    """Return (loss, metrics, preds) for a model on data with the REAL class ratio."""
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=batch_size)
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits_all.append(model(Xb.to(device)).cpu().view(-1))
            labels_all.append(yb.view(-1))
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = float(nn.BCEWithLogitsLoss()(logits, labels).item())

    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    preds = (probs >= threshold).astype(int)
    has_both = 0 < labels_np.sum() < len(labels_np)
    metrics = {
        "auc":       float(roc_auc_score(labels_np, probs)) if has_both else 0.0,
        "pr_auc":    float(average_precision_score(labels_np, probs)) if has_both else 0.0,
        "f1":        float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall":    float(recall_score(labels_np, preds, zero_division=0)),
    }
    return loss, metrics, preds


def train_epochs(model, loader, epochs: int, lr: float):
    """Train with a weighted loss (pos_weight) computed from the training labels."""
    device = next(model.parameters()).device
    all_y = torch.cat([yb for _, yb in loader])
    n_pos = (all_y == 1).sum().float() + 1e-8
    n_neg = (all_y == 0).sum().float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=(n_neg / n_pos).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb).view(-1), yb.view(-1))
            loss.backward()
            optimizer.step()


# ─────────────────────────────────────────────
# CLIENT FACTORY
# ─────────────────────────────────────────────

def make_client_fn(partitions_dir: str, batch_size: int, local_epochs: int, lr: float, threshold: float):
    """Returns a Flower client_fn that instantiates a client (bank) by ID."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_id = int(cid)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model().to(device)

        with open(f"{partitions_dir}/client_{client_id}.pkl", "rb") as f:
            part = pickle.load(f)

        # Local train (SMOTE already applied to train only) and local validation (real ratio)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(part["X_train"]), torch.tensor(part["y_train"])),
            batch_size=batch_size, shuffle=True,
        )
        X_val, y_val = part["X_val"], part["y_val"]

        class _Client(fl.client.NumPyClient):

            def get_parameters(self, config):
                return get_parameters(model)

            def fit(self, parameters, config):
                set_parameters(model, parameters)
                train_epochs(model, train_loader, local_epochs, lr)
                return get_parameters(model), len(train_loader.dataset), {}

            def evaluate(self, parameters, config):
                set_parameters(model, parameters)
                loss, metrics, _ = evaluate_model(model, X_val, y_val, threshold)
                return loss, len(y_val), metrics

        return _Client()

    return client_fn


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

global_history = []   # global model on the GLOBAL TEST SET, one entry per round
local_history = []    # weighted average of client-side local validation


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client-side local validation metrics (real fraud ratio)."""
    total = sum(n for n, _ in metrics)
    agg = {k: sum(n * m.get(k, 0.0) for n, m in metrics) / total for k in METRIC_KEYS}
    local_history.append(agg)
    return agg


def make_global_evaluate_fn(X_test: np.ndarray, y_test: np.ndarray, threshold: float):
    """Server-side evaluation of the global model on the shared global test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    def evaluate(server_round: int, parameters, config) -> Tuple[float, Dict]:
        set_parameters(model, parameters)
        loss, metrics, _ = evaluate_model(model, X_test, y_test, threshold)
        if server_round > 0:                      # round 0 = untrained initial model
            global_history.append({"round": server_round, "loss": loss, **metrics})
            print(f"\n  Round {server_round:>2} [global test] → AUC: {metrics['auc']:.4f} | "
                  f"PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        return loss, metrics

    return evaluate


# ─────────────────────────────────────────────
# CENTRALISED BASELINE
# ─────────────────────────────────────────────

def run_centralised_baseline(X_train, y_train, X_test, y_test, apply_smote: bool,
                             threshold: float, results_dir: str, epochs: int = 20):
    """Train one model on the full train pool; evaluate on the SAME global test set."""
    print("\n" + "=" * 55)
    print("  Running centralised baseline...")
    print("=" * 55)

    if apply_smote:   # same imbalance handling as the clients, for a fair comparison
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=512, shuffle=True)
    train_epochs(model, loader, epochs, lr=1e-3)

    _, results, preds = evaluate_model(model, X_test, y_test, threshold)
    print(f"  Centralised [global test] → AUC: {results['auc']:.4f} | "
          f"PR-AUC: {results['pr_auc']:.4f} | F1: {results['f1']:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/centralised_results.json", "w") as f:
        json.dump(results, f, indent=2)
    save_confusion_matrix(y_test, preds, "Centralised Baseline — Confusion Matrix (global test)",
                          f"{results_dir}/centralised_confusion_matrix.png")
    return results


def save_confusion_matrix(y_true, y_pred, title: str, path: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ─────────────────────────────────────────────
# RESULTS PLOTTING
# ─────────────────────────────────────────────

def plot_results(history, centralised, results_dir: str):
    """Federated global model vs centralised baseline, both on the global test set."""
    os.makedirs(results_dir, exist_ok=True)
    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Federated vs Centralised — Global Test Set", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flat,
                                 ["auc", "pr_auc", "precision", "recall"],
                                 ["ROC-AUC", "PR-AUC", "Precision", "Recall"]):
        ax.plot(rounds, [h[metric] for h in history], "b-o", label="Federated (FL)",
                linewidth=2, markersize=5)
        ax.axhline(y=centralised[metric], color="r", linestyle="--", linewidth=2,
                   label="Centralised baseline")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{results_dir}/fl_vs_centralised.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[OK] Performance plot saved → {path}")


# ─────────────────────────────────────────────
# LOCAL SIMULATION (no Ray needed)
# ─────────────────────────────────────────────

def run_local_simulation(client_fn, strategy, n_clients: int, n_rounds: int, initial_params):
    """
    Runs the same Flower clients and the same FedAvg strategy as start_simulation,
    but sequentially in one process. Works on Windows / Python 3.12 without Ray.
    Each round: global weights → every client trains locally → FedAvg aggregation
    → global model evaluated on the global test set + client-side validation.
    """
    from flwr.common import (Code, Status, FitRes, EvaluateRes,
                             ndarrays_to_parameters, parameters_to_ndarrays)
    ok = Status(code=Code.OK, message="")
    parameters = initial_params
    strategy.evaluate(0, parameters)                       # round 0 (not logged)

    for rnd in range(1, n_rounds + 1):
        print(f"\n[Round {rnd}/{n_rounds}] training {n_clients} clients...")
        global_nd = parameters_to_ndarrays(parameters)

        fit_results = []
        for cid in range(n_clients):
            client = client_fn(str(cid))
            nd, n, metrics = client.fit(global_nd, {})
            fit_results.append((None, FitRes(ok, ndarrays_to_parameters(nd), n, metrics)))
        parameters, _ = strategy.aggregate_fit(rnd, fit_results, [])   # FedAvg

        strategy.evaluate(rnd, parameters)                 # global test set
        new_nd = parameters_to_ndarrays(parameters)
        eval_results = []
        for cid in range(n_clients):
            loss, n, metrics = client_fn(str(cid)).evaluate(new_nd, {})
            eval_results.append((None, EvaluateRes(ok, loss, n, metrics)))
        strategy.aggregate_evaluate(rnd, eval_results, [])  # local validation

    return parameters


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    apply_smote = not args.no_smote

    print("\n" + "=" * 55)
    print("  FEDERATED FRAUD DETECTION — SIMULATION")
    print(f"  Clients: {args.n_clients} | Rounds: {args.n_rounds}")
    print(f"  Local epochs: {args.local_epochs} | Strategy: {args.strategy} | SMOTE: {apply_smote} | Seed: {args.seed}")
    print("=" * 55)

    # 1. Data: global split + scaling (train only) + partitioning
    print("\n[Step 1] Preparing data...")
    df = load_raw_data(args.data_csv)
    X_train, y_train, X_test, y_test = preprocess(df)
    settings = dict(n_clients=args.n_clients, strategy=args.strategy, apply_smote=apply_smote)
    if partitions_match(args.partitions_dir, **settings):
        print(f"[OK] Reusing partitions in '{args.partitions_dir}/' (same settings)")
    else:
        partition_data(X_train, y_train, X_test, y_test, save_dir=args.partitions_dir, **settings)
    X_test, y_test = load_global_test(args.partitions_dir)

    # 2. Centralised baseline (same global test set)
    cent_results = run_centralised_baseline(X_train, y_train, X_test, y_test,
                                            apply_smote, args.threshold, args.results_dir)

    # 3. Federated simulation
    print("\n" + "=" * 55)
    print("  Starting Federated Learning Simulation...")
    print("=" * 55 + "\n")

    client_fn = make_client_fn(args.partitions_dir, args.batch_size, args.local_epochs,
                               args.lr, args.threshold)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(get_model()))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.n_clients,
        min_evaluate_clients=args.n_clients,
        min_available_clients=args.n_clients,
        initial_parameters=initial_params,
        evaluate_fn=make_global_evaluate_fn(X_test, y_test, args.threshold),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    backend = args.backend
    if backend == "auto":
        try:
            import ray  # noqa: F401
            backend = "ray"
        except ImportError:
            backend = "local"
    print(f"[INFO] Simulation backend: {backend}")

    if backend == "ray":
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.n_clients,
            config=fl.server.ServerConfig(num_rounds=args.n_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
    else:
        run_local_simulation(client_fn, strategy, args.n_clients, args.n_rounds, initial_params)

    # 4. Save history & plots
    with open(f"{args.results_dir}/fl_metrics_history.json", "w") as f:
        json.dump(global_history, f, indent=2)
    with open(f"{args.results_dir}/fl_local_validation_history.json", "w") as f:
        json.dump(local_history, f, indent=2)
    print(f"\n[OK] FL metrics saved → {args.results_dir}/fl_metrics_history.json")

    if global_history:
        plot_results(global_history, cent_results, args.results_dir)

        # 5. Final summary (both on the same global test set)
        final = global_history[-1]
        print("\n" + "=" * 55)
        print("  FINAL RESULTS SUMMARY  (global test set, real fraud ratio)")
        print("=" * 55)
        print(f"  {'Metric':<15} {'Federated (FL)':>18} {'Centralised':>15}")
        print("  " + "-" * 50)
        for m in METRIC_KEYS:
            print(f"  {m.upper():<15} {final[m]:>18.4f} {cent_results[m]:>15.4f}")
        print("\n  [DONE] All results saved to 'results/' folder.\n")


if __name__ == "__main__":
    main()
