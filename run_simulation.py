"""
run_simulation.py
-----------------
Run the ENTIRE federated learning experiment in a SINGLE process.
This uses Flower's built-in simulation mode (no separate server/client terminals needed).

Usage:
    python run_simulation.py
    python run_simulation.py --n_clients 5 --n_rounds 10 --strategy non_iid
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mlp import get_model, get_parameters, set_parameters
from utils.data_loader import load_raw_data, preprocess, partition_data


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Federated Fraud Detection Simulation")
    p.add_argument("--n_clients",     type=int,   default=5,        help="Number of FL clients (banks)")
    p.add_argument("--n_rounds",      type=int,   default=10,       help="Number of FL rounds")
    p.add_argument("--local_epochs",  type=int,   default=3,        help="Local training epochs per round")
    p.add_argument("--lr",            type=float, default=1e-3,     help="Learning rate")
    p.add_argument("--strategy",      type=str,   default="non_iid",choices=["iid","non_iid"])
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--no_smote",      action="store_true",          help="Disable SMOTE")
    p.add_argument("--data_csv",      type=str,   default="data/creditcard.csv")
    p.add_argument("--partitions_dir",type=str,   default="data/partitions")
    p.add_argument("--results_dir",   type=str,   default="results")
    return p.parse_args()


# ─────────────────────────────────────────────
# CLIENT FACTORY
# ─────────────────────────────────────────────

def make_client_fn(partitions_dir: str, batch_size: int, local_epochs: int, lr: float):
    """Returns a Flower client_fn that instantiates a client by ID."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_id = int(cid)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = get_model().to(device)

        # Load partition
        path = f"{partitions_dir}/client_{client_id}.pkl"
        with open(path, "rb") as f:
            partition = pickle.load(f)
        X = torch.tensor(partition["X"], dtype=torch.float32)
        y = torch.tensor(partition["y"], dtype=torch.float32)

        # Split train/val
        n     = len(y)
        n_val = max(1, int(n * 0.2))
        perm  = torch.randperm(n, generator=torch.Generator().manual_seed(client_id))

        X_val,   y_val   = X[perm[:n_val]],   y[perm[:n_val]]
        X_train, y_train = X[perm[n_val:]],   y[perm[n_val:]]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

        class _Client(fl.client.NumPyClient):

            def get_parameters(self, config):
                return get_parameters(model)

            def fit(self, parameters, config):
                set_parameters(model, parameters)
                # Weighted loss for class imbalance
                all_y      = torch.cat([yb for _, yb in train_loader])
                n_neg      = (all_y == 0).sum().float()
                n_pos      = (all_y == 1).sum().float() + 1e-8
                pos_weight = (n_neg / n_pos).to(device)
                criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                model.train()
                for _ in range(local_epochs):
                    for Xb, yb in train_loader:
                        Xb, yb = Xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(Xb), yb)
                        loss.backward()
                        optimizer.step()

                return get_parameters(model), len(train_loader.dataset), {}

            def evaluate(self, parameters, config):
                set_parameters(model, parameters)
                model.eval()
                logits_all, labels_all = [], []
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        logits_all.append(model(Xb.to(device)).cpu())
                        labels_all.append(yb)
                logits_np = torch.cat(logits_all).numpy()
                labels_np = torch.cat(labels_all).numpy()
                probs_np  = torch.sigmoid(torch.tensor(logits_np)).numpy()
                preds_np  = (probs_np >= 0.5).astype(int)

                loss = float(nn.BCEWithLogitsLoss()(
                    torch.tensor(logits_np), torch.tensor(labels_np)
                ).item())
                auc  = float(roc_auc_score(labels_np, probs_np))    if labels_np.sum() > 0 else 0.0
                f1   = float(f1_score(labels_np, preds_np,          zero_division=0))
                prec = float(precision_score(labels_np, preds_np,   zero_division=0))
                rec  = float(recall_score(labels_np, preds_np,      zero_division=0))

                return loss, len(val_loader.dataset), {"auc": auc, "f1": f1, "precision": prec, "recall": rec}

        return _Client()

    return client_fn


# ─────────────────────────────────────────────
# METRICS AGGREGATION
# ─────────────────────────────────────────────

history_log = []   # Filled during simulation

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    agg   = {}
    for key in ["auc", "f1", "precision", "recall"]:
        agg[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    history_log.append(agg)
    print(f"  → AUC: {agg['auc']:.4f} | F1: {agg['f1']:.4f} | "
          f"Precision: {agg['precision']:.4f} | Recall: {agg['recall']:.4f}")
    return agg


# ─────────────────────────────────────────────
# CENTRALISED BASELINE
# ─────────────────────────────────────────────

def run_centralised_baseline(X: np.ndarray, y: np.ndarray, results_dir: str):
    """Train a single model on all data — the upper-bound comparison."""
    print("\n" + "="*55)
    print("  Running centralised baseline...")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model().to(device)

    # Split 80/20
    n     = len(y)
    n_val = int(n * 0.2)
    perm  = np.random.default_rng(42).permutation(n)
    idx_train, idx_val = perm[n_val:], perm[:n_val]

    X_tr = torch.tensor(X[idx_train], dtype=torch.float32)
    y_tr = torch.tensor(y[idx_train], dtype=torch.float32)
    X_vl = torch.tensor(X[idx_val],   dtype=torch.float32)
    y_vl = torch.tensor(y[idx_val],   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=512, shuffle=False)

    n_pos      = float(y_tr.sum())
    n_neg      = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(20):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(Xb.to(device)), yb.to(device)).backward()
            optimizer.step()

    # Evaluate
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits_all.append(model(Xb.to(device)).cpu())
            labels_all.append(yb)
    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    probs_np  = torch.sigmoid(torch.tensor(logits_np)).numpy()
    preds_np  = (probs_np >= 0.5).astype(int)

    results = {
        "auc"      : float(roc_auc_score(labels_np, probs_np)),
        "f1"       : float(f1_score(labels_np, preds_np, zero_division=0)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall"   : float(recall_score(labels_np, preds_np, zero_division=0)),
    }
    print(f"  Centralised → AUC: {results['auc']:.4f} | F1: {results['f1']:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/centralised_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix plot
    cm = confusion_matrix(labels_np, preds_np)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax.set_title("Centralised Baseline — Confusion Matrix")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/centralised_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {results_dir}/centralised_confusion_matrix.png")

    return results


# ─────────────────────────────────────────────
# RESULTS PLOTTING
# ─────────────────────────────────────────────

def plot_results(history, centralised, results_dir: str):
    """Generate comparison plots of federated vs centralised metrics."""
    os.makedirs(results_dir, exist_ok=True)
    rounds = list(range(1, len(history) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Federated vs Centralised — Performance Comparison", fontsize=14, fontweight="bold")

    metrics = ["auc", "f1", "precision", "recall"]
    titles  = ["AUC-ROC", "F1 Score", "Precision", "Recall"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        fl_values    = [h.get(metric, 0) for h in history]
        cent_value   = centralised.get(metric, 0)

        ax.plot(rounds, fl_values, "b-o", label="Federated (FL)", linewidth=2, markersize=5)
        ax.axhline(y=cent_value, color="r", linestyle="--", linewidth=2, label="Centralised baseline")
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
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "="*55)
    print("  FEDERATED FRAUD DETECTION — SIMULATION")
    print(f"  Clients: {args.n_clients} | Rounds: {args.n_rounds}")
    print(f"  Local epochs: {args.local_epochs} | Strategy: {args.strategy}")
    print("="*55)

    # 1. Prepare data (only if partitions don't exist yet)
    if not os.path.exists(f"{args.partitions_dir}/client_0.pkl"):
        print("\n[Step 1] Preparing data...")
        df   = load_raw_data(args.data_csv)
        X, y = preprocess(df)
        partition_data(
            X, y,
            n_clients    = args.n_clients,
            strategy     = args.strategy,
            apply_smote  = not args.no_smote,
            save_dir     = args.partitions_dir,
        )
    else:
        print(f"\n[Step 1] Partitions already exist in '{args.partitions_dir}/' — skipping prep.")
        df   = load_raw_data(args.data_csv)
        X, y = preprocess(df)

    # 2. Centralised baseline
    cent_results = run_centralised_baseline(X, y, args.results_dir)

    # 3. Federated simulation
    print("\n" + "="*55)
    print("  Starting Federated Learning Simulation...")
    print("="*55 + "\n")

    client_fn = make_client_fn(
        partitions_dir = args.partitions_dir,
        batch_size     = args.batch_size,
        local_epochs   = args.local_epochs,
        lr             = args.lr,
    )

    initial_model  = get_model()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit                    = 1.0,
        fraction_evaluate               = 1.0,
        min_fit_clients                 = args.n_clients,
        min_evaluate_clients            = args.n_clients,
        min_available_clients           = args.n_clients,
        initial_parameters              = initial_params,
        evaluate_metrics_aggregation_fn = weighted_average,
        on_fit_config_fn                = lambda rnd: {
            "local_epochs": args.local_epochs,
            "lr"          : args.lr,
        },
    )

    fl.simulation.start_simulation(
        client_fn         = client_fn,
        num_clients       = args.n_clients,
        config            = fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy          = strategy,
        client_resources  = {"num_cpus": 1, "num_gpus": 0.0},
    )

    # 4. Save history & plots
    history_path = f"{args.results_dir}/fl_metrics_history.json"
    with open(history_path, "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\n[OK] FL metrics history saved → {history_path}")

    if history_log:
        plot_results(history_log, cent_results, args.results_dir)

    # 5. Final summary
    if history_log:
        final = history_log[-1]
        print("\n" + "="*55)
        print("  FINAL RESULTS SUMMARY")
        print("="*55)
        print(f"  {'Metric':<15} {'Federated (FL)':>18} {'Centralised':>15}")
        print("  " + "-"*50)
        for m in ["auc", "f1", "precision", "recall"]:
            fl_val   = final.get(m, 0)
            cent_val = cent_results.get(m, 0)
            print(f"  {m.upper():<15} {fl_val:>18.4f} {cent_val:>15.4f}")
        print("\n  [DONE] All results saved to 'results/' folder.\n")


if __name__ == "__main__":
    main()
