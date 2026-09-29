"""
clients/fl_client.py
--------------------
Flower FL Client.

Each client:
  1. Receives global model weights from the server
  2. Trains locally on its private partition
  3. Sends updated weights (gradients stay local — raw data NEVER leaves)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import flwr as fl

from models.mlp import get_model, get_parameters, set_parameters


# ─────────────────────────────────────────────
# DATASET HELPER
# ─────────────────────────────────────────────

def load_partition(client_id: int, data_dir: str = "data/partitions"):
    """Load a single client's partition from disk."""
    path = f"{data_dir}/client_{client_id}.pkl"
    with open(path, "rb") as f:
        partition = pickle.load(f)
    X = torch.tensor(partition["X"], dtype=torch.float32)
    y = torch.tensor(partition["y"], dtype=torch.float32)
    return X, y


def make_loaders(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2, batch_size: int = 256):
    """Split into train/val and return DataLoaders."""
    n     = len(y)
    n_val = int(n * val_ratio)
    perm  = torch.randperm(n)

    X_val,   y_val   = X[perm[:n_val]],   y[perm[:n_val]]
    X_train, y_train = X[perm[n_val:]],   y[perm[n_val:]]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# LOCAL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_one_round(model, loader, device, epochs: int = 3, lr: float = 1e-3):
    """Train the model for `epochs` local epochs on this client's data."""
    # Compute positive class weight to handle class imbalance in loss
    all_labels = torch.cat([y for _, y in loader])
    n_neg      = (all_labels == 0).sum().float()
    n_pos      = (all_labels == 1).sum().float()
    pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (len(loader) * epochs)


def evaluate(model, loader, device):
    """Compute loss + AUC + F1 + Precision + Recall on validation set."""
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np  = torch.sigmoid(torch.tensor(logits_np)).numpy()
    preds_np  = (probs_np >= 0.5).astype(int)

    metrics = {
        "loss"      : total_loss / len(loader),
        "auc"       : roc_auc_score(labels_np, probs_np)              if labels_np.sum() > 0 else 0.0,
        "f1"        : f1_score(labels_np, preds_np, zero_division=0),
        "precision" : precision_score(labels_np, preds_np, zero_division=0),
        "recall"    : recall_score(labels_np, preds_np, zero_division=0),
    }
    return metrics


# ─────────────────────────────────────────────
# FLOWER CLIENT CLASS
# ─────────────────────────────────────────────

class FraudDetectionClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, data_dir: str = "data/partitions"):
        self.client_id = client_id
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = get_model().to(self.device)

        X, y = load_partition(client_id, data_dir)
        self.train_loader, self.val_loader = make_loaders(X, y)

        print(f"[Client {client_id}] Ready | Device: {self.device} | "
              f"Train: {len(self.train_loader.dataset):,} | Val: {len(self.val_loader.dataset):,}")

    def get_parameters(self, config):
        """Return current local model weights to the server."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """
        1. Accept global weights from server
        2. Train locally
        3. Return updated weights + sample count
        """
        set_parameters(self.model, parameters)
        loss = train_one_round(
            self.model, self.train_loader, self.device,
            epochs=config.get("local_epochs", 3),
            lr=config.get("lr", 1e-3)
        )
        print(f"  [Client {self.client_id}] Train loss: {loss:.4f}")
        return get_parameters(self.model), len(self.train_loader.dataset), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        """Evaluate global model on local validation set."""
        set_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.val_loader, self.device)
        print(f"  [Client {self.client_id}] Val loss: {metrics['loss']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
        return float(metrics["loss"]), len(self.val_loader.dataset), {
            "auc"       : float(metrics["auc"]),
            "f1"        : float(metrics["f1"]),
            "precision" : float(metrics["precision"]),
            "recall"    : float(metrics["recall"]),
        }
