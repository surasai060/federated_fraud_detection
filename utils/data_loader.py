"""
utils/data_loader.py
--------------------
Handles dataset download, preprocessing, and federated partitioning.
Run this ONCE before training to prepare your data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle


# ─────────────────────────────────────────────
# STEP 1 ─ LOAD RAW DATA
# ─────────────────────────────────────────────

def load_raw_data(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the raw Kaggle credit card fraud dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{csv_path}'.\n"
            "Please download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Then place 'creditcard.csv' inside the 'data/' folder.\n"
        )
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"     Fraud cases  : {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")
    return df


# ─────────────────────────────────────────────
# STEP 2 ─ PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    - Normalize 'Amount' and 'Time'
    - Drop originals
    - Return X (numpy), y (numpy)
    """
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    print(f"[OK] Preprocessing done. Feature shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────
# STEP 3 ─ FEDERATED PARTITIONING
# ─────────────────────────────────────────────

def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    strategy: str = "non_iid",   # "iid" or "non_iid"
    apply_smote: bool = True,
    random_state: int = 42,
    save_dir: str = "data/partitions"
):
    """
    Split data across N clients.

    - "iid"     : Stratified random split (same fraud ratio per client)
    - "non_iid" : Time-window split (different fraud ratios per client)

    Optionally applies SMOTE per client to handle class imbalance.
    Saves partitions as pickle files.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    if strategy == "iid":
        # Stratified random split
        indices = np.arange(len(y))
        fraud_idx  = indices[y == 1]
        normal_idx = indices[y == 0]

        rng.shuffle(fraud_idx)
        rng.shuffle(normal_idx)

        fraud_splits  = np.array_split(fraud_idx, n_clients)
        normal_splits = np.array_split(normal_idx, n_clients)

        client_splits = [
            np.concatenate([f, n])
            for f, n in zip(fraud_splits, normal_splits)
        ]

    elif strategy == "non_iid":
        # Time-window based split (simulates real banks with different periods)
        time_order = np.argsort(np.arange(len(y)))  # already time-ordered in raw CSV
        client_splits = np.array_split(time_order, n_clients)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n[OK] Partitioning strategy: '{strategy}' across {n_clients} clients")
    print(f"{'Client':<10} {'Samples':>10} {'Fraud':>10} {'Fraud%':>10}")
    print("-" * 45)

    for cid, idx in enumerate(client_splits):
        X_c = X[idx]
        y_c = y[idx]

        fraud_count = int(y_c.sum())
        fraud_pct   = fraud_count / len(y_c) * 100

        if apply_smote and fraud_count >= 2:
            # Only apply SMOTE if there are enough minority samples
            try:
                sm = SMOTE(random_state=random_state, k_neighbors=min(5, fraud_count - 1))
                X_c, y_c = sm.fit_resample(X_c, y_c)
                smote_note = f" [+SMOTE → {len(y_c):,} samples]"
            except Exception as e:
                smote_note = f" [SMOTE skipped: {e}]"
        else:
            smote_note = ""

        print(f"  Client {cid:<4} {len(y_c):>10,} {int(y_c.sum()):>10,} {y_c.mean()*100:>9.2f}%{smote_note}")

        partition = {"X": X_c.astype(np.float32), "y": y_c.astype(np.float32)}
        path = os.path.join(save_dir, f"client_{cid}.pkl")
        with open(path, "wb") as f:
            pickle.dump(partition, f)

    print(f"\n[OK] Partitions saved to '{save_dir}/'")
    return save_dir


# ─────────────────────────────────────────────
# MAIN  ─  run this script directly to prepare
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df       = load_raw_data("data/creditcard.csv")
    X, y     = preprocess(df)
    partition_data(X, y, n_clients=5, strategy="non_iid", apply_smote=True)
    print("\n[DONE] Data is ready. You can now start federated training.")
