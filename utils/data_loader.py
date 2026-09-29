"""
utils/data_loader.py
--------------------
Handles dataset loading, preprocessing, and federated partitioning.

Leakage-safe pipeline:
  1. Split the full dataset into a TRAIN POOL (80%) and a GLOBAL TEST SET (20%),
     stratified by class. The test set is never used for training or SMOTE.
  2. Fit the scaler on the train pool only, then apply it to the test set.
  3. Partition the train pool across N clients (banks).
  4. Inside each client: split local TRAIN / VALIDATION first,
     then apply SMOTE to the local TRAIN part only.
     Validation and test data keep the real fraud ratio (~0.17%).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


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
    print(f"     Fraud cases  : {int(df['Class'].sum()):,} ({df['Class'].mean()*100:.4f}%)")
    return df


# ─────────────────────────────────────────────
# STEP 2 ─ GLOBAL SPLIT + PREPROCESS (no leakage)
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    - Sort by Time (keeps the real time order for the non-IID split)
    - Split into train pool and global test set (stratified)
    - Fit StandardScaler on the TRAIN POOL ONLY for 'Time' and 'Amount'
    Returns: X_train, y_train, X_test, y_test  (numpy float32)
    """
    df = df.sort_values("Time").reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Class"], random_state=random_state
    )
    # keep time order inside the train pool (needed for the non-IID split)
    train_df = train_df.sort_values("Time")
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    train_df[["Amount_scaled", "Time_scaled"]] = scaler.fit_transform(train_df[["Amount", "Time"]])
    test_df[["Amount_scaled", "Time_scaled"]] = scaler.transform(test_df[["Amount", "Time"]])

    def to_xy(frame):
        frame = frame.drop(columns=["Amount", "Time"])
        X = frame.drop(columns=["Class"]).values.astype(np.float32)
        y = frame["Class"].values.astype(np.float32)
        return X, y

    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)

    print(f"[OK] Preprocessing done. Features: {X_train.shape[1]}")
    print(f"     Train pool     : {len(y_train):,} rows, {int(y_train.sum())} fraud")
    print(f"     Global test set: {len(y_test):,} rows, {int(y_test.sum())} fraud "
          f"({y_test.mean()*100:.4f}%, never oversampled)")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# STEP 3 ─ FEDERATED PARTITIONING
# ─────────────────────────────────────────────

def _client_indices(y: np.ndarray, n_clients: int, strategy: str, rng):
    if strategy == "iid":
        idx = np.arange(len(y))
        fraud_idx, normal_idx = idx[y == 1], idx[y == 0]
        rng.shuffle(fraud_idx)
        rng.shuffle(normal_idx)
        return [np.concatenate([f, n]) for f, n in
                zip(np.array_split(fraud_idx, n_clients), np.array_split(normal_idx, n_clients))]
    if strategy == "non_iid":
        # Train pool is sorted by Time -> each bank gets a different time window
        return np.array_split(np.arange(len(y)), n_clients)
    raise ValueError(f"Unknown strategy: {strategy}")


def partition_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 5,
    strategy: str = "non_iid",   # "iid" or "non_iid"
    apply_smote: bool = True,
    val_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "data/partitions",
):
    """
    Split the train pool across N clients. For each client:
      local train / local validation split FIRST, then SMOTE on local train only.
    Saves one pickle per client, the global test set, and a meta.json with the settings.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)
    splits = _client_indices(y_train, n_clients, strategy, rng)

    print(f"\n[OK] Partitioning strategy: '{strategy}' across {n_clients} clients")
    print(f"{'Client':<10} {'Train':>9} {'Fraud':>7} {'Val':>8} {'Val fraud':>10}  SMOTE")
    print("-" * 62)

    for cid, idx in enumerate(splits):
        X_c, y_c = X_train[idx], y_train[idx]

        stratify = y_c if y_c.sum() >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_c, y_c, test_size=val_size, stratify=stratify, random_state=random_state
        )

        smote_note = "-"
        fraud_tr = int(y_tr.sum())
        if apply_smote and fraud_tr >= 2:
            try:
                sm = SMOTE(random_state=random_state, k_neighbors=min(5, fraud_tr - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                smote_note = f"train → {len(y_tr):,} samples"
            except Exception as e:  # pragma: no cover
                smote_note = f"skipped: {e}"

        print(f"  Client {cid:<3} {len(y_tr):>9,} {fraud_tr:>7} {len(y_val):>8,} "
              f"{int(y_val.sum()):>10}  {smote_note}")

        partition = {
            "X_train": X_tr.astype(np.float32), "y_train": y_tr.astype(np.float32),
            "X_val": X_val.astype(np.float32), "y_val": y_val.astype(np.float32),
        }
        with open(os.path.join(save_dir, f"client_{cid}.pkl"), "wb") as f:
            pickle.dump(partition, f)

    with open(os.path.join(save_dir, "global_test.pkl"), "wb") as f:
        pickle.dump({"X": X_test.astype(np.float32), "y": y_test.astype(np.float32)}, f)

    meta = {"n_clients": n_clients, "strategy": strategy, "apply_smote": apply_smote,
            "val_size": val_size, "random_state": random_state}
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Partitions + global test set saved to '{save_dir}/'")
    return save_dir


def partitions_match(save_dir: str, **settings) -> bool:
    """True if saved partitions exist and were created with the same settings."""
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path) or not os.path.exists(os.path.join(save_dir, "global_test.pkl")):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return all(meta.get(k) == v for k, v in settings.items())


def load_global_test(save_dir: str = "data/partitions"):
    with open(os.path.join(save_dir, "global_test.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["X"], d["y"]


# ─────────────────────────────────────────────
# MAIN  ─  run this script directly to prepare
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_raw_data("data/creditcard.csv")
    X_train, y_train, X_test, y_test = preprocess(df)
    partition_data(X_train, y_train, X_test, y_test, n_clients=5, strategy="non_iid", apply_smote=True)
    print("\n[DONE] Data is ready. You can now start federated training.")
