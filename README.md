# 🏦 Federated Fraud Detection

> A PyTorch + Flower pipeline that trains one credit card fraud detection model across several simulated banks — without any bank sharing its raw transactions. Only model weights move between clients and server, using the FedAvg algorithm.

Built as a Federated Learning portfolio project by **Sai Sura** — Master's student in Intelligent Interactive Systems, Universität Bielefeld. Made for the L.E.A.D.D. (Learning from Distributed Data) seminar.

---

## 📸 Console Preview

```
=======================================================
  FEDERATED FRAUD DETECTION — SIMULATION
  Clients: 5 | Rounds: 10
  Local epochs: 3 | Strategy: non_iid
=======================================================

[Step 1] Preparing data...
[OK] Loaded dataset: 284,807 rows, 31 columns
     Fraud cases  : 492 (0.1727%)
[OK] Partitioning strategy: 'non_iid' across 5 clients
  Client 0     57,089        98     0.17%  [+SMOTE → 114,080 samples]
  Client 1     57,089       112     0.20%  [+SMOTE → 113,932 samples]
  ...

  Centralised → AUC: 0.9847 | F1: 0.8312

Starting Federated Learning Simulation...

Round 1
  → AUC: 0.9201 | F1: 0.7654 | Precision: 0.8021 | Recall: 0.7312
...
Round 10
  → AUC: 0.9712 | F1: 0.8201 | Precision: 0.8456 | Recall: 0.7965

  FINAL RESULTS SUMMARY
  Metric          Federated (FL)     Centralised
  ──────────────────────────────────────────────
  AUC               0.9712              0.9847
  F1                0.8201              0.8312
  PRECISION         0.8456              0.8521
  RECALL            0.7965              0.8104
```
> Numbers above are illustrative. Run `python run_simulation.py` yourself and paste your real numbers here — see [Results](#-results).

---

## 🎯 Problem This Solves

Banks want to detect fraud better by training on more data, but transaction records are highly private. GDPR, banking secrecy laws, and competitive concerns stop banks from pooling raw data in one place.

**Federated Learning (FL)** solves this: instead of moving data to one place, the model moves to the data. Each bank trains locally on its own transactions, and only the trained *weights* are sent to a central server. The server averages the weights from all banks (FedAvg) into one global model and sends it back. This repeats for several rounds, and no bank ever exposes a single raw transaction to anyone else.

This project simulates that setup end-to-end: 5 simulated banks, one Flower server, and a centralized baseline to check how much (if anything) privacy costs in performance.

---

## 🔍 What It Does

| Step | What Happens | Where |
|---|---|---|
| **Preprocessing** | Scales `Time` and `Amount`, drops the raw columns | `utils/data_loader.py` |
| **Partitioning** | Splits data across N simulated banks — `iid` (even fraud ratio) or `non_iid` (time-window split) | `utils/data_loader.py` |
| **Imbalance handling** | SMOTE oversampling per bank + a weighted loss (`pos_weight` in `BCEWithLogitsLoss`) | `utils/data_loader.py`, `clients/fl_client.py` |
| **Local training** | Each bank trains the same MLP on its own partition for a fixed number of epochs | `run_simulation.py` (client_fn), `clients/fl_client.py` |
| **Aggregation** | Server averages all client weights, weighted by how much data each bank has (FedAvg) | Flower's `FedAvg` strategy |
| **Evaluation** | AUC, F1, Precision, Recall computed per bank and aggregated after every round | `run_simulation.py` |
| **Baseline** | One model trained on the full, non-partitioned dataset, for comparison | `run_simulation.py` (`run_centralised_baseline`) |

---

## 🏗️ Architecture

```
                ┌─────────────────────┐
                │  creditcard.csv     │
                │  (Kaggle dataset)   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Preprocessing     │  ← Scale Time & Amount
                │ (utils/data_loader) │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Partitioning      │  ← iid / non_iid split
                │ across N banks      │    + SMOTE per bank
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Bank 0     │   │   Bank 1     │   │   Bank N     │
│ Local MLP    │   │ Local MLP    │   │ Local MLP    │  ← models/mlp.py
│ train_one_   │   │ train_one_   │   │ train_one_   │    (Flower NumPyClient)
│ round()      │   │ round()      │   │ round()      │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        └──────────────────┼──────────────────┘
                           ▼
                ┌─────────────────────┐
                │   Flower Server     │  ← FedAvg: weighted average
                │   (FedAvg strategy) │    of all client weights
                └──────────┬──────────┘
                           │
              (repeat for N rounds)
                           │
                           ▼
                ┌─────────────────────┐
                │  Global Model Eval  │  ← AUC, F1, Precision, Recall
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Results & Plots    │  ← JSON + PNG in results/
                │ vs Centralised      │
                │ Baseline            │
                └─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository

```
git clone https://github.com/surasai060/federated_fraud_detection.git
cd federated_fraud_detection
```

### 2. Create a virtual environment and install dependencies

```
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Get the dataset

Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

### 4. Run the full simulation

```
python run_simulation.py
```

This preprocesses the data, partitions it across 5 simulated banks, trains a centralized baseline, runs 10 rounds of federated training, and saves all metrics and plots to `results/`.

### Other useful commands

```
python notebooks/exploration.py     # Explore the dataset before training
python results/plot_results.py      # Regenerate plots without retraining
python run_simulation.py --help     # Show all CLI options
```

---

## ⚙️ Customization

```
# More rounds (better convergence, slower)
python run_simulation.py --n_rounds 20

# More clients (more banks)
python run_simulation.py --n_clients 10

# IID split instead of non-IID (same fraud ratio per bank)
python run_simulation.py --strategy iid

# More local training per round
python run_simulation.py --local_epochs 5

# Combine options
python run_simulation.py --n_clients 8 --n_rounds 15 --strategy non_iid --local_epochs 5
```

> If you change `--n_clients` or `--strategy`, delete `data/partitions/` first so the data is re-split correctly.

---

## 📄 Sample Metrics Output (JSON)

`results/fl_metrics_history.json`:

```json
[
  {"round": 1,  "loss": 0.31, "auc": 0.9201, "f1": 0.7654, "precision": 0.8021, "recall": 0.7312},
  {"round": 2,  "loss": 0.24, "auc": 0.9389, "f1": 0.7891, "precision": 0.8145, "recall": 0.7601},
  {"round": 10, "loss": 0.11, "auc": 0.9712, "f1": 0.8201, "precision": 0.8456, "recall": 0.7965}
]
```

---

## 📁 Project Structure

```
federated_fraud_detection/
│
├── run_simulation.py            # MAIN FILE — runs the entire experiment
│   ├── client_fn                #   Builds one Flower client per bank
│   ├── run_centralised_baseline #   Trains the non-federated comparison model
│   └── plot_results             #   FL vs centralized comparison charts
│
├── models/
│   └── mlp.py                   # FraudMLP: 30 → 64 → 32 → 16 → 1 (PyTorch)
│
├── clients/
│   └── fl_client.py             # Flower NumPyClient: local train/eval logic
│
├── server/
│   └── fl_server.py             # FedAvg server + per-round metrics logging
│
├── utils/
│   └── data_loader.py           # Load, scale, partition (iid/non_iid) + SMOTE
│
├── notebooks/
│   └── exploration.py           # Class balance, Amount/Time histograms, correlations
│
├── results/
│   └── plot_results.py          # Regenerate plots from saved JSON
│
├── data/                        # creditcard.csv goes here (not tracked in Git)
├── requirements.txt
├── GUIDE.md                     # Full setup walkthrough
└── .gitignore
```

---

## 🔗 How This Relates to Real-World Privacy-Preserving ML

| This Project | Real-World Equivalent |
|---|---|
| Simulated banks (`client_fn`) | Separate financial institutions, each with private customer data |
| FedAvg weight averaging | The core algorithm behind Google's on-device keyboard prediction (Gboard) and cross-hospital medical AI |
| `pos_weight` + SMOTE | Standard techniques for fraud/rare-event detection in production ML systems |
| Centralized baseline comparison | How a bank would justify the "privacy cost" of FL to leadership before adopting it |
| Non-IID time-window split | Real banks each seeing a different slice of transactions, not a random shuffle |

---

## 🛠️ Tech Stack

- **Python 3.10/3.11** — project language
- **PyTorch** — neural network (MLP) and training loop
- **Flower (flwr)** — federated learning simulation and FedAvg strategy
- **scikit-learn** — AUC, F1, precision, recall
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **pandas / numpy** — data loading and preprocessing
- **matplotlib / seaborn** — result plots and confusion matrix

---

## ⚠️ Limitations

- All banks run in one process on one machine — this is a simulation, not physically separate institutions.
- SMOTE is applied before the local train/validation split, so synthetic samples can leak between the two — treat validation scores as optimistic.
- Metrics are evaluated on each bank's own validation split, not a single global hold-out set.
- No formal privacy guarantee (no Differential Privacy or Secure Aggregation) — FL alone only keeps raw data local.

## 🔮 Future Work

- Add a global hold-out test set shared across banks
- Try alternative strategies (FedProx, FedNova, FedAvgM) for non-IID robustness
- Add Differential Privacy or Secure Aggregation
- Deploy clients on physically separate machines

---

## 👤 Author

**Sai Sura**
Master's in Intelligent Interactive Systems — Universität Bielefeld
1 year SOC experience — Tech Mahindra (IBM QRadar, HP ArcSight)
📧 <surasai060@gmail.com>
🔗 [LinkedIn](https://linkedin.com/in/sai-sura-945032284)

---

## 📜 License

MIT License — free to use, modify, and distribute.

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* (FedAvg)
- Beutel et al. (2020). *Flower: A Friendly Federated Learning Framework.*
- Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* (dataset)
