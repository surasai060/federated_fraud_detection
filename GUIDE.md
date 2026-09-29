# Federated Fraud Detection — Complete Setup Guide
**Project: L.E.A.D.D. Seminar — Summer 2026**
**By: Sai Sura & Indra Karan Nukala**

---

## What This Guide Covers

This is a zero-to-finish guide. By the end you will have:
- A working federated learning pipeline
- 5 simulated bank clients training collaboratively
- Comparison of federated vs centralised performance
- Plots of AUC, F1, Precision, Recall across training rounds

**Time estimate:** ~30 minutes setup, then training runs automatically.

---

## Folder Structure (what everything is)

```
federated_fraud_detection/
│
├── data/                        ← Put your CSV here; partitions auto-generated
│   ├── creditcard.csv           ← YOU DOWNLOAD THIS (see Step 2)
│   └── partitions/              ← Auto-created when you run the simulation
│       ├── client_0.pkl
│       ├── client_1.pkl
│       └── ...
│
├── models/
│   ├── __init__.py
│   └── mlp.py                   ← Neural network definition (MLP)
│
├── clients/
│   ├── __init__.py
│   └── fl_client.py             ← Flower client logic (local training)
│
├── server/
│   ├── __init__.py
│   └── fl_server.py             ← FedAvg server (for manual mode)
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py           ← Download/preprocess/partition data
│
├── notebooks/
│   └── exploration.py           ← Data exploration script
│
├── results/
│   └── plot_results.py          ← Regenerate plots from saved results
│
├── run_simulation.py            ← MAIN FILE — run this to train
├── requirements.txt             ← All Python packages
├── .gitignore
└── GUIDE.md                     ← This file
```

---

## STEP 1 — Install Prerequisites

### 1a. Install Python 3.10 or 3.11
- Download from https://www.python.org/downloads/
- **During install:** tick ✅ "Add Python to PATH"
- Verify: open a terminal and type `python --version`

### 1b. Install VS Code
- Download from https://code.visualstudio.com/
- Install the **Python extension** (search "Python" in Extensions panel, it's by Microsoft)

---

## STEP 2 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Create a free Kaggle account if you don't have one
3. Click **Download** → you get `archive.zip`
4. Extract it — you'll find `creditcard.csv` (144 MB)
5. **Move `creditcard.csv` into the `data/` folder** of this project

> The file contains ~284,000 transactions with 492 fraudulent ones.

---

## STEP 3 — Open the Project in VS Code

1. Open VS Code
2. **File → Open Folder** → navigate to `federated_fraud_detection/` → click **Open**
3. You should see the full folder tree on the left side

---

## STEP 4 — Create a Virtual Environment

Open the **integrated terminal** in VS Code:
- **Terminal → New Terminal** (or press `Ctrl + `` ` ``)

Run these commands one by one:

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You'll see `(venv)` appear at the start of your terminal prompt. This means it's active.

**Select it in VS Code:**
- Press `Ctrl+Shift+P` → type "Python: Select Interpreter" → choose the one that says `venv`

---

## STEP 5 — Install All Packages

With your virtual environment active, run:

```bash
pip install -r requirements.txt
```

This installs:
- `flwr` — Flower federated learning framework
- `torch` — PyTorch (neural network training)
- `scikit-learn` — metrics, evaluation
- `imbalanced-learn` — SMOTE for class imbalance
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — plotting

> ⏳ This takes 3–5 minutes. You'll see a progress bar.

---

## STEP 6 — (Optional) Explore the Data

To understand the dataset before training:

```bash
python notebooks/exploration.py
```

This generates two plots in `results/`:
- `data_exploration.png` — class distribution, amount & time histograms
- `correlation_matrix.png` — feature correlations

---

## STEP 7 — Run the Full Simulation

This is the **main command**. It does everything:
1. Preprocesses the data
2. Partitions it across 5 simulated bank clients (non-IID)
3. Applies SMOTE to handle fraud class imbalance
4. Trains the global model with FedAvg for 10 rounds
5. Runs a centralised baseline for comparison
6. Saves all results and plots to `results/`

```bash
python run_simulation.py
```

### What you'll see in the terminal:
```
=======================================================
  FEDERATED FRAUD DETECTION — SIMULATION
  Clients: 5 | Rounds: 10
  Local epochs: 3 | Strategy: non_iid
=======================================================

[Step 1] Preparing data...
[OK] Loaded dataset: 284,807 rows, 31 columns
     Fraud cases: 492 (0.1727%)
[OK] Partitioning strategy: 'non_iid' across 5 clients
  Client 0     57,089     98     0.17%  [+SMOTE → 114,080 samples]
  ...

  Centralised → AUC: 0.9847 | F1: 0.8312

Starting Federated Learning Simulation...

Round 1
  → AUC: 0.9201 | F1: 0.7654 | Precision: 0.8021 | Recall: 0.7312

Round 2
  → AUC: 0.9389 | F1: 0.7891 | ...
...

FINAL RESULTS SUMMARY
  Metric          Federated (FL)     Centralised
  ──────────────────────────────────────────────
  AUC               0.9712              0.9847
  F1                0.8201              0.8312
  PRECISION         0.8456              0.8521
  RECALL            0.7965              0.8104
```

---

## STEP 8 — View Your Results

After training, open the `results/` folder:

| File | What it shows |
|---|---|
| `fl_vs_centralised.png` | Line chart: FL metrics vs centralised baseline per round |
| `fl_metrics_per_round.png` | Per-round AUC, F1, Precision, Recall for the FL model |
| `fl_vs_centralised_bar.png` | Bar chart comparison of final round vs centralised |
| `centralised_confusion_matrix.png` | Confusion matrix for the baseline model |
| `fl_metrics_history.json` | Raw numbers — useful for your report |
| `centralised_results.json` | Centralised model metrics |

---

## Customisation Options

You can change parameters when running:

```bash
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

> **Tip:** If you change `n_clients`, delete `data/partitions/` first so the data is re-split correctly.

---

## Troubleshooting

**"Dataset not found"**
→ Make sure `creditcard.csv` is in the `data/` folder (not inside a zip).

**"Module not found: flwr / torch / etc."**
→ Your virtual environment isn't active. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux), then retry.

**"SMOTE error: not enough minority samples"**
→ Some clients may have very few fraud cases. This is expected in non-IID splits; SMOTE is skipped for those clients automatically.

**Training is slow**
→ Normal on CPU. Reduce `--n_rounds 5` and `--n_clients 3` for a quick test run.

**ModuleNotFoundError on models/utils**
→ Always run scripts from the **project root** (`federated_fraud_detection/`), not from inside a subfolder.

---

## How Federated Learning Works (for your report)

```
Round 1:
  Server  ──── sends global weights ────→  Client 0 (Bank A)
                                        →  Client 1 (Bank B)
                                        →  Client 2 (Bank C)
                                        →  Client 3 (Bank D)
                                        →  Client 4 (Bank E)

  Each client trains locally on its own data.
  Raw transaction data NEVER leaves the client.

  Clients  ──── send updated weights ──→  Server
  Server   ──── FedAvg: averages all weights (weighted by data size)

Round 2: repeat with improved global model
...
Round 10: final global model ready
```

**Why FedAvg?**
- McMahan et al. (2017): each client's updated weights are averaged proportionally to the number of samples it contributed.
- This is privacy-preserving: only weight tensors (numbers) move, not actual transactions.

---

## Key Files to Edit for Your Report

| What you might want to change | Where |
|---|---|
| Model architecture (more layers, dropout) | `models/mlp.py` |
| Local training loop, loss function | `clients/fl_client.py` → `train_one_round()` |
| Aggregation strategy (FedProx, FedNova) | `run_simulation.py` → `strategy = ...` |
| Data partitioning logic | `utils/data_loader.py` → `partition_data()` |
| Number of rounds, clients, epochs | `run_simulation.py --help` |

---

## Quick Reference Commands

```bash
# Activate environment (always do this first)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Explore data
python notebooks/exploration.py

# Run full experiment (default: 5 clients, 10 rounds)
python run_simulation.py

# Regenerate plots without retraining
python results/plot_results.py

# Get help on all options
python run_simulation.py --help
```
