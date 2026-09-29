"""
notebooks/exploration.py
------------------------
Data Exploration Script.
Run this to understand your dataset before training.

    python notebooks/exploration.py

Or open as a Jupyter notebook by converting:
    pip install jupytext
    jupytext --to notebook notebooks/exploration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/creditcard.csv"

# ── 1. Load ──────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"\nShape: {df.shape}")
print(f"\nClass distribution:")
print(df["Class"].value_counts())
print(f"\nFraud %: {df['Class'].mean() * 100:.4f}%")

# ── 2. Basic stats ───────────────────────────
print("\nBasic statistics:")
print(df[["Amount", "Time"]].describe())

# ── 3. Plots ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Class imbalance
axes[0].bar(["Legitimate", "Fraudulent"], df["Class"].value_counts().values,
            color=["steelblue", "tomato"])
axes[0].set_title("Class Distribution (Imbalanced!)")
axes[0].set_ylabel("Count")
for i, v in enumerate(df["Class"].value_counts().values):
    axes[0].text(i, v + 100, f"{v:,}", ha="center", fontweight="bold")

# Amount distribution by class
df[df["Class"] == 0]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Legit",  color="steelblue")
df[df["Class"] == 1]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Fraud",  color="tomato")
axes[1].set_title("Transaction Amount by Class")
axes[1].set_xlabel("Amount")
axes[1].legend()

# Time distribution
df[df["Class"] == 0]["Time"].hist(bins=50, ax=axes[2], alpha=0.6, label="Legit",  color="steelblue")
df[df["Class"] == 1]["Time"].hist(bins=50, ax=axes[2], alpha=0.6, label="Fraud",  color="tomato")
axes[2].set_title("Time Distribution by Class")
axes[2].set_xlabel("Time (seconds)")
axes[2].legend()

plt.suptitle("Credit Card Fraud Dataset — Exploration", fontsize=14, fontweight="bold")
plt.tight_layout()
out = "results/data_exploration.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out, dpi=150)
plt.close()
print(f"\n[OK] Saved exploration plot → {out}")

# ── 4. Feature correlation heatmap ───────────
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, ax=ax,
            linewidths=0.3, annot=False)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = "results/correlation_matrix.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"[OK] Saved correlation matrix → {out2}")
print("\n[DONE] Exploration complete.")
