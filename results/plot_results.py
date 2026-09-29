"""
results/plot_results.py
-----------------------
Standalone script to regenerate plots from saved JSON results.
Run after training is complete:
    python results/plot_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_fl_metrics(history, out_dir):
    rounds = list(range(1, len(history) + 1))
    metrics = ["auc", "f1", "precision", "recall"]
    titles  = ["AUC-ROC", "F1 Score", "Precision", "Recall"]
    colors  = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Federated Learning — Per-Round Metrics", fontsize=14, fontweight="bold")

    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        values = [h.get(metric, 0) for h in history]
        ax.plot(rounds, values, "-o", color=color, linewidth=2, markersize=5, label=title)
        ax.fill_between(rounds, values, alpha=0.1, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fl_metrics_per_round.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_comparison(history, centralised, out_dir):
    if not history:
        return
    final   = history[-1]
    metrics = ["auc", "f1", "precision", "recall"]
    labels  = ["AUC-ROC", "F1", "Precision", "Recall"]

    fl_values   = [final.get(m, 0) for m in metrics]
    cent_values = [centralised.get(m, 0) for m in metrics]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, fl_values,   width, label="Federated (FL)", color="#2563eb", alpha=0.85)
    ax.bar(x + width/2, cent_values, width, label="Centralised",    color="#dc2626", alpha=0.85)

    ax.set_title("Federated vs Centralised — Final Round Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (fv, cv) in enumerate(zip(fl_values, cent_values)):
        ax.text(i - width/2, fv + 0.02, f"{fv:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "fl_vs_centralised_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    fl_path   = os.path.join(RESULTS_DIR, "fl_metrics_history.json")
    cent_path = os.path.join(RESULTS_DIR, "centralised_results.json")

    if not os.path.exists(fl_path):
        print(f"[ERROR] {fl_path} not found. Run run_simulation.py first.")
        exit(1)

    history     = load_json(fl_path)
    centralised = load_json(cent_path) if os.path.exists(cent_path) else {}

    plot_fl_metrics(history, RESULTS_DIR)
    plot_comparison(history, centralised, RESULTS_DIR)
    print("\n[DONE] All plots regenerated.")
