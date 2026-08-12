# src/miarag/plots.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(scores_by_attack, out_path: Path) -> Path:
    fig, ax = plt.subplots()
    for name, (scores, labels) in scores_by_attack.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path

def plot_distribution(scores, labels, out_path: Path) -> Path:
    fig, ax = plt.subplots()
    m = [s for s, l in zip(scores, labels) if l == 1]
    n = [s for s, l in zip(scores, labels) if l == 0]
    ax.hist(m, alpha=0.6, label="member"); ax.hist(n, alpha=0.6, label="non-member")
    ax.set_xlabel("score"); ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path

def plot_budget_sequences(sequences, out_path: Path) -> Path:
    fig, ax = plt.subplots()
    for name, seq in sequences.items():
        ax.plot(range(len(seq)), seq, marker="o", label=name)
    ax.set_xlabel("budget step"); ax.set_ylabel("similarity"); ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path
