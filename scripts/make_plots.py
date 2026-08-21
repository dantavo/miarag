# scripts/make_plots.py
"""Rigenera i grafici della vetrina/tesi dai CSV degli attacchi (44-doc, doc-split).

Mappa CSV → (attacco, target) ESPLICITA e verificata:
- AZURE  (GPT-4o-mini): results/scores_s2mia_azure44.csv, results/scores_budgetleak.csv,
  results/scores_rag_mia.csv, results/scores_rag_mia_graybox.csv
- OLLAMA (llama3.1:8b): results/ollama44/scores_{s2mia,budgetleak,rag_mia}.csv

Genera SOLO i 3 grafici che dipendono da S2MIA (auc_barchart, roc_azure, roc_ollama)
dopo il fix di scoring S²MIA-M. I 3 grafici solo-RAG-MIA (roc_ragmia_graybox,
roc_target_compare, dist_ragmia_graybox) sono invariati e NON vengono toccati.

Output: results/plots/ + copia in assets/ (vetrina) e docs_private/plots/ (tesi).

Uso:  OMP_NUM_THREADS=1 PYTHONPATH=src uv run python scripts/make_plots.py
"""
from __future__ import annotations
import csv, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
COPY_TO = [ROOT / "assets", ROOT / "docs_private" / "plots"]

def load(rel: str):
    scores, labels = [], []
    with (ROOT / rel).open() as f:
        for r in csv.DictReader(f):
            scores.append(float(r["score"])); labels.append(int(r["label"]))
    return scores, labels

def auc(rel: str) -> float:
    s, l = load(rel); return roc_auc_score(l, s)

# --- mappa esplicita ---
AZURE = [
    ("S2MIA",               "results/scores_s2mia_azure44.csv",   "C0"),
    ("BudgetLeak",          "results/scores_budgetleak.csv",      "C1"),
    ("RAG-MIA black-box",   "results/scores_rag_mia.csv",         "C2"),
    ("RAG-MIA gray-box",    "results/scores_rag_mia_graybox.csv", "C3"),
]
OLLAMA = [
    ("S2MIA",      "results/ollama44/scores_s2mia.csv",      "C0"),
    ("BudgetLeak", "results/ollama44/scores_budgetleak.csv", "C1"),
    ("RAG-MIA",    "results/ollama44/scores_rag_mia.csv",    "C2"),
]

def _roc_fig(curves, title, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, rel, color in curves:
        s, l = load(rel)
        fpr, tpr, _ = roc_curve(l, s)
        a = roc_auc_score(l, s)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="random (0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)

def make_roc_azure():
    _roc_fig(AZURE, "ROC — target GPT-4o-mini (Azure), 44-doc doc-split",
             OUT / "roc_azure.png")

def make_roc_ollama():
    _roc_fig(OLLAMA, "ROC — target llama3.1:8b (Ollama), 44-doc doc-split",
             OUT / "roc_ollama.png")

def make_auc_barchart():
    import numpy as np
    attacks = ["S2MIA", "BudgetLeak", "RAG-MIA"]
    azure = [auc("results/scores_s2mia_azure44.csv"), auc("results/scores_budgetleak.csv"),
             auc("results/scores_rag_mia.csv")]
    ollama = [auc("results/ollama44/scores_s2mia.csv"), auc("results/ollama44/scores_budgetleak.csv"),
              auc("results/ollama44/scores_rag_mia.csv")]
    gray = auc("results/scores_rag_mia_graybox.csv")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(attacks)); w = 0.38
    b1 = ax.bar(x - w/2, azure, w, label="GPT-4o-mini (Azure)", color="C0")
    b2 = ax.bar(x + w/2, ollama, w, label="llama3.1 (Ollama)", color="C1")
    xg = len(attacks)
    b3 = ax.bar([xg], [gray], w, label="RAG-MIA gray-box (Azure)", color="C2")

    for b in list(b1) + list(b2):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.2f}", ha="center", va="bottom")
    for b in b3:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.3f}", ha="center",
                va="bottom", fontweight="bold")

    ax.axhline(0.5, ls="--", color="gray")
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.4, 1.02)
    ax.set_title("AUC per attacco e target (44-doc, doc-split)")
    ax.set_xticks(list(x) + [xg])
    ax.set_xticklabels(attacks + ["RAG-MIA\ngray-box"])
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    (OUT).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "auc_barchart.png", dpi=120, bbox_inches="tight"); plt.close(fig)

def main():
    print("AUC verificati:")
    for tag, rel in [("azure S2MIA", "results/scores_s2mia_azure44.csv"),
                     ("azure BudgetLeak", "results/scores_budgetleak.csv"),
                     ("azure RAG-MIA bb", "results/scores_rag_mia.csv"),
                     ("azure RAG-MIA gray", "results/scores_rag_mia_graybox.csv"),
                     ("ollama S2MIA", "results/ollama44/scores_s2mia.csv"),
                     ("ollama BudgetLeak", "results/ollama44/scores_budgetleak.csv"),
                     ("ollama RAG-MIA", "results/ollama44/scores_rag_mia.csv")]:
        print(f"  {tag:<22} {auc(rel):.3f}")
    make_roc_azure(); make_roc_ollama(); make_auc_barchart()
    made = ["roc_azure.png", "roc_ollama.png", "auc_barchart.png"]
    for dst in COPY_TO:
        dst.mkdir(parents=True, exist_ok=True)
        for f in made:
            shutil.copy2(OUT / f, dst / f)
    print(f"rigenerati {made} in {OUT} + copiati in {[str(d) for d in COPY_TO]}")

if __name__ == "__main__":
    main()
