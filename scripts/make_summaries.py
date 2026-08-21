# scripts/make_summaries.py
"""Summary per-target TRACCIABILI (Cap06) da CSV mappati esplicitamente.

Problema risolto: results/summary.csv è un MIX di target — run_eval.py legge
results/scores_{s2mia,budgetleak,rag_mia}.csv, ma dopo il regen S2MIA quella dir
contiene s2mia=OLLAMA (sovrascritto) + budgetleak/rag_mia=AZURE (run azure, che
scriveva in results/ senza --results-dir). Provenienza confermata dai log:
  - run_azure_full.log  → llm=azure_openai, output dir: results/        → AZURE
  - run_ollama44.log    → llm=ollama,       output dir: results/ollama44 → OLLAMA

Qui la mappa file→(target,attacco) è ESPLICITA e VERIFICATA (assert su AUC attesi:
fail-fast se un file non è quello che crediamo). Ogni riga porta la colonna
`source_csv` per tracciabilità.

Advantage: riportiamo ENTRAMBE le definizioni per sciogliere il conflitto
def↔codice segnalato:
  - adv_youden  = max_t (TPR(t) − FPR(t))  → "membership advantage" standard
    (Yeom 2018), è quello che metrics.evaluate() e summary.csv usano.
  - adv_at_1fpr = TPR@1%FPR − FPR_op       → advantage vincolato al punto a
    basso FPR (quello che Cap05 §5.4.4 sembra descrivere a parole).

Uso:  OMP_NUM_THREADS=1 PYTHONPATH=src uv run python scripts/make_summaries.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent

# attacco -> (csv_relativo, AUC_atteso) per ciascun target. AUC_atteso protegge
# da rietichettature: se il file cambia provenienza, l'assert scatta.
AZURE = {
    "s2mia":            ("results/scores_s2mia_azure44.csv",   0.687),
    "budgetleak":       ("results/scores_budgetleak.csv",      0.592),
    "rag_mia":          ("results/scores_rag_mia.csv",         0.796),
    "s2mia_graybox":    ("results/scores_s2mia_graybox.csv",   0.713),
    "rag_mia_graybox":  ("results/scores_rag_mia_graybox.csv", 0.988),
}
OLLAMA = {
    "s2mia":      ("results/ollama44/scores_s2mia.csv",      0.688),
    "budgetleak": ("results/ollama44/scores_budgetleak.csv", 0.560),
    "rag_mia":    ("results/ollama44/scores_rag_mia.csv",    0.769),
}

def _load(rel):
    s, l, hp = [], [], []
    with (ROOT / rel).open() as f:
        for r in csv.DictReader(f):
            s.append(float(r["score"])); l.append(int(r["label"])); hp.append(int(r["has_person"]))
    return np.array(s), np.array(l), np.array(hp)

def _metrics(scores, labels, prior=0.1, target_fpr=0.01):
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    # TPR@<=1%FPR
    ok = fpr <= target_fpr
    if ok.any():
        idx = np.where(ok)[0]; b = idx[int(np.argmax(tpr[idx]))]
        tpr1, fpr_op = float(tpr[b]), float(fpr[b])
    else:
        tpr1, fpr_op = 0.0, 0.0
    ppv = (prior * tpr1) / (prior * tpr1 + (1 - prior) * fpr_op) if (prior*tpr1 + (1-prior)*fpr_op) > 0 else 0.0
    adv_youden = float(np.max(tpr - fpr))
    adv_1fpr = float(tpr1 - fpr_op)
    return dict(auc=auc, tpr_at_1fpr=tpr1, fpr_op=fpr_op, ppv=ppv,
                adv_youden=adv_youden, adv_at_1fpr=adv_1fpr)

def build(target_name, mapping):
    rows = []
    for attack, (rel, auc_exp) in mapping.items():
        s, l, hp = _load(rel)
        m = _metrics(s, l)
        assert abs(m["auc"] - auc_exp) < 0.01, \
            f"AUC mismatch {rel}: atteso ~{auc_exp}, calcolato {m['auc']:.3f} — mappa target ERRATA!"
        rows.append({"target": target_name, "attack": attack, "n": len(s),
                     "source_csv": rel, **{k: round(v, 6) for k, v in m.items()}})
    return rows

def main():
    cols = ["target", "attack", "n", "auc", "tpr_at_1fpr", "fpr_op", "ppv",
            "adv_youden", "adv_at_1fpr", "source_csv"]
    for name, mapping, out in [("azure", AZURE, "summary_azure.csv"),
                               ("ollama", OLLAMA, "summary_ollama.csv")]:
        rows = build(name, mapping)
        with (ROOT / "results" / out).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\n=== {name.upper()} → results/{out} ===")
        print(f"{'attack':<16}{'AUC':>7}{'TPR@1%':>8}{'PPV':>7}{'adv_youden':>12}{'adv_1fpr':>10}")
        for r in rows:
            print(f"{r['attack']:<16}{r['auc']:>7.3f}{r['tpr_at_1fpr']:>8.3f}{r['ppv']:>7.3f}"
                  f"{r['adv_youden']:>12.3f}{r['adv_at_1fpr']:>10.3f}")

if __name__ == "__main__":
    main()
