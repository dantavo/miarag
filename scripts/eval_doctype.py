# scripts/eval_doctype.py
"""Disaggregazione per TIPO DOCUMENTO (wiki vs società) usando una mappa
doc_id→tipo ESPLICITA ed EDITABILE: `docs_private/doc_types.csv`.

Motivazione: il `chunk_id` da solo NON separa in modo pulito wiki e società
(i wiki hanno prefisso `wiki_`, ma i documenti caricati hanno nomi eterogenei;
alcuni tecnici tipo "DATI IMMOBILIARI" sono ambigui). La mappa è quindi un file
rivedibile a mano (colonna `review` segnala i casi dubbi). Cambiando le etichette
e ri-eseguendo si ottengono numeri tracciabili per Cap06 §6.3.2.

Default della mappa: `wiki_*` = wiki (scraped), resto = società (uploaded) → 26/18.
NB: NON coincide col "33 wiki / 11 società" citato altrove — quel conteggio
richiede una riclassificazione manuale (rivedere i 2 doc `DATI*`/`Dati*`).

Uso:  OMP_NUM_THREADS=1 PYTHONPATH=src uv run python scripts/eval_doctype.py
"""
from __future__ import annotations
import csv
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DOC_TYPES = ROOT / "docs_private" / "doc_types.csv"

# scores CSV per (attacco, target) — mappa verificata (vedi make_summaries.py)
ATTACKS = {
    "S2MIA":            {"Azure": "results/scores_s2mia_azure44.csv",
                          "Ollama": "results/ollama44/scores_s2mia.csv"},
    "S2MIA gray-box":   {"Azure": "results/scores_s2mia_graybox.csv"},
    "BudgetLeak":       {"Azure": "results/scores_budgetleak.csv",
                          "Ollama": "results/ollama44/scores_budgetleak.csv"},
    "RAG-MIA":          {"Azure": "results/scores_rag_mia.csv",
                          "Ollama": "results/ollama44/scores_rag_mia.csv"},
}

def load_types():
    m = {}
    with DOC_TYPES.open() as f:
        for r in csv.DictReader(f):
            m[r["doc_id"]] = r["type"]
    return m

def doc_id_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0]

def per_type_auc(scores_csv: str, types: dict):
    by = {}
    unknown = 0
    with (ROOT / scores_csv).open() as f:
        for r in csv.DictReader(f):
            t = types.get(doc_id_of(r["chunk_id"]))
            if t is None:
                unknown += 1; continue
            by.setdefault(t, ([], []))
            by[t][0].append(float(r["score"])); by[t][1].append(int(r["label"]))
    out = {}
    for t, (s, l) in by.items():
        out[t] = (roc_auc_score(l, s) if len(set(l)) > 1 else float("nan"),
                  len(l), sum(l), len(l) - sum(l))
    return out, unknown

def main():
    types = load_types()
    nwiki = sum(1 for v in types.values() if v == "wiki")
    print(f"mappa: {len(types)} doc ({nwiki} wiki / {len(types)-nwiki} società) da {DOC_TYPES}\n")
    rows = []
    for attack, targets in ATTACKS.items():
        for tgt, csvp in targets.items():
            res, unk = per_type_auc(csvp, types)
            cells = "  ".join(f"{t}={a:.3f}(n={n},m={m})" for t, (a, n, m, nn) in sorted(res.items()))
            print(f"{attack:<16} {tgt:<7} {cells}" + (f"  [!{unk} chunk senza tipo]" if unk else ""))
            for t, (a, n, m, nn) in res.items():
                rows.append({"attack": attack, "target": tgt, "doc_type": t,
                             "auc": round(a, 4), "n": n, "members": m, "non_members": nn,
                             "source_csv": csvp})
    outp = ROOT / "results" / "summary_doctype.csv"
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "target", "doc_type", "auc", "n",
                                          "members", "non_members", "source_csv"])
        w.writeheader(); w.writerows(rows)
    print(f"\nscritto {outp}")

if __name__ == "__main__":
    main()
