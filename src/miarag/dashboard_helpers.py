"""Logica pura per la dashboard: nessun import di streamlit qui, così è testabile offline."""
import csv
from pathlib import Path
from miarag.pseudonymize import pseudonymize_text
from miarag.attacks.s2mia import s2mia_features
from miarag.attacks.budgetleak import budget_sequence, budget_features

def load_summary_rows(path: Path) -> list[dict]:
    """Legge results/summary.csv → lista di dict. [] se il file non esiste."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))

def pii_demo(text: str, ner=None) -> str:
    """Demo gate PII: testo grezzo → pseudonimizzato. ner=None → solo regex (no modello, veloce per UI).
    Per la demo va bene la sola regex; il NER reale è lento e non necessario a mostrare il meccanismo."""
    return pseudonymize_text(text, ner=ner)

def live_s2mia_on_chunk(rag, chunk_text: str) -> dict:
    """Attacco S2MIA su UN chunk. Ritorna feature + score monotono (BLEU + 1/(1+ppl))."""
    f = s2mia_features(rag, chunk_text)
    inv_ppl = 1.0 / (1.0 + f["perplexity"])
    return {"bleu": f["bleu"], "perplexity": f["perplexity"], "score": float(f["bleu"] + inv_ppl)}

def live_budgetleak_on_chunk(rag, chunk_text: str) -> dict:
    """Attacco BudgetLeak su UN chunk. Ritorna la sequenza tri-budget + feature + score (roc+final)."""
    seq = budget_sequence(rag, chunk_text)
    bf = budget_features(seq)
    return {"sequence": list(seq), **bf, "score": float(bf["rate_of_change"] + bf["final_sim"])}
