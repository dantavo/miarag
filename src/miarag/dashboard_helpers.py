"""Logica pura per la dashboard: nessun import di streamlit qui, così è testabile offline."""
import csv
from pathlib import Path
from miarag.pseudonymize import pseudonymize_text
from miarag.attacks.s2mia import s2mia_features
from miarag.attacks.budgetleak import budget_sequence, budget_features
from miarag.attacks.rag_mia import rag_mia_features

def load_summary_rows(path: Path) -> list[dict]:
    """Legge results/summary.csv → lista di dict. [] se il file non esiste."""
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))

def pii_demo(text: str, ner=None) -> str:
    """Demo gate PII: testo grezzo → pseudonimizzato. ner=None → solo regex (no modello, veloce per UI).
    Per la demo va bene la sola regex; il NER reale è lento e non necessario a mostrare il meccanismo."""
    return pseudonymize_text(text, ner=ner)

def live_s2mia_on_chunk(rag, chunk_text: str) -> dict:
    """Attacco S2MIA su UN chunk (demo dashboard). Ritorna le due feature GREZZE
    del paper (BLEU = S_sem, perplexity = PPL_gen).

    NOTA: il metodo di valutazione reale (S²MIA-M) è un classificatore XGBoost
    addestrato sull'intero dataset — non calcolabile su un singolo chunk. Qui lo
    `score` è solo un indicatore monotono DIMOSTRATIVO per la UI (alto ⇒ più
    probabile membro: BLEU alta e perplexity bassa), NON il verdetto di attacco.
    """
    import math
    f = s2mia_features(rag, chunk_text)
    ppl = f["perplexity"]
    demo_score = f["bleu"] - (math.log(ppl) if ppl > 0 and ppl != float("inf") else 0.0)
    return {
        "bleu": f["bleu"],
        "perplexity": f["perplexity"],
        "score": float(demo_score),          # indicatore demo, non il metodo di valutazione
        "score_is_demo": True,
    }

def live_budgetleak_on_chunk(rag, chunk_text: str) -> dict:
    """Attacco BudgetLeak su UN chunk. Ritorna la sequenza tri-budget + feature + score (roc+final)."""
    seq = budget_sequence(rag, chunk_text)
    bf = budget_features(seq)
    return {"sequence": list(seq), **bf, "score": float(bf["rate_of_change"] + bf["final_sim"])}

def live_rag_mia_on_chunk(rag, chunk_text: str, language: str = "it") -> dict:
    """Attacco RAG-MIA (Anderson 2025) su UN chunk. Prompt injection black-box.
    Ritorna la risposta LLM + parsed yes_score ∈ [0, 1] come score membership."""
    f = rag_mia_features(rag, chunk_text, language=language)
    return {
        "answer": f["answer"],
        "retrieved_ids": f["retrieved_ids"],
        "score": float(f["yes_score"]),
    }
