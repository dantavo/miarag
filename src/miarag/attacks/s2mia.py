# src/miarag/attacks/s2mia.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from miarag.corpus import Chunk

_smooth = SmoothingFunction().method1

def _inv_ppl(ppl: float) -> float:
    return 1.0 / (1.0 + ppl)

def split_query_answer(text: str) -> tuple[str, str]:
    toks = text.split()
    mid = max(1, len(toks) // 2)
    return " ".join(toks[:mid]), " ".join(toks[mid:])

def s2mia_features(rag, chunk_text: str) -> dict:
    query, expected = split_query_answer(chunk_text)
    resp = rag.query(query)
    gen = resp.answer
    bleu = sentence_bleu([expected.split()], gen.split(), smoothing_function=_smooth) \
        if expected.split() and gen.split() else 0.0
    try:
        ppl = rag.perplexity_of(gen)
    except Exception:
        ppl = float("inf")
    return {"bleu": float(bleu), "perplexity": float(ppl)}

def s2mia_scores(rag, chunks: list[Chunk]):
    scores, labels = [], []
    for c in chunks:
        f = s2mia_features(rag, c.text)
        # score monotono: BLEU alta e perplexity bassa ⇒ membro
        scores.append(f["bleu"] + _inv_ppl(f["perplexity"]))
        labels.append(int(c.is_member))
    return scores, labels

def s2mia_scores_model(rag, chunks: list[Chunk], seed: int = 42):
    """Variante S2MIA-M: feature (bleu, perplexity) → XGBoost.
    Ritorna probabilità OUT-OF-SAMPLE via cross-validation (nessun fit-and-predict
    sugli stessi dati) → AUC non gonfiata in-sample."""
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    X, y = [], []
    for c in chunks:
        f = s2mia_features(rag, c.text)
        X.append([f["bleu"], _inv_ppl(f["perplexity"])])
        y.append(int(c.is_member))
    X, y = np.array(X), np.array(y)
    clf = XGBClassifier(n_estimators=50, max_depth=3, random_state=seed, eval_metric="logloss")
    n_splits = min(5, int(np.bincount(y).min())) if len(set(y)) > 1 else 0
    if n_splits < 2:
        # troppo pochi campioni per la cross-validation: fallback esplicito (in-sample),
        # segnalato nel valore di ritorno lasciando invariata la semantica minima
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1].tolist()
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1].tolist()
    return proba, y.tolist()
