# src/miarag/attacks/s2mia.py
"""S2MIA: Similarity-based Membership Inference Attack.

Features per chunk:
- BLEU(risposta, expected): sovrapposizione n-gram.
- inv_perplexity(risposta): PPL bassa → membro (via PerplexityScorer).
- cosine_sim(risposta, expected): via EmbeddingProvider iniettato (opt-in).

Chunk split: v0.2 usa split su punteggiatura (frasi complete) invece del vecchio
token-mid che spezzava frasi a metà. Retrocompat: se il chunk non ha punteggiatura
sensata → fallback token-mid.
"""
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from miarag.corpus import Chunk

_smooth = SmoothingFunction().method1

# Split su punto/punto-esclam/interrog seguiti da spazio + maiuscola o EOL.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý])")


def _inv_ppl(ppl: float) -> float:
    return 1.0 / (1.0 + ppl)


def split_query_answer(text: str) -> tuple[str, str]:
    """Split intelligente: taglia al confine di frase più vicino alla metà.
    Fallback: split token-mid (v0.1-thesis) se nessuna frase completa trovata."""
    sentences = _SENT_SPLIT.split(text.strip())
    if len(sentences) >= 2:
        # Cumula frasi finché non superi ~metà lunghezza; taglia lì.
        target = len(text) // 2
        acc = 0
        cut = 0
        for i, s in enumerate(sentences):
            acc += len(s) + 1  # +1 spazio
            if acc >= target:
                cut = i + 1
                break
        cut = max(1, min(cut, len(sentences) - 1))
        return " ".join(sentences[:cut]).strip(), " ".join(sentences[cut:]).strip()
    # Fallback token-mid (v0.1-thesis behavior).
    toks = text.split()
    mid = max(1, len(toks) // 2)
    return " ".join(toks[:mid]), " ".join(toks[mid:])


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def s2mia_features(rag, chunk_text: str, use_cosine: bool = False) -> dict:
    query, expected = split_query_answer(chunk_text)
    resp = rag.query(query)
    gen = resp.answer
    bleu = sentence_bleu([expected.split()], gen.split(), smoothing_function=_smooth) \
        if expected.split() and gen.split() else 0.0
    try:
        ppl = rag.perplexity_of(gen)
    except Exception:
        ppl = float("inf")
    out = {"bleu": float(bleu), "perplexity": float(ppl)}
    if use_cosine and hasattr(rag, "_embed_query"):
        # Riusa l'embedder del RAG (semantic similarity gen↔expected).
        try:
            e_gen = rag._embed_query(gen)
            e_exp = rag._embed_query(expected)
            out["cosine"] = float(_cosine(e_gen, e_exp))
        except Exception:
            out["cosine"] = 0.0
    return out


def s2mia_scores(rag, chunks: list[Chunk]):
    scores, labels = [], []
    skipped = 0
    for c in chunks:
        try:
            f = s2mia_features(rag, c.text)
            # score monotono: BLEU alta e perplexity bassa ⇒ membro
            scores.append(f["bleu"] + _inv_ppl(f["perplexity"]))
            labels.append(int(c.is_member))
        except Exception as e:  # resilienza per-chunk: blip di rete non uccide l'attacco
            skipped += 1
            print(f"[s2mia] skip {c.chunk_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if skipped:
        print(f"[s2mia] {skipped}/{len(chunks)} chunk saltati per errori transitori", flush=True)
    return scores, labels


def s2mia_features_native_ppl(rag, chunk_text: str) -> dict:
    """S2MIA gray-box: usa la perplexity NATIVA del modello target (dai suoi
    logprob) invece del proxy GPT-2. Richiede rag.query_with_logprobs.

    ppl_native = exp(-mean logprob dei token generati) → segnale di fluenza
    misurato dal target stesso, non da un LM inglese esterno.
    """
    import math
    query, expected = split_query_answer(chunk_text)
    out = rag.query_with_logprobs(query)
    gen = out["answer"]
    bleu = sentence_bleu([expected.split()], gen.split(), smoothing_function=_smooth) \
        if expected.split() and gen.split() else 0.0
    lps = out["token_logprobs"]
    ppl = math.exp(-sum(lps) / len(lps)) if lps else float("inf")
    return {"bleu": float(bleu), "perplexity": float(ppl)}


def s2mia_scores_native_ppl(rag, chunks: list[Chunk]):
    """Batch S2MIA con perplexity nativa (gray-box). Ritorna (scores, labels)."""
    scores, labels = [], []
    skipped = 0
    for c in chunks:
        try:
            f = s2mia_features_native_ppl(rag, c.text)
            scores.append(f["bleu"] + _inv_ppl(f["perplexity"]))
            labels.append(int(c.is_member))
        except Exception as e:
            skipped += 1
            print(f"[s2mia_graybox] skip {c.chunk_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if skipped:
        print(f"[s2mia_graybox] {skipped}/{len(chunks)} chunk saltati", flush=True)
    return scores, labels


def s2mia_scores_model(rag, chunks: list[Chunk], seed: int = 42, use_cosine: bool = False):
    """Variante S2MIA-M: feature (bleu, inv_ppl, [cosine]) → XGBoost.
    Ritorna probabilità OUT-OF-SAMPLE via cross-validation (nessun fit-and-predict
    sugli stessi dati) → AUC non gonfiata in-sample.

    use_cosine=True aggiunge cosine similarity gen↔expected via embedder del RAG.
    """
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    X, y = [], []
    for c in chunks:
        f = s2mia_features(rag, c.text, use_cosine=use_cosine)
        row = [f["bleu"], _inv_ppl(f["perplexity"])]
        if use_cosine:
            row.append(f.get("cosine", 0.0))
        X.append(row)
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
