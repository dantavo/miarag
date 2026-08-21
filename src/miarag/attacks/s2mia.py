# src/miarag/attacks/s2mia.py
"""S2MIA: Semantic Similarity + Perplexity Membership Inference Attack (Li 2025).

Feature per chunk (S_sem, PPL_gen nella notazione del paper):
- BLEU(risposta, expected): sovrapposizione n-gram → S_sem. Alta ⇒ membro.
- perplexity(risposta): PPL_gen. Bassa ⇒ membro (via GPT-2 proxy o logprob nativi).
- cosine_sim(risposta, expected): via EmbeddingProvider iniettato (opt-in, extra).

METODO DI SCORING (fedele al paper, Sez. III.B):
Il paper NON combina le due feature in uno score scalare (nessun `bleu + f(ppl)`).
Definisce due meccanismi distinti:
- S²MIA-T: due soglie separate, membro ⟺ (S_sem ≥ θ_sem) AND (PPL ≤ θ_gen).
- S²MIA-M: classificatore supervisionato f_MIA(bleu, ppl) — neural net o XGBoost.

Questo PoC usa **S²MIA-M (XGBoost)** come scoring primario: produce P(membro)
continua, out-of-sample via cross-validation, compatibile con la pipeline AUC.
Le feature GREZZE (bleu, ppl) sono esposte e salvate su CSV per verificabilità.

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
    """Estrae le feature S2MIA (black-box, perplexity da proxy GPT-2)."""
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
    if lps:
        mean_nll = -sum(lps) / len(lps)
        try:
            ppl = math.exp(mean_nll)
        except OverflowError:
            ppl = float("inf")   # perplexity enorme: sanitizzata in _xgb_proba
    else:
        ppl = float("inf")       # nessun token generato: answer degenere
    return {"bleu": float(bleu), "perplexity": float(ppl)}


# ---------------------------------------------------------------------------
# Estrazione feature + scoring S²MIA-M (XGBoost). Il RAG viene interrogato UNA
# sola volta per chunk; le feature grezze sono restituite per il salvataggio.
# ---------------------------------------------------------------------------

def _extract_features(rag, chunks: list[Chunk], feature_fn, name: str,
                      use_cosine: bool = False):
    """Interroga il RAG per ogni chunk e raccoglie (X, y, feats, kept_chunks).

    feature_fn(rag, text) -> dict con almeno {bleu, perplexity}.
    Resiliente: un blip su un chunk lo salta senza uccidere l'attacco.

    Parallelismo: MIARAG_S2MIA_WORKERS>1 usa un ThreadPoolExecutor. La query
    verso Ollama (~95% del tempo per-chunk) è I/O-bound (GIL rilasciato) e
    Ollama la batcha via continuous-batching (richiede OLLAMA_NUM_PARALLEL>=W).
    Le operazioni MPS (embedder MiniLM, gpt2 perplexity) serializzano via
    MPS_LOCK nei provider → thread-safe. L'ordine di (X,y,feats,kept) è
    preservato per indice; gli skippati (eccezioni) sono esclusi mantenendo
    l'allineamento. Default W=1 → path identico al sequenziale (test invariati).
    """
    import os, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(chunks)
    if n == 0:
        return [], [], [], []
    workers = max(1, int(os.environ.get("MIARAG_S2MIA_WORKERS", "1")))
    step = max(1, n // 20)
    t0 = time.time()
    print(f"[{name}] estrazione feature su {n} chunk (workers={workers})", flush=True)

    def _one(idx: int, c: Chunk):
        f = feature_fn(rag, c.text, use_cosine=use_cosine) if use_cosine \
            else feature_fn(rag, c.text)
        row = [f["bleu"], f["perplexity"]]
        if use_cosine:
            row.append(f.get("cosine", 0.0))
        return idx, c, row, {"bleu": f["bleu"], "ppl": f["perplexity"]}

    results = [None] * n          # results[idx] = (chunk, row, feat) o None se skip
    skipped = 0
    done = 0

    def _log_progress():
        el = time.time() - t0
        rate = el / done if done else 0.0
        eta = rate * (n - done)
        print(f"[{name}] {done}/{n} ({100*done//n}%) · {el:.0f}s · ETA ~{eta/60:.1f}min · "
              f"{rate:.2f}s/chunk · {skipped} skip", flush=True)

    # NB: il consumer (loop sotto) gira nel thread principale → done/skipped
    # sono aggiornati solo qui, nessuna race. I worker eseguono solo _one().
    def _consume(idx: int, c: Chunk, fut_or_call):
        nonlocal skipped, done
        try:
            _, cc, row, ft = fut_or_call()
            results[idx] = (cc, row, ft)
        except Exception as e:
            skipped += 1
            print(f"[{name}] skip {c.chunk_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        done += 1
        if done % step == 0 or done == n:
            _log_progress()

    if workers == 1:
        for idx, c in enumerate(chunks):
            _consume(idx, c, lambda idx=idx, c=c: _one(idx, c))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, idx, c): (idx, c) for idx, c in enumerate(chunks)}
            for fut in as_completed(futs):
                idx, c = futs[fut]
                _consume(idx, c, fut.result)

    X, y, feats, kept = [], [], [], []
    for r in results:
        if r is None:
            continue
        cc, row, ft = r
        X.append(row); y.append(int(cc.is_member)); feats.append(ft); kept.append(cc)
    if skipped:
        print(f"[{name}] TOTALE {skipped}/{n} chunk saltati", flush=True)
    return X, y, feats, kept


def _xgb_proba(X, y, seed: int = 42):
    """S²MIA-M: XGBoost su feature grezze → P(membro) out-of-sample via CV.
    Fallback in-sample se troppo pochi campioni per la stratificazione."""
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    X, y = np.array(X, dtype=float), np.array(y)
    # Sanitizza inf/nan E valori finiti enormi: la perplexity (nativa gray-box o
    # gpt2 su answer degeneri) può valere +inf O un finito gigantesco tipo
    # exp(700)≈1e304 (math.exp non solleva OverflowError sotto ~710) → XGBoost
    # rifiuta entrambi ('inf or a value too large'). Le feature servono solo a
    # split di soglia (ordinamento monotono): +inf/nan → sentinel, poi clip dei
    # finiti fuori scala. Preserva l'ordinamento ("ppl altissima" ⇒ non-membro).
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=0.0)
    X = np.clip(X, -1e9, 1e9)
    clf = XGBClassifier(n_estimators=50, max_depth=3, random_state=seed, eval_metric="logloss")
    n_splits = min(5, int(np.bincount(y).min())) if len(set(y)) > 1 else 0
    if n_splits < 2:
        # troppo pochi campioni per la CV: fallback esplicito (in-sample).
        clf.fit(X, y)
        return clf.predict_proba(X)[:, 1].tolist()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1].tolist()


def s2mia_scores_with_feats(rag, chunks: list[Chunk], seed: int = 42,
                            use_cosine: bool = False, graybox: bool = False):
    """Scoring S²MIA-M primario. Ritorna (scores, labels, feats, kept_chunks).

    scores = P(membro) da XGBoost su feature grezze (bleu, ppl[, cosine]).
    feats  = lista di {bleu, ppl} allineata a scores/labels/kept_chunks, per il
             salvataggio su CSV (verificabilità: le due feature restano ispezionabili).
    kept_chunks = i chunk effettivamente valutati (esclusi quelli saltati per errori).
    """
    feature_fn = s2mia_features_native_ppl if graybox else s2mia_features
    name = "s2mia_graybox" if graybox else "s2mia"
    # gray-box non supporta cosine (nessun _embed_query sul percorso logprob)
    uc = use_cosine and not graybox
    X, y, feats, kept = _extract_features(rag, chunks, feature_fn, name, use_cosine=uc)
    if not X:
        return [], [], [], []
    try:
        proba = _xgb_proba(X, y, seed=seed)
    except Exception as e:
        # INSURANCE: l'estrazione feature è la parte costosa (query al target).
        # Se lo scoring XGBoost fallisce, NON perdere le feature: dumpale su disco
        # così si può ri-scorare offline in secondi invece di rifare l'estrazione.
        import os, csv as _csv, time as _time
        dump = os.environ.get("MIARAG_S2MIA_FEATURE_DUMP",
                              f"results/_feat_dump_{name}_{int(_time.time())}.csv")
        try:
            os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
            with open(dump, "w", newline="") as f:
                w = _csv.writer(f); w.writerow(["chunk_id", "label", "has_person", "bleu", "ppl"])
                for c, ft in zip(kept, feats):
                    w.writerow([c.chunk_id, int(c.is_member), int(c.has_person), ft["bleu"], ft["ppl"]])
            print(f"[{name}] SCORING FALLITO ({type(e).__name__}) — feature grezze salvate in {dump} "
                  f"(ri-scorabili offline)", flush=True)
        except Exception as de:
            print(f"[{name}] scoring fallito E dump fallito: {de}", flush=True)
        raise
    return proba, y, feats, kept


def s2mia_scores(rag, chunks: list[Chunk]):
    """Wrapper a 2 valori (scores, labels) per retrocompat con run_eval/dashboard/test.
    Usa S²MIA-M (XGBoost). Per le feature grezze usa s2mia_scores_with_feats."""
    scores, labels, _feats, _kept = s2mia_scores_with_feats(rag, chunks)
    return scores, labels


def s2mia_scores_native_ppl(rag, chunks: list[Chunk]):
    """Wrapper a 2 valori gray-box (perplexity nativa dai logprob del target)."""
    scores, labels, _feats, _kept = s2mia_scores_with_feats(rag, chunks, graybox=True)
    return scores, labels


def s2mia_scores_model(rag, chunks: list[Chunk], seed: int = 42, use_cosine: bool = False):
    """Alias storico di S²MIA-M: (proba, labels). Mantenuto per compat API/test."""
    scores, labels, _feats, _kept = s2mia_scores_with_feats(
        rag, chunks, seed=seed, use_cosine=use_cosine)
    return scores, labels
