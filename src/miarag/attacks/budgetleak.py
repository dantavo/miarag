# src/miarag/attacks/budgetleak.py
import numpy as np
from miarag.corpus import Chunk
from miarag.attacks.s2mia import split_query_answer

BUDGETS = (32, 96, 256)

# Ordine features in _feature_vector. Centralizzato per evitare indici magici.
FEATURE_NAMES = ("rate_of_change", "cumulative_fluctuation", "final_sim")
FINAL_SIM_IDX = FEATURE_NAMES.index("final_sim")

def _similarity(a: str, b: str) -> float:
    """Jaccard su token: dipendenza-free, monotona, sufficiente per il side-channel."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def budget_sequence(rag, chunk_text: str, budgets=BUDGETS) -> list[float]:
    query, expected = split_query_answer(chunk_text)
    seq = []
    for b in budgets:
        resp = rag.query(query, max_tokens=b)
        seq.append(_similarity(resp.answer, expected))
    return seq

def budget_features(seq: list[float]) -> dict:
    arr = np.array(seq, dtype=float)
    diffs = np.diff(arr) if len(arr) > 1 else np.array([0.0])
    return {
        "rate_of_change": float(diffs.mean()),
        "cumulative_fluctuation": float(np.abs(diffs).sum()),
        "final_sim": float(arr[-1]),
    }

def _feature_vector(rag, chunk_text: str) -> list[float]:
    f = budget_features(budget_sequence(rag, chunk_text))
    return [f["rate_of_change"], f["cumulative_fluctuation"], f["final_sim"]]

def budgetleak_scores(rag, chunks: list[Chunk]):
    scores, labels = [], []
    skipped = 0
    for c in chunks:
        try:
            roc, fluct, final = _feature_vector(rag, c.text)
            scores.append(roc + final)        # membro: sale in fretta + finisce alto
            labels.append(int(c.is_member))
        except Exception as e:  # resilienza per-chunk (budgetleak = 3 call/chunk, più esposto ai blip)
            skipped += 1
            print(f"[budgetleak] skip {c.chunk_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if skipped:
        print(f"[budgetleak] {skipped}/{len(chunks)} chunk saltati per errori transitori", flush=True)
    return scores, labels

def budgetleak_scores_fcm(rag, chunks: list[Chunk], seed: int = 42):
    import skfuzzy as fuzz
    X = np.array([_feature_vector(rag, c.text) for c in chunks], dtype=float)
    labels = [int(c.is_member) for c in chunks]
    cntr, u, *_ = fuzz.cluster.cmeans(X.T, c=2, m=2.0, error=1e-4, maxiter=200, seed=seed)
    # Cluster con final_sim medio più alto = cluster membri. Indice via costante
    # (FINAL_SIM_IDX) invece di magic number → robusto a riordino features.
    member_cluster = int(np.argmax(cntr[:, FINAL_SIM_IDX]))
    scores = u[member_cluster].tolist()
    return scores, labels
