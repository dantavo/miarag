# src/miarag/attacks/budgetleak.py
import numpy as np
from miarag.corpus import Chunk
from miarag.attacks.s2mia import split_query_answer

BUDGETS = (32, 96, 256)

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
    for c in chunks:
        roc, fluct, final = _feature_vector(rag, c.text)
        scores.append(roc + final)        # membro: sale in fretta + finisce alto
        labels.append(int(c.is_member))
    return scores, labels

def budgetleak_scores_fcm(rag, chunks: list[Chunk], seed: int = 42):
    import skfuzzy as fuzz
    X = np.array([_feature_vector(rag, c.text) for c in chunks], dtype=float)
    labels = [int(c.is_member) for c in chunks]
    cntr, u, *_ = fuzz.cluster.cmeans(X.T, c=2, m=2.0, error=1e-4, maxiter=200, seed=seed)
    # ASSUMPTION: _feature_vector order = [rate_of_change, cumulative_fluctuation, final_sim]
    # → cntr[:, 2] = cluster centroids on final_sim (index 2). Fragile if feature order changes.
    member_cluster = int(np.argmax(cntr[:, 2]))   # cluster con final_sim medio più alto
    scores = u[member_cluster].tolist()
    return scores, labels
