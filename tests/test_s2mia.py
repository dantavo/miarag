# tests/test_s2mia.py
from miarag.corpus import Chunk
from miarag.attacks.s2mia import (
    split_query_answer, s2mia_scores, s2mia_scores_model, s2mia_scores_with_feats,
)

def test_split_halves():
    q, a = split_query_answer("uno due tre quattro cinque sei")
    assert q and a and q != a

class _RAG:
    """Membro: restituisce la risposta attesa (BLEU alta, ppl bassa).
    Non-membro: testo diverso (BLEU ~0, ppl alta)."""
    def query(self, question, max_tokens=256):
        from miarag.rag import RAGResponse
        # simulazione: se la question contiene 'MEM' rispondi 'combacia esattamente'
        ans = "combacia esattamente" if "MEM" in question else "parole totalmente diverse qui"
        return RAGResponse(answer=ans, retrieved_ids=["x"], perplexity=None)
    def perplexity_of(self, text):
        # membro (testo atteso) → ppl bassa; non-membro → ppl alta
        return 5.0 if "combacia" in text else 200.0

def _balanced_chunks(n_per_class=12):
    chunks = []
    for i in range(n_per_class):
        chunks.append(Chunk(f"m{i}", "d", f"MEM chunk {i} combacia esattamente",
                            is_member=True, has_person=False))
    for i in range(n_per_class):
        chunks.append(Chunk(f"n{i}", "d", f"XXX chunk {i} parole diverse originali",
                            is_member=False, has_person=False))
    return chunks

def test_s2mia_scores_shape_and_range():
    """S²MIA-M: score = P(membro) in [0,1], allineati alle label."""
    chunks = _balanced_chunks()
    scores, labels = s2mia_scores(_RAG(), chunks)
    assert len(scores) == len(labels) == len(chunks)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert labels == [1]*12 + [0]*12

def test_s2mia_scores_separate_members():
    """Con feature separabili (BLEU alta+ppl bassa per i membri), lo score medio
    dei membri deve superare quello dei non-membri (out-of-sample XGBoost)."""
    chunks = _balanced_chunks()
    scores, labels = s2mia_scores(_RAG(), chunks)
    mem = [s for s, l in zip(scores, labels) if l == 1]
    non = [s for s, l in zip(scores, labels) if l == 0]
    assert sum(mem)/len(mem) > sum(non)/len(non)

def test_s2mia_scores_with_feats_exposes_raw():
    """s2mia_scores_with_feats espone bleu/ppl grezze per il salvataggio CSV."""
    chunks = _balanced_chunks()
    scores, labels, feats, kept = s2mia_scores_with_feats(_RAG(), chunks)
    assert len(scores) == len(labels) == len(feats) == len(kept)
    assert all("bleu" in f and "ppl" in f for f in feats)
    # membri: BLEU alta e ppl bassa; non-membri: BLEU ~0 e ppl alta
    mem_ppl = [f["ppl"] for f, l in zip(feats, labels) if l == 1]
    non_ppl = [f["ppl"] for f, l in zip(feats, labels) if l == 0]
    assert sum(mem_ppl)/len(mem_ppl) < sum(non_ppl)/len(non_ppl)

def test_s2mia_scores_model_offline():
    """Test s2mia_scores_model with fake RAG and synthetic separable data."""
    # Build >= 10 chunks per class with distinguishable BLEU/perplexity signals
    chunks = []
    for i in range(12):
        # Members: query contains "MEM", will get high BLEU
        chunks.append(Chunk(f"m{i}", "d", f"MEM chunk {i} combacia esattamente", is_member=True, has_person=False))
    for i in range(12):
        # Non-members: different query, will get low BLEU
        chunks.append(Chunk(f"n{i}", "d", f"XXX chunk {i} parole diverse originali", is_member=False, has_person=False))

    proba, labels = s2mia_scores_model(_RAG(), chunks, seed=42)

    # Shape and type checks
    assert isinstance(proba, list)
    assert isinstance(labels, list)
    assert len(proba) == len(labels) == len(chunks)

    # All probabilities in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in proba)

    # Labels match is_member
    expected_labels = [1]*12 + [0]*12
    assert labels == expected_labels

def test_xgb_proba_handles_inf_nan():
    """Regressione: la perplexity nativa (gray-box) o gpt2 su answer vuota può
    valere +inf → XGBoost crasha ('Input data contains inf'). _xgb_proba deve
    sanitizzare inf/nan e restituire probabilità finite in [0,1]."""
    from miarag.attacks.s2mia import _xgb_proba
    X, y = [], []
    for i in range(12):
        X.append([0.8, float("inf") if i % 4 == 0 else 5.0]); y.append(1)
    for i in range(12):
        X.append([0.01, 300.0]); y.append(0)
    X[0][1] = float("nan")          # anche nan non deve far crashare
    proba = _xgb_proba(X, y, seed=42)
    assert len(proba) == len(y)
    assert all(0.0 <= p <= 1.0 for p in proba)
