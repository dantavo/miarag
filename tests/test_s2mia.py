# tests/test_s2mia.py
from miarag.corpus import Chunk
from miarag.attacks.s2mia import split_query_answer, s2mia_scores, s2mia_scores_model

def test_split_halves():
    q, a = split_query_answer("uno due tre quattro cinque sei")
    assert q and a and q != a

class _RAG:
    """Membro: restituisce la risposta attesa (BLEU alta). Non-membro: testo diverso."""
    def query(self, question, max_tokens=256):
        from miarag.rag import RAGResponse
        # simulazione: se la question contiene 'MEM' rispondi 'combacia esattamente'
        ans = "combacia esattamente" if "MEM" in question else "parole totalmente diverse qui"
        return RAGResponse(answer=ans, retrieved_ids=["x"], perplexity=None)
    def perplexity_of(self, text): return 5.0

def test_scores_separate_members():
    chunks = [
        Chunk("m", "d", "MEM combacia esattamente", is_member=True, has_person=False),
        Chunk("n", "d", "XXX parole diverse originali", is_member=False, has_person=False),
    ]
    scores, labels = s2mia_scores(_RAG(), chunks)
    assert labels == [1, 0]
    assert scores[0] > scores[1]     # il membro ha score più alto

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
