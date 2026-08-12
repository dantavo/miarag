# tests/test_s2mia.py
from miarag.corpus import Chunk
from miarag.attacks.s2mia import split_query_answer, s2mia_scores

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
