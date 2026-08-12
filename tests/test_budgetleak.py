# tests/test_budgetleak.py
from miarag.corpus import Chunk
from miarag.attacks.budgetleak import budget_features, budgetleak_scores, BUDGETS

def test_features_monotone_sequence():
    f = budget_features([0.1, 0.5, 0.9])
    assert f["final_sim"] == 0.9
    assert f["rate_of_change"] > 0

class _RAG:
    """Membro: similarità sale in fretta col budget. Non-membro: resta bassa."""
    def query(self, question, max_tokens=256):
        from miarag.rag import RAGResponse
        member = "MEM" in question
        # più alto max_tokens ⇒ per il membro risposta più vicina all'atteso
        sim_answer = ("expected " * max_tokens) if member else "noise noise noise"
        return RAGResponse(answer=sim_answer, retrieved_ids=["x"], perplexity=None)

def test_budgetleak_separates():
    chunks = [
        Chunk("m", "d", "MEM expected expected expected expected", is_member=True, has_person=False),
        Chunk("n", "d", "XXX expected expected expected expected", is_member=False, has_person=False),
    ]
    scores, labels = budgetleak_scores(_RAG(), chunks)
    assert labels == [1, 0]
    assert scores[0] > scores[1]
    assert len(BUDGETS) == 3
