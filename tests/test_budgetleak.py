# tests/test_budgetleak.py
from miarag.corpus import Chunk
from miarag.attacks.budgetleak import budget_features, budgetleak_scores, budget_sequence, BUDGETS

def test_features_monotone_sequence():
    f = budget_features([0.1, 0.5, 0.9])
    assert f["final_sim"] == 0.9
    assert f["rate_of_change"] > 0

def test_features_edge_case_single_value():
    """Single-value sequence should not crash."""
    f = budget_features([0.5])
    assert f["rate_of_change"] == 0.0
    assert f["cumulative_fluctuation"] == 0.0
    assert f["final_sim"] == 0.5

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

class _RAG_Growing:
    """RAG that produces growing similarity for members, flat for non-members."""
    def __init__(self):
        # Budgets: (32, 96, 256)
        # Member: similarity grows with budget → [0.3, 0.7, 1.0]
        # Non-member: stays flat and low → [0.1, 0.1, 0.1]
        self.budget_to_sim_member = {32: 0.3, 96: 0.7, 256: 1.0}
        self.budget_to_sim_nonmember = {32: 0.1, 96: 0.1, 256: 0.1}

    def query(self, question, max_tokens=256):
        from miarag.rag import RAGResponse
        member = "MEM" in question
        target_sim = self.budget_to_sim_member.get(max_tokens, 0.0) if member else self.budget_to_sim_nonmember.get(max_tokens, 0.0)

        # The expected answer (from split_query_answer) will be "expected expected expected"
        # To control Jaccard similarity, we need to control the intersection and union.
        # Jaccard(A,B) = |A∩B| / |A∪B|
        #
        # Strategy: vary the mix of "expected" (shared) and unique tokens
        # Budget 32 (target 0.3): answer = "expected noise1 noise2 noise3" → Jaccard = 1/4 = 0.25
        # Budget 96 (target 0.7): answer = "expected expected noise4" → Jaccard = 2/3 = 0.67
        # Budget 256 (target 1.0): answer = "expected expected expected" → Jaccard = 3/3 = 1.0

        if target_sim >= 0.9:
            answer = "expected expected expected"
        elif target_sim >= 0.6:
            answer = "expected expected noise4"
        elif target_sim >= 0.2:
            answer = "expected noise1 noise2 noise3"
        else:
            answer = "noise5 noise6 noise7"

        return RAGResponse(answer=answer, retrieved_ids=["x"], perplexity=None)

def test_rate_of_change_contributes():
    """Validates that rate_of_change drives separation, not just final_sim."""
    rag = _RAG_Growing()
    chunks = [
        Chunk("m", "d", "MEM expected expected expected", is_member=True, has_person=False),
        Chunk("n", "d", "XXX expected expected expected", is_member=False, has_person=False),
    ]

    # Check sequences: member should grow, non-member flat
    seq_member = budget_sequence(rag, chunks[0].text)
    seq_nonmember = budget_sequence(rag, chunks[1].text)

    # Member: growing sequence
    assert seq_member[0] < seq_member[1] < seq_member[2], f"Member seq should grow: {seq_member}"

    # Non-member: flat sequence
    assert abs(seq_nonmember[0] - seq_nonmember[1]) < 0.01, f"Non-member should be flat: {seq_nonmember}"
    assert abs(seq_nonmember[1] - seq_nonmember[2]) < 0.01

    # Check features
    f_member = budget_features(seq_member)
    f_nonmember = budget_features(seq_nonmember)

    # Member has positive rate_of_change, non-member near zero
    assert f_member["rate_of_change"] > f_nonmember["rate_of_change"], \
        f"Member roc={f_member['rate_of_change']} should > non-member roc={f_nonmember['rate_of_change']}"

    # Even if final_sim were similar, rate_of_change should separate
    # In budgetleak_scores, score = roc + final_sim
    scores, labels = budgetleak_scores(rag, chunks)
    assert scores[0] > scores[1], \
        f"Member score={scores[0]} should > non-member score={scores[1]} (labels={labels})"
