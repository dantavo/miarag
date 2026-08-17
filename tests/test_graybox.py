# tests/test_graybox.py
"""Test gray-box (logprobs) logic. Offline: nessuna chiamata Azure."""
import math
from miarag.corpus import Chunk
from miarag.attacks.rag_mia import (
    rag_mia_graybox_score, rag_mia_graybox_scores, _logsumexp,
)


def test_logsumexp_basic():
    # logsumexp([0]) = 0 ; logsumexp([0,0]) = ln 2
    assert abs(_logsumexp([0.0]) - 0.0) < 1e-9
    assert abs(_logsumexp([0.0, 0.0]) - math.log(2)) < 1e-9
    assert _logsumexp([]) == float("-inf")


def test_graybox_score_yes_dominant():
    # Sì con logprob alto, No basso → score verso 1
    s = rag_mia_graybox_score({"Sì": -0.1, "No": -5.0})
    assert s > 0.9


def test_graybox_score_no_dominant():
    s = rag_mia_graybox_score({"Sì": -5.0, "No": -0.1})
    assert s < 0.1


def test_graybox_score_balanced():
    s = rag_mia_graybox_score({"Sì": -1.0, "No": -1.0})
    assert abs(s - 0.5) < 1e-6


def test_graybox_score_no_yes_no_tokens():
    # nessun token Sì/No riconosciuto → 0.5 incerto
    assert rag_mia_graybox_score({"forse": -0.2, "boh": -0.3}) == 0.5


def test_graybox_score_continuous_not_discrete():
    # a differenza del black-box (0/0.5/1), qui lo score è continuo
    s1 = rag_mia_graybox_score({"Sì": -0.5, "No": -1.0})
    s2 = rag_mia_graybox_score({"Sì": -0.5, "No": -2.0})
    assert s1 != s2 and 0.0 < s1 < 1.0 and 0.0 < s2 < 1.0


class _FakeLogprobRAG:
    """RAG fittizio con query_with_logprobs: membri (MEM) → Sì dominante."""
    def query_with_logprobs(self, question, max_tokens=8):
        if "MEM" in question:
            first_top = {"Sì": -0.05, "No": -4.0}
        else:
            first_top = {"Sì": -3.0, "No": -0.2}
        return {"answer": "Sì" if "MEM" in question else "No",
                "retrieved_ids": ["c0"], "token_logprobs": [-0.05], "first_top": first_top}


def test_graybox_scores_separates_and_continuous():
    chunks = [
        Chunk("m1", "d", "MEM contratto", is_member=True, has_person=False),
        Chunk("n1", "d", "XXX diverso", is_member=False, has_person=False),
    ]
    scores, labels = rag_mia_graybox_scores(_FakeLogprobRAG(), chunks)
    assert labels == [1, 0]
    assert scores[0] > scores[1]                 # membro più alto
    assert all(0.0 <= s <= 1.0 for s in scores)  # in [0,1]
    assert scores[0] not in (0.0, 0.5, 1.0)      # continuo, non discreto
