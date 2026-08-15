# tests/test_rag_mia.py
"""Test RAG-MIA (Anderson 2025 prompt injection attack). Offline, no network."""
from miarag.corpus import Chunk
from miarag.rag import RAGResponse
from miarag.attacks.rag_mia import (
    parse_yes_no,
    rag_mia_features,
    rag_mia_scores,
    _truncate,
)


class _RAG:
    """Simulazione RAG:
    - Membri: risponde 'Sì' (ha visto il testo nel contesto).
    - Non-membri: risponde 'No' (contesto irrilevante).
    Distinzione via marker 'MEM' nella query.
    """
    def query(self, question, max_tokens=256):
        ans = "Sì" if "MEM" in question else "No"
        return RAGResponse(answer=ans, retrieved_ids=["c0"], perplexity=None)


class _RAGRefusal:
    """RAG con prompt hardening: rifiuta la domanda."""
    def query(self, question, max_tokens=256):
        return RAGResponse(answer="unanswerable", retrieved_ids=[], perplexity=None)


# ─── parse_yes_no ────────────────────────────────────────────────────────

def test_parse_yes_no_english():
    assert parse_yes_no("Yes") == 1.0
    assert parse_yes_no("yes.") == 1.0
    assert parse_yes_no("Yes, it appears in the context.") == 1.0
    assert parse_yes_no("No") == 0.0
    assert parse_yes_no("No, absolutely not.") == 0.0


def test_parse_yes_no_italian():
    assert parse_yes_no("Sì") == 1.0
    assert parse_yes_no("Sì, il testo compare.") == 1.0
    assert parse_yes_no("No, non compare") == 0.0


def test_parse_yes_no_refusal_returns_neutral():
    """Prompt hardening: 'unanswerable' → 0.5 (neutral)."""
    assert parse_yes_no("unanswerable") == 0.5
    assert parse_yes_no("I cannot answer") == 0.5
    assert parse_yes_no("Non posso rispondere") == 0.5


def test_parse_yes_no_ambiguous_returns_neutral():
    assert parse_yes_no("") == 0.5
    assert parse_yes_no("   ") == 0.5
    assert parse_yes_no("boh non lo so davvero") == 0.5


def test_parse_yes_no_yes_before_no_wins():
    """'Yes, but no' → Yes vince (primo match)."""
    assert parse_yes_no("Yes, but no") == 1.0
    assert parse_yes_no("No, but yes") == 0.0


# ─── _truncate ──────────────────────────────────────────────────────────

def test_truncate_short_stays_intact():
    assert _truncate("breve testo") == "breve testo"


def test_truncate_long_at_word_boundary():
    text = "parola " * 200  # 1400 chars
    out = _truncate(text)
    assert len(out) <= 500
    # Non tronca a metà parola
    assert not out.endswith("paro")


# ─── rag_mia_features ────────────────────────────────────────────────────

def test_features_member_returns_yes():
    f = rag_mia_features(_RAG(), "MEM contratto assicurativo importante")
    assert f["yes_score"] == 1.0
    assert "Sì" in f["answer"]


def test_features_non_member_returns_no():
    f = rag_mia_features(_RAG(), "XXX qualcosa di completamente diverso")
    assert f["yes_score"] == 0.0


def test_features_refusal_neutral():
    f = rag_mia_features(_RAGRefusal(), "MEM testo qualunque")
    assert f["yes_score"] == 0.5


# ─── rag_mia_scores (contratto uniforme S2MIA/BudgetLeak) ───────────────

def test_scores_separate_members_from_non_members():
    chunks = [
        Chunk("m1", "d", "MEM primo membro", is_member=True, has_person=False),
        Chunk("m2", "d", "MEM secondo membro", is_member=True, has_person=False),
        Chunk("n1", "d", "XXX primo non membro", is_member=False, has_person=False),
        Chunk("n2", "d", "XXX secondo non membro", is_member=False, has_person=False),
    ]
    scores, labels = rag_mia_scores(_RAG(), chunks)
    assert scores == [1.0, 1.0, 0.0, 0.0]
    assert labels == [1, 1, 0, 0]


def test_scores_english_prompt():
    """Language switch: prompt cambia ma contratto scores invariato."""
    chunks = [Chunk("m", "d", "MEM sample", is_member=True, has_person=False)]
    scores, labels = rag_mia_scores(_RAG(), chunks, language="en")
    assert scores == [1.0]
    assert labels == [1]


def test_score_range_bounded_0_1():
    """Score sempre in [0, 1]: contratto per metriche/plots."""
    chunks = [
        Chunk("m", "d", "MEM x", is_member=True, has_person=False),
        Chunk("n", "d", "YYY z", is_member=False, has_person=False),
    ]
    scores, _ = rag_mia_scores(_RAG(), chunks)
    for s in scores:
        assert 0.0 <= s <= 1.0
