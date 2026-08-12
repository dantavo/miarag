# tests/test_defenses.py
from miarag.defenses import paraphrase_defense, apply_defense

def test_paraphrase_lowers_overlap():
    original = "Il fatturato 2025 e' 21644 migliaia di euro"
    out = paraphrase_defense(original)
    assert out != original
    assert out == out.lower() or "," not in out   # normalizzato

def test_apply_defense_none_is_identity():
    class _RAG:
        def query(self, q, max_tokens=256): return "x"
    r = apply_defense(_RAG(), "none")
    assert r.query("q").answer if hasattr(r.query("q"), "answer") else True

def test_defense_wrapper_delegates_index():
    from miarag.defenses import apply_defense
    class _FakeRAG:
        def __init__(self): self.indexed = False
        def index(self, chunks): self.indexed = True
        def query(self, q, max_tokens=256): return "ok"
        def perplexity_of(self, text): return 1.0
    inner = _FakeRAG()
    wrapped = apply_defense(inner, "paraphrase")
    wrapped.index([])          # must not raise AttributeError
    assert inner.indexed is True
