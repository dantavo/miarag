# tests/test_rag.py
import pytest
from unittest.mock import MagicMock
from miarag.corpus import Chunk
from miarag.rag import TargetRAG, RAGResponse

class _FakeEmbed:
    def embed_documents(self, texts): return [[float(len(t)), 1.0] for t in texts]
    def embed_query(self, text): return [float(len(text)), 1.0]

def _chunks():
    return [Chunk(f"c{i}", "d0", f"contenuto numero {i} " * 3, True, False) for i in range(5)]

def test_index_and_retrieve(monkeypatch):
    rag = TargetRAG.__new__(TargetRAG)          # bypass __init__ per iniettare fake
    rag._configure_for_test(embedder=_FakeEmbed(), generate=lambda prompt, max_tokens: "risposta", top_k=2)
    rag.index(_chunks())
    resp = rag.query("contenuto numero 1", max_tokens=32)
    assert isinstance(resp, RAGResponse)
    assert len(resp.retrieved_ids) == 2
    assert resp.answer == "risposta"

def test_max_tokens_passed_through():
    seen = {}
    rag = TargetRAG.__new__(TargetRAG)
    rag._configure_for_test(embedder=_FakeEmbed(),
                            generate=lambda prompt, max_tokens: seen.setdefault("mt", max_tokens) or "ok",
                            top_k=1)
    rag.index(_chunks())
    rag.query("x", max_tokens=7)
    assert seen["mt"] == 7

def test_perplexity_fallback_deterministic():
    """Test perplexity path with mocked _perplexity_hf to avoid network download."""
    rag = TargetRAG.__new__(TargetRAG)
    rag._ollama_url = "http://localhost:11434"
    rag._ollama_model = "llama3.1:8b"
    # Mock the HF perplexity method to return deterministic value
    rag._perplexity_hf = lambda text: 42.0
    ppl = rag.perplexity_of("test text")
    assert ppl == 42.0

def test_ollama_generate_puts_num_predict_in_options():
    """Verify max_tokens reaches Ollama correctly via options dict (BudgetLeak critical)."""
    rag = TargetRAG.__new__(TargetRAG)
    rag._llm = MagicMock()
    rag._llm.invoke.return_value = "ok"
    out = rag._ollama_generate("prompt", max_tokens=7)
    assert out == "ok"
    assert rag._llm.invoke.call_args.kwargs.get("options") == {"num_predict": 7}
