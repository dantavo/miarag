# tests/test_config.py
from pathlib import Path
from miarag.config import get_settings

def test_settings_defaults():
    s = get_settings()
    assert s.seed == 42
    assert s.top_k == 4
    assert s.ollama_model  # non vuoto
    assert isinstance(s.data_dir, Path)
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

def test_get_settings_reads_env_at_call_time(monkeypatch):
    """Regression: get_settings deve leggere le env AL MOMENTO della chiamata,
    non ai default congelati all'import. Blindare il bug del flag --llm che
    non aveva effetto (config valutava os.getenv all'import del modulo)."""
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_embed")
    monkeypatch.setenv("PERPLEXITY_PROVIDER", "hf_causal")
    monkeypatch.setenv("TOP_K", "8")
    s = get_settings()
    assert s.llm_provider == "azure_openai"
    assert s.embedding_provider == "openai_embed"
    assert s.perplexity_provider == "hf_causal"
    assert s.top_k == 8
