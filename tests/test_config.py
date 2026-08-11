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
