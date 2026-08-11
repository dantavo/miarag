# src/miarag/config.py
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]  # master/poc/

@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_TARGET_MODEL", "llama3.1:8b")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    seed: int = 42
    data_dir: Path = _ROOT / "data"
    results_dir: Path = _ROOT / "results"
    corpus_dir: Path = _ROOT / "documenti"

def get_settings() -> Settings:
    return Settings()
