# src/miarag/config.py
"""Settings PoC MIA-RAG. Provider-agnostic dalla v0.2.

Backcompat: default = comportamento v0.1-thesis (Ollama + MiniLM + GPT-2 PPL).
Override via env: LLM_PROVIDER, EMBEDDING_PROVIDER, PERPLEXITY_PROVIDER.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]  # master/miarag/


@dataclass(frozen=True)
class Settings:
    # ─── Provider selection ───────────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_tf")
    perplexity_provider: str = os.getenv("PERPLEXITY_PROVIDER", "gpt2")

    # ─── Ollama (backcompat v0.1-thesis) ──────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_TARGET_MODEL", "llama3.1:8b")

    # ─── Embeddings (nome modello per sentence_tf; ignorato da altri) ────
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ─── RAG params ───────────────────────────────────────────────────────
    top_k: int = int(os.getenv("TOP_K", "4"))
    seed: int = int(os.getenv("SEED", "42"))

    # ─── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = _ROOT / "data"
    results_dir: Path = _ROOT / "results"
    corpus_dir: Path = _ROOT / "documenti"

    def validate(self) -> None:
        """Fail-fast su config incoerenti. Chiama esplicitamente dagli entry-point."""
        errs: list[str] = []
        if self.llm_provider == "azure_openai":
            for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                      "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"):
                if not os.getenv(k):
                    errs.append(f"env var mancante: {k} (richiesta da llm_provider=azure_openai)")
        if self.llm_provider == "bedrock":
            if not os.getenv("BEDROCK_CLAUDE_MODEL_ID"):
                errs.append("env var mancante: BEDROCK_CLAUDE_MODEL_ID")
        if self.embedding_provider == "openai_embed":
            for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                      "AZURE_OPENAI_API_VERSION"):
                if not os.getenv(k):
                    errs.append(f"env var mancante: {k} (richiesta da embedding_provider=openai_embed)")
        if errs:
            raise RuntimeError("Config invalida:\n  - " + "\n  - ".join(errs))


def get_settings() -> Settings:
    """Legge le variabili d'ambiente AL MOMENTO DELLA CHIAMATA.

    NB: i default dei campi della dataclass sono valutati all'import del modulo;
    quindi impostare os.environ dopo l'import NON li aggiornerebbe. Qui passiamo
    esplicitamente i valori correnti così che override runtime (es. flag CLI
    --llm che setta LLM_PROVIDER prima di get_settings) funzionino davvero.
    """
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_tf"),
        perplexity_provider=os.getenv("PERPLEXITY_PROVIDER", "gpt2"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_TARGET_MODEL", "llama3.1:8b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=int(os.getenv("TOP_K", "4")),
        seed=int(os.getenv("SEED", "42")),
    )
