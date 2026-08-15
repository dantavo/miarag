# src/miarag/providers/base.py
"""Provider protocols per LLM / Embeddings / Perplexity.

Design: interfacce minimali. Ogni backend (Ollama, Azure OpenAI, Bedrock, HF…)
implementa la Protocol corrispondente. TargetRAG resta thin e provider-agnostic.

Chiave: `budget_kwargs(max_tokens)` mappa il concetto astratto "budget di generazione"
al param nativo del provider (num_predict / max_tokens / maxTokens). BudgetLeak
funziona così su qualunque backend senza toccare l'attacco.
"""
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Backend di generazione testo. Black-box: prompt → risposta."""

    name: str

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Genera risposta. `max_tokens` = budget generation (mappato internamente)."""
        ...

    def supports_logprobs(self) -> bool:
        """True se il provider espone logprobs per calcolo PPL nativo.

        Ollama v0.32.7 → False (usa proxy). Azure OpenAI → True. Bedrock → dipende.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Backend di embedding. Locale (sentence-transformers) o API (OpenAI/Bedrock)."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class PerplexityScorer(Protocol):
    """Calcolo perplexity per feature S2MIA. Indipendente dal target LLM
    (l'attaccante usa il proprio LM di riferimento)."""

    name: str

    def perplexity(self, text: str) -> float:
        """Restituisce exp(mean NLL). +inf se il testo è vuoto o degenere."""
        ...
