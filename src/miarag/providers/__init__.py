# src/miarag/providers/__init__.py
"""Provider factory. Chiave: build_* legge Settings e istanzia il backend giusto.

Uso:
    from miarag.providers import build_llm, build_embedder, build_perplexity
    llm = build_llm(settings)
    emb = build_embedder(settings)
    ppl = build_perplexity(settings)
"""
from __future__ import annotations
from typing import Callable, Any

from miarag.providers.base import LLMProvider, EmbeddingProvider, PerplexityScorer

__all__ = [
    "LLMProvider", "EmbeddingProvider", "PerplexityScorer",
    "build_llm", "build_embedder", "build_perplexity",
    "register_llm", "register_embedder", "register_perplexity",
]


# ─── Registry: nome → factory(settings) ───────────────────────────────────────

_LLM_REGISTRY: dict[str, Callable[[Any], LLMProvider]] = {}
_EMBED_REGISTRY: dict[str, Callable[[Any], EmbeddingProvider]] = {}
_PPL_REGISTRY: dict[str, Callable[[Any], PerplexityScorer]] = {}


def register_llm(name: str, factory: Callable[[Any], LLMProvider]) -> None:
    _LLM_REGISTRY[name] = factory


def register_embedder(name: str, factory: Callable[[Any], EmbeddingProvider]) -> None:
    _EMBED_REGISTRY[name] = factory


def register_perplexity(name: str, factory: Callable[[Any], PerplexityScorer]) -> None:
    _PPL_REGISTRY[name] = factory


# ─── Built-in providers (import lazy dentro factory per evitare heavy deps a boot) ──

def _make_ollama(s):
    from miarag.providers.ollama import OllamaProvider
    return OllamaProvider(base_url=s.ollama_base_url, model=s.ollama_model)


def _make_azure(s):
    from miarag.providers.azure_openai import AzureOpenAIProvider
    import os
    return AzureOpenAIProvider(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    )


def _make_bedrock(s):
    from miarag.providers.bedrock import BedrockProvider
    import os
    return BedrockProvider(
        model_id=os.environ["BEDROCK_CLAUDE_MODEL_ID"],
        region=os.environ.get("AWS_REGION", "eu-west-1"),
    )


def _make_sentence_tf(s):
    from miarag.providers.embeddings.sentence_tf import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model=s.embedding_model)


def _make_openai_embed(s):
    from miarag.providers.embeddings.openai_embed import OpenAIEmbedder
    import os
    return OpenAIEmbedder(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        deployment=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small"),
    )


def _make_gpt2_ppl(s):
    from miarag.providers.perplexity.gpt2 import GPT2Perplexity
    return GPT2Perplexity()


def _make_hf_causal_ppl(s):
    from miarag.providers.perplexity.hf_causal import HFCausalPerplexity
    import os
    model = os.environ.get("PERPLEXITY_HF_MODEL", "gpt2")
    return HFCausalPerplexity(model_name=model)


register_llm("ollama", _make_ollama)
register_llm("azure_openai", _make_azure)
register_llm("bedrock", _make_bedrock)

register_embedder("sentence_tf", _make_sentence_tf)
register_embedder("openai_embed", _make_openai_embed)

register_perplexity("gpt2", _make_gpt2_ppl)
register_perplexity("hf_causal", _make_hf_causal_ppl)


# ─── Factory pubbliche ────────────────────────────────────────────────────────

def build_llm(settings) -> LLMProvider:
    name = settings.llm_provider
    if name not in _LLM_REGISTRY:
        raise ValueError(f"LLM provider '{name}' sconosciuto. Disponibili: {sorted(_LLM_REGISTRY)}")
    return _LLM_REGISTRY[name](settings)


def build_embedder(settings) -> EmbeddingProvider:
    name = settings.embedding_provider
    if name not in _EMBED_REGISTRY:
        raise ValueError(f"Embedding provider '{name}' sconosciuto. Disponibili: {sorted(_EMBED_REGISTRY)}")
    return _EMBED_REGISTRY[name](settings)


def build_perplexity(settings) -> PerplexityScorer:
    name = settings.perplexity_provider
    if name not in _PPL_REGISTRY:
        raise ValueError(f"Perplexity provider '{name}' sconosciuto. Disponibili: {sorted(_PPL_REGISTRY)}")
    return _PPL_REGISTRY[name](settings)
