# src/miarag/providers/embeddings/openai_embed.py
"""OpenAI/Azure embeddings via langchain_openai."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class OpenAIEmbedder:
    """Azure OpenAI embeddings (text-embedding-3-small/large)."""
    api_key: str
    endpoint: str
    api_version: str
    deployment: str  # embedding deployment name
    name: str = "openai_embed"
    dim: int = field(init=False, default=0)

    def __post_init__(self):
        from langchain_openai import AzureOpenAIEmbeddings
        self._embed = AzureOpenAIEmbeddings(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            azure_deployment=self.deployment,
        )
        # dim: probe con dummy embed (1 chiamata API, accettabile a init).
        probe = self._embed.embed_query("probe")
        self.dim = len(probe)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed.embed_query(text)
