# src/miarag/providers/embeddings/sentence_tf.py
"""SentenceTransformerEmbedder: locale, no rete a runtime dopo primo download."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SentenceTransformerEmbedder:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "sentence_tf"
    dim: int = field(init=False, default=0)

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(self.model)
        # dim: probe con dummy encode
        self.dim = int(self._st.get_sentence_embedding_dimension() or 0)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._st.encode(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._st.encode([text])[0].tolist()
