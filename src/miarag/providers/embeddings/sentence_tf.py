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
        # get_embedding_dimension (nuovo) con fallback su get_sentence_embedding_dimension (vecchio).
        _dim_fn = getattr(self._st, "get_embedding_dimension", None) \
            or getattr(self._st, "get_sentence_embedding_dimension", None)
        self.dim = int(_dim_fn() or 0) if _dim_fn else 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # MPS non thread-safe: serializza (no-op se estrazione single-thread).
        from miarag.providers._mps import MPS_LOCK
        with MPS_LOCK:
            return self._st.encode(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        from miarag.providers._mps import MPS_LOCK
        with MPS_LOCK:
            return self._st.encode([text])[0].tolist()
