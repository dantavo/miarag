# src/miarag/rag.py
"""TargetRAG: RAG black-box provider-agnostic.

Firma v0.2 (raccomandata):
    TargetRAG(llm=<LLMProvider>, embedder=<EmbeddingProvider>, ppl=<PerplexityScorer>, top_k=4)

Firma v0.1-thesis (backcompat, deprecated ma funzionante):
    TargetRAG(embedding_model, ollama_base_url, ollama_model, top_k=4)

Se chiamato con la firma vecchia, costruisce internamente Ollama + SentenceTransformer + GPT-2
esattamente come v0.1-thesis → nessun test si rompe.
"""
from __future__ import annotations
from dataclasses import dataclass
import chromadb
from miarag.corpus import Chunk


@dataclass
class RAGResponse:
    answer: str
    retrieved_ids: list[str]
    perplexity: float | None = None


_PROMPT = (
    "Usa il contesto per rispondere.\n\nContesto:\n{context}\n\nDomanda: {q}\nRisposta:"
)


class TargetRAG:
    def __init__(self, *args, llm=None, embedder=None, ppl=None, top_k: int = 4,
                 collection_name: str = "miarag", **kwargs):
        # ─── Backcompat v0.1-thesis: positional (embedding_model, url, model, top_k) ──
        if args and llm is None and embedder is None:
            embedding_model = args[0]
            ollama_base_url = args[1] if len(args) > 1 else kwargs.get("ollama_base_url")
            ollama_model = args[2] if len(args) > 2 else kwargs.get("ollama_model")
            top_k = args[3] if len(args) > 3 else top_k
            from miarag.providers.ollama import OllamaProvider
            from miarag.providers.embeddings.sentence_tf import SentenceTransformerEmbedder
            llm = OllamaProvider(base_url=ollama_base_url, model=ollama_model)
            embedder = SentenceTransformerEmbedder(model=embedding_model)
            # ppl None → lazy: creato al primo perplexity_of() (evita download GPT-2 se non serve)

        self._llm = llm
        self._embedder = embedder
        self._ppl = ppl
        self.top_k = top_k

        # Adattatori interni (test-friendly: _configure_for_test li rimpiazza).
        if embedder is not None:
            self._embed_docs = embedder.embed_documents
            self._embed_query = embedder.embed_query
        if llm is not None:
            self._generate = lambda prompt, max_tokens: llm.generate(prompt, max_tokens)

        self._client = chromadb.EphemeralClient()
        self._coll = self._client.create_collection(collection_name)

    # ─── Test helper (v0.1-thesis API) ────────────────────────────────────
    def _configure_for_test(self, embedder, generate, top_k):
        import uuid
        self._embed_docs = embedder.embed_documents
        self._embed_query = embedder.embed_query
        self._generate = generate
        self.top_k = top_k
        self._client = chromadb.EphemeralClient()
        self._coll = self._client.create_collection(f"test_{uuid.uuid4().hex[:8]}")

    # ─── Core RAG ─────────────────────────────────────────────────────────
    def index(self, chunks: list[Chunk]) -> None:
        embs = self._embed_docs([c.text for c in chunks])
        self._coll.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embs,
        )

    def query(self, question: str, max_tokens: int = 256) -> RAGResponse:
        q_emb = self._embed_query(question)
        res = self._coll.query(query_embeddings=[q_emb], n_results=self.top_k)
        ids = res["ids"][0]
        ctx = "\n".join(res["documents"][0])
        answer = self._generate(_PROMPT.format(context=ctx, q=question), max_tokens)
        return RAGResponse(answer=answer, retrieved_ids=ids, perplexity=None)

    # ─── Perplexity (delegato a PerplexityScorer iniettato o lazy GPT-2) ──
    def perplexity_of(self, text: str) -> float:
        if self._ppl is None:
            # Lazy backcompat: costruisci GPT-2 al primo uso.
            from miarag.providers.perplexity.gpt2 import GPT2Perplexity
            self._ppl = GPT2Perplexity()
        return self._ppl.perplexity(text)

    # ─── Compat shim: alcuni test montano _perplexity_hf direttamente ─────
    @property
    def _perplexity_hf(self):
        # Se un test ha fatto rag._perplexity_hf = lambda ...: ..., torna quella closure.
        return self.__dict__.get("_perplexity_hf_override")

    @_perplexity_hf.setter
    def _perplexity_hf(self, fn):
        # Setter usato dai test v0.1-thesis: reindirizza perplexity_of() al fn iniettato.
        self.__dict__["_perplexity_hf_override"] = fn

        class _InlinePpl:
            name = "test_inline"
            def perplexity(self, text): return fn(text)
        self._ppl = _InlinePpl()

    # Backcompat: test v0.1-thesis chiama rag._ollama_generate("prompt", 7) direttamente
    # dopo aver montato rag._llm = MagicMock(). Espone stesso contratto:
    # chiama self._llm.invoke(prompt, options={"num_predict": max_tokens}).
    def _ollama_generate(self, prompt: str, max_tokens: int) -> str:
        # Preferisci path Ollama-native se disponibile (test lo mockano con MagicMock).
        if self._llm is not None and hasattr(self._llm, "invoke"):
            return self._llm.invoke(prompt, options={"num_predict": max_tokens})
        # Fallback: usa il provider astratto.
        return self._llm.generate(prompt, max_tokens)
