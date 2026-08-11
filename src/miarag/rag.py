# src/miarag/rag.py
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
    def __init__(self, embedding_model, ollama_base_url, ollama_model, top_k=4, collection_name="miarag"):
        from langchain_ollama import OllamaLLM
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(embedding_model)
        self._embed_docs = lambda texts: self._st.encode(list(texts)).tolist()
        self._embed_query = lambda t: self._st.encode([t])[0].tolist()
        self._ollama_url = ollama_base_url
        self._ollama_model = ollama_model
        self._llm = OllamaLLM(base_url=ollama_base_url, model=ollama_model)
        self._generate = self._ollama_generate
        self.top_k = top_k
        self._client = chromadb.EphemeralClient()
        self._coll = self._client.create_collection(collection_name)

    def _configure_for_test(self, embedder, generate, top_k):
        import uuid
        self._embed_docs = embedder.embed_documents
        self._embed_query = embedder.embed_query
        self._generate = generate
        self.top_k = top_k
        self._client = chromadb.EphemeralClient()
        self._coll = self._client.create_collection(f"test_{uuid.uuid4().hex[:8]}")

    def _ollama_generate(self, prompt: str, max_tokens: int) -> str:
        return self._llm.invoke(prompt, options={"num_predict": max_tokens})

    def index(self, chunks: list[Chunk]) -> None:
        embs = self._embed_docs([c.text for c in chunks])
        self._coll.add(ids=[c.chunk_id for c in chunks],
                       documents=[c.text for c in chunks],
                       embeddings=embs)

    def query(self, question: str, max_tokens: int = 256) -> RAGResponse:
        q_emb = self._embed_query(question)
        res = self._coll.query(query_embeddings=[q_emb], n_results=self.top_k)
        ids = res["ids"][0]
        ctx = "\n".join(res["documents"][0])
        answer = self._generate(_PROMPT.format(context=ctx, q=question), max_tokens)
        return RAGResponse(answer=answer, retrieved_ids=ids, perplexity=None)

    def perplexity_of(self, text: str) -> float:
        """
        Computes perplexity of text under the target model.

        Standard Ollama /api/generate does NOT expose per-token logprobs,
        so we fall back to a local HF causal LM (gpt2) for perplexity estimation.
        This is the production-safe implementation per Task 4 brief Step 5.
        """
        return self._perplexity_hf(text)

    def _perplexity_hf(self, text: str) -> float:
        """
        Fallback perplexity using local HF causal LM (gpt2).
        Computes true per-token NLL → perplexity = exp(mean NLL).
        """
        import math
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Lazy-load model (cache in instance)
        if not hasattr(self, '_perplexity_model'):
            self._perplexity_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self._perplexity_model.eval()
            # Use MPS (Metal) if available on Apple Silicon
            if torch.backends.mps.is_available():
                self._perplexity_model = self._perplexity_model.to("mps")

        tokenizer = self._perplexity_tokenizer
        model = self._perplexity_model
        device = model.device

        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss is the mean NLL per token
            nll = outputs.loss.item()

        return math.exp(nll)
