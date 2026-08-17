# src/miarag/providers/azure_openai.py
"""AzureOpenAIProvider: Azure OpenAI chat completions via httpx diretto.

Richiede env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.

Usa httpx diretto (non langchain) per:
- accesso ai LOGPROBS (Azure li espone; Ollama no) → perplexity nativa + gray-box
- retry/backoff + cost tracking uniformi con OllamaProvider
- controllo esplicito del timeout

`generate()` → solo testo (contratto LLMProvider, uso black-box).
`generate_with_logprobs()` → testo + logprob per-token + top-logprob del 1° token
(per RAG-MIA gray-box e perplexity nativa).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from miarag.providers._cost import TRACKER


@dataclass
class AzureOpenAIProvider:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str
    name: str = "azure_openai"
    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    max_retries: int = 3
    top_logprobs: int = 5

    def __post_init__(self):
        import httpx
        base = self.endpoint.rstrip("/")
        self._url = (
            f"{base}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=self.connect_timeout, read=self.read_timeout,
                write=self.read_timeout, pool=self.read_timeout,
            ),
            headers={"api-key": self.api_key, "content-type": "application/json"},
        )

    # ─── core POST con retry ──────────────────────────────────────────────
    def _post(self, body: dict) -> dict:
        import time, httpx
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(self._url, json=body)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"Azure call failed after {self.max_retries} attempts: {last}")

    def _body(self, prompt: str, max_tokens: int, logprobs: bool = False) -> dict:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        return body

    # ─── LLMProvider contract ─────────────────────────────────────────────
    def generate(self, prompt: str, max_tokens: int) -> str:
        d = self._post(self._body(prompt, max_tokens))
        content = d["choices"][0]["message"].get("content") or ""
        TRACKER.record(self.name, prompt, content)
        return content

    def supports_logprobs(self) -> bool:
        return True

    # ─── Gray-box / perplexity: generazione + logprobs ────────────────────
    def generate_with_logprobs(self, prompt: str, max_tokens: int) -> dict:
        """Ritorna:
            {
              "text": str,
              "token_logprobs": [float, ...],   # logprob del token scelto, per token
              "first_top": {token(str): logprob(float), ...},  # alternative del 1° token
            }
        `token_logprobs` → perplexity nativa = exp(-mean(token_logprobs)).
        `first_top` → RAG-MIA gray-box (confronto Sì vs No sul primo token).
        """
        d = self._post(self._body(prompt, max_tokens, logprobs=True))
        choice = d["choices"][0]
        text = choice["message"].get("content") or ""
        TRACKER.record(self.name, prompt, text)

        content_lp = (choice.get("logprobs") or {}).get("content") or []
        token_logprobs = [tok["logprob"] for tok in content_lp]
        first_top: dict[str, float] = {}
        if content_lp:
            for alt in content_lp[0].get("top_logprobs", []):
                first_top[alt["token"]] = alt["logprob"]
        return {"text": text, "token_logprobs": token_logprobs, "first_top": first_top}

    def native_perplexity(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        """Genera e restituisce (testo, perplexity nativa della generazione).
        PPL = exp(-mean logprob dei token generati). +inf se nessun token."""
        out = self.generate_with_logprobs(prompt, max_tokens)
        lps = out["token_logprobs"]
        if not lps:
            return out["text"], float("inf")
        return out["text"], math.exp(-sum(lps) / len(lps))

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
