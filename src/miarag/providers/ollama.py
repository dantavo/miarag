# src/miarag/providers/ollama.py
"""OllamaProvider: wrap OllamaLLM (langchain_ollama). Mantiene contratto v0.1-thesis."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OllamaProvider:
    base_url: str
    model: str
    name: str = "ollama"

    def __post_init__(self):
        from langchain_ollama import OllamaLLM
        self._llm = OllamaLLM(base_url=self.base_url, model=self.model)

    def generate(self, prompt: str, max_tokens: int) -> str:
        # Ollama-native param: num_predict (in options dict).
        return self._llm.invoke(prompt, options={"num_predict": max_tokens})

    def supports_logprobs(self) -> bool:
        # Ollama /api/generate non espone per-token logprobs a v0.32.7.
        return False
