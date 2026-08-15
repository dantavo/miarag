# src/miarag/providers/perplexity/hf_causal.py
"""Generic HF causal-LM perplexity scorer. Utile per LM italiani (mistral-7b-italian,
gpt-neo-italian, minerva-3b) al posto di GPT-2 EN.

Uso:
    scorer = HFCausalPerplexity("sapienzanlp/Minerva-350M-base-v1.0")
    scorer.perplexity("il contratto prevede…")
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class HFCausalPerplexity:
    model_name: str
    name: str = "hf_causal"

    def __post_init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        if torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        elif torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._torch = torch

    def perplexity(self, text: str) -> float:
        if not text or not text.strip():
            return float("inf")
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc.input_ids.to(self._model.device)
        with self._torch.no_grad():
            out = self._model(input_ids, labels=input_ids)
            nll = out.loss.item()
        return math.exp(nll)
