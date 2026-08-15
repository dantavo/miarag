# src/miarag/providers/perplexity/gpt2.py
"""GPT-2 perplexity scorer (attaccante proxy LM).

NOTA: GPT-2 tokenizer è EN. Per corpus IT considerare LLaMA-IT o mistral-7b-italian
(vedi providers/perplexity/hf_causal.py per generalizzazione).
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class GPT2Perplexity:
    model_name: str = "gpt2"
    name: str = "gpt2"

    def __post_init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        if torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        self._torch = torch

    def perplexity(self, text: str) -> float:
        if not text or not text.strip():
            return float("inf")
        enc = self._tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(self._model.device)
        with self._torch.no_grad():
            out = self._model(input_ids, labels=input_ids)
            nll = out.loss.item()
        return math.exp(nll)
