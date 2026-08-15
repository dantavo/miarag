# src/miarag/providers/azure_openai.py
"""AzureOpenAIProvider: wrap langchain_openai.AzureChatOpenAI.

Richiede env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.

Nota: Azure OpenAI espone logprobs → supports_logprobs=True. Un PerplexityScorer
API-based può usare direttamente questi logprobs invece del proxy GPT-2.
"""
from __future__ import annotations
from dataclasses import dataclass

from miarag.providers._retry import api_retry
from miarag.providers._cost import TRACKER


@dataclass
class AzureOpenAIProvider:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str
    name: str = "azure_openai"

    def __post_init__(self):
        from langchain_openai import AzureChatOpenAI
        self._llm = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            azure_deployment=self.deployment,
            max_tokens=None,  # settato per invoke
        )

    @api_retry
    def _invoke(self, prompt: str, max_tokens: int):
        return self._llm.invoke(prompt, max_tokens=max_tokens)

    def generate(self, prompt: str, max_tokens: int) -> str:
        resp = self._invoke(prompt, max_tokens)
        content = resp.content if hasattr(resp, "content") else str(resp)
        TRACKER.record(self.name, prompt, content)
        return content

    def supports_logprobs(self) -> bool:
        return True
