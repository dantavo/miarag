# src/miarag/providers/bedrock.py
"""BedrockProvider: wrap langchain_aws.ChatBedrock per Claude/Titan/Llama su AWS.

Richiede credenziali AWS (env o profilo ~/.aws/credentials) + BEDROCK_CLAUDE_MODEL_ID.
"""
from __future__ import annotations
from dataclasses import dataclass

from miarag.providers._retry import api_retry
from miarag.providers._cost import TRACKER


@dataclass
class BedrockProvider:
    model_id: str
    region: str
    name: str = "bedrock"

    def __post_init__(self):
        from langchain_aws import ChatBedrock
        self._llm = ChatBedrock(
            model_id=self.model_id,
            region_name=self.region,
        )

    @api_retry
    def _invoke(self, prompt: str, max_tokens: int):
        return self._llm.invoke(prompt, model_kwargs={"max_tokens": max_tokens})

    def generate(self, prompt: str, max_tokens: int) -> str:
        resp = self._invoke(prompt, max_tokens)
        content = resp.content if hasattr(resp, "content") else str(resp)
        TRACKER.record(self.name, prompt, content)
        return content

    def supports_logprobs(self) -> bool:
        return False
