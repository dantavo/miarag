# src/miarag/providers/bedrock.py
"""BedrockProvider: wrap langchain_aws.ChatBedrock per Claude/Titan/Llama su AWS.

Richiede credenziali AWS (env o profilo ~/.aws/credentials) + BEDROCK_CLAUDE_MODEL_ID.
"""
from __future__ import annotations
from dataclasses import dataclass


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

    def generate(self, prompt: str, max_tokens: int) -> str:
        # Bedrock/Claude: max_tokens via model_kwargs.
        resp = self._llm.invoke(prompt, model_kwargs={"max_tokens": max_tokens})
        return resp.content if hasattr(resp, "content") else str(resp)

    def supports_logprobs(self) -> bool:
        # Claude su Bedrock non espone logprobs per-token.
        return False
