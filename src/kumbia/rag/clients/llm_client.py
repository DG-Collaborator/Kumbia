"""
   The LLM seam: embed text into a vector, and generate a completion from a
   prompt. Implement once per model provider; RagPipeline and persona pipelines
   depend on this protocol, not on any concrete client.

   temperature is optional: None keeps the provider's default; 0 asks for the
   most deterministic output, which routing and figure narration use so the
   same question gets the same answer.
"""
from __future__ import annotations

from typing import Protocol


class LlmClient(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def generate(
            self, prompt: str, system_prompt: str | None = None, temperature: float | None = None,
    ) -> str: ...