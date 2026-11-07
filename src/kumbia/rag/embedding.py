"""
   Embeds many texts at once with bounded concurrency, through any LlmClient.
   Shared by everything that loads a batch into a vector index (RagPipeline's
   ingest, a persona's memory load), so a 50-row slice costs roughly the time
   of 50 / concurrency embedding round-trips instead of 50 sequential ones.
   Order is preserved: vectors[i] belongs to texts[i].
"""
from __future__ import annotations

import asyncio

from kumbia.rag.clients.llm_client import LlmClient

DEFAULT_CONCURRENCY = 8


async def embed_all(llm: LlmClient, texts: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[list[float]]:
    gate = asyncio.Semaphore(max(1, concurrency))

    async def embed_one(text: str) -> list[float]:
        async with gate:
            return await llm.embed(text)

    return list(await asyncio.gather(*(embed_one(text) for text in texts)))
