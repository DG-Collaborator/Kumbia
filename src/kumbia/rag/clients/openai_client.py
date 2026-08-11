"""
   LlmClient backed by the OpenAI API. Uses text-embedding-3-small for
   embeddings and gpt-4o-mini for generation by default, both overridable at
   construction to target any OpenAI-compatible model or endpoint. Reuses the
   shared core httpx.AsyncClient — the API key is added per-request so one
   client instance can serve every LlmClient implementation side by side.
"""
from __future__ import annotations

import httpx

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class OpenAiClient:
    def __init__(
            self,
            client: httpx.AsyncClient,
            api_key: str,
            embed_model: str = "text-embedding-3-small",
            generate_model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._embed_model = embed_model
        self._generate_model = generate_model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._embed_model, "input": text},
        )
        if response.is_error:
            raise RuntimeError(
                f"OpenAI embeddings request failed ({response.status_code}) — "
                "check that OPENAI_API_KEY is set and has active billing/quota."
            )
        data = response.json()["data"]
        return data[0]["embedding"] if data else []

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._generate_model, "messages": messages},
        )
        if response.is_error:
            raise RuntimeError(
                f"OpenAI request failed ({response.status_code}) — "
                "check that OPENAI_API_KEY is set correctly."
            )
        choices = response.json().get("choices", [])
        return choices[0]["message"]["content"] if choices else ""