"""
       LlmClient backed by a local (or LAN) Ollama instance. Uses nomic-embed-text
       for embeddings and llama3.2 for generation by default, both overridable at
       construction. Configurable base_url so the same class works against
       localhost:11434 now and a real LAN host:port once it reaches this
       machine over the network. Reuses the shared core httpx.AsyncClient.
"""
from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(
            self,
            client: httpx.AsyncClient,
            base_url: str = "http://localhost:11434",
            embed_model: str = "nomic-embed-text",
            generate_model: str = "llama3.2",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._embed_model = embed_model
        self._generate_model = generate_model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._embed_model, "prompt": text},
        )
        if response.is_error:
            raise RuntimeError(
                f"Ollama embeddings request failed ({response.status_code}) — "
                f"is Ollama running at {self._base_url} with {self._embed_model} pulled?"
            )
        return response.json().get("embedding", [])

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._generate_model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt is not None:
            payload["system"] = system_prompt

        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        if response.is_error:
            raise RuntimeError(
                f"Ollama generate request failed ({response.status_code}) — "
                f"is Ollama running at {self._base_url} with {self._generate_model} pulled?"
            )
        return response.json().get("response", "")