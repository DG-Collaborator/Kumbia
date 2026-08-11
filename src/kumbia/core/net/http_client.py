"""
   Builds the shared HTTP client used by every HTTP-based source. One client
   instance is meant to be shared across JsonTableSource, ElasticsearchDataSource,
   and (later) the RAG layer's LlmClient implementations.
"""
from __future__ import annotations

import httpx


def create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30.0)