"""
   Composition root for Kumbia's MCP server. Builds shared dependencies once
   and registers each persona's tools against a single MCPServer instance —
   personas never construct their own duplicate infrastructure.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from typing import Literal, cast
from mcp.server.mcpserver import MCPServer

from kumbia.core.net.http_client import create_http_client
from kumbia.core.source.elasticsearch_data_source import ElasticsearchDataSource
from kumbia.core.source.source_query import Credentials
from kumbia.mcp_server.personas.network_monitoring.tools import (
    register_network_activity_prompts,
    register_network_activity_resources,
    register_network_activity_tools,
)
from kumbia.rag.clients.openai_client import OpenAiClient
from kumbia.rag.personas.network_monitoring.pipeline import NetworkActivityPipeline

load_dotenv()

mcp = MCPServer("kumbia")

# one HTTP client shared by every HTTP-based source and the LLM client
http_client = create_http_client()

# optional: enables network_activity_report/drilldown; registers no-op unless
# both ES_URL and OPENAI_API_KEY are set — the persona is useless without an
# LLM, so there's no value in exposing tools that are guaranteed to fail.
elasticsearch_url = os.environ.get("ES_URL")
elasticsearch_username = os.environ.get("ES_USERNAME")
elasticsearch_password = os.environ.get("ES_PASSWORD")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if elasticsearch_url and openai_api_key:
    credentials = (
        Credentials(elasticsearch_username, elasticsearch_password)
        if elasticsearch_username and elasticsearch_password
        else None
    )
    pipeline = NetworkActivityPipeline(
        data_source=ElasticsearchDataSource(http_client),
        source_address=elasticsearch_url,
        credentials=credentials,
        llm=OpenAiClient(http_client, api_key=openai_api_key),
    )
    register_network_activity_tools(mcp, pipeline)
    register_network_activity_resources(mcp, pipeline)
    register_network_activity_prompts(mcp)


Transport = Literal["stdio", "sse", "streamable-http"]
_VALID_TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")


def _resolve_transport() -> Transport:
    raw = os.environ.get("MCP_TRANSPORT", "stdio")
    if raw not in _VALID_TRANSPORTS:
        raise ValueError(f"MCP_TRANSPORT must be one of {_VALID_TRANSPORTS}, got: {raw!r}")
    return cast(Transport, raw)


if __name__ == "__main__":
    # streamable-http has no auth configured here — defaults to 127.0.0.1,
    # safe for local/trusted-network use only. Production hardening
    # (auth_server_provider/token_verifier, CORS, a real host/port) is
    # deliberately deferred; this exists so the transport path itself is
    # proven and easy to build on later.
    mcp.run(transport=_resolve_transport())