"""
        Composition root for Kumbia's MCP server. Builds shared dependencies once
        and registers each persona's tools against a single MCPServer instance —
        personas never construct their own duplicate infrastructure.
     """
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Literal, cast
from mcp.server.mcpserver import MCPServer

from kumbia.core.net.http_client import create_http_client
from kumbia.core.source.elasticsearch_data_source import ElasticsearchDataSource
from kumbia.core.source.empty_data_source import EmptyDataSource
from kumbia.core.source.jsonl_table_source import JsonlTableSource
from kumbia.core.source.prometheus_data_source import PrometheusDataSource
from kumbia.core.source.source_query import Credentials, SourceQuery
from kumbia.core.source.sql_data_source import SqlDataSource
from kumbia.core.source.tavily_search_source import TavilySearchSource
from kumbia.mcp_server.personas.assistant.tools import (
    register_assistant_tools,
    register_assistant_resources,
    register_correlated_telemetry_tool,
    register_power_telemetry_tool,
    register_web_search_tool,
)
from kumbia.mcp_server.personas.network_monitoring.tools import (
    register_network_activity_prompts,
    register_network_activity_resources,
    register_network_activity_tools,
)
from kumbia.mcp_server.telemetry_tools import register_telemetry_control_tools, register_telemetry_resources
from kumbia.rag.clients.ollama_client import OllamaClient
from kumbia.rag.clients.openai_client import OpenAiClient
from kumbia.rag.conversation_pipeline import ConversationPipeline
from kumbia.rag.persona_profile import PersonaProfile
from kumbia.rag.personas.network_monitoring.pipeline import NetworkActivityPipeline
from kumbia.telemetry.asset_registry import AssetRegistry
from kumbia.telemetry.collector import TelemetryCollector
from kumbia.telemetry.correlation_engine import SourceMapping
from kumbia.telemetry.pipeline import TelemetryCorrelationPipeline, TelemetryStream
from kumbia.telemetry.stream_queries import es_network_window, prometheus_compute_window, sql_power_window

load_dotenv()

mcp = MCPServer("kumbia")

# one HTTP client shared by every HTTP-based source and every LLM client
http_client = create_http_client()

# Elasticsearch connection, shared by network_monitoring and the telemetry
# network stream — built once here rather than inside either consumer.
elasticsearch_url = os.environ.get("ES_URL")
elasticsearch_username = os.environ.get("ES_USERNAME")
elasticsearch_password = os.environ.get("ES_PASSWORD")
elasticsearch_credentials = (
    Credentials(elasticsearch_username, elasticsearch_password)
    if elasticsearch_username and elasticsearch_password
    else None
)

# Prometheus connection, used by the telemetry compute stream — built once
# here, same as Elasticsearch above. Credentials are optional
prometheus_url = os.environ.get("PROMETHEUS_URL")
prometheus_username = os.environ.get("PROMETHEUS_USERNAME")
prometheus_password = os.environ.get("PROMETHEUS_PASSWORD")
prometheus_credentials = (
    Credentials(prometheus_username, prometheus_password)
    if prometheus_username and prometheus_password
    else None
)

openai_api_key = os.environ.get("OPENAI_API_KEY")

# optional: enables network_activity_report/drilldown; registers no-op unless
# both ES_URL and OPENAI_API_KEY are set — the persona is useless without an
# LLM, so there's no value in exposing tools that are guaranteed to fail.
if elasticsearch_url and openai_api_key:
    network_monitoring_pipeline = NetworkActivityPipeline(
        data_source=ElasticsearchDataSource(http_client),
        source_address=elasticsearch_url,
        credentials=elasticsearch_credentials,
        llm=OpenAiClient(http_client, api_key=openai_api_key),
    )
    register_network_activity_tools(mcp, network_monitoring_pipeline)
    register_network_activity_resources(mcp, network_monitoring_pipeline)
    register_network_activity_prompts(mcp)


# --- assistant persona -------------------------------------------------
# ASSISTANT_LLM_PROVIDER picks which LlmClient backs the assistant persona —
# "ollama" (default, local SLM, matches DAFNI's original local-first intent)
# or "openai" (reuses the same account already configured above for
# network_monitoring). Swapping providers never touches ConversationPipeline
# itself — both clients are interchangeable behind the LlmClient Protocol.
ASSISTANT_SYSTEM_PROMPT = (
    "You are Kumbia's assistant persona, a helpful conversational assistant. "
    "Keep answers short and conversational, since they may be spoken aloud. "
    "The remembered facts below are current and correct, even if they mention "
    "dates or events after your training data ends — trust them completely "
    "and state the answer directly. Never say you don't have up-to-date "
    "information if the answer is present in the remembered facts."
)

assistant_llm_provider = os.environ.get("ASSISTANT_LLM_PROVIDER", "ollama").strip().lower()

if assistant_llm_provider == "ollama":
    assistant_model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
    assistant_llm = OllamaClient(
        http_client,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        generate_model=assistant_model_name,
    )
elif assistant_llm_provider == "openai":
    if not openai_api_key:
        raise ValueError("ASSISTANT_LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
    assistant_model_name = "gpt-4o-mini"
    assistant_llm = OpenAiClient(http_client, api_key=openai_api_key, generate_model=assistant_model_name)
else:
    raise ValueError(f"ASSISTANT_LLM_PROVIDER must be 'ollama' or 'openai', got: {assistant_llm_provider!r}")

_assistant_data_dir = Path(__file__).resolve().parents[3] / "data" / "assistant"
assistant_memory_path = Path(os.environ.get("ASSISTANT_MEMORY_PATH", str(_assistant_data_dir / "memory.jsonl")))
assistant_cache_path = Path(os.environ.get("ASSISTANT_CACHE_PATH", str(_assistant_data_dir / "cache.jsonl")))

tavily_api_key = os.environ.get("TAVILY_API_KEY")
assistant_data_sources = (
    {"web_search": TavilySearchSource(http_client, api_key=tavily_api_key)} if tavily_api_key else {}
)

# optional: gives the assistant read access to the power telemetry sqlite
# seeded by scratch_power_seed.py — registers no-op if the file doesn't
# exist, same graceful-degrade shape as the rest of this composition root.
_repo_root = Path(__file__).resolve().parents[3]
power_telemetry_db_path = Path(
    os.environ.get("POWER_TELEMETRY_DB_PATH", str(_repo_root / "power_telemetry.sqlite"))
)
if power_telemetry_db_path.exists():
    assistant_data_sources["power_telemetry"] = SqlDataSource()

# optional: gives the assistant read access to the platform collector's
# correlated store (written by the telemetry section below) — gated the same
# way as the collector itself, so the tool exists exactly when the store can.
telemetry_store_path = Path(os.environ.get(
    "TELEMETRY_STORE_PATH", str(_repo_root / "data" / "telemetry" / "canonical_records.jsonl")
))
if power_telemetry_db_path.exists():
    assistant_data_sources["correlated_telemetry"] = JsonlTableSource()

assistant_persona = PersonaProfile(
    name="assistant",
    system_prompt=ASSISTANT_SYSTEM_PROMPT,
    data_sources=assistant_data_sources,
    rag_namespace="assistant",
    memory_path=assistant_memory_path,
    cache_path=assistant_cache_path,
)

assistant_pipeline = ConversationPipeline(assistant_llm, assistant_persona)

register_assistant_tools(mcp, assistant_pipeline)
register_assistant_resources(mcp, assistant_pipeline, model_name=assistant_model_name)
if tavily_api_key:
    register_web_search_tool(mcp, assistant_pipeline)
if power_telemetry_db_path.exists():
    register_power_telemetry_tool(mcp, assistant_pipeline, power_telemetry_db_path)
    register_correlated_telemetry_tool(mcp, assistant_pipeline, telemetry_store_path)


# --- platform: continuous telemetry collection ---------------------------
# Shared infrastructure, not owned by any persona: one collector feeding one
# canonical store (telemetry_store_path, defined above so the assistant can
# be given read access to it) that any persona's reasoning can draw on.
# Every stream always exists — backed by its real system when configured,
# otherwise by EmptyDataSource — so the pipeline's shape never depends on
# which env vars happen to be set. Registers no-op unless the power
# telemetry db exists. Code defaults are production cadence (5-min buckets,
# 300s poll); demo cadence is set in .env.
TELEMETRY_ASSET_ID = "FBA-1"           # demo registry: one synthetic asset
TELEMETRY_POWER_OUTLET = "R42-U18"     # matches scratch_power_seed.py / the power simulator

telemetry_bucket_minutes = int(os.environ.get("TELEMETRY_BUCKET_MINUTES", "5"))
telemetry_poll_interval_seconds = float(os.environ.get("TELEMETRY_POLL_INTERVAL_SECONDS", "300"))
telemetry_network_host_ip = os.environ.get("TELEMETRY_NETWORK_HOST_IP")
telemetry_es_index_pattern = os.environ.get("TELEMETRY_ES_INDEX_PATTERN", "network-activity-*")
telemetry_network_enabled = bool(elasticsearch_url and telemetry_network_host_ip)
telemetry_compute_instance = os.environ.get("TELEMETRY_COMPUTE_INSTANCE")
telemetry_compute_enabled = bool(prometheus_url and telemetry_compute_instance)


if power_telemetry_db_path.exists():
    asset_identities = {"pdu_outlet": TELEMETRY_POWER_OUTLET}
    if telemetry_network_enabled:
        asset_identities["ip_address"] = telemetry_network_host_ip
    if telemetry_compute_enabled:
        asset_identities["instance"] = telemetry_compute_instance
    telemetry_registry = AssetRegistry({TELEMETRY_ASSET_ID: asset_identities})

    telemetry_pipeline = TelemetryCorrelationPipeline(
        streams=[
            TelemetryStream(
                "power",
                SqlDataSource(),
                SourceMapping(
                    identity_type="pdu_outlet", identity_column="outlet", timestamp_column="ts",
                    metric_columns={"watts": "power_watts"}, source_system="synthetic_dcim",
                ),
            ),
            TelemetryStream(
                "compute",
                PrometheusDataSource(http_client) if telemetry_compute_enabled else EmptyDataSource(),
                SourceMapping(
                    identity_type="instance", identity_column="instance", timestamp_column="timestamp",
                    metric_columns={"value": "cpu_utilization"}, source_system="prometheus",
                ),
            ),
            TelemetryStream(
                "network",
                ElasticsearchDataSource(http_client) if telemetry_network_enabled else EmptyDataSource(),
                SourceMapping(
                    identity_type="ip_address", identity_column="key.host", timestamp_column="key.ts",
                    metric_columns={"doc_count": "network_event_count"}, source_system="elasticsearch",
                ),
            ),
        ],
        registry=telemetry_registry,
        bucket_minutes=telemetry_bucket_minutes,
    )

    telemetry_query_builders = {
        "power": lambda window: sql_power_window(str(power_telemetry_db_path), window),
        "compute": (
            (lambda window: prometheus_compute_window(
                prometheus_url, window,
                [telemetry_compute_instance], telemetry_bucket_minutes, prometheus_credentials,
            ))
            if telemetry_compute_enabled
            else (lambda window: SourceQuery(address="unconfigured"))
        ),
        "network": (
            (lambda window: es_network_window(
                elasticsearch_url, telemetry_es_index_pattern, window,
                [telemetry_network_host_ip], telemetry_bucket_minutes, elasticsearch_credentials,
            ))
            if telemetry_network_enabled
            else (lambda window: SourceQuery(address="unconfigured"))
        ),
    }

    telemetry_collector = TelemetryCollector(
        telemetry_pipeline,
        telemetry_query_builders,
        telemetry_store_path,
        poll_interval_seconds=telemetry_poll_interval_seconds,
    )
    register_telemetry_control_tools(mcp, telemetry_collector)
    register_telemetry_resources(mcp, telemetry_collector)


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
