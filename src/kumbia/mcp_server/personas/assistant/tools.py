"""
       Conversational assistant persona: registers answer_question, remember,
       assistant://status, and — when their data sources are configured —
       web_search, power_telemetry_report, and correlated_telemetry_report.
       Tool bodies stay thin — all conversation/memory logic lives in the
       pipeline, and all telemetry figures are computed in telemetry.analysis;
       this file only adapts MCP's argument/return shape to those calls.
       Each data tool is also added to the pipeline as a Skill, so
       answer_question can route a chat message to the same fixed query.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from kumbia.core.source.source_query import SourceQuery
from kumbia.rag.conversation_pipeline import ConversationPipeline, Skill
from kumbia.telemetry.analysis import summary_text

_MAX_POWER_TELEMETRY_LIMIT = 500
_MAX_CORRELATED_TELEMETRY_LIMIT = 1440  # one day of 1-minute buckets

_started_at = time.monotonic()


def register_assistant_tools(mcp: MCPServer, pipeline: ConversationPipeline) -> None:

    @mcp.tool()
    async def answer_question(text: str) -> str:
        """
           Ask the assistant a question or continue the conversation.
           Maintains turn history across calls within this server process, so
           follow-up questions can refer back to earlier turns, and answers
           are grounded in the persona's remembered facts when relevant.
           Questions about data the persona can reach (telemetry, power,
           web search when asked) are routed to that data tool and answered
           from it; the reply names which one was used.
        """
        return await pipeline.ask(text)

    @mcp.tool()
    async def remember(fact: str) -> str:
        """
           Explicitly store a new fact in the persona's persistent memory.
           Only call this when the user directly asks to remember something
           (e.g. "remember that...") — never automatically, since an
           incorrectly stored fact would wrongly ground future answers.
        """
        await pipeline.remember(fact)
        return f"Got it, I'll remember: {fact}"


def register_assistant_resources(mcp: MCPServer, pipeline: ConversationPipeline, model_name: str) -> None:

    @mcp.resource("assistant://status")
    async def get_status() -> str:
        """
           The assistant's current status: which model is answering, how long
           this server has been running, how many turns the conversation has
           had, and how many facts are in memory (explicit + cached lookups
           combined). Read-only — no LLM call triggered by reading this.
        """
        await pipeline.ensure_memory_loaded()
        uptime_seconds = int(time.monotonic() - _started_at)
        lines = [
            "# Assistant Status",
            "",
            f"- Model: {model_name}",
            f"- Uptime: {uptime_seconds}s",
            f"- Turns so far: {pipeline.turn_count}",
            f"- Facts in memory: {pipeline.memory_count}",
        ]
        return "\n".join(lines)


def register_web_search_tool(mcp: MCPServer, pipeline: ConversationPipeline) -> None:

    async def search(query: str) -> str:
        return await pipeline.query_data_source("web_search", query)

    pipeline.add_skill(Skill(
        name="web_search",
        description="look something up on the internet. Only when the user explicitly asks to search or check online.",
        run=search,
    ))

    @mcp.tool()
    async def web_search(query: str) -> str:
        """
           Look up current information on the web when the assistant's local
           knowledge is missing or might be outdated. Only call this when
           explicitly asked to check/look something up online — not
           automatically, since this costs a real API call and requires
           internet. Successful lookups are cached, so the same question
           won't need a re-search next time.
        """
        return await search(query)


def register_power_telemetry_tool(mcp: MCPServer, pipeline: ConversationPipeline, db_path: Path) -> None:

    async def report(question: str, limit: int = 20) -> str:
        bounded_limit = max(1, min(int(limit), _MAX_POWER_TELEMETRY_LIMIT))
        query = SourceQuery(
            address=str(db_path),
            statement=f"SELECT * FROM power_readings ORDER BY ts DESC LIMIT {bounded_limit}",
        )
        return await pipeline.query_structured_source("power_telemetry", query, question)

    pipeline.add_skill(Skill(
        name="power_telemetry_report",
        description="the latest raw power readings (PDU outlet, timestamp, watts). "
                    "For the current or most recent power reading of an outlet.",
        run=report,
    ))

    @mcp.tool()
    async def power_telemetry_report(question: str, limit: int = 20) -> str:
        """
           Ask a free-text question grounded in recent power telemetry
           readings (outlet, timestamp, watts) from the persona's power
           telemetry data source. Fetches the most recent `limit` readings
           (newest first, capped at 500) and answers using only that data —
           the SQL itself is fixed here, never built from the question.
        """
        return await report(question, limit)


def register_correlated_telemetry_tool(mcp: MCPServer, pipeline: ConversationPipeline, store_path: Path) -> None:

    async def report(question: str, limit: int = 60) -> str:
        bounded_limit = max(1, min(int(limit), _MAX_CORRELATED_TELEMETRY_LIMIT))
        query = SourceQuery(address=str(store_path), statement=json.dumps({"last": bounded_limit}))
        return await pipeline.narrate_structured_source(
            "correlated_telemetry", query, question, summarize=summary_text,
        )

    pipeline.add_skill(Skill(
        name="correlated_telemetry_report",
        description="collected telemetry over time per asset: power draw (watts), CPU utilization, "
                    "and network activity (event counts per time bucket, incl. traffic/ingress/egress "
                    "questions). For trends, peaks, lows, changes, and how these metrics relate.",
        run=report,
    ))

    @mcp.tool()
    async def correlated_telemetry_report(question: str, limit: int = 60) -> str:
        """
           Ask about trends and relationships in the most recent correlated
           telemetry — power, compute, and network readings already joined
           per asset and per time bucket by the platform's background
           collector. Reads the newest `limit` records (capped at 1440) and
           answers from figures computed deterministically over all of them
           (start/end and change, peak and low, and how strongly each pair
           of metrics moves together); the model only narrates those figures,
           which are returned alongside its answer. Only has data once
           telemetry collection has been started; the query itself is fixed
           here, never built from the question.
        """
        return await report(question, limit)
