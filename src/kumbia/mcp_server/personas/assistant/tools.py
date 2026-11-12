"""
       Conversational assistant persona: registers answer_question, remember,
       assistant://status, and — when their data sources are configured —
       web_search, power_telemetry_report, power_consumption_report, and
       correlated_telemetry_report.
       Tool bodies stay thin — all conversation/memory logic lives in the
       pipeline, and all telemetry figures are computed in telemetry.analysis;
       this file only adapts MCP's argument/return shape to those calls.
       Each data tool is also added to the pipeline as a Skill, so
       answer_question can route a chat message to the same fixed query.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from kumbia.core.source.source_query import SourceQuery
from kumbia.rag.conversation_pipeline import ConversationPipeline, Skill
from kumbia.rag.question_parsing import MAX_HOURS_BACK, hours_from_question
from kumbia.telemetry import power_report
from kumbia.telemetry.analysis import summary_text
from kumbia.telemetry.asset_registry import AssetRegistry
from kumbia.telemetry.stream_queries import plan_window, sql_power_buckets
from kumbia.telemetry.window_source import CorrelatedWindowSource

_MAX_POWER_TELEMETRY_LIMIT = 500

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
        description="the few most recent raw power readings (PDU outlet, timestamp, watts). Only for the "
                    "current or latest reading right now, not for any period of time.",
        run=report,
    ))

    @mcp.tool()
    async def power_telemetry_report(question: str, limit: int = 20) -> str:
        """
           Ask a free-text question grounded in the most recent raw power
           telemetry readings (outlet, timestamp, watts). Fetches the newest
           `limit` readings (capped at 500) and answers using only that data
           — the SQL itself is fixed here, never built from the question. For
           consumption over a period, use power_consumption_report.
        """
        return await report(question, limit)


def register_power_consumption_tool(
        mcp: MCPServer,
        pipeline: ConversationPipeline,
        db_path: Path,
        registry: AssetRegistry,
        source_system: str,
        default_hours_back: int = 24,
) -> None:
    """
       Power consumption over a trailing window for every registered outlet:
       figures from telemetry.power_report (computed by code), narrated. The
       window comes from the question by regex (or the tool's hours_back),
       the outlets from the registry — never from the model.
    """

    def asset_for_outlet(outlet: str) -> str | None:
        return registry.resolve("pdu_outlet", outlet)

    async def report(question: str, hours_back: int | None = None) -> str:
        hours = _bounded_hours(hours_back) if hours_back else hours_from_question(question, default_hours_back)
        outlets = registry.identities("pdu_outlet")
        if not outlets:
            return "No PDU outlets are registered, so there is no power data to report."
        plan = plan_window(
            hours, _utc_now(), power_report.MIN_BUCKET_MINUTES, len(outlets), power_report.MAX_BUCKET_ROWS,
        )
        query = sql_power_buckets(str(db_path), plan.window, plan.bucket_minutes, outlets)
        return await pipeline.narrate_structured_source(
            "power_telemetry", query, question,
            summarize=lambda table: power_report.power_figures_text(table, plan, asset_for_outlet, source_system),
            on_empty=f"{plan.describe()}.\nNo power readings from the registered outlets "
                     f"({', '.join(outlets)}) in this window.",
        )

    pipeline.add_skill(Skill(
        name="power_consumption_report",
        description="power consumption of the monitored system over a time window (default the last "
                    f"{default_hours_back} hours): average, highest and lowest power draw in watts, when it "
                    "peaked, how it changed, and energy used in kWh, per asset and in total. For any overview, "
                    "summary, total, trend, or peak of power, electricity, or energy use.",
        run=report,
    ))

    @mcp.tool()
    async def power_consumption_report(question: str, hours_back: int | None = None) -> str:
        """
           Power consumption over a trailing window for every registered PDU
           outlet: average, highest, and lowest draw (watts) with when they
           happened, first-to-last change, coverage of the window, and energy
           used (kWh), per asset and as a system total. Figures are computed
           deterministically and returned alongside a narration of them.
           hours_back sets the window; when unset, it's read from the question
           ("past 3 days"), defaulting to the server's chat window.
        """
        return await report(question, hours_back)


def register_correlated_telemetry_tool(
        mcp: MCPServer,
        pipeline: ConversationPipeline,
        window_source: CorrelatedWindowSource,
        default_hours_back: int = 24,
) -> None:
    """
       window_source must also be the persona's "correlated_telemetry" data
       source; it's passed here as well so the tool can plan the window it
       reads (and say which window that was) before fetching.
    """

    async def report(question: str, hours_back: int | None = None) -> str:
        hours = _bounded_hours(hours_back) if hours_back else hours_from_question(question, default_hours_back)
        plan = window_source.plan(hours)
        return await pipeline.narrate_structured_source(
            "correlated_telemetry", window_source.query_for(plan), question,
            summarize=lambda table: summary_text(table, header=plan.describe()),
            on_empty=f"{plan.describe()}.\nNo power, compute, or network telemetry for any registered asset "
                     "in this window.",
        )

    pipeline.add_skill(Skill(
        name="correlated_telemetry_report",
        description="how the monitored assets' power draw, CPU utilization, and their own network traffic "
                    "(events, bytes, and other configured network metrics) relate over a time window: whether "
                    "power rises and falls with traffic or CPU, how strongly they move together, and each "
                    "one's peak, low, and change. For any question that combines power with network or "
                    "compute data. Not for network-wide traffic, or which hosts talked to which.",
        run=report,
    ))

    @mcp.tool()
    async def correlated_telemetry_report(question: str, hours_back: int | None = None) -> str:
        """
           Ask about trends and relationships across power, compute, and
           network telemetry, joined per registered asset and per time bucket
           straight from the live sources for the requested window — no
           background collection needed. Answers from figures computed
           deterministically over every bucket (start/end and change, peak
           and low, and how strongly each pair of metrics moves together);
           the model only narrates those figures, which are returned
           alongside its answer. hours_back sets the window; when unset, it's
           read from the question, defaulting to the server's chat window.
        """
        return await report(question, hours_back)


def _bounded_hours(hours_back: int) -> int:
    return max(1, min(int(hours_back), MAX_HOURS_BACK))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
