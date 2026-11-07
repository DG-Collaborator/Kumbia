"""
   Network Monitoring persona: registers network_activity_report,
   network_activity_drilldown, network_field_breakdown, and network_metric, backed by
   NetworkActivityPipeline. Tool bodies stay thin — all business logic lives
   in the pipeline; this file only adapts MCP's arguments/return shape to the
   pipeline's own method signatures.

   network_activity_skills() offers the same abilities as Skills, so any
   conversational persona the composition root chooses can route chat to
   them. The window and host come from the question via deterministic regexes
   (rag.question_parsing), the index scope from configuration — never from
   the LLM.

   on_result, when given, is called with (source, answer) after every
   data-backed answer from a tool or skill. The composition root uses it to
   feed these answers into another persona's memory without this persona
   knowing who is listening.
"""
from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable

from mcp.server.mcpserver import MCPServer

from kumbia.rag.conversation_pipeline import Skill
from kumbia.rag.personas.network_monitoring.pipeline import (
    NetworkActivityPipeline,
    field_report_text,
    metric_report_text,
    to_text,
)
from kumbia.rag.question_parsing import hours_from_question, ipv4_from_question

ResultListener = Callable[[str, str], Awaitable[None]]

_MAX_HOURS_BACK = 720
_MAX_DRILLDOWN_EVENTS = 200


def register_network_activity_tools(
        mcp: MCPServer,
        pipeline: NetworkActivityPipeline,
        default_index_pattern: str = "*",
        on_result: ResultListener | None = None,
) -> None:

    _publish = _publisher(on_result)

    @mcp.tool()
    async def network_activity_report(
            hours_back: int = 24,
            top_n: int = 10,
            question: str | None = None,
            index_pattern: str | None = None,
    ) -> str:
        """
           Report on network activity over a trailing time window.

           Runs aggregation queries against Elasticsearch (top source IPs by
           event volume with distinct DNS-name counts, event volume over time
           with statistically flagged spikes) and returns the figures plus a
           narrative grounded in them. index_pattern scopes which indices are
           searched — leave it unset to use the server's configured network
           index; narrow it only when asked to.
        """
        report = await pipeline.run(
            _bounded_hours(hours_back), top_n, question, index_pattern or default_index_pattern,
        )
        return await _publish("network_activity_report", to_text(report, pipeline.host_labels))

    @mcp.tool()
    async def network_activity_drilldown(
            host_ip: str,
            question: str,
            hours_back: int = 24,
            max_events: int = 20,
            index_pattern: str | None = None,
    ) -> str:
        """
           Profile one host over a trailing window and answer a question
           about it: events it sent and received, its top destinations, the
           top sources that sent to it, and its top value for each configured
           breakdown field — all computed by aggregations — plus its
           max_events most recent events (capped at 200) as extra evidence.
           host_ip may also be a configured host name (e.g. a service name).
           index_pattern scopes which indices are searched — leave it unset to
           use the server's configured network index.
        """
        drilldown = await pipeline.drilldown(
            pipeline.host_aliases.get(host_ip, host_ip), question, _bounded_hours(hours_back),
            max(1, min(int(max_events), _MAX_DRILLDOWN_EVENTS)), index_pattern or default_index_pattern,
        )
        return await _publish("network_activity_drilldown", drilldown.answer)

    @mcp.tool()
    async def network_field_breakdown(
            field_alias: str,
            hours_back: int = 24,
            question: str | None = None,
            top_n: int = 10,
            index_pattern: str | None = None,
    ) -> str:
        """
           Break network events down by one configured field (for example an
           application, port, or country field from the enriched data) over a
           trailing window: event count per value and number of distinct
           values, plus a narrative grounded in those figures. field_alias
           must be one of the server's configured breakdown aliases; an
           unknown alias returns the list of valid ones.
        """
        try:
            report = await pipeline.breakdown(
                field_alias, _bounded_hours(hours_back), question, top_n, index_pattern or default_index_pattern,
            )
        except ValueError as error:
            return str(error)
        return await _publish(f"network_breakdown_{field_alias}", field_report_text(report))

    @mcp.tool()
    async def network_metric(
            metric_alias: str,
            hours_back: int = 24,
            question: str | None = None,
            top_n: int = 5,
            index_pattern: str | None = None,
    ) -> str:
        """
           Report on one configured numeric network metric (for example bytes,
           packets, or round-trip time) over a trailing window: count, total,
           average, minimum, maximum, and 95th percentile; the hosts with the
           highest total and highest average; and the busiest and
           highest/lowest-average time buckets — plus a narrative grounded in
           those figures. metric_alias must be one of the server's configured
           metric aliases; an unknown alias returns the list of valid ones.
        """
        try:
            report = await pipeline.metric(
                metric_alias, _bounded_hours(hours_back), question, top_n, index_pattern or default_index_pattern,
            )
        except ValueError as error:
            return str(error)
        return await _publish(f"network_metric_{metric_alias}", metric_report_text(report))


def network_activity_skills(
        pipeline: NetworkActivityPipeline,
        index_pattern: str = "*",
        default_hours_back: int = 24,
        on_result: ResultListener | None = None,
        drilldown_max_events: int = 20,
) -> list[Skill]:
    """
       The pipeline's report, drilldown, and one breakdown per configured
       field, as Skills a conversational persona can route chat to.
    """

    _publish = _publisher(on_result)

    async def report(question: str) -> str:
        hours = hours_from_question(question, default_hours_back)
        result = await pipeline.run(hours_back=hours, question=question, index_pattern=index_pattern)
        return await _publish("network_activity_report", to_text(result, pipeline.host_labels))

    async def drilldown(question: str) -> str:
        host_ip = ipv4_from_question(question, pipeline.host_aliases)
        if host_ip is None:
            named = f" or one of its names ({', '.join(sorted(pipeline.host_aliases))})" if pipeline.host_aliases else ""
            return f"Which host? Include its IP address{named}."
        hours = hours_from_question(question, default_hours_back)
        result = await pipeline.drilldown(
            host_ip, question, hours_back=hours, max_events=drilldown_max_events, index_pattern=index_pattern,
        )
        return await _publish("network_activity_drilldown", result.answer)

    def breakdown_runner(alias: str) -> Callable[[str], Awaitable[str]]:
        async def run(question: str) -> str:
            hours = hours_from_question(question, default_hours_back)
            result = await pipeline.breakdown(alias, hours_back=hours, question=question, index_pattern=index_pattern)
            return await _publish(f"network_breakdown_{alias}", field_report_text(result))
        return run

    def metric_runner(alias: str) -> Callable[[str], Awaitable[str]]:
        async def run(question: str) -> str:
            hours = hours_from_question(question, default_hours_back)
            result = await pipeline.metric(alias, hours_back=hours, question=question, index_pattern=index_pattern)
            return await _publish(f"network_metric_{alias}", metric_report_text(result))
        return run

    skills = [
        Skill(
            name="network_activity_report",
            description="network traffic across all hosts over a time window: number of connection "
                        "records, top talkers (busiest source IPs), distinct destinations per host, "
                        "volume over time, spikes"
                        + (" and traffic totals with top hosts by volume" if pipeline.report_metrics else "")
                        + ".",
            run=report,
        ),
        Skill(
            name="network_activity_drilldown",
            description="questions about one specific host named by its IP address"
                        + (f" or by name ({', '.join(sorted(pipeline.host_aliases))})" if pipeline.host_aliases else "")
                        + ": what it is, who it talked to, who talked to it, its apps and ports, anything unusual.",
            run=drilldown,
        ),
    ]
    for alias, field_name in pipeline.breakdown_fields.items():
        skills.append(Skill(
            name=f"network_breakdown_{alias}",
            description=f"network events broken down by {alias} ({field_name}): which {alias} values are "
                        f"most common, how many distinct ones, what stands out.",
            run=breakdown_runner(alias),
        ))
    for alias, field_name in pipeline.metric_fields.items():
        skills.append(Skill(
            name=f"network_metric_{alias}",
            description=f"the {alias} metric ({field_name}): total, average, min, max, 95th percentile, "
                        f"which hosts have the most or highest {alias}, and when it peaked.",
            run=metric_runner(alias),
        ))
    return skills


def register_network_activity_resources(mcp: MCPServer, pipeline: NetworkActivityPipeline,
                                        default_index_pattern: str = "*") -> None:

    @mcp.resource("network://recent-talkers")
    async def get_recent_talkers() -> str:
        """
           List the top source IPs by event volume from the most recent 24-hour
           network activity window, with distinct DNS-name counts. Read-only —
           deterministic math only, no LLM narration triggered by reading this.
        """
        summary = await pipeline.summary(hours_back=24, top_n=10, index_pattern=default_index_pattern)
        lines = [f"# Recent Top Talkers (last 24h, {default_index_pattern})", ""]
        if not summary.top_talkers:
            lines.append("No data available.")
            if summary.newest_event:
                lines.append(f"Newest event in scope: {summary.newest_event}")
        else:
            for t in summary.top_talkers:
                lines.append(f"- {t.host_ip} — {t.doc_count} events, {t.unique_dns_names} distinct DNS names")
        return "\n".join(lines)


def register_network_activity_prompts(mcp: MCPServer) -> None:

    @mcp.prompt()
    def investigate_host(host_ip: str, hours_back: int = 24) -> str:
        """Generate a guided investigation prompt for a specific host's network activity."""
        return (
            f"Investigate host {host_ip}'s network activity over the last {hours_back} hour(s). "
            f"Follow these steps:\n"
            f"1. Call network_activity_drilldown(host_ip=\"{host_ip}\", hours_back={hours_back}, "
            f"question=\"Summarize this host's activity and flag anything unusual.\")\n"
            f"2. From the answer, identify:\n"
            f"   - Overall volume and whether it looks typical or anomalous\n"
            f"   - Any unusual DNS names or destinations contacted\n"
            f"   - Any patterns worth flagging to a network operator\n"
            f"3. Present findings as a short, structured summary with clear headings."
        )


def _publisher(on_result: ResultListener | None) -> Callable[[str, str], Awaitable[str]]:
    """
       Wraps on_result so every answer is handed to the listener (if any) and
       then returned unchanged. A failing listener is reported on stderr (the
       stdio transport's log channel) but never costs the caller its answer.
    """
    async def publish(source: str, answer: str) -> str:
        if on_result is not None:
            try:
                await on_result(source, answer)
            except Exception as error:  # noqa: BLE001 — a listener must not break the tool
                print(f"network_monitoring: result listener failed for {source}: {error}", file=sys.stderr)
        return answer
    return publish


def _bounded_hours(hours_back: int) -> int:
    return max(1, min(int(hours_back), _MAX_HOURS_BACK))
