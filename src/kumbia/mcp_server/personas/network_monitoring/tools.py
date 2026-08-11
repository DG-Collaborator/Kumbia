"""
   Network Monitoring persona: registers network_activity_report and
   network_activity_drilldown, backed by NetworkActivityPipeline. Tool bodies
   stay thin — all business logic lives in the pipeline; this file only adapts
   MCP's arguments/return shape to the pipeline's own method signatures.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from kumbia.rag.personas.network_monitoring.pipeline import NetworkActivityPipeline, to_text


def register_network_activity_tools(mcp: MCPServer, pipeline: NetworkActivityPipeline) -> None:

    @mcp.tool()
    async def network_activity_report(
            hours_back: int = 24,
            top_n: int = 10,
            question: str | None = None,
            index_pattern: str = "*",
    ) -> str:
        """
           Report on network activity over a trailing time window.

           Runs aggregation queries against Elasticsearch (top source IPs by
           event volume with distinct DNS-name counts, hourly event volume) and
           returns a grounded narrative. index_pattern scopes which indices are
           searched — defaults to "*" (all network activity); narrow it to a
           specific log source (like Gigamon AMI's "logs-gigamon.ami*") only
           when asked to.
        """
        report = await pipeline.run(hours_back, top_n, question, index_pattern)
        return to_text(report)

    @mcp.tool()
    async def network_activity_drilldown(
            host_ip: str,
            question: str,
            hours_back: int = 24,
            max_events: int = 200,
            index_pattern: str = "*",
    ) -> str:
        """
           Ask a free-text question grounded in raw network events for one
           specific host (not aggregate counts).

           Fetches a bounded, capped slice of recent events for that host and
           answers from those real documents. index_pattern scopes which
           indices are searched — defaults to "*" (all network activity).
        """
        drilldown = await pipeline.drilldown(host_ip, question, hours_back, max_events, index_pattern)
        return drilldown.answer


def register_network_activity_resources(mcp: MCPServer, pipeline: NetworkActivityPipeline) -> None:

    @mcp.resource("network://recent-talkers")
    async def get_recent_talkers() -> str:
        """
           List the top source IPs by event volume from the most recent 24-hour
           network activity window, with distinct DNS-name counts. Read-only —
           deterministic math only, no LLM narration triggered by reading this.
        """
        summary = await pipeline.summary(hours_back=24, top_n=10)
        lines = ["# Recent Top Talkers (last 24h, all activity)", ""]
        if not summary.top_talkers:
            lines.append("No data available.")
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