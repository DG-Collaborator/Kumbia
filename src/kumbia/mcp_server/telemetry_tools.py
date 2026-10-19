"""
     Platform-level telemetry collection controls: registers
     start_telemetry_collection, stop_telemetry_collection, and
     telemetry://collection-status. Lives directly under mcp_server/, beside
     personas/ rather than inside it, because continuous collection is shared
     infrastructure any persona's reasoning can draw on — not something one
     persona owns. Tool bodies stay thin: all collection logic lives in
     TelemetryCollector; this file only adapts it to MCP's call/return shape.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from kumbia.telemetry.canonical_record_store import load_records
from kumbia.telemetry.collector import TelemetryCollector


def register_telemetry_control_tools(mcp: MCPServer, collector: TelemetryCollector) -> None:

    @mcp.tool()
    async def start_telemetry_collection() -> str:
        """
           Start continuously collecting and correlating power, compute, and
           network telemetry in the background, on a fixed interval, until
           stopped. Only call this when the user explicitly asks to start
           collecting — it keeps running (and querying the telemetry
           sources) after this call returns.
        """
        if not collector.start():
            return "Telemetry collection is already running."
        return (
            f"Started telemetry collection: every {collector.poll_interval_seconds:g}s, "
            f"{collector.bucket_minutes}-minute buckets, collecting from {collector.collected_through}."
        )

    @mcp.tool()
    async def stop_telemetry_collection() -> str:
        """
           Stop background telemetry collection. Waits for any collection
           cycle already in progress to finish writing, so nothing is left
           half-stored. Only call this when the user explicitly asks to stop.
        """
        if not await collector.stop():
            return "Telemetry collection isn't running."
        return (
            f"Stopped telemetry collection after {collector.cycles_completed} cycle(s). "
            f"{collector.records_collected} record(s) stored this session; "
            f"everything before {collector.collected_through} has been collected."
        )


def register_telemetry_resources(mcp: MCPServer, collector: TelemetryCollector) -> None:

    @mcp.resource("telemetry://collection-status")
    async def get_collection_status() -> str:
        """
           Whether background telemetry collection is running, its interval
           and bucket size, how far it has collected, how many records it
           has stored (this session and in total), and the last error if the
           most recent cycle failed. Read-only — never starts, stops, or
           triggers a collection cycle.
        """
        lines = [
            "# Telemetry Collection Status",
            "",
            f"- Running: {'yes' if collector.is_running else 'no'}",
            f"- Poll interval: {collector.poll_interval_seconds:g}s",
            f"- Bucket size: {collector.bucket_minutes} min",
            f"- Collected through: {collector.collected_through or 'not started yet'}",
            f"- Cycles completed (this session): {collector.cycles_completed}",
            f"- Records stored (this session): {collector.records_collected}",
            f"- Records in store (total): {len(load_records(collector.store_path))}",
            f"- Last error: {collector.last_error or 'none'}",
        ]
        return "\n".join(lines)
