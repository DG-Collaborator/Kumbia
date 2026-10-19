"""Dev-only: drives one persistent MCP session end to end — the thing
    dev_client.py can't do, since it spawns a fresh server per call and a
    background collector dies with that server. Within a single session it
    starts telemetry collection, watches telemetry://collection-status as
    records accumulate, asks the assistant's correlated_telemetry_report a
    question over what was collected, then stops collection. Not shipped in
    the installable package (lives outside src/), same as dev_client.py.

    Start the simulators first, each in its own terminal, so there is live
    data to collect:
        python tools/telemetry_sim/power_simulator.py
        .venv/bin/python tools/telemetry_sim/network_simulator.py

    Usage:
        python tools/mcp_client/session_client.py
        python tools/mcp_client/session_client.py --watch-minutes 3 --question "Is power rising?"
    """
from __future__ import annotations

import argparse
import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from dev_client import SERVER_PARAMS

REQUIRED_TOOLS = {"start_telemetry_collection", "stop_telemetry_collection", "correlated_telemetry_report"}

DEFAULT_QUESTION = (
    "Over the recorded minutes, how did FBA-1's power draw and network event count change? "
    "Do they tend to rise and fall together?"
)


async def run(watch_minutes: float, status_every_seconds: float, question: str, limit: int) -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = {t.name for t in (await session.list_tools()).tools}
            missing = REQUIRED_TOOLS - tools
            if missing:
                print(f"Server is missing {sorted(missing)} — does power_telemetry.sqlite exist?")
                return

            print(await _call(session, "start_telemetry_collection"))
            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + watch_minutes * 60
                while True:
                    print()
                    print(await _status(session))
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(status_every_seconds, remaining))

                print(f"\nAsking the assistant: {question}\n")
                print(await _call(session, "correlated_telemetry_report", {"question": question, "limit": limit}))
            finally:
                print()
                print(await _call(session, "stop_telemetry_collection"))


async def _call(session: ClientSession, name: str, arguments: dict | None = None) -> str:
    result = await session.call_tool(name, arguments=arguments or {})
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return f"[{name} failed] {text}" if result.is_error else text


async def _status(session: ClientSession) -> str:
    result = await session.read_resource("telemetry://collection-status")
    return "\n".join(content.text for content in result.contents if hasattr(content, "text"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start, watch, query, and stop telemetry collection in one MCP session.")
    parser.add_argument("--watch-minutes", type=float, default=4.0,
                        help="how long to let collection run before asking the question")
    parser.add_argument("--status-every-seconds", type=float, default=30.0)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--limit", type=int, default=20, help="newest correlated records to ground the answer in (max 60)")
    args = parser.parse_args()
    asyncio.run(run(args.watch_minutes, args.status_every_seconds, args.question, args.limit))


if __name__ == "__main__":
    main()