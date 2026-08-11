"""Dev-only harness for exercising Kumbia's MCP server without a full host
  application. Not shipped in the installable package (lives outside src/).
  Complements the Inspector: this is for quick scriptable checks; the
  Inspector is for interactive browsing.

  Usage:
      python tools/mcp_client/dev_client.py list
      python tools/mcp_client/dev_client.py call network_activity_report '{"hours_back": 12}'
  """
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "kumbia" / "mcp_server" / "main.py"

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,  # the same interpreter running this script — guarantees the venv's kumbia/mcp installs are used
    args=[str(SERVER_SCRIPT)],
    env=None,
)


async def list_capabilities() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            prompts = await session.list_prompts()

            print("Tools:")
            for t in tools.tools:
                first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
                print(f"  {t.name} — {first_line}")

            print("Resources:")
            for r in resources.resources:
                print(f"  {r.uri}")

            print("Prompts:")
            for p in prompts.prompts:
                print(f"  {p.name}")


async def call_tool(name: str, arguments: dict) -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            for block in result.content:
                if hasattr(block, "text"):
                    print(block.text)
                else:
                    print(block)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    if command == "list":
        asyncio.run(list_capabilities())
    elif command == "call":
        if len(sys.argv) < 4:
            print("Usage: dev_client.py call <tool_name> '<json_arguments>'")
            return
        tool_name = sys.argv[2]
        arguments = json.loads(sys.argv[3])
        asyncio.run(call_tool(tool_name, arguments))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
