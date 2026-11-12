"""Dev-only: a small local web UI for demoing Kumbia. It is just another MCP
   host: at startup it opens ONE persistent MCP session to the Kumbia server
   (launched exactly like dev_client.py launches it) and keeps it for the life
   of the app, so chat keeps its turn history and background telemetry
   collection keeps running between clicks. Each endpoint below makes one tool
   call or resource read; nothing here imports Kumbia's own code. Also starts
   and stops the telemetry simulators, so the whole demo runs from one window.

   Every tool call and simulator action, from any visitor, is recorded in the
   history store (history.py) with the visitor's IP; /api/history lists it
   for the page's History tab, which everyone can see.

   Usage:
       .venv/bin/python tools/demo_ui/server.py            # then open http://localhost:8000
       .venv/bin/python tools/demo_ui/server.py --host 19.51.0.74 --port 8080

   Listens on 127.0.0.1 unless --host names an interface address (or 0.0.0.0
   for all). There is no auth: anyone who can reach that address can chat,
   spend API credits, start/stop the simulators, and read every visitor's
   history. Stopping it (Ctrl+C)
   stops collection first, so an in-flight cycle finishes writing, then stops
   the simulators.
   """
from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcp_client"))

from dev_client import SERVER_PARAMS  # noqa: E402  (one definition of how to launch the server)
from history import HistoryStore  # noqa: E402
from simulators import SimulatorManager  # noqa: E402

HISTORY_PATH = HERE.parents[1] / "data" / "history" / "history.jsonl"

STATUS_URI = "telemetry://collection-status"


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with contextlib.AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(SERVER_PARAMS))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        app.state.session = session
        app.state.tools = {tool.name for tool in (await session.list_tools()).tools}
        app.state.simulators = SimulatorManager()
        app.state.history = HistoryStore(HISTORY_PATH)
        try:
            yield
        finally:
            if "stop_telemetry_collection" in app.state.tools:
                with contextlib.suppress(Exception):
                    await session.call_tool("stop_telemetry_collection", arguments={})
            await run_in_threadpool(app.state.simulators.stop_all)


# ---------- helpers ----------

def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _visitor(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _record(request: Request, kind: str, tool: str, prompt: str, answer: str, ok: bool) -> None:
    request.app.state.history.record(
        who=_visitor(request), kind=kind, tool=tool, prompt=prompt, answer=answer, ok=ok,
    )


async def _call_tool(request: Request, name: str, arguments: dict, kind: str, prompt: str) -> JSONResponse:
    """One MCP tool call, recorded in history (success or failure) as kind/prompt."""
    if name not in request.app.state.tools:
        message = f"The Kumbia server didn't register {name}; check its data source is configured."
        _record(request, kind, name, prompt, message, ok=False)
        return _error(404, message)
    try:
        result = await request.app.state.session.call_tool(name, arguments=arguments)
    except Exception as error:  # transport-level failure: report it instead of a blank 500
        message = f"{name} failed: {error}"
        _record(request, kind, name, prompt, message, ok=False)
        return _error(502, message)
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    if result.is_error:
        message = text or f"{name} failed."
        _record(request, kind, name, prompt, message, ok=False)
        return _error(502, message)
    _record(request, kind, name, prompt, text, ok=True)
    return JSONResponse({"text": text})


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _required_text(body: dict, field: str) -> str | None:
    value = body.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int(body: dict, field: str, default: int) -> int:
    try:
        return int(body.get(field, default))
    except (TypeError, ValueError):
        return default


# ---------- page ----------
async def index(request: Request) -> FileResponse | PlainTextResponse:
    page = HERE / "index.html"
    if not page.exists():
        return PlainTextResponse(f"The demo page is missing: expected {page}", status_code=404)
    return FileResponse(page)


# ---------- assistant ----------

async def tools(request: Request) -> JSONResponse:
    return JSONResponse({"tools": sorted(request.app.state.tools)})


async def chat(request: Request) -> JSONResponse:
    text = _required_text(await _json_body(request), "text")
    if text is None:
        return _error(400, "Type a message first.")
    return await _call_tool(request, "answer_question", {"text": text}, "chat", text)


async def remember(request: Request) -> JSONResponse:
    fact = _required_text(await _json_body(request), "fact")
    if fact is None:
        return _error(400, "Type the fact to remember first.")
    return await _call_tool(request, "remember", {"fact": fact}, "remember", f"Remember: {fact}")


async def web_search(request: Request) -> JSONResponse:
    query = _required_text(await _json_body(request), "query")
    if query is None:
        return _error(400, "Type what to search for first.")
    return await _call_tool(request, "web_search", {"query": query}, "web_search", f"Search the web: {query}")


async def network_report(request: Request) -> JSONResponse:
    body = await _json_body(request)
    # index scope comes from the server's NETWORK_ES_INDEX_PATTERN, not from here
    hours = max(1, _int(body, "hours_back", 24))
    return await _call_tool(request, "network_activity_report", {"hours_back": hours},
                            "report", f"Network activity report, last {hours} hour(s)")


async def power(request: Request) -> JSONResponse:
    body = await _json_body(request)
    question = _required_text(body, "question") or "What was the most recent power reading for R42-U18?"
    return await _call_tool(request, "power_telemetry_report", {"question": question, "limit": _int(body, "limit", 5)},
                            "power", question)


# ---------- telemetry ----------

async def telemetry_start(request: Request) -> JSONResponse:
    return await _call_tool(request, "start_telemetry_collection", {}, "telemetry", "Start telemetry collection")


async def telemetry_stop(request: Request) -> JSONResponse:
    return await _call_tool(request, "stop_telemetry_collection", {}, "telemetry", "Stop telemetry collection")


async def telemetry_status(request: Request) -> JSONResponse:
    """The status resource's text, plus its "- Label: value" lines as fields for the page to lay out."""
    if "start_telemetry_collection" not in request.app.state.tools:
        return _error(404, "Telemetry collection isn't registered; does power_telemetry.sqlite exist?")
    try:
        result = await request.app.state.session.read_resource(STATUS_URI)
    except Exception as error:
        return _error(502, f"Reading {STATUS_URI} failed: {error}")
    text = "\n".join(content.text for content in result.contents if hasattr(content, "text"))
    fields = {}
    for line in text.splitlines():
        if line.startswith("- ") and ": " in line:
            label, value = line[2:].split(": ", 1)
            fields[label] = value
    return JSONResponse({"text": text, "fields": fields})


async def telemetry_ask(request: Request) -> JSONResponse:
    body = await _json_body(request)
    question = _required_text(body, "question")
    if question is None:
        return _error(400, "Type a question about the telemetry first.")
    arguments = {"question": question}
    if body.get("hours_back") is not None:
        arguments["hours_back"] = max(1, _int(body, "hours_back", 24))
    return await _call_tool(request, "correlated_telemetry_report", arguments, "telemetry", question)


# ---------- simulators ----------

def _simulator_list(manager: SimulatorManager) -> list[dict]:
    return [{"name": s.name, "running": s.running, "last_line": s.last_line} for s in manager.status()]


async def simulators(request: Request) -> JSONResponse:
    return JSONResponse({"simulators": _simulator_list(request.app.state.simulators)})


async def simulator_action(request: Request) -> JSONResponse:
    manager: SimulatorManager = request.app.state.simulators
    name, action = request.path_params["name"], request.path_params["action"]
    try:
        if action == "start":
            await run_in_threadpool(manager.start, name)
        elif action == "stop":
            await run_in_threadpool(manager.stop, name)
        else:
            return _error(404, f"Unknown action {action!r}; use start or stop.")
    except ValueError as error:
        return _error(404, str(error))
    _record(request, "simulator", f"{name} simulator", f"{action.capitalize()} the {name} simulator",
            f"{name.capitalize()} simulator {'started' if action == 'start' else 'stopped'}.", ok=True)
    return JSONResponse({"simulators": _simulator_list(manager)})


# ---------- history ----------

async def history_list(request: Request) -> JSONResponse:
    """Newest-first summaries (no full answers), paged with ?before=<id>&limit=N; also says who is asking."""
    limit = max(1, min(_int(dict(request.query_params), "limit", 50), 200))
    entries, more = request.app.state.history.page(limit, request.query_params.get("before") or None)
    return JSONResponse({"entries": [e.summary() for e in entries], "more": more, "viewer": _visitor(request)})


async def history_entry(request: Request) -> JSONResponse:
    entry = request.app.state.history.get(request.path_params["entry_id"])
    if entry is None:
        return _error(404, "That history entry doesn't exist.")
    return JSONResponse(asdict(entry))


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/tools", tools),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/remember", remember, methods=["POST"]),
        Route("/api/web-search", web_search, methods=["POST"]),
        Route("/api/network-report", network_report, methods=["POST"]),
        Route("/api/power", power, methods=["POST"]),
        Route("/api/telemetry/start", telemetry_start, methods=["POST"]),
        Route("/api/telemetry/stop", telemetry_stop, methods=["POST"]),
        Route("/api/telemetry/status", telemetry_status),
        Route("/api/telemetry/ask", telemetry_ask, methods=["POST"]),
        Route("/api/simulators", simulators),
        Route("/api/simulators/{name}/{action}", simulator_action, methods=["POST"]),
        Route("/api/history", history_list),
        Route("/api/history/{entry_id}", history_entry),
    ],
    lifespan=lifespan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local demo UI for Kumbia.")
    parser.add_argument("--host", default="127.0.0.1", help="interface IP to bind; 0.0.0.0 for all")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    shown = "localhost" if args.host == "127.0.0.1" else args.host
    print(f"Kumbia demo UI: http://{shown}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
