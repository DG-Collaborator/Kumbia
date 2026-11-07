"""Dev-only: the demo UI's interaction history. Every question, report, and
   tool call any visitor makes through the UI is appended here — who (their
   IP address as the server sees it), when, what they asked, which tool
   answered, and the full answer — so the History tab can list them for
   everyone, newest first.

   One JSON object per line in data/history/history.jsonl (gitignored),
   append-only, so history survives restarts and a crash can at worst lose
   the line being written. Entries are also kept in memory for fast listing;
   the file is only read once, at startup. This lives in the demo host, not
   in Kumbia's backend: recording what a UI's visitors did is the UI's
   concern, and any other host can keep its own history the same way.
   """
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_PREVIEW_CHARS = 160


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    ts: str          # ISO 8601, UTC
    who: str         # visitor's IP address as the server sees it
    kind: str        # chat, report, power, telemetry, remember, web_search, simulator, ...
    tool: str        # the tool (or action) that answered
    prompt: str      # what the visitor asked or did
    answer: str      # the full answer, or the error text
    ok: bool

    def summary(self) -> dict:
        """Everything but the full answer, plus a short preview of it — what the list shows."""
        data = asdict(self)
        answer = data.pop("answer")
        data["preview"] = answer if len(answer) <= _PREVIEW_CHARS else answer[:_PREVIEW_CHARS] + "…"
        return data


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: list[HistoryEntry] = self._load()  # oldest first
        self._by_id = {entry.id: entry for entry in self._entries}

    def record(self, *, who: str, kind: str, tool: str, prompt: str, answer: str, ok: bool) -> HistoryEntry:
        entry = HistoryEntry(
            id=uuid.uuid4().hex[:12],
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            who=who, kind=kind, tool=tool, prompt=prompt, answer=answer, ok=ok,
        )
        line = json.dumps(asdict(entry), ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._entries.append(entry)
            self._by_id[entry.id] = entry
        return entry

    def page(self, limit: int = 50, before: str | None = None) -> tuple[list[HistoryEntry], bool]:
        """
           Up to limit entries, newest first, starting just before the entry
           with id before (or at the newest when before is None). The bool
           says whether older entries remain.
        """
        with self._lock:
            newest_first = self._entries[::-1]
        start = 0
        if before is not None:
            start = next((i + 1 for i, entry in enumerate(newest_first) if entry.id == before), len(newest_first))
        chunk = newest_first[start:start + limit]
        return chunk, start + limit < len(newest_first)

    def get(self, entry_id: str) -> HistoryEntry | None:
        with self._lock:
            return self._by_id.get(entry_id)

    def _load(self) -> list[HistoryEntry]:
        if not self._path.exists():
            return []
        entries = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(HistoryEntry(**json.loads(line)))
                except (ValueError, TypeError):
                    continue  # a torn final line from a crash, or a hand edit: skip it, keep the rest
        return entries
