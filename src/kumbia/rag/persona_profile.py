"""
    Per-persona configuration, injected into ConversationPipeline rather than
    baked into it. One pipeline class, many PersonaProfile instances — this is
    the seam that makes adding a new persona a matter of constructing a new
    profile, never editing the pipeline itself.

    data_sources is keyed by name (e.g. "web_search", "power_telemetry") so a
    persona's ConversationPipeline can only reach the sources listed here —
    a persona simply has no way to look up a source it wasn't given, which is
    what makes tool visibility structural rather than instructional.

    rag_namespace exists so two personas' in-memory vector indexes never mix,
    even if they happen to share a process. memory_path/cache_path are kept
    separate per persona for the same reason: no persona's remembered facts or
    cached lookups are reachable from another persona's pipeline instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumbia.core.source.data_source import DataSource


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    system_prompt: str
    data_sources: dict[str, DataSource]
    rag_namespace: str
    memory_path: Path
    cache_path: Path