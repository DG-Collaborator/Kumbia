"""
   Network Monitoring persona: network activity over a trailing time window,
   across whatever Elasticsearch index/indices are in scope. Not scoped to any
   single log source by default — index_pattern controls scope and defaults to
   "*" (everything), with callers able to narrow it per question.
"""
from __future__ import annotations

from dataclasses import dataclass

from kumbia.core.source.data_source import DataSource
from kumbia.core.source.source_query import Credentials, SourceQuery
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.personas.network_monitoring import engine, queries
from kumbia.rag.personas.network_monitoring.engine import Summary
from kumbia.rag.personas.network_monitoring.queries import DEFAULT_FIELD_MAPPING, FieldMapping
from kumbia.rag.rag_pipeline import RagPipeline

DEFAULT_INDEX_PATTERN = "*"  # unscoped: all network activity, not any one log source

_DEFAULT_NARRATIVE_PROMPT = (
    "Summarize this network activity in 3-4 short paragraphs. Call out any top "
    "talker that looks unusual (e.g. very high volume against very few distinct "
    "destinations), and note whether traffic volume was steady or spiky across "
    "the window."
)


@dataclass(frozen=True)
class NetworkActivityReport:
    hours_back: int
    index_pattern: str
    summary: Summary
    narrative: str


@dataclass(frozen=True)
class NetworkActivityDrilldown:
    host_ip: str
    hours_back: int
    index_pattern: str
    matched_event_count: int
    question: str
    answer: str


class NetworkActivityPipeline:
    def __init__(
            self,
            data_source: DataSource,
            source_address: str,
            credentials: Credentials | None,
            llm: LlmClient | None,
            fields: FieldMapping = DEFAULT_FIELD_MAPPING,
    ) -> None:
        self._data_source = data_source
        self._source_address = source_address
        self._credentials = credentials
        self._llm = llm
        self._fields = fields

    async def summary(
            self,
            hours_back: int = 24,
            top_n: int = 10,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> Summary:
        """The deterministic-math half only — same aggregation run() uses,
        with no LLM narration. Cheap and side-effect-free by design; this is
        what resources call, never run()'s narrative path, so reading a
        resource never triggers an LLM call once Step 4 lands.
        """
        window = queries.last_hours(hours_back)
        url = queries.search_url(self._source_address, index_pattern)

        top_talkers_table = await self._data_source.fetch(
            SourceQuery(
                address=url,
                statement=queries.top_talkers(window, top_n, self._fields),
                credentials=self._credentials,
            )
        )
        volume_table = await self._data_source.fetch(
            SourceQuery(
                address=url,
                statement=queries.volume_over_time(window, self._fields),
                credentials=self._credentials,
            )
        )
        return engine.summarize(top_talkers_table, volume_table)

    async def run(
            self,
            hours_back: int = 24,
            top_n: int = 10,
            question: str | None = None,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> NetworkActivityReport:
        summary = await self.summary(hours_back, top_n, index_pattern)

        if summary.total_doc_count == 0:
            narrative = f"No activity found in the last {hours_back} hour(s)."
        else:
            system_prompt = (
                    "You are a network operations analyst. Use only the figures given "
                    "below — they were computed deterministically from Elasticsearch "
                    "aggregations, do not recalculate or contradict them.\n\n"
                    + _summary_text(hours_back, index_pattern, summary)
            )
            narrative = await self._llm.generate(
                prompt=question or _DEFAULT_NARRATIVE_PROMPT,
                system_prompt=system_prompt,
            )

        return NetworkActivityReport(
            hours_back=hours_back,
            index_pattern=index_pattern,
            summary=summary,
            narrative=narrative,
        )

    async def drilldown(
            self,
            host_ip: str,
            question: str,
            hours_back: int = 24,
            max_events: int = 200,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> NetworkActivityDrilldown:
        """
           Free-text follow-up grounded in the actual events for one flagged
           host. Fetches a bounded slice and hands it to a fresh RagPipeline,
           which embeds/retrieves/answers from that slice.
        """
        window = queries.last_hours(hours_back)
        url = queries.search_url(self._source_address, index_pattern)

        raw_table = await self._data_source.fetch(
            SourceQuery(
                address=url,
                statement=queries.raw_hits(window, host_ip, max_events, self._fields),
                credentials=self._credentials,
            )
        )

        if raw_table.is_empty:
            answer = f"No events found for {host_ip} in the last {hours_back} hour(s)."
        else:
            rag = RagPipeline(self._llm)
            await rag.ingest(raw_table)
            answer = await rag.answer(question)

        return NetworkActivityDrilldown(
            host_ip=host_ip,
            hours_back=hours_back,
            index_pattern=index_pattern,
            matched_event_count=len(raw_table.rows),
            question=question,
            answer=answer,
        )


def to_text(report: NetworkActivityReport) -> str:
    return _summary_text(report.hours_back, report.index_pattern, report.summary) + "\n\n" + report.narrative


def _summary_text(hours_back: int, index_pattern: str, summary: Summary) -> str:
    lines = [
        f"Window: last {hours_back} hour(s)",
        f"Scope: {index_pattern}",
        f"Total events: {summary.total_doc_count}",
    ]
    if summary.busiest_bucket is not None:
        lines.append(
            f"Busiest hour: {summary.busiest_bucket.timestamp} "
            f"({summary.busiest_bucket.doc_count} events)"
        )
    lines.append("")
    lines.append("Top talkers:")
    for t in summary.top_talkers:
        lines.append(f"  {t.host_ip} — {t.doc_count} events, {t.unique_dns_names} distinct DNS names")
    return "\n".join(lines)

