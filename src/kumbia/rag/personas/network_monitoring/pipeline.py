"""
   Network Monitoring persona: network activity over a trailing time window,
   across whatever Elasticsearch index/indices are in scope. Not scoped to any
   single log source by default — index_pattern controls scope and defaults to
   "*" (everything), with callers able to narrow it per question.

   Every figure (counts, top talkers, spikes, per-field breakdowns) is
   computed by an Elasticsearch aggregation plus engine.py; the LLM only
   narrates those figures, and a narration citing any figure the summary
   doesn't contain is withheld (rag.grounding.withhold_unsupported).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from kumbia.core.source.data_source import DataSource
from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.source_query import Credentials, SourceQuery
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.grounding import withhold_unsupported
from kumbia.rag.personas.network_monitoring import engine, queries
from kumbia.rag.personas.network_monitoring.engine import FieldBreakdown, Summary
from kumbia.rag.personas.network_monitoring.queries import DEFAULT_FIELD_MAPPING, FieldMapping
from kumbia.rag.rag_pipeline import RagPipeline

DEFAULT_INDEX_PATTERN = "*"  # unscoped: all network activity, not any one log source

_DEFAULT_NARRATIVE_PROMPT = (
    "Summarize this network activity in 3-4 short paragraphs. Call out any top "
    "talker that looks unusual (e.g. very high volume against very few distinct "
    "destinations), any flagged volume spikes, and whether traffic volume was "
    "steady or spiky across the window."
)

_NARRATION_RULES = (
    "Use only the figures given below — they were computed deterministically from "
    "Elasticsearch aggregations. Do not recalculate, estimate, or contradict them, "
    "and do not introduce any number, host, or value that is not listed. If the "
    "figures don't answer the question, say what they do show instead."
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


@dataclass(frozen=True)
class NetworkFieldReport:
    hours_back: int
    index_pattern: str
    breakdown: FieldBreakdown
    newest_event: str | None
    narrative: str


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

    @property
    def breakdown_fields(self) -> dict[str, str]:
        """Alias -> field for every field this pipeline can break traffic down by."""
        return self._fields.breakdowns()

    async def summary(
            self,
            hours_back: int = 24,
            top_n: int = 10,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> Summary:
        """The deterministic-math half only — same aggregation run() uses,
        with no LLM narration. Cheap and side-effect-free by design; this is
        what resources call, never run()'s narrative path, so reading a
        resource never triggers an LLM call. When the window is empty, also
        looks up the newest event in scope, so "nothing found" says why.
        """
        window = queries.last_hours(hours_back)
        interval = queries.bucket_interval(hours_back)

        top_talkers_table = await self._fetch(index_pattern, queries.top_talkers(window, top_n, self._fields))
        volume_table = await self._fetch(index_pattern, queries.volume_over_time(window, self._fields, interval))
        summary = engine.summarize(top_talkers_table, volume_table, interval)

        if summary.total_doc_count == 0:
            newest = await self.newest_event(index_pattern)
            summary = replace(summary, newest_event=newest)
        return summary

    async def newest_event(self, index_pattern: str = DEFAULT_INDEX_PATTERN) -> str | None:
        return engine.newest_event(await self._fetch(index_pattern, queries.latest_event(self._fields)))

    async def run(
            self,
            hours_back: int = 24,
            top_n: int = 10,
            question: str | None = None,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> NetworkActivityReport:
        summary = await self.summary(hours_back, top_n, index_pattern)

        if summary.total_doc_count == 0:
            narrative = _empty_window_text(hours_back, summary.newest_event)
        else:
            narrative = await self._narrate(_summary_text(hours_back, index_pattern, summary), question)

        return NetworkActivityReport(
            hours_back=hours_back,
            index_pattern=index_pattern,
            summary=summary,
            narrative=narrative,
        )

    async def breakdown(
            self,
            alias: str,
            hours_back: int = 24,
            question: str | None = None,
            top_n: int = 10,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> NetworkFieldReport:
        """
           Event counts per value of one configured field (alias from
           breakdown_fields — never a field name taken from the question),
           plus how many distinct values it had, narrated from those figures.
        """
        field_name = self.breakdown_fields.get(alias)
        if field_name is None:
            raise ValueError(
                f"Unknown breakdown {alias!r}; configured: {', '.join(sorted(self.breakdown_fields))}."
            )
        window = queries.last_hours(hours_back)
        values_table = await self._fetch(
            index_pattern, queries.field_breakdown(window, field_name, top_n, self._fields),
        )
        distinct_table = await self._fetch(
            index_pattern, queries.distinct_values(window, field_name, self._fields),
        )
        result = engine.field_breakdown(alias, field_name, values_table, distinct_table)

        newest = None
        if not result.values:
            newest = await self.newest_event(index_pattern)
            narrative = _empty_window_text(hours_back, newest, f" with a value for {field_name}")
        else:
            prompt = question or f"Describe how network events break down by {alias}, and what stands out."
            narrative = await self._narrate(_breakdown_text(hours_back, index_pattern, result), prompt)

        return NetworkFieldReport(
            hours_back=hours_back,
            index_pattern=index_pattern,
            breakdown=result,
            newest_event=newest,
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
           which embeds/retrieves/answers from that slice (and withholds an
           answer citing figures the retrieved events don't contain).
        """
        window = queries.last_hours(hours_back)
        raw_table = await self._fetch(
            index_pattern, queries.raw_hits(window, host_ip, max_events, self._fields),
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

    async def _fetch(self, index_pattern: str, statement: str) -> DataTable:
        return await self._data_source.fetch(
            SourceQuery(
                address=queries.search_url(self._source_address, index_pattern),
                statement=statement,
                credentials=self._credentials,
            )
        )

    async def _narrate(self, figures: str, question: str | None) -> str:
        prompt = question or _DEFAULT_NARRATIVE_PROMPT
        system_prompt = f"You are a network operations analyst. {_NARRATION_RULES}\n\n{figures}"
        narration = await self._llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0)
        return withhold_unsupported(narration, figures, prompt)


def to_text(report: NetworkActivityReport) -> str:
    return _summary_text(report.hours_back, report.index_pattern, report.summary) + "\n\n" + report.narrative


def field_report_text(report: NetworkFieldReport) -> str:
    return _breakdown_text(report.hours_back, report.index_pattern, report.breakdown) + "\n\n" + report.narrative


def _empty_window_text(hours_back: int, newest_event: str | None, qualifier: str = "") -> str:
    if newest_event is None:
        return f"No activity found in the last {hours_back} hour(s){qualifier}, and the index scope has no events at all."
    return (
        f"No activity found in the last {hours_back} hour(s){qualifier}. "
        f"The newest event in scope is at {newest_event}; widen the window to include it."
    )


def _summary_text(hours_back: int, index_pattern: str, summary: Summary) -> str:
    lines = [
        f"Window: last {hours_back} hour(s)",
        f"Scope: {index_pattern}",
        f"Total events: {summary.total_doc_count}",
    ]
    if summary.busiest_bucket is not None:
        lines.append(
            f"Busiest {summary.interval} bucket: {summary.busiest_bucket.timestamp} "
            f"({summary.busiest_bucket.doc_count} events)"
        )
    if summary.volume:
        lines.append(f"Volume buckets ({summary.interval}): {len(summary.volume)}")
    if summary.anomalies:
        lines.append(f"Volume spikes (more than {engine.ANOMALY_SIGMA:g} standard deviations above the mean):")
        lines.extend(f"  {bucket.timestamp} — {bucket.doc_count} events" for bucket in summary.anomalies)
    else:
        lines.append("Volume spikes: none flagged")
    lines.append("")
    lines.append("Top talkers:")
    for t in summary.top_talkers:
        lines.append(f"  {t.host_ip} — {t.doc_count} events, {t.unique_dns_names} distinct DNS names")
    return "\n".join(lines)


def _breakdown_text(hours_back: int, index_pattern: str, breakdown: FieldBreakdown) -> str:
    lines = [
        f"Window: last {hours_back} hour(s)",
        f"Scope: {index_pattern}",
        f"Breakdown by {breakdown.alias} ({breakdown.field_name})",
        f"Distinct values: {breakdown.distinct_count}",
        "Top values:",
    ]
    lines.extend(f"  {v.value} — {v.doc_count} events" for v in breakdown.values)
    return "\n".join(lines)
