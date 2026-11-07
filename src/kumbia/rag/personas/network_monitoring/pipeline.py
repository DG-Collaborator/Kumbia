"""
   Network Monitoring persona: network activity over a trailing time window,
   across whatever Elasticsearch index/indices are in scope. Not scoped to any
   single log source by default — index_pattern controls scope and defaults to
   "*" (everything), with callers able to narrow it per question.

   Every figure (counts, top talkers, spikes, per-field breakdowns, host
   profiles) is computed by an Elasticsearch aggregation plus engine.py; the
   LLM only narrates those figures, and a narration citing any figure the
   evidence doesn't contain is withheld (rag.grounding.withhold_unsupported).

   host_aliases ({name: ip}) gives stable addresses human names — service
   ClusterIPs, gateways — so figures read "10.96.0.10 (kube-dns)" and a
   question can name the service instead of the address.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from kumbia.core.source.data_source import DataSource
from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.source_query import Credentials, SourceQuery
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.grounding import withhold_unsupported
from kumbia.rag.personas.network_monitoring import engine, queries
from kumbia.rag.personas.network_monitoring.engine import FieldBreakdown, FieldValue, HostProfile, Summary
from kumbia.rag.personas.network_monitoring.queries import DEFAULT_FIELD_MAPPING, FieldMapping

DEFAULT_INDEX_PATTERN = "*"  # unscoped: all network activity, not any one log source
_PROFILE_TOP_N = 5

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
    profile: HostProfile | None = None


@dataclass(frozen=True)
class NetworkFieldReport:
    hours_back: int
    index_pattern: str
    breakdown: FieldBreakdown
    newest_event: str | None
    narrative: str
    labels: dict[str, str] | None = None  # ip -> name, for display


class NetworkActivityPipeline:
    def __init__(
            self,
            data_source: DataSource,
            source_address: str,
            credentials: Credentials | None,
            llm: LlmClient | None,
            fields: FieldMapping = DEFAULT_FIELD_MAPPING,
            host_aliases: dict[str, str] | None = None,
    ) -> None:
        self._data_source = data_source
        self._source_address = source_address
        self._credentials = credentials
        self._llm = llm
        self._fields = fields
        self._host_aliases = dict(host_aliases or {})
        self._names = {ip: name for name, ip in self._host_aliases.items()}

    @property
    def breakdown_fields(self) -> dict[str, str]:
        """Alias -> field for every field this pipeline can break traffic down by."""
        return self._fields.breakdowns()

    @property
    def host_aliases(self) -> dict[str, str]:
        """Name -> IP for every host this pipeline knows by name."""
        return dict(self._host_aliases)

    @property
    def host_labels(self) -> dict[str, str]:
        """IP -> name, for labeling addresses in figures."""
        return dict(self._names)

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
            narrative = await self._narrate(_summary_text(hours_back, index_pattern, summary, self._names), question)

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
            in_window = engine.single_count(await self._fetch(index_pattern, queries.event_count(window, self._fields)))
            if in_window > 0:
                narrative = (
                    f"There are {in_window} events in the last {hours_back} hour(s), but none of them "
                    f"record a value for {field_name}, so there is nothing to break down by {alias}."
                )
            else:
                newest = await self.newest_event(index_pattern)
                narrative = _empty_window_text(hours_back, newest)
        else:
            prompt = question or f"Describe how network events break down by {alias}, and what stands out."
            narrative = await self._narrate(_breakdown_text(hours_back, index_pattern, result, self._names), prompt)

        return NetworkFieldReport(
            hours_back=hours_back,
            index_pattern=index_pattern,
            breakdown=result,
            newest_event=newest,
            narrative=narrative,
            labels=self._names,
        )

    async def drilldown(
            self,
            host_ip: str,
            question: str,
            hours_back: int = 24,
            max_events: int = 20,
            index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> NetworkActivityDrilldown:
        """
           Everything about one host in the window, from both ends of its
           connections: event counts as source and as destination, who it
           talked to, who talked to it, and its top value for each breakdown
           field — all aggregations, fetched concurrently. The max_events
           most recent events it sent are given to the model as extra
           evidence (not shown), and the narration must stay within both.
        """
        window = queries.last_hours(hours_back)
        host, dest = self._fields.host_field, self._fields.dest_field
        breakdowns = {alias: name for alias, name in self.breakdown_fields.items() if name != dest}

        fetches = [
            self._fetch(index_pattern, queries.event_count(window, self._fields, host, host_ip)),
            self._fetch(index_pattern, queries.event_count(window, self._fields, dest, host_ip)),
            self._fetch(index_pattern, queries.host_terms(window, host, host_ip, dest, _PROFILE_TOP_N, self._fields)),
            self._fetch(index_pattern, queries.host_terms(window, dest, host_ip, host, _PROFILE_TOP_N, self._fields)),
            self._fetch(index_pattern, queries.raw_hits(window, host_ip, max(0, max_events), self._fields)),
            *(
                self._fetch(index_pattern, queries.host_terms(window, host, host_ip, name, _PROFILE_TOP_N, self._fields))
                for name in breakdowns.values()
            ),
        ]
        as_source, as_dest, top_dests, top_sources, recent, *breakdown_tables = await asyncio.gather(*fetches)

        profile = HostProfile(
            host_ip=host_ip,
            events_as_source=engine.single_count(as_source),
            events_as_destination=engine.single_count(as_dest),
            top_destinations=engine.field_values(top_dests),
            top_sources=engine.field_values(top_sources),
            breakdowns={alias: engine.field_values(table) for alias, table in zip(breakdowns, breakdown_tables)},
        )
        matched = profile.events_as_source + profile.events_as_destination

        if matched == 0:
            answer = f"No events found for {self._label(host_ip)} in the last {hours_back} hour(s)."
        else:
            figures = _profile_text(hours_back, index_pattern, profile, self._fields, self._names)
            narrative = await self._narrate(figures, question, extra_evidence=_rows_text(recent))
            answer = f"{figures}\n\n{narrative}"

        return NetworkActivityDrilldown(
            host_ip=host_ip,
            hours_back=hours_back,
            index_pattern=index_pattern,
            matched_event_count=matched,
            question=question,
            answer=answer,
            profile=profile,
        )

    def _label(self, ip: str) -> str:
        return _label(ip, self._names)

    async def _fetch(self, index_pattern: str, statement: str) -> DataTable:
        return await self._data_source.fetch(
            SourceQuery(
                address=queries.search_url(self._source_address, index_pattern),
                statement=statement,
                credentials=self._credentials,
            )
        )

    async def _narrate(self, figures: str, question: str | None, extra_evidence: str = "") -> str:
        prompt = question or _DEFAULT_NARRATIVE_PROMPT
        evidence = figures + (f"\n\nMost recent events sent by this host:\n{extra_evidence}" if extra_evidence else "")
        system_prompt = f"You are a network operations analyst. {_NARRATION_RULES}\n\n{evidence}"
        narration = await self._llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0)
        return withhold_unsupported(narration, evidence, prompt)


def to_text(report: NetworkActivityReport, labels: dict[str, str] | None = None) -> str:
    return _summary_text(report.hours_back, report.index_pattern, report.summary, labels) + "\n\n" + report.narrative


def field_report_text(report: NetworkFieldReport) -> str:
    if not report.breakdown.values:
        return report.narrative
    return (
        _breakdown_text(report.hours_back, report.index_pattern, report.breakdown, report.labels)
        + "\n\n" + report.narrative
    )


def _label(value: str, labels: dict[str, str] | None) -> str:
    name = (labels or {}).get(value)
    return f"{value} ({name})" if name else value


def _empty_window_text(hours_back: int, newest_event: str | None) -> str:
    if newest_event is None:
        return f"No activity found in the last {hours_back} hour(s), and the index scope has no events at all."
    return (
        f"No activity found in the last {hours_back} hour(s). "
        f"The newest event in scope is at {newest_event}; widen the window to include it."
    )


def _summary_text(hours_back: int, index_pattern: str, summary: Summary,
                  labels: dict[str, str] | None = None) -> str:
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

    dns_recorded = any(t.unique_dns_names for t in summary.top_talkers)
    lines.append("")
    lines.append("Top talkers (by events sent):")
    for t in summary.top_talkers:
        dns = f", {t.unique_dns_names} distinct DNS names" if dns_recorded else ""
        lines.append(
            f"  {_label(t.host_ip, labels)} — {t.doc_count} events, "
            f"{t.unique_destinations} distinct destinations{dns}"
        )
    if summary.top_talkers and not dns_recorded:
        lines.append("DNS names: not recorded for these hosts in this window")
    return "\n".join(lines)


def _breakdown_text(hours_back: int, index_pattern: str, breakdown: FieldBreakdown,
                    labels: dict[str, str] | None = None) -> str:
    lines = [
        f"Window: last {hours_back} hour(s)",
        f"Scope: {index_pattern}",
        f"Breakdown by {breakdown.alias} ({breakdown.field_name})",
        f"Distinct values: {breakdown.distinct_count}",
        "Top values:",
    ]
    lines.extend(f"  {_label(v.value, labels)} — {v.doc_count} events" for v in breakdown.values)
    return "\n".join(lines)


def _profile_text(hours_back: int, index_pattern: str, profile: HostProfile, fields: FieldMapping,
                  labels: dict[str, str] | None = None) -> str:
    def listing(values: list[FieldValue]) -> list[str]:
        return [f"  {_label(v.value, labels)} — {v.doc_count} events" for v in values] or ["  (none)"]

    lines = [
        f"Host: {_label(profile.host_ip, labels)}",
        f"Window: last {hours_back} hour(s)",
        f"Scope: {index_pattern}",
        f"Events sent (as {fields.host_field}): {profile.events_as_source}",
        f"Events received (as {fields.dest_field}): {profile.events_as_destination}",
        "Top destinations it sent to:",
        *listing(profile.top_destinations),
        "Top sources that sent to it:",
        *listing(profile.top_sources),
    ]
    for alias, values in profile.breakdowns.items():
        if values:
            lines.append(f"Top {alias} values in what it sent:")
            lines.extend(listing(values))
    return "\n".join(lines)


def _rows_text(table: DataTable) -> str:
    return "\n".join(
        " | ".join(f"{column.key}: {row.value(column.key)}" for column in table.columns if row.value(column.key))
        for row in table.rows
    )
