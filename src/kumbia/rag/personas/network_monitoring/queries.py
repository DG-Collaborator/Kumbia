"""
   Network-activity queries against Elasticsearch. Field names are
   configurable via FieldMapping rather than hardcoded literals scattered
   through the query bodies — swapping to a different ES source with different
   field conventions is a config change here, not a rewrite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class FieldMapping:
    """
       Where this persona finds things in the index. breakdown_fields maps a
       short alias (lowercase letters, digits, underscores — it becomes part
       of a skill/tool name) to any aggregatable field worth breaking traffic
       down by (an app name, a destination port, a country, ...). Empty means
       "just the DNS field", which every Gigamon AMI-shaped index has.

       drilldown_fields limits which fields a drilldown fetches per event.
       Wide enriched records can carry hundreds of fields; fetching only the
       ones worth reasoning about keeps each event small to embed. Empty
       means every field (the index's whole _source).

       dest_field is the other end of a connection from host_field; host
       profiles use it for "who did this host talk to" and "who talked to
       this host".
    """
    host_field: str = "host.ip"
    dest_field: str = "destination.ip"
    dns_field: str = "gigamon.ami.dns_name"
    timestamp_field: str = "@timestamp"
    breakdown_fields: dict[str, str] = field(default_factory=dict)
    drilldown_fields: tuple[str, ...] = ()

    def breakdowns(self) -> dict[str, str]:
        return dict(self.breakdown_fields) or {"dns": self.dns_field}


DEFAULT_FIELD_MAPPING = FieldMapping()


@dataclass(frozen=True)
class TimeWindow:
    from_iso: str
    to_iso: str


def last_hours(hours_back: int) -> TimeWindow:
    now = datetime.now(timezone.utc)
    from_time = now - timedelta(hours=hours_back)
    return TimeWindow(from_iso=_iso(from_time), to_iso=_iso(now))


def search_url(elasticsearch_base_url: str, index_pattern: str) -> str:
    return f"{elasticsearch_base_url.rstrip('/')}/{index_pattern}/_search"


def top_talkers(window: TimeWindow, top_n: int, fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Top source IPs by doc_count, with per-host distinct DNS-name and destination counts nested in."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {
            "top_hosts": {
                "terms": {"field": fields.host_field, "size": top_n},
                "aggs": {
                    "unique_dns_names": {"cardinality": {"field": fields.dns_field}},
                    "unique_destinations": {"cardinality": {"field": fields.dest_field}},
                },
            }
        },
    })


def event_count(window: TimeWindow, fields: FieldMapping = DEFAULT_FIELD_MAPPING,
                filter_field: str | None = None, filter_value: str | None = None,
                present_field: str | None = None) -> str:
    """
       Events in the window, optionally only those where filter_field equals
       filter_value and/or present_field has a value. One "events" cell.
    """
    return json.dumps({
        "size": 0,
        "query": _window_query(window, fields, filter_field, filter_value, present_field),
        "aggs": {"events": {"value_count": {"field": fields.timestamp_field}}},
    })


def host_terms(window: TimeWindow, filter_field: str, host_ip: str, agg_field: str, top_n: int,
               fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Top values of agg_field among the window's events where filter_field is host_ip."""
    _require_ip(host_ip)
    return json.dumps({
        "size": 0,
        "query": _window_query(window, fields, filter_field, host_ip),
        "aggs": {"values": {"terms": {"field": agg_field, "size": top_n}}},
    })


def bucket_interval(hours_back: int) -> str:
    """
       A histogram bucket size that keeps any window readable: fine enough to
       show a spike in a short window, coarse enough not to drown a long one.
    """
    if hours_back <= 6:
        return "5m"
    if hours_back <= 72:
        return "1h"
    return "1d"


def volume_over_time(
        window: TimeWindow, fields: FieldMapping = DEFAULT_FIELD_MAPPING, interval: str = "1h",
) -> str:
    """Doc-count volume per interval across the window, for spotting spikes/lulls."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {
            "over_time": {
                "date_histogram": {"field": fields.timestamp_field, "fixed_interval": interval}
            }
        },
    })


def latest_event(fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Newest event timestamp anywhere in scope (no window), to explain an empty window."""
    return json.dumps({
        "size": 0,
        "aggs": {"newest": {"max": {"field": fields.timestamp_field}}},
    })


def field_breakdown(window: TimeWindow, field_name: str, top_n: int,
                    fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Event count per value of one field across the window, most frequent first."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {"values": {"terms": {"field": field_name, "size": top_n}}},
    })


def distinct_values(window: TimeWindow, field_name: str, fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Approximate count of distinct values of one field across the window."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {"distinct": {"cardinality": {"field": field_name}}},
    })


def raw_hits(window: TimeWindow, host_ip: str, max_events: int, fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """
       Raw hit slice for one host. Grounds a free-text question in the actual
       events instead of the aggregate counts.
    """
    _require_ip(host_ip)

    body: dict = {
        "size": max_events,
        "sort": [{fields.timestamp_field: "desc"}],
        "query": {
            "bool": {
                "filter": [
                    {"term": {fields.host_field: host_ip}},
                    {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
                ]
            }
        },
    }
    if fields.drilldown_fields:
        body["_source"] = list(fields.drilldown_fields)
    return json.dumps(body)


def _window_query(window: TimeWindow, fields: FieldMapping, filter_field: str | None = None,
                  filter_value: str | None = None, present_field: str | None = None) -> dict:
    clauses: list[dict] = [{"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}}]
    if filter_field is not None and filter_value is not None:
        clauses.append({"term": {filter_field: filter_value}})
    if present_field is not None:
        clauses.append({"exists": {"field": present_field}})
    return {"bool": {"filter": clauses}}


def _require_ip(host_ip: str) -> None:
    if not _is_ip_literal(host_ip):
        raise ValueError(f"host_ip must be a plain IPv4/IPv6 address, got: {host_ip!r}")


_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_IPV6_RE = re.compile(r"^[0-9a-fA-F:]+$")


def _is_ip_literal(value: str) -> bool:
    v4_match = _IPV4_RE.match(value)
    if v4_match:
        return all(0 <= int(octet) <= 255 for octet in v4_match.groups())
    return bool(_IPV6_RE.match(value)) and ":" in value


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")