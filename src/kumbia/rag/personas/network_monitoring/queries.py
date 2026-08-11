"""
   Network-activity queries against Elasticsearch. Field names are
   configurable via FieldMapping rather than hardcoded literals scattered
   through the query bodies — swapping to a different ES source with different
   field conventions is a config change here, not a rewrite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class FieldMapping:
    host_field: str = "host.ip"
    dns_field: str = "gigamon.ami.dns_name"
    timestamp_field: str = "@timestamp"


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
    """Top source IPs by doc_count, with a per-host unique-DNS-name count nested in."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {
            "top_hosts": {
                "terms": {"field": fields.host_field, "size": top_n},
                "aggs": {
                    "unique_dns_names": {"cardinality": {"field": fields.dns_field}}
                },
            }
        },
    })


def volume_over_time(window: TimeWindow, fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """Hourly doc-count volume across the window, for spotting spikes/lulls."""
    return json.dumps({
        "size": 0,
        "query": {"range": {fields.timestamp_field: {"gte": window.from_iso, "lte": window.to_iso}}},
        "aggs": {
            "over_time": {
                "date_histogram": {"field": fields.timestamp_field, "fixed_interval": "1h"}
            }
        },
    })


def raw_hits(window: TimeWindow, host_ip: str, max_events: int, fields: FieldMapping = DEFAULT_FIELD_MAPPING) -> str:
    """
       Raw hit slice for one host. Grounds a free-text question in the actual
       events instead of the aggregate counts.
    """
    if not _is_ip_literal(host_ip):
        raise ValueError(f"host_ip must be a plain IPv4/IPv6 address, got: {host_ip!r}")

    return json.dumps({
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
    })


_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_IPV6_RE = re.compile(r"^[0-9a-fA-F:]+$")


def _is_ip_literal(value: str) -> bool:
    v4_match = _IPV4_RE.match(value)
    if v4_match:
        return all(0 <= int(octet) <= 255 for octet in v4_match.groups())
    return bool(_IPV6_RE.match(value)) and ":" in value


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")