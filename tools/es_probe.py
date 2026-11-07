"""
   Dev-only, read-only: shows what network data the Elasticsearch in .env
   actually holds, so NETWORK_* settings can be filled in correctly. Prints
   index names with document counts, and for the chosen pattern the newest
   event time and every field's type, marking which fields can be aggregated
   (and so used in NETWORK_BREAKDOWN_FIELDS) and how many distinct values the
   top candidates have. Prints field names and types only, never document
   values, so its output is safe to paste into a chat.

   With --host IP, instead checks whether one machine appears in the traffic
   (as source, and on either side, over the last hour and day) and lists the
   busiest source IPs in its /16, to pick a TELEMETRY_NETWORK_HOST_IP the
   tap actually sees.

   Usage:
       .venv/bin/python tools/es_probe.py                       # list indices
       .venv/bin/python tools/es_probe.py "logs-gigamon.ami-*"  # fields of a pattern
       .venv/bin/python tools/es_probe.py "logs-gigamon.ami-*" --host 10.20.0.241
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
_AGGREGATABLE = {"keyword", "ip", "long", "integer", "short", "byte", "double", "float",
                 "half_float", "scaled_float", "date", "date_nanos", "boolean", "constant_keyword"}
_CARDINALITY_SAMPLE = 40


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    base = os.environ["ES_URL"].rstrip("/")
    user, password = os.environ.get("ES_USERNAME"), os.environ.get("ES_PASSWORD")
    auth = (user, password) if user and password else None
    timestamp_field = os.environ.get("NETWORK_TIMESTAMP_FIELD", "@timestamp")

    with httpx.Client(auth=auth, timeout=30.0) as client:
        indices = client.get(f"{base}/_cat/indices", params={"format": "json", "h": "index,docs.count", "s": "index"})
        indices.raise_for_status()
        print("Indices (hidden/system ones skipped):")
        for row in indices.json():
            if not row["index"].startswith("."):
                print(f"  {row['index']}  ({row['docs.count']} docs)")

        if len(sys.argv) < 2:
            print("\nRe-run with an index pattern to see its fields.")
            return
        pattern = sys.argv[1]

        if len(sys.argv) >= 4 and sys.argv[2] == "--host":
            _probe_host(client, base, pattern, sys.argv[3], timestamp_field)
            return

        newest = client.post(f"{base}/{pattern}/_search", json={
            "size": 0, "aggs": {"newest": {"max": {"field": timestamp_field}},
                                "oldest": {"min": {"field": timestamp_field}}},
        })
        newest.raise_for_status()
        aggs = newest.json().get("aggregations", {})
        total = newest.json().get("hits", {}).get("total", {})
        print(f"\nPattern {pattern}: {total.get('value')}{'+' if total.get('relation') == 'gte' else ''} events, "
              f"{timestamp_field} from {aggs.get('oldest', {}).get('value_as_string')} "
              f"to {aggs.get('newest', {}).get('value_as_string')}")

        mapping = client.get(f"{base}/{pattern}/_mapping")
        mapping.raise_for_status()
        fields: dict[str, str] = {}
        for index_mapping in mapping.json().values():
            _flatten(index_mapping.get("mappings", {}).get("properties", {}), "", fields)

        print("\nFields (A = aggregatable, usable for host/DNS/breakdowns):")
        aggregatable = []
        for name in sorted(fields):
            flag = "A" if fields[name] in _AGGREGATABLE else " "
            if flag == "A":
                aggregatable.append(name)
            print(f"  [{flag}] {name}: {fields[name]}")

        print(f"\nDistinct values (approx.) for the first {_CARDINALITY_SAMPLE} aggregatable fields:")
        for name in aggregatable[:_CARDINALITY_SAMPLE]:
            response = client.post(f"{base}/{pattern}/_search", json={
                "size": 0, "aggs": {"distinct": {"cardinality": {"field": name}}},
            })
            value = response.json().get("aggregations", {}).get("distinct", {}).get("value") \
                if response.is_success else f"error {response.status_code}"
            print(f"  {name}: {value}")


def _probe_host(client: httpx.Client, base: str, pattern: str, ip: str, timestamp_field: str) -> None:
    host_field = os.environ.get("NETWORK_HOST_FIELD", "source.ip")
    print(f"\nHost {ip} in {pattern}:")
    for field_name in (host_field, "related.ip"):
        for window in ("now-1h", "now-24h"):
            response = client.post(f"{base}/{pattern}/_count", json={"query": {"bool": {"filter": [
                {"term": {field_name: ip}}, {"range": {timestamp_field: {"gte": window}}},
            ]}}})
            count = response.json().get("count") if response.is_success else f"error {response.status_code}"
            side = "as source" if field_name == host_field else "either side"
            print(f"  {side:<12} ({field_name}), since {window}: {count}")

    subnet = ".".join(ip.split(".")[:2]) + ".0.0/16"
    response = client.post(f"{base}/{pattern}/_search", json={
        "size": 0,
        "query": {"bool": {"filter": [
            {"term": {host_field: subnet}}, {"range": {timestamp_field: {"gte": "now-24h"}}},
        ]}},
        "aggs": {"ips": {"terms": {"field": host_field, "size": 15}}},
    })
    print(f"\nBusiest {host_field} values in {subnet}, last 24h:")
    if not response.is_success:
        print(f"  error {response.status_code}: {response.text[:300]}")
        return
    buckets = response.json().get("aggregations", {}).get("ips", {}).get("buckets", [])
    if not buckets:
        print("  none: the tap sees no traffic from this subnet")
    for bucket in buckets:
        print(f"  {bucket['key']}: {bucket['doc_count']} events")


def _flatten(properties: dict, prefix: str, out: dict[str, str]) -> None:
    for name, spec in properties.items():
        path = f"{prefix}{name}"
        if "properties" in spec:
            _flatten(spec["properties"], f"{path}.", out)
        else:
            out[path] = spec.get("type", "object")
        for sub_name, sub_spec in spec.get("fields", {}).items():  # multi-fields like name.keyword
            out[f"{path}.{sub_name}"] = sub_spec.get("type", "object")


if __name__ == "__main__":
    main()
