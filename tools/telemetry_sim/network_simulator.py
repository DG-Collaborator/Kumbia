"""
      Dev-only: makes the network stream live. Once per interval, indexes a
      batch of network events for one host into an Elasticsearch index
      (default network-activity-live), timestamped now. The batch size follows
      load_profile — shared with the power simulator — so network activity and
      power draw rise and fall together.

      Creates the index first, with the same field types as
      network-activity-sample (host.ip as ip, @timestamp as date, gigamon
      fields as keyword); left to dynamic mapping, host.ip would become text
      and the collector's terms filter on it would stop matching. Never
      touches any other index. Reads ES_URL / ES_USERNAME / ES_PASSWORD from
      the repo's .env. Not part of the server. Stop with Ctrl+C.

      Usage:
          python tools/telemetry_sim/network_simulator.py
          python tools/telemetry_sim/network_simulator.py --interval-seconds 60 --count 5
  """
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from load_profile import load_at

REPO_ROOT = Path(__file__).resolve().parents[2]
_EDGE_OFFSET_SECONDS = 2

_INDEX_MAPPINGS = {
    "mappings": {
        "properties": {
            "@timestamp": {"type": "date"},
            "host": {"properties": {"ip": {"type": "ip"}}},
            "gigamon": {"properties": {"ami": {"properties": {
                "dns_name": {"type": "keyword"},
                "event_name": {"type": "keyword"},
            }}}},
        }
    }
}

_DNS_NAMES = ["telemetry.example.net", "cdn.example.com", "api.example.org", "updates.example.net"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Index live synthetic network events into Elasticsearch.")
    parser.add_argument("--index", default="network-activity-live",
                        help="must match TELEMETRY_ES_INDEX_PATTERN (default network-activity-*)")
    parser.add_argument("--host-ip", default="203.0.113.5", help="must match TELEMETRY_NETWORK_HOST_IP")
    parser.add_argument("--interval-seconds", type=float, default=60.0,
                        help="one batch per interval; match TELEMETRY_BUCKET_MINUTES * 60")
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=12)
    parser.add_argument("--count", type=int, default=0, help="stop after N batches (0 = run until Ctrl+C)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    base_url = os.environ["ES_URL"].rstrip("/")
    username, password = os.environ.get("ES_USERNAME"), os.environ.get("ES_PASSWORD")
    auth = (username, password) if username and password else None

    with httpx.Client(auth=auth, timeout=30.0) as client:
        _ensure_index(client, base_url, args.index)
        print(f"Indexing events for {args.host_ip} into {args.index} every {args.interval_seconds:g}s (Ctrl+C to stop)")

        written = 0
        try:
            while True:
                _sleep_until_next_interval(args.interval_seconds)
                now = datetime.now(timezone.utc)
                event_count = max(args.min_events, round(
                    args.min_events + (args.max_events - args.min_events) * load_at(now) + random.uniform(-1, 1)))
                _index_batch(client, base_url, args.index, args.host_ip, now, event_count)
                written += 1
                print(f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}  {args.host_ip}  {event_count} event(s)")
                if args.count and written >= args.count:
                    break
        except KeyboardInterrupt:
            pass
    print(f"Stopped after {written} batch(es).")


def _ensure_index(client: httpx.Client, base_url: str, index: str) -> None:
    if client.head(f"{base_url}/{index}").status_code == 200:
        return
    response = client.put(f"{base_url}/{index}", json=_INDEX_MAPPINGS)
    response.raise_for_status()
    print(f"Created index {index} with explicit field types.")


def _index_batch(client: httpx.Client, base_url: str, index: str, host_ip: str,
                 now: datetime, event_count: int) -> None:
    lines: list[str] = []
    for i in range(event_count):
        timestamp = now + timedelta(milliseconds=i)  # distinct, all inside this interval's bucket
        lines.append(json.dumps({"index": {}}))
        lines.append(json.dumps({
            "@timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "host": {"ip": host_ip},
            "gigamon": {"ami": {"dns_name": random.choice(_DNS_NAMES), "event_name": "metadata generation"}},
        }))
    response = client.post(
        f"{base_url}/{index}/_bulk?refresh=wait_for",
        content="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    response.raise_for_status()
    if response.json().get("errors"):
        raise RuntimeError(f"Bulk indexing reported errors: {response.text[:500]}")


def _sleep_until_next_interval(interval_seconds: float) -> None:
    now = time.time()
    next_mark = (now // interval_seconds + 1) * interval_seconds + _EDGE_OFFSET_SECONDS
    time.sleep(next_mark - now)


if __name__ == "__main__":
    main()