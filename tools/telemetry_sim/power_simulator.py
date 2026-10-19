"""
      Dev-only: makes the power stream live. Once per interval, appends one
      reading for a PDU outlet to power_telemetry.sqlite's power_readings
      table, timestamped now (ISO-8601 UTC, the same format the seed data and
      sql_power_window use). Watts follow load_profile — shared with the
      network simulator — plus a little noise, so power tracks network
      activity the way a real loaded server would.

      Writes a couple of seconds into each interval so every reading lands
      inside its time bucket, never on the edge: with --interval-seconds
      equal to TELEMETRY_BUCKET_MINUTES * 60, each bucket gets exactly one
      reading. Never drops or rewrites existing rows (the Sept seed data stays).
      Not part of the server — stdlib only, no kumbia imports. Stop with Ctrl+C.

      Usage:
          python tools/telemetry_sim/power_simulator.py
          python tools/telemetry_sim/power_simulator.py --interval-seconds 60 --count 5
  """
from __future__ import annotations

import argparse
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from load_profile import load_at

REPO_ROOT = Path(__file__).resolve().parents[2]
_EDGE_OFFSET_SECONDS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Append live synthetic power readings to the power telemetry sqlite.")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "power_telemetry.sqlite")
    parser.add_argument("--outlet", default="R42-U18", help="must match the asset registry's pdu_outlet")
    parser.add_argument("--interval-seconds", type=float, default=60.0,
                        help="one reading per interval; match TELEMETRY_BUCKET_MINUTES * 60")
    parser.add_argument("--idle-watts", type=float, default=450.0)
    parser.add_argument("--peak-watts", type=float, default=530.0)
    parser.add_argument("--count", type=int, default=0, help="stop after N readings (0 = run until Ctrl+C)")
    args = parser.parse_args()

    _ensure_table(args.db)
    print(f"Writing one reading for {args.outlet} every {args.interval_seconds:g}s to {args.db} (Ctrl+C to stop)")

    written = 0
    try:
        while True:
            _sleep_until_next_interval(args.interval_seconds)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            watts = round(
                args.idle_watts + (args.peak_watts - args.idle_watts) * load_at(now) + random.uniform(-3, 3), 1
            )
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            with sqlite3.connect(args.db) as connection:
                connection.execute(
                    "INSERT INTO power_readings (outlet, ts, watts) VALUES (?, ?, ?)",
                    (args.outlet, timestamp, watts),
                )
            written += 1
            print(f"{timestamp}  {args.outlet}  {watts} W")
            if args.count and written >= args.count:
                break
    except KeyboardInterrupt:
        pass
    print(f"Stopped after {written} reading(s).")


def _ensure_table(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS power_readings (outlet TEXT, ts TEXT, watts REAL)")


def _sleep_until_next_interval(interval_seconds: float) -> None:
    now = time.time()
    next_mark = (now // interval_seconds + 1) * interval_seconds + _EDGE_OFFSET_SECONDS
    time.sleep(next_mark - now)


if __name__ == "__main__":
    main()
