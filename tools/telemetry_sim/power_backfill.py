"""
      Dev-only: fills power_telemetry.sqlite with synthetic history, so
      questions about "the past 24 hours" or "this week" have power data to
      answer from. Replaces scratch_power_seed.py, which dropped the table
      and wrote four readings from September.

      Writes one reading per --interval-seconds for every PDU outlet in the
      asset registry (data/telemetry/assets.json, the same file the server
      reads; --outlet when there is none), over the last --days. Readings sit
      on the same grid the live power simulator writes to (2 s past each
      interval), and a reading that already exists for an outlet at a
      timestamp is never rewritten — so the backfill can be rerun any time,
      before or alongside the simulator, and only fills gaps. Never drops
      the table or deletes rows.

      Watts follow load_profile (a daily cycle plus a short one, shared with
      the simulators) plus a little noise; each outlet gets its own steady
      offset so several assets don't draw identical power. This is synthetic
      data: any relationship it has with REAL network traffic is coincidence,
      and the server labels it with POWER_TELEMETRY_SOURCE_SYSTEM
      (synthetic_dcim by default) in every answer. Stdlib only, no kumbia
      imports.

      Usage:
          python tools/telemetry_sim/power_backfill.py                 # last 7 days, 60 s readings
          python tools/telemetry_sim/power_backfill.py --days 30 --interval-seconds 300
          python tools/telemetry_sim/power_backfill.py --registry path/to/assets.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from load_profile import load_at, outlet_offset_watts

REPO_ROOT = Path(__file__).resolve().parents[2]
_EDGE_OFFSET_SECONDS = 2  # matches power_simulator.py, so both write to the same timestamps
_BATCH_SIZE = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill synthetic power readings for every registered outlet.")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "power_telemetry.sqlite")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "data" / "telemetry" / "assets.json",
                        help="asset registry JSON; its pdu_outlet identities are the outlets written")
    parser.add_argument("--outlet", default="R42-U18", help="outlet to write when the registry file doesn't exist")
    parser.add_argument("--days", type=float, default=7.0, help="how far back to fill, from now")
    parser.add_argument("--interval-seconds", type=int, default=60,
                        help="one reading per outlet per interval; match the power simulator's")
    parser.add_argument("--idle-watts", type=float, default=450.0)
    parser.add_argument("--peak-watts", type=float, default=530.0)
    parser.add_argument("--seed", type=int, default=None, help="random seed for repeatable noise")
    args = parser.parse_args()

    if args.interval_seconds < 1 or args.days <= 0:
        sys.exit("--interval-seconds must be >= 1 and --days > 0.")
    rng = random.Random(args.seed)
    outlets = _outlets(args.registry, args.outlet)
    _ensure_table(args.db)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(days=args.days)
    grid = list(_grid(start, now, args.interval_seconds))
    print(f"Backfilling {len(outlets)} outlet(s) {', '.join(outlets)} from {_iso(grid[0]) if grid else '-'} "
          f"to {_iso(grid[-1]) if grid else '-'} every {args.interval_seconds}s into {args.db}")

    with sqlite3.connect(args.db) as connection:
        for outlet in outlets:
            existing = {
                ts for (ts,) in connection.execute(
                    "SELECT ts FROM power_readings WHERE outlet = ? AND ts >= ?", (outlet, _iso(start)),
                )
            }
            offset = outlet_offset_watts(outlet)
            rows = [
                (outlet, _iso(moment), _watts(moment, args.idle_watts, args.peak_watts, offset, rng))
                for moment in grid
                if _iso(moment) not in existing
            ]
            for i in range(0, len(rows), _BATCH_SIZE):
                connection.executemany(
                    "INSERT INTO power_readings (outlet, ts, watts) VALUES (?, ?, ?)", rows[i:i + _BATCH_SIZE],
                )
            print(f"  {outlet}: wrote {len(rows)} reading(s), kept {len(grid) - len(rows)} already present")


def _outlets(registry_path: Path, fallback: str) -> list[str]:
    if not registry_path.exists():
        print(f"No registry at {registry_path}; using --outlet {fallback}")
        return [fallback]
    assets = json.loads(registry_path.read_text(encoding="utf-8"))
    outlets = sorted({identities["pdu_outlet"] for identities in assets.values() if "pdu_outlet" in identities})
    if not outlets:
        sys.exit(f"{registry_path} registers no pdu_outlet identities; nothing to backfill.")
    return outlets


def _ensure_table(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS power_readings (outlet TEXT, ts TEXT, watts REAL)")
        connection.execute("CREATE INDEX IF NOT EXISTS power_readings_outlet_ts ON power_readings (outlet, ts)")


def _grid(start: datetime, end: datetime, interval_seconds: int):
    """Every interval mark (+ the edge offset) in [start, end), the same marks the live simulator writes at."""
    first = (int(start.timestamp()) // interval_seconds + 1) * interval_seconds + _EDGE_OFFSET_SECONDS
    moment = first
    while moment < end.timestamp():
        yield datetime.fromtimestamp(moment, tz=timezone.utc)
        moment += interval_seconds


def _watts(moment: datetime, idle: float, peak: float, offset: float, rng: random.Random) -> float:
    return round(idle + offset + (peak - idle) * load_at(moment) + rng.uniform(-3, 3), 1)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    main()
