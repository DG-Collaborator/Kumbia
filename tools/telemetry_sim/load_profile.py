"""
      Dev-only: the shared synthetic "load" curve both telemetry simulators
      (and the power backfill) follow, so the power and network streams they
      produce rise and fall together — a real relationship for the
      correlation layer (and the assistant) to find, instead of two unrelated
      random walks.

      Load is 0..1, computed from the wall clock alone, so any simulator in
      any process gets the same load for the same moment without
      coordinating with the others. It mixes two cycles: a daily one (low
      overnight, highest mid-afternoon UTC), so a 24-hour or multi-day view
      has a real shape, and a short PERIOD_SECONDS one on top, so a demo
      watched for a few minutes still visibly moves.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime

PERIOD_SECONDS = 600
DAY_SECONDS = 86_400
_DAILY_PEAK_HOUR_UTC = 15
_DAILY_SHARE = 0.6  # how much of the load the daily cycle carries; the rest is the short cycle


def load_at(moment: datetime) -> float:
    return _DAILY_SHARE * daily_load_at(moment) + (1 - _DAILY_SHARE) * short_load_at(moment)


def short_load_at(moment: datetime) -> float:
    phase = (moment.timestamp() % PERIOD_SECONDS) / PERIOD_SECONDS
    return 0.5 - 0.5 * math.cos(2 * math.pi * phase)


def daily_load_at(moment: datetime) -> float:
    seconds_from_peak = (moment.timestamp() - _DAILY_PEAK_HOUR_UTC * 3600) % DAY_SECONDS
    return 0.5 + 0.5 * math.cos(2 * math.pi * seconds_from_peak / DAY_SECONDS)


def outlet_offset_watts(outlet: str) -> float:
    """A steady per-outlet shift in watts (-15..+15), the same in every process, so several assets differ."""
    digest = hashlib.sha256(outlet.encode("utf-8")).digest()
    return (digest[0] / 255) * 30 - 15
