"""
      Dev-only: the shared synthetic "load" curve both telemetry simulators
      follow, so the power and network streams they produce rise and fall
      together — a real relationship for the correlation layer (and the
      assistant) to find, instead of two unrelated random walks.

      Load is a smooth 0 → 1 → 0 cycle over PERIOD_SECONDS, computed from the
      wall clock alone. Any simulator, in any process, gets the same load for
      the same moment without coordinating with the others.
"""
from __future__ import annotations

import math
from datetime import datetime

PERIOD_SECONDS = 600  


def load_at(moment: datetime) -> float:
    phase = (moment.timestamp() % PERIOD_SECONDS) / PERIOD_SECONDS
    return 0.5 - 0.5 * math.cos(2 * math.pi * phase)