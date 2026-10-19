"""
     The correlation engine's output shape — a single asset's telemetry at a
     point in time, after identity resolution and any power/compute join has
     happened. Not a fetch-time representation: raw DataSource fetches stay
     generic DataTable, same as everywhere else in core. Only timestamp,
     asset_id, and source_system are guaranteed; every reading is optional
     since a given window may only have power data, only compute data, or
     both.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalRecord:
    timestamp: str
    asset_id: str
    source_system: str
    site: str | None = None
    vendor: str | None = None
    device_type: str | None = None
    power_watts: float | None = None
    energy_kwh: float | None = None
    cpu_utilization: float | None = None
    gpu_utilization: float | None = None
    memory_utilization: float | None = None
    network_event_count: float | None = None