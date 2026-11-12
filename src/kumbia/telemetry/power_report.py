"""
     Power consumption over a window, computed deterministically from the
     per-bucket rows stream_queries.sql_power_buckets returns (outlet, bucket
     start, average / min / max watts, reading count). No I/O, no LLM: a model
     only ever narrates these figures.

     Per asset: how many readings and how much of the window they cover,
     average draw, the highest and lowest single reading and the bucket each
     fell in, the first and last bucket's average (how draw changed), and
     energy used. With more than one asset, also the system total: summed
     average draw, total energy, and the bucket where combined draw peaked.

     Energy is estimated as each covered bucket's average watts times the
     bucket's length — exact for steady sampling, and never extrapolated
     into buckets with no readings: coverage is reported beside it instead,
     so a gap in the data reads as a gap, not as zero or as a guess.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kumbia.core.domain.data_table import DataTable
from kumbia.telemetry.stream_queries import WindowPlan

MIN_BUCKET_MINUTES = 5  # coarser than typical PDU polling, so a bucket with any reading represents the bucket
MAX_BUCKET_ROWS = 2000


@dataclass(frozen=True)
class _Bucket:
    timestamp: str
    average: float
    low: float
    high: float
    readings: int


@dataclass(frozen=True)
class AssetPower:
    asset_id: str
    outlets: list[str]
    readings: int
    buckets_covered: int
    average_watts: float
    peak_watts: float
    peak_bucket: str
    low_watts: float
    low_bucket: str
    first_bucket: _Bucket
    last_bucket: _Bucket
    energy_kwh: float


def summarize_power(
        table: DataTable,
        plan: WindowPlan,
        asset_for_outlet: Callable[[str], str | None],
        identity_column: str = "outlet",
        timestamp_column: str = "ts",
        value_column: str = "watts",
) -> list[AssetPower]:
    """One AssetPower per asset with readings in the window, assets in sorted order."""
    bucket_hours = plan.bucket_minutes / 60
    by_asset: dict[str, dict[str, list[_Bucket]]] = {}
    outlets_by_asset: dict[str, set[str]] = {}
    for row in table.rows:
        outlet = row.value(identity_column)
        asset_id = asset_for_outlet(outlet)
        average = _to_float(row.value(value_column))
        if asset_id is None or average is None or not row.value(timestamp_column):
            continue
        bucket = _Bucket(
            timestamp=row.value(timestamp_column),
            average=average,
            low=_or(_to_float(row.value(f"{value_column}_min")), average),
            high=_or(_to_float(row.value(f"{value_column}_max")), average),
            readings=int(_or(_to_float(row.value("readings")), 1)),
        )
        by_asset.setdefault(asset_id, {}).setdefault(bucket.timestamp, []).append(bucket)
        outlets_by_asset.setdefault(asset_id, set()).add(outlet)

    summaries: list[AssetPower] = []
    for asset_id in sorted(by_asset):
        # an asset fed by several outlets draws their sum in each bucket
        buckets = [_combine(timestamp, parts) for timestamp, parts in sorted(by_asset[asset_id].items())]
        readings = sum(b.readings for b in buckets)
        peak = max(buckets, key=lambda b: b.high)
        low = min(buckets, key=lambda b: b.low)
        summaries.append(AssetPower(
            asset_id=asset_id,
            outlets=sorted(outlets_by_asset[asset_id]),
            readings=readings,
            buckets_covered=len(buckets),
            average_watts=sum(b.average for b in buckets) / len(buckets),
            peak_watts=peak.high,
            peak_bucket=peak.timestamp,
            low_watts=low.low,
            low_bucket=low.timestamp,
            first_bucket=buckets[0],
            last_bucket=buckets[-1],
            energy_kwh=sum(b.average for b in buckets) * bucket_hours / 1000,
        ))
    return summaries


def power_figures_text(
        table: DataTable,
        plan: WindowPlan,
        asset_for_outlet: Callable[[str], str | None],
        source_system: str,
) -> str:
    """The figures as plain text — what a model narrates, and what the user sees above that narration."""
    summaries = summarize_power(table, plan, asset_for_outlet)
    header = f"{plan.describe()}. Power data source: {source_system}."
    if not summaries:
        return f"{header}\nNo power readings from registered outlets in this window."

    paragraphs = [header]
    for s in summaries:
        coverage = s.buckets_covered / plan.bucket_count * 100
        change = s.last_bucket.average - s.first_bucket.average
        percent = f" ({change / s.first_bucket.average * 100:+.1f}%)" if s.first_bucket.average else ""
        paragraphs.append("\n".join([
            f"Asset {s.asset_id} (outlet {', '.join(s.outlets)}): {s.readings} reading(s) in "
            f"{s.buckets_covered} of {plan.bucket_count} buckets ({coverage:.0f}% of the window)",
            f"- average draw: {_fmt(s.average_watts)} W",
            f"- highest reading: {_fmt(s.peak_watts)} W, in the bucket starting {s.peak_bucket}",
            f"- lowest reading: {_fmt(s.low_watts)} W, in the bucket starting {s.low_bucket}",
            f"- first bucket ({s.first_bucket.timestamp}) averaged {_fmt(s.first_bucket.average)} W, "
            f"last bucket ({s.last_bucket.timestamp}) averaged {_fmt(s.last_bucket.average)} W, "
            f"change {change:+.1f} W{percent}",
            f"- energy used in covered buckets: {s.energy_kwh:.3f} kWh",
        ]))

    if len(summaries) > 1:
        paragraphs.append(_system_text(table, plan, asset_for_outlet, summaries))
    return "\n\n".join(paragraphs)


def _system_text(
        table: DataTable,
        plan: WindowPlan,
        asset_for_outlet: Callable[[str], str | None],
        summaries: list[AssetPower],
) -> str:
    totals: dict[str, float] = {}
    for row in table.rows:
        average = _to_float(row.value("watts"))
        if average is not None and asset_for_outlet(row.value("outlet")) is not None:
            totals[row.value("ts")] = totals.get(row.value("ts"), 0.0) + average
    peak_bucket = max(totals, key=totals.get)
    return "\n".join([
        f"System total ({len(summaries)} assets)",
        f"- combined average draw: {_fmt(sum(s.average_watts for s in summaries))} W",
        f"- combined draw peaked at {_fmt(totals[peak_bucket])} W in the bucket starting {peak_bucket}",
        f"- total energy used in covered buckets: {sum(s.energy_kwh for s in summaries):.3f} kWh",
    ])


def _combine(timestamp: str, parts: list[_Bucket]) -> _Bucket:
    if len(parts) == 1:
        return parts[0]
    return _Bucket(
        timestamp=timestamp,
        average=sum(p.average for p in parts),
        low=sum(p.low for p in parts),
        high=sum(p.high for p in parts),
        readings=sum(p.readings for p in parts),
    )


def _or(value: float | None, default: float) -> float:
    return default if value is None else value


def _fmt(value: float) -> str:
    return f"{round(value, 1):g}"


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
