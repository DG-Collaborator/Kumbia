"""
     Descriptive analytics over correlated telemetry — the first of the FBA
     architecture's analysis levels ("what happened?"), computed here in the
     shared layer so no persona has to re-derive it, and so an LLM only ever
     narrates figures instead of calculating them. Deterministic: no I/O, no
     LLM.

     Works on a generic DataTable of correlated records (one row per asset per
     time bucket — e.g. the collector's store read through JsonlTableSource).
     Every column with at least one numeric value, other than the timestamp
     and asset columns, is treated as a metric. Per asset it reports each
     metric's first/last value and change, its peak and low with their
     buckets, and the Pearson correlation of every metric pair over the
     buckets where both were measured — with the plain-language strength
     label decided here, not left to the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from kumbia.core.domain.data_table import DataRow, DataTable

_MIN_CORRELATION_SAMPLES = 3


@dataclass(frozen=True)
class Reading:
    timestamp: str
    value: float


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    samples: int
    first: Reading
    last: Reading
    peak: Reading
    low: Reading

    @property
    def change(self) -> float:
        return self.last.value - self.first.value

    @property
    def percent_change(self) -> float | None:
        return None if self.first.value == 0 else self.change / abs(self.first.value) * 100


@dataclass(frozen=True)
class Correlation:
    metric_a: str
    metric_b: str
    samples: int
    coefficient: float | None  # None: fewer than 3 shared buckets, or a metric that never varied

    @property
    def description(self) -> str:
        if self.coefficient is None:
            return "not measurable (too few shared buckets, or one metric never changed)"
        magnitude = abs(self.coefficient)
        if magnitude < 0.2:
            return "no meaningful relationship"
        strength = "strong" if magnitude >= 0.7 else "moderate" if magnitude >= 0.4 else "weak"
        if self.coefficient > 0:
            return f"{strength} positive: they rise and fall together"
        return f"{strength} negative: one rises as the other falls"


@dataclass(frozen=True)
class AssetSummary:
    asset_id: str
    bucket_count: int
    first_bucket: str
    last_bucket: str
    metrics: list[MetricSummary]
    correlations: list[Correlation]


def summarize(
        table: DataTable,
        timestamp_column: str = "timestamp",
        asset_column: str = "asset_id",
) -> list[AssetSummary]:
    """One AssetSummary per asset in the table, assets in sorted order."""
    metric_columns = [c.key for c in table.columns if c.key not in (timestamp_column, asset_column)]

    rows_by_asset: dict[str, list[DataRow]] = {}
    for row in table.rows:
        if row.value(timestamp_column):
            rows_by_asset.setdefault(row.value(asset_column), []).append(row)

    summaries: list[AssetSummary] = []
    for asset_id in sorted(rows_by_asset):
        rows = rows_by_asset[asset_id]
        buckets = sorted({row.value(timestamp_column) for row in rows})

        series: dict[str, list[Reading]] = {}
        for column in metric_columns:
            readings = sorted(
                (Reading(row.value(timestamp_column), value)
                 for row in rows
                 if (value := _to_float(row.value(column))) is not None),
                key=lambda reading: reading.timestamp,
            )
            if readings:
                series[column] = readings

        summaries.append(AssetSummary(
            asset_id=asset_id,
            bucket_count=len(buckets),
            first_bucket=buckets[0],
            last_bucket=buckets[-1],
            metrics=[_metric_summary(metric, readings) for metric, readings in series.items()],
            correlations=[_correlation(a, series[a], b, series[b]) for a, b in combinations(series, 2)],
        ))
    return summaries
def summary_text(table: DataTable) -> str:
    """The figures as plain text — what a model is given to narrate, and what the user sees alongside it."""
    summaries = summarize(table)
    if not summaries:
        return "No correlated telemetry records to summarize."
    return "\n\n".join(_asset_text(summary) for summary in summaries)


def _metric_summary(metric: str, readings: list[Reading]) -> MetricSummary:
    return MetricSummary(
        metric=metric,
        samples=len(readings),
        first=readings[0],
        last=readings[-1],
        peak=max(readings, key=lambda reading: reading.value),
        low=min(readings, key=lambda reading: reading.value),
    )


def _correlation(metric_a: str, readings_a: list[Reading], metric_b: str, readings_b: list[Reading]) -> Correlation:
    """Pearson r over the buckets where both metrics have a reading."""
    values_a = {reading.timestamp: reading.value for reading in readings_a}
    values_b = {reading.timestamp: reading.value for reading in readings_b}
    shared = sorted(values_a.keys() & values_b.keys())
    coefficient = None
    if len(shared) >= _MIN_CORRELATION_SAMPLES:
        coefficient = _pearson([values_a[t] for t in shared], [values_b[t] for t in shared])
    if len(shared) >= _MIN_CORRELATION_SAMPLES:
        coefficient = _pearson([values_a[t] for t in shared], [values_b[t] for t in shared])
    return Correlation(metric_a, metric_b, len(shared), coefficient)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    spread_y = sum((y - mean_y) ** 2 for y in ys)
    if spread_x == 0 or spread_y == 0:
        return None
    return covariance / math.sqrt(spread_x * spread_y)


def _asset_text(summary: AssetSummary) -> str:
    lines = [
        f"Asset {summary.asset_id}: {summary.bucket_count} time bucket(s), "
        f"first {summary.first_bucket}, last {summary.last_bucket}"
    ]
    for m in summary.metrics:
        percent = f" ({m.percent_change:+.1f}%)" if m.percent_change is not None else ""
        lines.append(
            f"- {m.metric} ({m.samples} reading(s)): started at {_fmt(m.first.value)} ({m.first.timestamp}), "
            f"ended at {_fmt(m.last.value)} ({m.last.timestamp}), change {m.change:+g}{percent}; "
            f"peak {_fmt(m.peak.value)} at {m.peak.timestamp}; low {_fmt(m.low.value)} at {m.low.timestamp}"
        )
    for c in summary.correlations:
        r = f"r = {c.coefficient:.2f} " if c.coefficient is not None else ""
        lines.append(
            f"- {c.metric_a} vs {c.metric_b}: {r}over {c.samples} shared bucket(s) — {c.description}"
        )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{round(value, 2):g}"


def _to_float(value: str) -> float | None:
    """Cell values are pre-stringified text; '' (a missing cell) and anything non-numeric become None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None