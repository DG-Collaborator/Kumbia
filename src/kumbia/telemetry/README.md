# telemetry/

Shared, persona-agnostic layer for correlating power, compute, and network
telemetry by physical asset. Added in Step 5, in response to collaborator
feedback proposing "Power & Compute Intelligence" as a first-class capability
of the broader FBA platform this backend prototypes for; extended to a third
(network) stream during the live-telemetry work, per the FBA doc's
power ↔ compute ↔ network model.

It sits alongside `rag/` (a fourth top-level module next to `core`, `rag`,
`mcp_server`) rather than inside any one persona, because the correlation it
produces is meant to be consumed by more than one future persona (a Network
Operator asking "how efficiently am I operating," a Vendor/Supplier asking
"how does my equipment perform") with the same underlying data. Nothing in
`rag` or `mcp_server` was changed to add it; the only `core` changes are
generic source improvements listed below.

## What's here

- **`canonical_record.py`** — `CanonicalRecord`, the correlation engine's
  output shape: one asset's telemetry at a point in time. Only `timestamp`,
  `asset_id`, and `source_system` are required; every reading
  (`power_watts`, `energy_kwh`, `cpu_utilization`, `gpu_utilization`,
  `memory_utilization`, `network_event_count`) and every descriptive field
  (`site`, `vendor`, `device_type`) is optional, since a given record may
  only have data from some domains.
- **`asset_registry.py`** — `AssetRegistry`, identity resolution. Maps a raw
  identity reported by one telemetry system (a hostname, a GPU UUID, a PDU
  outlet, an IP address) to a single canonical asset ID, so systems that
  each name the same physical asset differently can still be joined.
  In-memory, seeded at construction; one-directional lookup
  (`resolve(identity_type, value) -> asset_id | None`).
- **`correlation_engine.py`** — `SourceMapping` (describes how to read one
  source's generic `DataTable`: which column is the identity, which is the
  timestamp, which columns map to which `CanonicalRecord` reading fields)
  and `correlate_streams()`, a pure, deterministic function that resolves
  each row's identity through an `AssetRegistry`, floors its timestamp into
  a time bucket, and joins any number of streams sharing an
  `(asset_id, bucket)` key. Valid reading targets are derived from
  `CanonicalRecord` itself, so a new reading field needs no engine edit; a
  mapping naming an unknown field fails loudly. No I/O, no LLM. A bucket
  with data from only some streams still produces a record — nothing is
  silently dropped.
- **`pipeline.py`** — `TelemetryCorrelationPipeline`, the orchestration
  layer. Holds a list of named `TelemetryStream`s (`DataSource` +
  `SourceMapping`) plus a shared `AssetRegistry`; `correlate(queries)`
  takes one already-built `SourceQuery` per stream name, fetches all
  streams concurrently, and delegates the join to `correlate_streams()`.
  Stream names must be unique, and `queries` must name exactly the
  pipeline's streams. It does not know how to build a query for any
  particular telemetry system, and it stays LLM-free by design
  (narration/reasoning is a persona's job, not this shared layer's).
- **`stream_queries.py`** — builds each stream's `SourceQuery` for one
  half-open `[start, end)` `TimeWindow`: `sql_power_window` (sqlite, ISO
  text timestamps) and `es_network_window` (an Elasticsearch composite
  aggregation of event counts per host per bucket, filtered to registered
  IPs, refusing any window that could exceed one page of buckets). Only
  code-supplied values reach query text: windows are real datetimes, SQL
  identifiers are validated, IPs come from the registry.

Supporting pieces live elsewhere, in their existing seams rather than here,
because they're general-purpose fetch mechanisms with no telemetry- or
correlation-specific knowledge:

- **`core/source/prometheus_data_source.py`** — generic Prometheus HTTP API
  connector (same category as `ElasticsearchDataSource`).
- **`core/source/empty_data_source.py`** — a `DataSource` that always
  returns nothing; fills a stream slot whose real system isn't reachable yet
  (currently compute).
- **`core/source/elasticsearch_data_source.py`** — flattens composite
  aggregation keys into `key.<source>` columns, which is what lets the
  network stream's `key.host` / `key.ts` be read like any other column.
- **`scratch_power_seed.py`** (repo root) — throwaway script seeding
  synthetic power data into a local sqlite file for standalone verification;
  not a seam file, safe to delete/rerun.

## Why this is still a basic/minimal-scope layer

Every gap below is a deliberate scoping decision, not an oversight — each was
left out so the join mechanism itself could be proven before adding anything
that has no consumer yet.

- **Network is event counts only.** The network stream carries
  `network_event_count` (documents per host per bucket), not throughput or
  bytes — the current index has no byte fields. Metrics like GB/kWh need a
  source that reports bytes.
- **Last reading wins within a bucket.** Multiple readings from one stream
  in the same `(asset_id, bucket)` aren't averaged or summed; the last one
  processed is kept. Fine for pre-aggregated sources (Elasticsearch returns
  one row per bucket) and one-reading-per-bucket feeds, but not yet the
  time-window averaging the FBA doc describes.
- **No workload/business identity layer.** The collaborator's model treats
  "which workload/application/customer caused this consumption" as a
  *second* correlation layer built on top of the asset-level one. Kumbia's
  `CanonicalRecord` has no `Workload_ID`, `Application_ID`, `Customer_ID`,
  or `Service_Type` fields yet — this layer stops at "which physical asset,"
  not "which business activity."
- **No derived KPIs.** `correlate_streams()` produces joined raw readings
  only. None of the collaborator's suggested metrics (kWh/GPU-hour,
  watts/server idle-vs-load, compute-per-watt, stranded capacity,
  idle-power cost, etc.) are computed anywhere yet.
- **`AssetRegistry` is a seeded in-memory map**, not backed by a real
  CMDB/DCIM identity source. Every mapping used so far has been written by
  hand for testing.
- **`Confidence` and `Data_Quality`** (fields the collaborator's canonical
  record includes) were left off `CanonicalRecord` entirely for v1.
- **The compute connector is unverified against a real system.**
  `PrometheusDataSource` has only been tested against synthetic,
  hand-written Prometheus-API-shaped JSON — there's no Prometheus/
  node_exporter reachable from the machine this was built on, so the
  compute stream currently runs on `EmptyDataSource`.
- **No persona consumes this yet.** Nothing in `network_monitoring` (or any
  other persona) calls into `telemetry/`, and no MCP tools expose it. That
  wiring is planned as a distinct next step, after this layer is fully
  verified standalone.
- **No RAG-side static knowledge is wired in.** Vendor spec sheets, thermal
  design guides, and tariff documents — the kind of content the
  collaborator's feedback says belongs in RAG rather than here — aren't
  connected to anything in this layer yet, though the existing generic
  `rag/rag_pipeline.py` already has everything needed to do that once a
  persona asks for it.

In short: this layer currently proves *that* power, compute, and network
telemetry from different systems can be reliably joined by asset identity —
verified against real Elasticsearch network data. It does not yet answer any
of the actual business questions (efficiency, cost, stranded capacity,
benchmarking) the collaborator's feedback is ultimately after — that's the
work still ahead of it.
