# langgraph-runtime-inmem 0.33.3 TTL, persistence, and lifecycle fixes

> **Modified distribution:** this is not an official LangChain release. See
> [NOTICE.md](NOTICE.md), [LICENSE](LICENSE), and the source-only Docker transfer
> instructions in [TRANSFER.md](TRANSFER.md). The final wheel is under `wheel/`
> and the delta from the local post4 baseline is under `patches/`.

## Local PostgreSQL review candidate (2026-09-19)

The local version `0.33.3.post4+review.20260921` adds wheel-packaged PostgreSQL
hooks, bounded reconnecting pools, numeric Store range filters, fractional TTL
refresh, run checkpoint deletion and delta-safe `keep_latest` pruning. Built-in
Store snapshots are loaded only when that Store is actually used.

After installing this wheel with its `[postgres]` extra, a user graph can use:

```json
{
  "dependencies": ["langgraph-runtime-inmem"],
  "graphs": {"agent": "./graph.py:graph"},
  "checkpointer": {
    "path": "langgraph_runtime_inmem.postgres.checkpointer",
    "ttl": {"strategy": "keep_latest", "default_ttl": 43200, "sweep_interval_minutes": 1}
  },
  "store": {"path": "langgraph_runtime_inmem.postgres.store"}
}
```

Set `DATABASE_URL`, or separate `STORE_DATABASE_URL` and
`CHECKPOINT_DATABASE_URL`. The Store TTL defaults and sweep interval are still
controlled by the `STORE_TTL_*` environment variables described below. The hooks
perform schema setup and a one-time `store.ttl_minutes` conversion to double
precision, so their startup account must own the schema. This does not restore
fractional values already rounded by the old integer column.

These hooks were verified with `langgraph-api==0.13.3`,
`langgraph-checkpoint-postgres==3.1.2`, and PostgreSQL 17. Each hook lifecycle has
a pool of 1–8 connections with a 5-second acquisition timeout and at most 64
waiters. Total connections multiply with event loops and replicas.

This is **not a durable production Agent Server**: thread/run/assistant metadata
still uses local in-memory runtime snapshots, flushed roughly every 10 seconds.
An acknowledged new thread was observed to return HTTP 404 after immediate
SIGKILL even though its three checkpoints and Store item survived in PostgreSQL.
Graceful restarts and older flushed threads were recovered successfully. Use a
durable runtime metadata backend before requiring crash-safe acknowledgements
or multiple replicas; retain `.langgraph_api` for this single-replica dev runtime.

Pruning is transactional, preserves shared blobs and required DeltaChannel
ancestors, and conservatively serializes maintenance using a table-level lock.
It scans checkpoint metadata for each affected thread. Large histories or
high-write deployments need a different maintenance strategy and further load
testing. Application side effects in Store or external services are not undone
by checkpoint rollback.

The original post4 documentation follows; its historical version/test claims
describe the starting artifact, not certification of this review candidate.

This is a source-compatible patch for `langgraph-runtime-inmem==0.33.3`.  The
wheel version is `0.33.3.post4` so pip can distinguish it from the unpatched
upstream wheel.

## What is implemented

- Persists per-thread TTL overrides and applies the global
  `checkpointer.ttl.default_ttl` to every thread, including threads implicitly
  created by a run.
- Implements `Threads.sweep_ttl()` for both `delete` and `keep_latest`.
- Implements `InMemorySaver.prune()/aprune()` and preserves the ancestor chain
  required to reconstruct `DeltaChannel` state.
- Restores a background TTL loop to the in-memory runtime lifespan, including
  the `Starting thread TTL sweeper ...` and sweep-completion logs.
- Makes thread deletion release checkpoint storage, pending writes, and blobs;
  the upstream 0.33.3 deletion path can leave blobs allocated.
- Supports `GET /threads/{thread_id}?include=ttl` for the in-memory backend.
- Does not prune a thread while it has a pending or running run.
- Implements the standard LangGraph `BaseStore` item TTL contract, including
  `store.ttl.default_ttl`, per-item `ttl` overrides, and disabling expiration
  with `ttl=None`.
- Refreshes item expiration on `get` and `search` according to
  `refresh_on_read` and each operation's `refresh_ttl` override.
- Persists item TTL metadata alongside the disk-backed data and vector files,
  and deletes the item, its vectors, and its TTL metadata in one sweep.
- Tracks estimated serialized bytes when Store/checkpoint data is written,
  overwritten, copied, pruned, or deleted, then emits size metrics for both
  sweepers.
- Runs the store sweeper in the background and emits startup and completion
  logs on every sweep, including sweeps that remove zero items.
- Periodically flushes live checkpoint and Store dictionaries even while they
  are empty, with serialized registry ownership and safe same-filename close.
- Keeps serializer-specific checkpointer views from replacing or closing the
  singleton checkpoint files.
- Releases all run/thread stream queues, replay buffers, and control state on
  run deletion, thread deletion/TTL expiry, and global stream shutdown.
- Implements `delete_for_runs` for the built-in saver, releasing checkpoint
  rows, writes, and unreferenced blobs while preserving shared blobs and
  DeltaChannel ancestors required by surviving checkpoints.

For `keep_latest`, an expired inactive thread is pruned once.  It becomes
eligible again only after its `updated_at` advances and another full TTL period
passes.  This prevents the same retained checkpoint from being processed every
sweep interval.

## Install

```bash
pip install --force-reinstall --no-deps \
  release/langgraph_runtime_inmem-0.33.3.post4-py3-none-any.whl
```

Verify that `langgraph dev` uses the patched runtime:

```bash
python -c "import langgraph_runtime_inmem as m; print(m.__version__)"
```

Expected output: `0.33.3.post4`.

The primary tested dependency set is:

```text
langgraph==1.2.11
langgraph-api==0.13.3
langgraph-cli[inmem]==0.4.31
langgraph-checkpoint==4.2.0
langgraph-runtime-inmem==0.33.3.post4
```

All tests also pass with `langgraph-api==0.13.2`, so the patch can be installed
into that environment without requiring an API upgrade.

## Configuration

### PostgreSQL-backed graph persistence with the in-memory runtime

The example keeps the API runtime in memory while persisting graph checkpoints
and Store key/value data in PostgreSQL. Install the optional dependencies,
start PostgreSQL, and copy the environment template:

```bash
pip install -e '.[postgres]'
docker compose -f example/docker-compose.postgres.yml up -d
cp example/.env.example example/.env
langgraph dev --config example/langgraph.json
```

`example/postgres_checkpointer.py` and `example/postgres_store.py` expose async
context managers for the LangGraph API custom persistence hooks. They use
`DATABASE_URL` by default; `CHECKPOINT_DATABASE_URL` and `STORE_DATABASE_URL`
can override it when separate databases are desired. Both initialize their
PostgreSQL schemas with `setup()` and close their connections on API shutdown.
The graph remains compiled without a checkpointer so the API can inject the
configured saver.

The custom Store retains TTL. Its default TTL, refresh-on-read behavior, and
sweep interval come from `STORE_TTL_DEFAULT_MINUTES`,
`STORE_TTL_REFRESH_ON_READ`, and `STORE_TTL_SWEEP_INTERVAL_MINUTES`; its own
sweeper starts and stops with the custom Store lifecycle. The matching values
remain visible in `langgraph.json` as deployment documentation.

Thread and run metadata, assistants, crons, and the queue remain owned by the
in-memory runtime and its local `.langgraph_api` persistence. Run a single
runtime replica and preserve that directory if those records must survive a
container replacement.

To return to built-in in-memory persistence, remove `checkpointer.path` and
`store.path` from `example/langgraph.json` (and remove the PostgreSQL packages
from its `dependencies` list). The existing TTL settings can remain.

An opt-in integration test verifies Store persistence across two independent
connection lifecycles, which models an API process restart:

```bash
TEST_POSTGRES_DATABASE_URL="$DATABASE_URL" \
  pytest -q tests/integration/test_postgres_store_restart.py
```

The behavior follows LangGraph's official
[store item TTL configuration](https://docs.langchain.com/langsmith/configure-ttl#configuring-store-item-ttl).
The requested `langgraph.json` configuration works unchanged and can be
combined with Store TTL:

```json
{
  "checkpointer": {
    "ttl": {
      "strategy": "keep_latest",
      "sweep_interval_minutes": 1,
      "default_ttl": 43200
    }
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080
    }
  }
}
```

All TTL values are in minutes, so `43200` is 30 days and `10080` is 7 days.
Both sweepers first run after one full sweep interval, and startup logs confirm
that the loops are active.  Store reads refresh an item's expiration by default;
pass `refresh_ttl=False` to `get` or `search` to suppress that refresh.  Passing
`ttl` to `put` overrides the default for that item, while `ttl=None` makes it
non-expiring.  Enabling a default does not retroactively add TTL metadata to
items that already exist.

With production-like TTL values, newly active data is intentionally not removed
during a short smoke test.  Use a per-thread or per-item TTL of `0` in a test
request, or run the included unit tests, to exercise expiration immediately.

## Sweep size instrumentation

Every completed Store and checkpoint sweep logs both human-readable sizes and
raw byte estimates. Example fields:

```text
Store TTL sweep completed
deleted_items=1824
deleted_size=143.7 MB
deleted_size_bytes=150680371
total_before=512.4 MB
total_before_bytes=537290342
deleted_ratio=28.0%
deleted_ratio_percent=28.0
```

The checkpoint log uses the same fields and also includes
`threads_processed` and `threads_deleted`:

```text
Checkpoint TTL sweep completed
deleted_items=1824
threads_processed=120
threads_deleted=25
deleted_size=143.7 MB
total_before=512.4 MB
deleted_ratio=28.0%
```

These values estimate serialized payload size; they are not a direct RSS or
Python heap measurement. Store items and vectors are measured with
`pickle.dumps(...)` only when they are written. Checkpoints, pending writes,
and channel blobs reuse lengths from the runtime's already-serialized payloads.
Overwrites and deletes update a small in-memory tracker, and persisted data is
scanned once when the runtime starts. At the sweep boundary, totals and deleted
metrics use tracked per-item/per-thread integer lookups. `keep_latest` already
walks the checkpoint entries it removes; its accounting only adds constant-time
`len(...)` calls on their serialized buffers and never serializes them again.
Deleting the underlying expired records still costs time proportional to the
records actually removed. Python's allocator may retain freed arenas, so
process RSS does not necessarily fall by the logged amount immediately.

## Test

From this directory, with the LangGraph dev dependencies installed:

```bash
pytest -q
```

The tests cover thread TTL visibility and overrides, cascade deletion, blob
release, `keep_latest`, DeltaChannel safety, repeat-sweep prevention, sweep
limits, active-run protection, and the thread background loop. They also cover
empty-dictionary periodic flushes, registry races and ownership, transient
checkpointer views, complete stream cleanup, targeted run cleanup, shared blob
retention, upstream `delete_for_runs` conformance, and live Store persistence of
data, vectors, and TTL changes. Store TTL tests continue to cover default and
per-item TTLs, read/search refresh, write reset, non-expiring items,
item/vector cleanup, non-retroactive defaults, background sweeping,
disk-backed restart persistence, size accounting/logging, and the Agent Server
store wrapper.

The in-memory checkpoint integration tests run a real graph and background
sweeper to verify `delete`, `keep_latest`, and protection for threads with
pending or running runs. They run without PostgreSQL:

```bash
pytest -q tests/integration/test_inmem_checkpoint_sweeper.py
```

The in-memory Store integration tests inject the batched `Store()` into a real
graph and run the background TTL sweeper. They cover default and per-item TTLs,
non-expiring items, vector cleanup, get/search refresh, write reset, and sweeper
stop/restart. Time-dependent cases control the expiry clock while the sweeper
uses its real timer. No PostgreSQL or external embedding service is required:

```bash
pytest -q tests/integration/test_inmem_store_sweeper.py
```
