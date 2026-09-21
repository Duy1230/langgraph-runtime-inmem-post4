# Test report

> Historical Task 2 (`0.33.3.post3`) verification. For the current Task 3
> persistence/lifecycle patch (`0.33.3.post4`), see `TASK3_REPORT.md`.

## Environment

- Python 3.12
- `langgraph==1.2.11`
- `langgraph-api==0.13.3`
- `langgraph-cli[inmem]==0.4.31`
- `langgraph-checkpoint==4.2.0`
- Patched runtime: `langgraph-runtime-inmem==0.33.3.post3`

An additional compatibility run replaced `langgraph-api==0.13.3` with
`langgraph-api==0.13.2`; all 26 tests passed there as well.

## Automated tests

```text
..........................                                               [100%]
26 passed in 0.38s
```

Compatibility run:

```text
langgraph-api=0.13.2
langgraph-runtime-inmem=0.33.3.post3
26 passed in 0.64s
```

The eleven thread/checkpointer TTL tests cover:

1. Per-thread TTL overrides and `include=ttl` visibility.
2. Cascade deletion of threads, runs, crons, checkpoint storage, writes,
   blobs, and TTL metadata.
3. Global `keep_latest` pruning and rearming after later activity.
4. DeltaChannel ancestor-chain preservation while obsolete forks are removed.
5. Sweep limits and protection for pending/running runs.
6. Manual `keep_latest` pruning.
7. Serialized-size accounting across writes, overwrites, per-run deletion,
   full-thread deletion, pruning, persisted-data rebuild, and thread copies.
8. Checkpoint sweep log fields, including integer aggregation of the tracked
   per-thread totals without reserializing payloads.
9. Background-loop interval and sweep-limit propagation.

The fifteen Store TTL test cases cover:

1. `default_ttl`, per-item overrides, and `ttl=None`.
2. Default and per-operation `get` refresh behavior.
3. `search` refresh limited to returned items.
4. TTL reset on writes and metadata removal on deletes.
5. Non-retroactive defaults and per-item TTL without a global default.
6. Atomic removal of item data, vectors, and TTL metadata.
7. Background sweeper startup, deduplication, deletion, and shutdown.
8. TTL metadata persistence across a disk-backed restart.
9. Propagation of TTL configuration through the Agent Server `BatchedStore`.
10. No-op behavior without TTL configuration and validation of sweep intervals.
11. Pickle-based size accounting on writes and overwrites, integer-only size
    lookup during deletion, structured size logs, and zero-deletion sweeps.

`pytest -W error`, Python bytecode compilation, targeted Ruff checks, Ruff
format checks, and `git diff --check` all passed. Files inherited from upstream
outside this change still contain pre-existing warnings under Ruff 0.16.6, so
lint checks were scoped to the changed code and relevant rule sets.

## Wheel verification

The final wheel was force-installed into the test environment, replacing the
editable source. Tests were then run from `/tmp` with a separate pytest root so
the repository's `pythonpath = ["src"]` setting could not shadow the wheel.

```text
version=0.33.3.post3
loaded_from=.../site-packages/langgraph_runtime_inmem/__init__.py
26 passed in 0.41s
```

Wheel SHA-256:

```text
12bd64786a3b4e96084ce8de5c7cb2d01620fb194347ed0f4a9c7f84449079b8
```

## `langgraph dev` startup smoke test

`langgraph dev --no-browser --no-reload --port 8123` reached application-ready
state with the combined thread and Store TTL example configuration. Relevant
startup logs:

```text
Starting In-Memory runtime with langgraph-api=0.13.3 and in-memory runtime=0.33.3.post3
Starting store TTL sweeper with interval 1 minutes
Starting thread TTL sweeper with interval 1 minutes strategy=keep_latest
Application started up
```

The deterministic automated tests use a controlled clock and immediate expiry
to verify actual deletion without waiting for production-length TTL windows.
They also assert the Store and checkpoint completion events and all requested
size fields.
