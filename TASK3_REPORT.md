# Task 3 report: persistence and memory lifecycle

Date: 2026-09-09

## Outcome

The built-in in-memory runtime now keeps its real checkpoint and Store
`PersistentDict` instances on the periodic flush loop, rejects duplicate live
owners for a filename, and prevents serializer-only checkpointer views from
creating or closing competing files. Run/thread deletion and stream shutdown
release their owned stream state. Built-in per-run checkpoint deletion now
removes writes and unreferenced blobs and keeps accounting consistent.

The package version is `0.33.3.post4`.

## Root causes and fixes

1. Periodic persistence

   - Root cause: `_flush_loop` used `if store := weakref()`. An empty
     `PersistentDict` is false, so a live store was classified as dead and
     silently removed from the registry.
   - Root cause: every `Checkpointer(unpack_hook=...)` constructed three new
     persistent dictionaries at the singleton filenames, overwrote their
     weakrefs, then closed/flushed the temporary empty dictionaries through its
     `ExitStack`.
   - Root cause: registration, dead-reference removal, final close, and flush
     were unsynchronized. A late old-instance close could overwrite a newer
     same-filename owner.
   - Fix: use explicit `is None` liveness checks; serialize the registry and
     sync operations with an `RLock`; retain the first live owner; permit dead
     or explicitly closed owners to be replaced; isolate sync exceptions; and
     atomically unregister/final-sync owners. A denied duplicate is cleared
     without touching the owner's file.
   - Fix: serializer-specific saver instances now use non-persistent temporary
     containers and only alias the singleton's dictionaries. Closing one no
     longer stops the process-wide flush loop or closes singleton state.

2. Stream lifecycle

   - Root cause: removal methods left empty outer dictionaries, thread stream
     subscriptions used the run-queue cleanup path, read helpers allocated
     entries through `defaultdict`, and neither run/thread deletion nor global
     shutdown covered all five state maps.
   - Fix: add targeted `delete_run`, `delete_thread`, and complete `close`
     operations for `queues`, `control_queues`, `control_keys`,
     `message_stores`, and `thread_streams`. Queue removal now prunes empty
     parents, read/publish paths avoid accidental empty allocations, and
     subscribers receive a terminal signal before manager references are
     dropped.
   - Fix: active run contexts are tracked only for the duration of execution.
     If a run is deleted concurrently, a temporary pending-delete guard drops
     its late final publications and performs a final cleanup when the context
     exits. Global shutdown permanently closes the old manager, so old tasks
     cannot repopulate it.
   - Fix: cancellation selected by assistant/status now publishes under each
     run's actual thread ID instead of the optional request-level thread ID.

3. Per-run checkpoint deletion

   - Root cause: the built-in path manually removed only matching checkpoint
     rows. It leaked pending writes and channel blobs, and `Runs.delete` invoked
     that destructive cleanup before confirming the requested run existed.
   - Fix: implement sync/async `delete_for_runs`; scan every thread/namespace
     by metadata `run_id`; delete matching ordinary checkpoint rows and writes;
     delete only blobs no retained checkpoint references; remove empty
     containers; and rebuild aggregate size/count tracking from retained data.
   - Fix: validate and authorize the run before cleanup, then invoke the saver
     and targeted StreamManager cleanup. Unknown run IDs are now true no-ops at
     the saver layer and a non-destructive 404 at the runtime operation layer.
   - Fix: DeltaChannel protection is activated only by its dedicated
     `counters_since_delta_snapshot` metadata. Targeted ancestors required to
     reconstruct a surviving delta checkpoint remain intact through the
     nearest materialized snapshot; ordinary missing blobs do not cause broad
     retention.
   - Fix: empty pending-write groups created as a read side effect count as
     zero bytes, keeping rebuild accounting stable.

4. Store durability

   - Root cause: Store persistence depended on the broken shared registry, so
     data/vector/TTL dictionaries could be dropped from periodic flushing; its
     close path also lacked identity-safe registry removal.
   - Fix: Store data, vector, and TTL dictionaries use the repaired registry
     and identity-safe close path. Periodic snapshots now include item writes,
     generated vectors, TTL refreshes, TTL removal, and item/vector deletion.
     The global operations store and retry counter use the same close ordering.

## Changed files

- `src/langgraph_runtime_inmem/_persistence.py`
- `src/langgraph_runtime_inmem/checkpoint.py`
- `src/langgraph_runtime_inmem/inmem_stream.py`
- `src/langgraph_runtime_inmem/ops.py`
- `src/langgraph_runtime_inmem/store.py`
- `src/langgraph_runtime_inmem/database.py`
- `src/langgraph_runtime_inmem/__init__.py`
- `tests/test_persistence.py`
- `tests/test_inmem_stream.py`
- `tests/test_thread_ttl.py`
- `pyproject.toml`
- `README.md`
- `TEST_REPORT.md` (marks the previous `post3` report as historical)
- `TASK3_REPORT.md`

## Regression coverage

New tests cover:

- a live empty dictionary remaining registered and later flushing a write;
- live-owner rejection, dead-owner replacement, non-owner close safety, and
  registration during a blocked flush;
- transient checkpointer views preserving singleton dictionaries, registry
  weakrefs, files, and the live flush thread;
- periodic Store snapshots and restart reads for data, embeddings, TTL refresh,
  and deletion;
- non-allocating stream lookups; per-run isolation; per-thread cleanup;
  subscription removal; active-delete finalization; and complete shutdown;
- manual and TTL-driven operation wiring into stream cleanup;
- run validation before destructive cleanup;
- writes/blob release, shared-blob retention, exact accounting, namespace
  handling, and conservative DeltaChannel ancestor preservation.

Validation results:

```text
pytest -q -W error
59 passed, 1 skipped

LangGraph checkpoint-conformance delete_for_runs suite
7 passed, 0 failed

ruff check src tests
All checks passed!

python -m compileall -q src
passed

git diff --check
passed

wheel-isolated pytest -q -W error
version=0.33.3.post4
loaded_from=/tmp/.../site-packages/langgraph_runtime_inmem/__init__.py
59 passed, 1 skipped
```

The skipped test is the opt-in external PostgreSQL restart test; the in-memory
unit and integration suites ran. The conformance cases came from LangGraph's
official `checkpoint-conformance` package version `0.0.2`.

The built artifact is
`release/langgraph_runtime_inmem-0.33.3.post4-py3-none-any.whl` with SHA-256:

```text
7b1c96d09ae9274d65cd5e7173d55fb3070c92936727573e26109f1b3db6f239
```

## Intentional remaining behavior and risks

- If a deleted run owns an ancestor that a surviving DeltaChannel checkpoint
  still needs, that ancestor checkpoint, its writes, and its snapshot blob are
  deliberately retained. They become reclaimable when the dependent survivor
  is deleted or safely pruned. Dropping them immediately would silently change
  reconstructed state.
- Persistence remains a development, single-process pickle backend. Registry
  ownership is coordinated inside one process; it is not a cross-process file
  lock.
- The three checkpoint files and three Store files are each atomically
  replaced, but a process crash between individual file syncs is not a
  multi-file transaction. The patch preserves the runtime's existing periodic
  flush model rather than redesigning its storage format.
- Releasing Python objects does not guarantee an immediate RSS decrease because
  the Python allocator may retain arenas.
