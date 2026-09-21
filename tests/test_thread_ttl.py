from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from starlette.exceptions import HTTPException

from langgraph_runtime_inmem import checkpoint as checkpoint_module
from langgraph_runtime_inmem import ops
from langgraph_runtime_inmem.checkpoint import InMemorySaver
from langgraph_runtime_inmem.inmem_stream import Message, StreamManager


class FakeConnection:
    def __init__(self) -> None:
        self.store = {
            "threads": [],
            "runs": [],
            "crons": [],
            "assistants": [],
            "assistant_versions": [],
            "thread_ttls": {},
        }

    @asynccontextmanager
    async def pipeline(self):
        yield None


@pytest.mark.parametrize("kind", ["threads", "runs"])
async def test_concurrent_deletions_preserve_unrelated_records(conn, monkeypatch, kind):
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)
    threads = [await create_thread(conn) for _ in range(3)]
    runs = [{"run_id": uuid4(), "thread_id": t["thread_id"], "status": "success"} for t in threads]
    conn.store["runs"].extend(runs)
    arrived = 0
    barrier = asyncio.Event()

    async def cleanup(*args, **kwargs):
        nonlocal arrived
        arrived += 1
        ordinal = arrived
        if arrived == 2:
            barrier.set()
        await barrier.wait()
        if ordinal == 2:
            await asyncio.sleep(0.02)

    monkeypatch.setattr(ops, "_delete_checkpoints_for_thread", cleanup)
    async def delete(i):
        if kind == "threads":
            result = await ops.Threads.delete(conn, threads[i]["thread_id"])
            assert await anext(result) == threads[i]["thread_id"]
        else:
            result = await ops.Runs.delete(conn, runs[i]["run_id"], thread_id=threads[i]["thread_id"])
            assert await anext(result) == runs[i]["run_id"]
    await asyncio.wait_for(asyncio.gather(delete(0), delete(1)), timeout=2)
    assert conn.store["runs"] == [runs[2]]
    assert conn.store["threads"] == ([threads[2]] if kind == "threads" else threads)


async def test_checkpoint_delete_failure_preserves_thread_metadata(conn, monkeypatch):
    thread = await create_thread(conn)
    run = {"run_id": uuid4(), "thread_id": thread["thread_id"], "status": "success"}
    conn.store["runs"].append(run)
    cron = {"thread_id": str(thread["thread_id"])}
    conn.store["crons"].append(cron)
    async def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(ops, "_delete_checkpoints_for_thread", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        await ops.Threads.delete(conn, thread["thread_id"])
    assert conn.store["threads"] == [thread]
    assert conn.store["runs"] == [run]
    assert conn.store["crons"] == [cron]


@pytest.fixture
def conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> InMemorySaver:
    # Avoid disk persistence in unit tests while exercising the patched saver.
    monkeypatch.setattr(checkpoint_module, "DISABLE_FILE_PERSISTENCE", True)
    instance = InMemorySaver()
    monkeypatch.setattr(ops, "Checkpointer", lambda *args, **kwargs: instance)
    return instance


@pytest.fixture(autouse=True)
def reset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph_api import config as api_config

    monkeypatch.setattr(api_config, "THREAD_TTL", None)
    monkeypatch.setattr(api_config, "USE_CUSTOM_CHECKPOINTER", False)


def add_checkpoint(
    saver: InMemorySaver,
    thread_id: UUID,
    checkpoint_id: str,
    *,
    parent_id: str | None = None,
    namespace: str = "",
    value: object = None,
    materialized: bool = True,
    run_id: UUID | None = None,
    metadata_extra: dict | None = None,
) -> None:
    config = {
        "configurable": {
            "thread_id": str(thread_id),
            "checkpoint_ns": namespace,
            **({"checkpoint_id": parent_id} if parent_id else {}),
        }
    }
    channel_values = {"state": value} if materialized else {}
    saver.put(
        config,
        {
            "v": 1,
            "ts": datetime.now(UTC).isoformat(),
            "id": checkpoint_id,
            "channel_values": channel_values,
            "channel_versions": {"state": checkpoint_id},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": ["state"],
        },
        {
            "source": "loop",
            "step": int(checkpoint_id[:4]),
            **({"run_id": str(run_id)} if run_id else {}),
            **(metadata_extra or {}),
        },
        {"state": checkpoint_id},
    )
    saver.put_writes(
        {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": namespace,
                "checkpoint_id": checkpoint_id,
            }
        },
        [("state", value)],
        f"task-{checkpoint_id}",
    )


async def create_thread(
    conn: FakeConnection,
    *,
    thread_id: UUID | None = None,
    ttl: dict | None = None,
    age_minutes: float = 0,
):
    thread_id = thread_id or uuid4()
    iterator = await ops.Threads.put(
        conn,
        thread_id,
        metadata={},
        if_exists="raise",
        ttl=ttl,
    )
    thread = await anext(iterator)
    if age_minutes:
        old = datetime.now(UTC) - timedelta(minutes=age_minutes)
        thread["created_at"] = old
        thread["updated_at"] = old
        thread["state_updated_at"] = old
    return thread


def test_checkpoint_size_tracking_handles_overwrite_and_delete(saver) -> None:
    thread_id = uuid4()
    saver.put_writes(
        {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": "empty",
            }
        },
        [],
        "task-empty",
    )
    assert not saver.writes
    assert saver.estimated_size_bytes == 0

    add_checkpoint(saver, thread_id, "0001", value="small")
    first_size = saver.estimated_size_bytes

    add_checkpoint(saver, thread_id, "0001", value="x" * 4096)
    assert saver.estimated_checkpoint_count == 1
    assert saver.estimated_size_bytes > first_size

    saver.delete_thread(str(thread_id))
    assert saver.estimated_checkpoint_count == 0
    assert saver.estimated_size_bytes == 0


def test_checkpoint_size_tracking_can_be_rebuilt(saver) -> None:
    thread_id = uuid4()
    add_checkpoint(saver, thread_id, "0001", value={"payload": "x" * 1024})
    expected_size = saver.estimated_size_bytes

    saver._size_state.clear()
    assert saver.estimated_size_bytes == 0
    saver._rebuild_size_tracking()

    assert saver.estimated_size_bytes == expected_size
    assert saver.estimated_checkpoint_count == 1


@pytest.mark.asyncio
async def test_thread_copy_tracks_copied_checkpoint_size(conn, saver) -> None:
    thread = await create_thread(conn)
    source_id = thread["thread_id"]
    add_checkpoint(saver, source_id, "0001", value={"payload": "x" * 1024})
    source_size = saver.estimated_size_for_thread(source_id)

    copied_iterator = await ops.Threads.copy(conn, source_id)
    copied_thread = await anext(copied_iterator)
    copied_id = copied_thread["thread_id"]

    assert saver.estimated_size_for_thread(copied_id) == source_size
    assert saver.estimated_checkpoint_count_for_thread(copied_id) == 1
    assert saver.estimated_size_bytes == source_size * 2


@pytest.mark.asyncio
async def test_run_checkpoint_delete_updates_size_tracking(conn, saver) -> None:
    thread_id = uuid4()
    deleted_run_id = uuid4()
    add_checkpoint(
        saver,
        thread_id,
        "0001",
        value="deleted run",
        run_id=deleted_run_id,
    )
    add_checkpoint(saver, thread_id, "0002", value="retained run")
    total_before = saver.estimated_size_bytes

    await ops._delete_checkpoints_for_thread(thread_id, conn, deleted_run_id)

    assert set(saver.storage[str(thread_id)][""]) == {"0002"}
    assert {key[2] for key in saver.writes} == {"0002"}
    assert {key[3] for key in saver.blobs} == {"0002"}
    assert saver.estimated_checkpoint_count == 1
    assert 0 < saver.estimated_size_bytes < total_before


def test_run_checkpoint_delete_preserves_blob_referenced_by_survivor(saver) -> None:
    thread_id = uuid4()
    deleted_run_id = uuid4()
    survivor_run_id = uuid4()
    add_checkpoint(
        saver,
        thread_id,
        "0001",
        value={"shared": True},
        run_id=deleted_run_id,
    )

    # The survivor advances the checkpoint ID but intentionally reuses the
    # existing channel version/blob.
    saver.put(
        {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": "0001",
            }
        },
        {
            "v": 1,
            "ts": datetime.now(UTC).isoformat(),
            "id": "0002",
            "channel_values": {},
            "channel_versions": {"state": "0001"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": [],
        },
        {"source": "loop", "step": 2, "run_id": str(survivor_run_id)},
        {},
    )
    shared_blob_key = (str(thread_id), "", "state", "0001")

    saver.delete_for_runs([str(deleted_run_id)])

    assert set(saver.storage[str(thread_id)][""]) == {"0002"}
    assert (str(thread_id), "", "0001") not in saver.writes
    assert shared_blob_key in saver.blobs
    survivor = saver.get_tuple(
        {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": "0002",
            }
        }
    )
    assert survivor is not None
    assert survivor.checkpoint["channel_values"]["state"] == {"shared": True}
    assert saver.estimated_checkpoint_count == 1
    tracked_size = saver.estimated_size_bytes
    saver._rebuild_size_tracking()
    assert saver.estimated_size_bytes == tracked_size


def test_run_checkpoint_delete_retains_required_delta_ancestor(saver) -> None:
    thread_id = uuid4()
    ancestor_run_id = uuid4()
    survivor_run_id = uuid4()
    add_checkpoint(
        saver,
        thread_id,
        "0001",
        value=_DeltaSnapshot(["seed"]),
        run_id=ancestor_run_id,
    )
    add_checkpoint(
        saver,
        thread_id,
        "0002",
        parent_id="0001",
        value=["delta"],
        materialized=False,
        run_id=survivor_run_id,
        metadata_extra={"counters_since_delta_snapshot": {"state": (1, 1)}},
    )

    saver.delete_for_runs([str(ancestor_run_id)])

    # Generic checkpoint storage cannot synthesize a new reducer snapshot, so
    # preserving this targeted ancestor is the only lossless option.
    assert set(saver.storage[str(thread_id)][""]) == {"0001", "0002"}
    history = saver.get_delta_channel_history(
        config={
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": "0002",
            }
        },
        channels=["state"],
    )
    assert history["state"]["seed"] == _DeltaSnapshot(["seed"])
    assert history["state"]["writes"]


@pytest.mark.asyncio
async def test_runs_delete_validates_before_cleanup_and_cleans_streams(
    conn, saver, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)
    thread = await create_thread(conn)
    thread_id = thread["thread_id"]
    deleted_run_id = uuid4()
    sibling_run_id = uuid4()
    missing_run_id = uuid4()
    conn.store["runs"].extend(
        [
            {
                "run_id": deleted_run_id,
                "thread_id": thread_id,
                "status": "success",
            },
            {
                "run_id": sibling_run_id,
                "thread_id": thread_id,
                "status": "success",
            },
        ]
    )
    add_checkpoint(
        saver, thread_id, "0001", value="delete", run_id=deleted_run_id
    )
    add_checkpoint(
        saver, thread_id, "0002", value="keep", run_id=sibling_run_id
    )
    await manager.add_queue(deleted_run_id, thread_id)
    sibling_queue = await manager.add_queue(sibling_run_id, thread_id)
    await manager.put(
        deleted_run_id,
        thread_id,
        Message(topic=f"run:{deleted_run_id}:stream".encode(), data=b"delete"),
        resumable=True,
    )
    await manager.put(
        sibling_run_id,
        thread_id,
        Message(topic=f"run:{sibling_run_id}:stream".encode(), data=b"keep"),
        resumable=True,
    )

    before_missing_delete = dict(saver.storage[str(thread_id)][""])
    with pytest.raises(HTTPException) as exc_info:
        await ops.Runs.delete(conn, missing_run_id, thread_id=thread_id)
    assert exc_info.value.status_code == 404
    assert dict(saver.storage[str(thread_id)][""]) == before_missing_delete

    deleted = await ops.Runs.delete(conn, deleted_run_id, thread_id=thread_id)
    assert await anext(deleted) == deleted_run_id

    assert {run["run_id"] for run in conn.store["runs"]} == {sibling_run_id}
    assert set(saver.storage[str(thread_id)][""]) == {"0002"}
    assert deleted_run_id not in manager.queues.get(thread_id, {})
    assert deleted_run_id not in manager.message_stores.get(thread_id, {})
    assert manager.queues[thread_id][sibling_run_id] == [sibling_queue]
    assert sibling_run_id in manager.message_stores[thread_id]


@pytest.mark.asyncio
async def test_threads_delete_wires_stream_cleanup(
    conn, saver, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)
    thread = await create_thread(conn)
    thread_id = thread["thread_id"]
    run_id = uuid4()
    conn.store["runs"].append(
        {"run_id": run_id, "thread_id": thread_id, "status": "success"}
    )
    add_checkpoint(saver, thread_id, "0001", value="delete")
    await manager.add_queue(run_id, thread_id)
    await manager.add_control_queue(run_id, thread_id)
    await manager.add_thread_stream(thread_id)
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:stream".encode(), data=b"delete"),
        resumable=True,
    )

    deleted = await ops.Threads.delete(conn, thread_id)
    assert await anext(deleted) == thread_id

    assert thread_id not in manager.queues
    assert thread_id not in manager.control_queues
    assert thread_id not in manager.control_keys
    assert thread_id not in manager.message_stores
    assert thread_id not in manager.thread_streams


@pytest.mark.asyncio
async def test_multi_thread_cancel_uses_each_runs_actual_thread_key(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)
    first_thread = await create_thread(conn)
    second_thread = await create_thread(conn)
    assistant_id = uuid4()
    first_run_id = uuid4()
    second_run_id = uuid4()
    conn.store["runs"].extend(
        [
            {
                "run_id": first_run_id,
                "thread_id": first_thread["thread_id"],
                "assistant_id": assistant_id,
                "status": "pending",
            },
            {
                "run_id": second_run_id,
                "thread_id": second_thread["thread_id"],
                "assistant_id": assistant_id,
                "status": "pending",
            },
        ]
    )

    await ops.Runs.cancel(conn, assistant_id=assistant_id)

    assert manager.get_control_key(
        first_run_id, first_thread["thread_id"]
    ).data == b"interrupt"
    assert manager.get_control_key(
        second_run_id, second_thread["thread_id"]
    ).data == b"interrupt"
    assert "no-thread" not in manager.control_keys


@pytest.mark.asyncio
async def test_explicit_ttl_is_visible_and_overrides_global(conn) -> None:
    from langgraph_api import config as api_config

    api_config.THREAD_TTL = {
        "strategy": "keep_latest",
        "default_ttl": 43200,
        "sweep_interval_minutes": 1,
    }
    thread = await create_thread(
        conn,
        ttl={"strategy": "delete", "ttl": 10},
    )

    iterator = await ops.Threads.get(conn, thread["thread_id"], include_ttl=True)
    result = await anext(iterator)

    assert result["ttl"]["strategy"] == "delete"
    assert result["ttl"]["ttl_minutes"] == 10
    assert result["ttl"]["expires_at"] == thread["updated_at"] + timedelta(minutes=10)
    assert "ttl" not in thread


@pytest.mark.asyncio
async def test_delete_sweep_cascades_and_releases_all_checkpoint_memory(
    conn, saver, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)
    thread = await create_thread(
        conn,
        ttl={"strategy": "delete", "ttl": 1},
        age_minutes=2,
    )
    thread_id = thread["thread_id"]
    run_id = uuid4()
    conn.store["runs"].append(
        {
            "run_id": run_id,
            "thread_id": thread_id,
            "status": "success",
        }
    )
    conn.store["crons"].append({"cron_id": uuid4(), "thread_id": thread_id})
    add_checkpoint(saver, thread_id, "0001", value={"large": "payload"})
    await manager.add_queue(run_id, thread_id)
    await manager.add_control_queue(run_id, thread_id)
    await manager.add_thread_stream(thread_id)
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:stream".encode(), data=b"event"),
        resumable=True,
    )

    total_before = saver.estimated_size_bytes
    stats = {}
    assert saver.storage and saver.writes and saver.blobs
    assert total_before > 0
    assert saver.estimated_checkpoint_count == 1
    assert await ops.Threads.sweep_ttl(conn, stats=stats) == (1, 1)
    assert conn.store["threads"] == []
    assert conn.store["runs"] == []
    assert conn.store["crons"] == []
    assert conn.store["thread_ttls"] == {}
    assert not saver.storage
    assert not saver.writes
    assert not saver.blobs
    assert saver.estimated_size_bytes == 0
    assert thread_id not in manager.queues
    assert thread_id not in manager.control_keys
    assert thread_id not in manager.control_queues
    assert thread_id not in manager.message_stores
    assert thread_id not in manager.thread_streams
    assert stats == {
        "deleted_items": 1,
        "deleted_size_bytes": total_before,
        "total_before_bytes": total_before,
        "size_tracking_available": True,
    }


@pytest.mark.asyncio
async def test_global_keep_latest_prunes_once_then_rearms_after_activity(
    conn, saver
) -> None:
    from langgraph_api import config as api_config

    api_config.THREAD_TTL = {
        "strategy": "keep_latest",
        "default_ttl": 1,
        "sweep_interval_minutes": 1,
        "sweep_limit": 100,
    }
    thread = await create_thread(conn, age_minutes=2)
    thread_id = thread["thread_id"]
    add_checkpoint(saver, thread_id, "0001", value="old")
    add_checkpoint(saver, thread_id, "0002", parent_id="0001", value="latest")
    add_checkpoint(saver, thread_id, "0001", namespace="child", value="child-old")
    add_checkpoint(
        saver,
        thread_id,
        "0002",
        parent_id="0001",
        namespace="child",
        value="child-latest",
    )

    total_before = saver.estimated_size_bytes
    stats = {}
    assert saver.estimated_checkpoint_count == 4
    assert await ops.Threads.sweep_ttl(conn, stats=stats) == (1, 0)
    assert len(conn.store["threads"]) == 1
    assert set(saver.storage[str(thread_id)][""]) == {"0002"}
    assert set(saver.storage[str(thread_id)]["child"]) == {"0002"}
    assert len([key for key in saver.writes if key[0] == str(thread_id)]) == 2
    assert len([key for key in saver.blobs if key[0] == str(thread_id)]) == 2
    assert saver.estimated_checkpoint_count == 2
    assert 0 < saver.estimated_size_bytes < total_before
    assert stats["deleted_items"] == 2
    assert stats["deleted_size_bytes"] == total_before - saver.estimated_size_bytes
    assert stats["total_before_bytes"] == total_before

    # The same inactive state is not repeatedly pruned every sweep.
    assert await ops.Threads.sweep_ttl(conn) == (0, 0)
    iterator = await ops.Threads.get(conn, thread_id, include_ttl=True)
    assert (await anext(iterator))["ttl"]["expires_at"] is None

    # A subsequent write/activity advances updated_at and rearms the TTL.
    thread["updated_at"] = thread["updated_at"] + timedelta(seconds=1)
    assert await ops.Threads.sweep_ttl(conn) == (1, 0)


def test_keep_latest_preserves_delta_ancestor_chain(saver) -> None:
    thread_id = uuid4()
    add_checkpoint(
        saver,
        thread_id,
        "0001",
        value=_DeltaSnapshot(["seed"]),
    )
    add_checkpoint(
        saver,
        thread_id,
        "0002",
        parent_id="0001",
        value=["delta-1"],
        materialized=False,
    )
    add_checkpoint(
        saver,
        thread_id,
        "0003",
        parent_id="0002",
        value=["delta-2"],
        materialized=False,
    )
    # An obsolete fork should still be removed.
    add_checkpoint(
        saver,
        thread_id,
        "0002-fork",
        parent_id="0001",
        value=["fork"],
        materialized=False,
    )

    saver.prune([str(thread_id)], strategy="keep_latest")

    assert set(saver.storage[str(thread_id)][""]) == {"0001", "0002", "0003"}
    assert (str(thread_id), "", "0002-fork") not in saver.writes


@pytest.mark.asyncio
async def test_sweep_limit_and_active_run_protection(conn, saver) -> None:
    first = await create_thread(
        conn, ttl={"strategy": "delete", "ttl": 1}, age_minutes=3
    )
    second = await create_thread(
        conn, ttl={"strategy": "delete", "ttl": 1}, age_minutes=2
    )
    active = await create_thread(
        conn, ttl={"strategy": "delete", "ttl": 1}, age_minutes=4
    )
    conn.store["runs"].append(
        {
            "run_id": uuid4(),
            "thread_id": active["thread_id"],
            "status": "running",
        }
    )

    assert await ops.Threads.sweep_ttl(conn, limit=1) == (1, 1)
    remaining = {thread["thread_id"] for thread in conn.store["threads"]}
    assert first["thread_id"] not in remaining
    assert second["thread_id"] in remaining
    assert active["thread_id"] in remaining


@pytest.mark.asyncio
async def test_manual_keep_latest_prune_uses_checkpointer(
    conn, saver, monkeypatch
) -> None:
    thread = await create_thread(conn)
    add_checkpoint(saver, thread["thread_id"], "0001", value="old")
    add_checkpoint(saver, thread["thread_id"], "0002", parent_id="0001", value="new")

    @asynccontextmanager
    async def use_test_connection(*args, **kwargs):
        yield conn

    monkeypatch.setattr(ops, "connect", use_test_connection)
    assert (
        await ops.Threads.prune(
            [thread["thread_id"]], strategy="keep_latest", batch_size=1
        )
        == 1
    )
    assert set(saver.storage[str(thread["thread_id"])][""]) == {"0002"}
