from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from starlette.exceptions import HTTPException

from langgraph_runtime_inmem import checkpoint, database, ops, thread_ttl


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class CounterState(TypedDict):
    counter: int


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from langgraph_api import config as api_config

    # Keep the real connection/checkpointer factories wired together, with fresh
    # state for each test and no access to the developer's persisted checkpoints.
    monkeypatch.setattr(
        database, "GLOBAL_STORE", database.GlobalStore(filename=str(tmp_path / "ops"))
    )
    monkeypatch.setattr(checkpoint, "DISABLE_FILE_PERSISTENCE", True)
    monkeypatch.setattr(checkpoint, "MEMORY", None)
    monkeypatch.setattr(api_config, "USE_CUSTOM_CHECKPOINTER", False)
    monkeypatch.setattr(api_config, "THREAD_TTL", {"sweep_interval_minutes": 0})
    return checkpoint.Checkpointer()


@pytest.fixture
def graph(saver):
    async def increment(state: CounterState) -> CounterState:
        return {"counter": state["counter"] + 1}

    child = StateGraph(CounterState)
    child.add_node("increment", increment)
    child.add_edge(START, "increment")
    child.add_edge("increment", END)

    builder = StateGraph(CounterState)
    builder.add_node("child", child.compile())
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    return builder.compile(checkpointer=saver)


@asynccontextmanager
async def running_sweeper() -> AsyncIterator[asyncio.Task[None]]:
    # Run the production loop, including its real timer, connection and sweep.
    task = asyncio.create_task(thread_ttl.thread_ttl_sweep_loop())
    try:
        yield task
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert task.cancelled()


async def wait_until(task: asyncio.Task[None], predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            if task.done():
                await task
                pytest.fail("Checkpoint sweeper stopped before processing the thread")
            await asyncio.sleep(0.01)


async def create_thread(conn, *, ttl=None):
    return await anext(
        await ops.Threads.put(conn, uuid4(), metadata={}, if_exists="raise", ttl=ttl)
    )


def thread_config(thread):
    return {"configurable": {"thread_id": str(thread["thread_id"])}}


def checkpoint_ids(saver, thread_id):
    return {
        namespace: set(entries)
        for namespace, entries in saver.storage[str(thread_id)].items()
    }


def latest_checkpoint_ids(saver, thread_id):
    return {
        namespace: {max(ids)}
        for namespace, ids in checkpoint_ids(saver, thread_id).items()
    }


def assert_checkpoint_memory_released(saver, thread_id) -> None:
    thread_key = str(thread_id)
    assert thread_key not in saver.storage
    assert not any(key[0] == thread_key for key in saver.writes)
    assert not any(key[0] == thread_key for key in saver.blobs)
    assert saver.estimated_checkpoint_count_for_thread(thread_key) == 0
    assert saver.estimated_size_for_thread(thread_key) == 0


async def test_sweeper_deletes_expired_checkpoint_memory_and_keeps_fresh_thread(
    saver, graph
) -> None:
    async with database.connect() as conn:
        expired = await create_thread(conn, ttl={"strategy": "delete", "ttl": 0})
        fresh = await create_thread(conn, ttl={"strategy": "delete", "ttl": 60})
        expired_id = expired["thread_id"]
        fresh_id = fresh["thread_id"]
        for thread in (expired, fresh):
            assert await graph.ainvoke({"counter": 0}, thread_config(thread)) == {
                "counter": 1
            }
            conn.store["runs"].append(
                {
                    "run_id": uuid4(),
                    "thread_id": thread["thread_id"],
                    "status": "success",
                }
            )
            conn.store["crons"].append(
                {"cron_id": uuid4(), "thread_id": thread["thread_id"]}
            )

        fresh_ids = checkpoint_ids(saver, fresh_id)
        fresh_size = saver.estimated_size_for_thread(fresh_id)
        fresh_writes = {
            key: value for key, value in saver.writes.items() if key[0] == str(fresh_id)
        }
        fresh_blobs = {
            key: value for key, value in saver.blobs.items() if key[0] == str(fresh_id)
        }
        assert saver.estimated_size_bytes > fresh_size > 0
        assert any(key[0] == str(expired_id) for key in saver.writes)
        assert any(key[0] == str(expired_id) for key in saver.blobs)

        async with running_sweeper() as task:
            await wait_until(task, lambda: expired not in conn.store["threads"])

            with pytest.raises(HTTPException) as exc_info:
                await ops.Threads.get(conn, expired_id)
            assert exc_info.value.status_code == 404
            assert_checkpoint_memory_released(saver, expired_id)
            assert await saver.aget_tuple(thread_config(expired)) is None
            assert checkpoint_ids(saver, fresh_id) == fresh_ids
            assert dict(saver.writes) == fresh_writes
            assert dict(saver.blobs) == fresh_blobs
            assert saver.estimated_size_bytes == fresh_size
            assert (await graph.aget_state(thread_config(fresh))).values == {
                "counter": 1
            }
            result = await anext(await ops.Threads.get(conn, fresh_id))
            assert result["thread_id"] == fresh_id
            assert {run["thread_id"] for run in conn.store["runs"]} == {fresh_id}
            assert {cron["thread_id"] for cron in conn.store["crons"]} == {fresh_id}
            assert set(conn.store["thread_ttls"]) == {str(fresh_id)}


async def test_sweeper_keeps_latest_state_and_rearms_after_thread_activity(
    saver, graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph_api import config as api_config

    monkeypatch.setattr(
        api_config,
        "THREAD_TTL",
        {"strategy": "keep_latest", "default_ttl": 0, "sweep_interval_minutes": 0},
    )
    async with database.connect() as conn:
        thread = await create_thread(conn)
        thread_id = thread["thread_id"]
        thread_key = str(thread_id)
        config = thread_config(thread)
        assert await graph.ainvoke({"counter": 0}, config) == {"counter": 1}
        previous_run = await graph.aget_state(config)
        assert await graph.ainvoke({}, config) == {"counter": 2}
        expected_ids = latest_checkpoint_ids(saver, thread_id)
        assert "" in expected_ids
        assert len(expected_ids) > 1  # The graph also checkpoints its child namespace.
        expected_checkpoints = {
            namespace: await saver.aget_tuple(
                {
                    "configurable": {
                        "thread_id": thread_key,
                        "checkpoint_ns": namespace,
                    }
                }
            )
            for namespace in expected_ids
        }
        size_before = saver.estimated_size_bytes
        count_before = saver.estimated_checkpoint_count
        writes_before = len(saver.writes)
        blobs_before = len(saver.blobs)

        async with running_sweeper() as task:
            await wait_until(
                task,
                lambda: (
                    conn.store["thread_ttls"].get(thread_key, {}).get(
                        "last_swept_updated_at"
                    )
                    == thread["updated_at"]
                ),
            )
            # Empty channel values can require retained ancestors; the newest
            # state in every namespace must survive while older history shrinks.
            assert latest_checkpoint_ids(saver, thread_id) == expected_ids
            assert 0 < saver.estimated_checkpoint_count < count_before
            assert await saver.aget_tuple(previous_run.config) is None
            assert 0 < saver.estimated_size_bytes < size_before
            assert len(saver.writes) < writes_before
            assert len(saver.blobs) < blobs_before
            for namespace, ids in expected_ids.items():
                latest = await saver.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": thread_key,
                            "checkpoint_ns": namespace,
                        }
                    }
                )
                assert latest is not None
                assert latest.checkpoint["id"] in ids
                assert latest.checkpoint == expected_checkpoints[namespace].checkpoint
            result = await anext(
                await ops.Threads.get(conn, thread_id, include_ttl=True)
            )
            assert result["ttl"]["expires_at"] is None
            assert (await graph.aget_state(config)).values == {"counter": 2}

            # Continue from the retained graph state. Until thread activity is
            # recorded, another sweep must leave these new checkpoints intact.
            assert await graph.ainvoke({}, config) == {"counter": 3}
            resumed_ids = checkpoint_ids(saver, thread_id)
            resumed_count = saver.estimated_checkpoint_count
            expected_ids = latest_checkpoint_ids(saver, thread_id)
            assert resumed_ids != expected_ids
            sentinel = await create_thread(
                conn, ttl={"strategy": "delete", "ttl": 0}
            )
            await wait_until(task, lambda: sentinel not in conn.store["threads"])
            assert checkpoint_ids(saver, thread_id) == resumed_ids

            updated = await anext(
                await ops.Threads.patch(
                    conn, thread_id, metadata={"activity": "resumed"}
                )
            )
            assert updated["updated_at"] > thread["updated_at"]
            await wait_until(
                task,
                lambda: (
                    conn.store["thread_ttls"][thread_key]["last_swept_updated_at"]
                    == updated["updated_at"]
                ),
            )
            assert latest_checkpoint_ids(saver, thread_id) == expected_ids
            assert 0 < saver.estimated_checkpoint_count < resumed_count
            assert (await graph.aget_state(config)).values == {"counter": 3}


@pytest.mark.parametrize("run_status", ["pending", "running"])
async def test_sweeper_preserves_active_run_checkpoints_until_completion(
    saver, graph, run_status: str
) -> None:
    async with database.connect() as conn:
        thread = await create_thread(conn, ttl={"strategy": "delete", "ttl": 0})
        thread_id = thread["thread_id"]
        assert await graph.ainvoke({"counter": 0}, thread_config(thread)) == {
            "counter": 1
        }
        before_ids = checkpoint_ids(saver, thread_id)
        before_size = saver.estimated_size_bytes
        run = {"run_id": uuid4(), "thread_id": thread_id, "status": run_status}
        conn.store["runs"].append(run)
        sentinel = await create_thread(conn, ttl={"strategy": "delete", "ttl": 0})

        async with running_sweeper() as task:
            # Sentinel deletion proves a real sweep ran while this run was active.
            await wait_until(task, lambda: sentinel not in conn.store["threads"])
            assert thread in conn.store["threads"]
            assert run in conn.store["runs"]
            assert checkpoint_ids(saver, thread_id) == before_ids
            assert saver.estimated_size_bytes == before_size
            assert (await graph.aget_state(thread_config(thread))).values == {
                "counter": 1
            }

            await ops.Threads.set_joint_status(
                conn, thread_id, run["run_id"], "success", "counter"
            )
            await wait_until(task, lambda: thread not in conn.store["threads"])
            assert_checkpoint_memory_released(saver, thread_id)
            assert conn.store["threads"] == []
            assert conn.store["runs"] == []
            assert conn.store["thread_ttls"] == {}
            assert saver.estimated_checkpoint_count == 0
            assert saver.estimated_size_bytes == 0
