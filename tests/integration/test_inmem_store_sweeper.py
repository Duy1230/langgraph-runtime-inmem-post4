from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import NotRequired, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langgraph_runtime_inmem import _persistence
from langgraph_runtime_inmem import store as store_module


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def advance(self, *, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> MutableClock:
    # Only expiration time is controlled; the sweeper uses its real async timer.
    value = MutableClock()
    monkeypatch.setattr(store_module, "_utcnow", lambda: value.now)
    return value


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch):
    async def embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    # Exercise production configuration and Store() without sharing runtime data
    # or loading the developer's on-disk Store. Embeddings need no external API.
    monkeypatch.setattr(_persistence, "DISABLE_FILE_PERSISTENCE", True)
    monkeypatch.setattr(store_module, "STORE", store_module.DiskBackedInMemStore())
    monkeypatch.setattr(store_module, "BATCHED_STORE", threading.local())
    monkeypatch.setattr(store_module, "_STORE_CONFIG", None)
    store_module.set_store_config(
        {
            "index": {"dims": 2, "embed": embed, "fields": ["text"]},
            "ttl": {
                "default_ttl": 10,
                "refresh_on_read": True,
                "sweep_interval_minutes": 0.001,
            },
        }
    )
    wrapped = store_module.Store()
    try:
        yield wrapped
    finally:
        try:
            assert await wrapped.stop_ttl_sweeper(timeout=5)
        finally:
            if wrapped._task is not None:
                wrapped._task.cancel()
                with suppress(asyncio.CancelledError):
                    await wrapped._task
            wrapped.close()


class MemoryState(TypedDict):
    namespace: tuple[str, ...]
    key: str
    text: str
    ttl: NotRequired[float | None]


@pytest.fixture
def graph(store):
    async def remember(state: MemoryState, runtime: Runtime) -> dict:
        assert runtime.store is store
        kwargs = {"ttl": state["ttl"]} if "ttl" in state else {}
        await runtime.store.aput(
            state["namespace"], state["key"], {"text": state["text"]}, **kwargs
        )
        return {}

    builder = StateGraph(MemoryState)
    builder.add_node("remember", remember)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", END)
    return builder.compile(store=store)


@asynccontextmanager
async def running_sweeper(store) -> AsyncIterator[asyncio.Task[None]]:
    task = await store.start_ttl_sweeper()
    try:
        yield task
    finally:
        assert await store.stop_ttl_sweeper(timeout=5)
    assert task.done()


async def wait_until(task: asyncio.Task[None], predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            if task.done():
                await task
                pytest.fail("Store sweeper stopped before processing expired items")
            await asyncio.sleep(0.01)


async def test_sweeper_cleans_graph_memories_vectors_and_ttl_overrides(
    store, graph, clock: MutableClock
) -> None:
    backend = store_module.STORE
    namespace = ("memories", "retained")
    expired_namespace = ("memories", "expired")
    await graph.ainvoke(
        {"namespace": namespace, "key": "fresh", "text": "fresh memory", "ttl": 60}
    )
    await graph.ainvoke(
        {"namespace": namespace, "key": "permanent", "text": "permanent", "ttl": None}
    )
    retained_size = store.estimated_size_bytes
    retained_vectors = dict(backend._vectors[namespace])
    await graph.ainvoke(
        {"namespace": namespace, "key": "default", "text": "uses default TTL"}
    )
    await graph.ainvoke(
        {"namespace": expired_namespace, "key": "fresh", "text": "expired", "ttl": 0}
    )
    assert backend._vectors[expired_namespace]["fresh"]["text"]
    assert backend._ttl[(namespace, "default")]["ttl_minutes"] == 10
    assert (namespace, "permanent") not in backend._ttl
    total_before = store.estimated_size_bytes
    assert total_before > retained_size > 0

    async with running_sweeper(store) as task:
        await wait_until(task, lambda: expired_namespace not in backend._data)
        assert expired_namespace not in backend._vectors
        assert (expired_namespace, "fresh") not in backend._ttl
        assert await store.aget(expired_namespace, "fresh", refresh_ttl=False) is None
        assert "default" in backend._data[namespace]

        clock.advance(minutes=11)
        await wait_until(task, lambda: "default" not in backend._data[namespace])
        assert set(backend._data[namespace]) == {"fresh", "permanent"}
        assert dict(backend._vectors[namespace]) == retained_vectors
        assert set(backend._ttl) == {(namespace, "fresh")}
        assert store.estimated_size_bytes == retained_size
        items = await store.asearch(("memories",), query="memory", refresh_ttl=False)
        assert {(item.namespace, item.key) for item in items} == {
            (namespace, "fresh"),
            (namespace, "permanent"),
        }
        assert all(item.score is not None for item in items)

        clock.advance(minutes=50)
        await wait_until(task, lambda: "fresh" not in backend._data[namespace])
        permanent = await store.aget(namespace, "permanent", refresh_ttl=False)
        assert permanent is not None
        assert permanent.value == {"text": "permanent"}
        assert set(backend._vectors[namespace]) == {"permanent"}
        assert backend._ttl == {}
        assert 0 < store.estimated_size_bytes < retained_size


@pytest.mark.parametrize("read_operation", ["get", "search"])
@pytest.mark.parametrize(
    "refresh", [True, False], ids=["default-refresh", "no-refresh"]
)
async def test_sweeper_honors_batched_read_refresh(
    store, clock: MutableClock, read_operation: str, refresh: bool
) -> None:
    backend = store_module.STORE
    namespace = ("refresh",)
    await asyncio.gather(
        store.aput(namespace, "target", {"text": "target memory", "group": "hit"}),
        store.aput(namespace, "untouched", {"text": "other memory", "group": "miss"}),
    )
    original_expiry = backend._ttl[(namespace, "target")]["expires_at"]

    async with running_sweeper(store) as task:
        clock.advance(minutes=9)
        # Omission exercises refresh_on_read from the real Store configuration.
        kwargs = {} if refresh else {"refresh_ttl": False}
        if read_operation == "get":
            item = await store.aget(namespace, "target", **kwargs)
            assert item is not None
            assert item.value["text"] == "target memory"
        else:
            items = await store.asearch(
                namespace, query="memory", filter={"group": "hit"}, **kwargs
            )
            assert [item.key for item in items] == ["target"]

        assert backend._ttl[(namespace, "target")]["expires_at"] == (
            clock.now + timedelta(minutes=10) if refresh else original_expiry
        )
        assert backend._ttl[(namespace, "untouched")]["expires_at"] == original_expiry
        clock.advance(minutes=2)
        await wait_until(
            task, lambda: "untouched" not in backend._data.get(namespace, {})
        )
        if refresh:
            target = await store.aget(namespace, "target", refresh_ttl=False)
            assert target is not None
            assert target.value["text"] == "target memory"
            assert set(backend._vectors[namespace]) == {"target"}
            clock.advance(minutes=9)
        await wait_until(task, lambda: namespace not in backend._data)
        assert namespace not in backend._vectors
        assert backend._ttl == {}
        assert store.estimated_size_bytes == 0
        assert await store.aget(namespace, "target", refresh_ttl=False) is None


async def test_sweeper_respects_write_reset_and_removing_ttl(
    store, clock: MutableClock
) -> None:
    backend = store_module.STORE
    namespace = ("rewrites",)
    for key in ("reset", "permanent", "untouched"):
        await store.aput(namespace, key, {"text": "original"}, ttl=3)

    async with running_sweeper(store) as task:
        clock.advance(minutes=2)
        await store.aput(namespace, "reset", {"text": "updated"}, ttl=5)
        await store.aput(namespace, "permanent", {"text": "keep forever"}, ttl=None)
        assert (namespace, "permanent") not in backend._ttl
        clock.advance(minutes=2)
        await wait_until(task, lambda: "untouched" not in backend._data[namespace])
        reset = await store.aget(namespace, "reset", refresh_ttl=False)
        assert reset is not None
        assert reset.value == {"text": "updated"}

        clock.advance(minutes=4)
        await wait_until(task, lambda: "reset" not in backend._data[namespace])
        permanent = await store.aget(namespace, "permanent", refresh_ttl=False)
        assert permanent is not None
        assert permanent.value == {"text": "keep forever"}
        assert set(backend._vectors[namespace]) == {"permanent"}
        assert backend._ttl == {}


async def test_sweeper_can_stop_and_restart_through_store_wrapper(store) -> None:
    backend = store_module.STORE
    namespace = ("lifecycle",)
    await store.aput(namespace, "first", {"text": "first sweep"}, ttl=0)
    async with running_sweeper(store) as first_task:
        assert await store.start_ttl_sweeper() is first_task
        await wait_until(first_task, lambda: namespace not in backend._data)
    assert first_task.done()
    assert not first_task.cancelled()

    await store.aput(namespace, "second", {"text": "after stop"}, ttl=0)
    # The stopped task is complete; an expired item remains until restart.
    assert await store.aget(namespace, "second", refresh_ttl=False) is not None
    async with running_sweeper(store) as second_task:
        assert second_task is not first_task
        await wait_until(second_task, lambda: namespace not in backend._data)
    assert second_task.done()
    assert not second_task.cancelled()
    assert namespace not in backend._vectors
    assert backend._ttl == {}
    assert store.estimated_size_bytes == 0
