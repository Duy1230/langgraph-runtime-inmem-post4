from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.store.base import TTLConfig

from langgraph_runtime_inmem import _persistence, size_tracking
from langgraph_runtime_inmem import store as store_module


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def advance(self, *, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> MutableClock:
    value = MutableClock()
    monkeypatch.setattr(store_module, "_utcnow", lambda: value.now)
    return value


@pytest.fixture
def store_factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_persistence, "DISABLE_FILE_PERSISTENCE", True)
    instances: list[store_module.DiskBackedInMemStore] = []

    def create(
        ttl: TTLConfig | None = None,
    ) -> store_module.DiskBackedInMemStore:
        instance = store_module.DiskBackedInMemStore(ttl=ttl)
        instances.append(instance)
        return instance

    yield create

    for instance in instances:
        instance.close()


@pytest.mark.asyncio
async def test_default_ttl_per_item_override_and_no_ttl(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 10})

    await store.aput(("memories",), "default", {"value": "default"})
    await store.aput(("memories",), "short", {"value": "short"}, ttl=1)
    await store.aput(("memories",), "permanent", {"value": "forever"}, ttl=None)

    assert store.supports_ttl is True
    assert store._ttl[(("memories",), "default")]["ttl_minutes"] == 10
    assert store._ttl[(("memories",), "short")]["ttl_minutes"] == 1
    assert (("memories",), "permanent") not in store._ttl

    clock.advance(minutes=2)
    assert await store.sweep_ttl() == 1
    assert await store.aget(("memories",), "short", refresh_ttl=False) is None
    assert await store.aget(("memories",), "default", refresh_ttl=False) is not None

    clock.advance(minutes=9)
    assert await store.sweep_ttl() == 1
    assert await store.aget(("memories",), "default", refresh_ttl=False) is None
    assert await store.aget(("memories",), "permanent") is not None


@pytest.mark.asyncio
async def test_get_refreshes_ttl_by_default_and_can_be_disabled(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 10, "refresh_on_read": True})
    namespace = ("users", "u1")
    await store.aput(namespace, "refreshed", {"value": 1})
    await store.aput(namespace, "not-refreshed", {"value": 2})

    clock.advance(minutes=9)
    assert await store.aget(namespace, "refreshed") is not None
    assert await store.aget(namespace, "not-refreshed", refresh_ttl=False) is not None

    clock.advance(minutes=2)
    assert await store.sweep_ttl() == 1
    assert await store.aget(namespace, "refreshed", refresh_ttl=False) is not None
    assert await store.aget(namespace, "not-refreshed", refresh_ttl=False) is None

    clock.advance(minutes=9)
    assert await store.sweep_ttl() == 1
    assert await store.aget(namespace, "refreshed", refresh_ttl=False) is None


@pytest.mark.asyncio
async def test_explicit_get_refresh_overrides_disabled_default(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 10, "refresh_on_read": False})
    namespace = ("users", "u2")
    await store.aput(namespace, "default", {"value": 1})
    await store.aput(namespace, "override", {"value": 2})

    clock.advance(minutes=9)
    assert await store.aget(namespace, "default") is not None
    assert await store.aget(namespace, "override", refresh_ttl=True) is not None

    clock.advance(minutes=2)
    assert await store.sweep_ttl() == 1
    assert await store.aget(namespace, "default", refresh_ttl=False) is None
    assert await store.aget(namespace, "override", refresh_ttl=False) is not None


@pytest.mark.asyncio
async def test_search_refreshes_only_returned_items(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 10, "refresh_on_read": True})
    namespace = ("search",)
    await store.aput(namespace, "hit", {"group": "hit"})
    await store.aput(namespace, "miss", {"group": "miss"})

    clock.advance(minutes=9)
    results = await store.asearch(namespace, filter={"group": "hit"})
    assert [item.key for item in results] == ["hit"]

    clock.advance(minutes=2)
    assert await store.sweep_ttl() == 1
    assert await store.aget(namespace, "hit", refresh_ttl=False) is not None
    assert await store.aget(namespace, "miss", refresh_ttl=False) is None


@pytest.mark.asyncio
async def test_write_resets_ttl_and_delete_removes_metadata(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 10})
    namespace = ("updates",)
    await store.aput(namespace, "item", {"version": 1}, ttl=3)

    clock.advance(minutes=2)
    await store.aput(namespace, "item", {"version": 2}, ttl=5)
    clock.advance(minutes=2)

    assert await store.sweep_ttl() == 0
    item = await store.aget(namespace, "item", refresh_ttl=False)
    assert item is not None
    assert item.value == {"version": 2}

    await store.aput(namespace, "item", {"version": 3}, ttl=None)
    assert (namespace, "item") not in store._ttl
    clock.advance(minutes=100)
    assert await store.sweep_ttl() == 0

    await store.adelete(namespace, "item")
    assert (namespace, "item") not in store._ttl
    assert namespace not in store._data


@pytest.mark.asyncio
async def test_default_ttl_is_not_applied_retroactively(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory()
    namespace = ("existing",)
    await store.aput(namespace, "item", {"value": "pre-existing"})

    store.ttl_config = {"default_ttl": 1}
    clock.advance(minutes=2)

    assert await store.sweep_ttl() == 0
    assert await store.aget(namespace, "item", refresh_ttl=False) is not None


@pytest.mark.asyncio
async def test_per_item_ttl_works_without_global_default(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory()
    await store.aput(("override-only",), "item", {"value": 1}, ttl=1)

    clock.advance(minutes=2)

    assert await store.sweep_ttl() == 1
    assert await store.aget(("override-only",), "item") is None


@pytest.mark.asyncio
async def test_sweep_removes_item_vectors_and_ttl_metadata(
    store_factory,
    clock: MutableClock,
) -> None:
    store = store_factory({"default_ttl": 1})
    namespace = ("vectors",)
    await store.aput(namespace, "item", {"text": "hello"})
    store._vectors[namespace]["item"]["text"] = [1.0, 0.0]

    clock.advance(minutes=2)
    assert await store.sweep_ttl() == 1
    assert namespace not in store._data
    assert namespace not in store._vectors
    assert (namespace, "item") not in store._ttl


@pytest.mark.asyncio
async def test_size_tracking_updates_on_write_delete_and_ttl_sweep(
    store_factory,
    clock: MutableClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_factory({"default_ttl": 1})
    namespace = ("size-tracking",)
    log_events = []

    class FakeLogger:
        def info(self, event, **fields):
            log_events.append((event, fields))

        def exception(self, *args, **kwargs):
            raise AssertionError("The Store sweep should not log an exception")

    monkeypatch.setattr(store_module, "logger", FakeLogger())

    await store.aput(namespace, "permanent", {"payload": "x"}, ttl=None)
    first_size = store.estimated_size_bytes
    await store.aput(namespace, "permanent", {"payload": "x" * 4096}, ttl=None)
    assert store.estimated_size_bytes > first_size

    await store.aput(namespace, "expired", {"payload": "y" * 2048})
    total_before = store.estimated_size_bytes
    clock.advance(minutes=2)

    def fail_if_reserialized(*args, **kwargs):
        raise AssertionError("TTL sweep must use tracked integers, not pickle.dumps")

    monkeypatch.setattr(size_tracking.pickle, "dumps", fail_if_reserialized)

    assert await store.sweep_ttl() == 1
    stats = store.last_ttl_sweep_stats
    assert stats["deleted_items"] == 1
    assert stats["deleted_size_bytes"] > 0
    assert stats["total_before_bytes"] == total_before
    assert store.estimated_size_bytes == total_before - stats["deleted_size_bytes"]
    assert 0 < stats["deleted_ratio_percent"] < 100

    _, fields = next(
        event for event in log_events if event[0] == "Store TTL sweep completed"
    )
    assert fields["deleted_items"] == 1
    assert fields["deleted_size_bytes"] == stats["deleted_size_bytes"]
    assert fields["total_before_bytes"] == total_before
    assert fields["deleted_ratio"].endswith("%")

    await store.adelete(namespace, "permanent")
    assert store.estimated_size_bytes == 0

    log_events.clear()
    assert await store.sweep_ttl() == 0
    assert len(log_events) == 1
    event, fields = log_events[0]
    assert event == "Store TTL sweep completed"
    assert fields.pop("duration") >= 0
    assert fields == {
        "deleted_items": 0,
        "deleted_size": "0 B",
        "deleted_size_bytes": 0,
        "total_before": "0 B",
        "total_before_bytes": 0,
        "deleted_ratio": "0.0%",
        "deleted_ratio_percent": 0.0,
    }


@pytest.mark.asyncio
async def test_background_sweeper_runs_once_and_can_be_stopped(
    store_factory,
) -> None:
    store = store_factory(
        {
            "default_ttl": 0,
            "sweep_interval_minutes": 0.001,
        }
    )
    await store.aput(("background",), "item", {"value": 1})

    task = await store.start_ttl_sweeper()
    assert await store.start_ttl_sweeper() is task

    async def wait_until_deleted() -> None:
        while store._data.get(("background",)):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_deleted(), timeout=1)
    assert await store.stop_ttl_sweeper(timeout=1) is True
    assert task.done()


@pytest.mark.asyncio
async def test_ttl_metadata_survives_disk_backed_restart(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    clock: MutableClock,
) -> None:
    monkeypatch.setattr(_persistence, "DISABLE_FILE_PERSISTENCE", False)
    monkeypatch.setattr(_persistence, "register_persistent_dict", lambda value: None)
    monkeypatch.setattr(store_module, "_STORE_FILE", str(tmp_path / "store.pckl"))
    monkeypatch.setattr(
        store_module,
        "_VECTOR_FILE",
        str(tmp_path / "store.vectors.pckl"),
    )
    monkeypatch.setattr(store_module, "_TTL_FILE", str(tmp_path / "store.ttl.pckl"))

    first = store_module.DiskBackedInMemStore(ttl={"default_ttl": 10})
    await first.aput(("persisted",), "item", {"value": "saved"})
    first.close()

    second = store_module.DiskBackedInMemStore(ttl={"default_ttl": 10})
    assert await second.aget(("persisted",), "item", refresh_ttl=False) is not None
    total_before = second.estimated_size_bytes
    assert total_before > 0
    assert second._ttl[(("persisted",), "item")]["expires_at"] == (
        clock.now + timedelta(minutes=10)
    )

    clock.advance(minutes=11)
    assert await second.sweep_ttl() == 1
    assert second.last_ttl_sweep_stats["deleted_size_bytes"] == total_before
    assert await second.aget(("persisted",), "item", refresh_ttl=False) is None
    second.close()


@pytest.mark.asyncio
async def test_server_store_wrapper_propagates_ttl_config(
    store_factory,
    monkeypatch: pytest.MonkeyPatch,
    clock: MutableClock,
) -> None:
    initial = store_factory()
    monkeypatch.setattr(store_module, "STORE", initial)
    monkeypatch.setattr(store_module, "BATCHED_STORE", threading.local())
    monkeypatch.setattr(store_module, "_STORE_CONFIG", None)

    store_module.set_store_config(
        {
            "ttl": {
                "default_ttl": 1,
                "refresh_on_read": False,
                "sweep_interval_minutes": 5,
            }
        }
    )
    wrapped = store_module.Store()

    assert wrapped.supports_ttl is True
    assert wrapped.ttl_config == store_module.STORE.ttl_config
    await wrapped.aput(("wrapped",), "item", {"value": 1})
    assert wrapped.estimated_size_bytes > 0
    clock.advance(minutes=2)
    assert await wrapped.sweep_ttl() == 1
    assert wrapped.last_ttl_sweep_stats["deleted_items"] == 1
    assert await wrapped.aget(("wrapped",), "item", refresh_ttl=False) is None

    wrapped._task.cancel()
    with suppress(asyncio.CancelledError):
        await wrapped._task
    store_module.STORE.close()


@pytest.mark.asyncio
async def test_store_without_ttl_config_keeps_noop_sweeper(store_factory) -> None:
    store = store_factory()
    task = await store.start_ttl_sweeper()
    await task
    assert task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [0, -1])
async def test_rejects_non_positive_sweep_interval(store_factory, interval) -> None:
    store = store_factory({"default_ttl": 1, "sweep_interval_minutes": interval})
    with pytest.raises(ValueError, match="greater than zero"):
        await store.start_ttl_sweeper()
