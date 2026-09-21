from __future__ import annotations

import gc
import pickle
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import PersistentDict

from langgraph_runtime_inmem import _persistence
from langgraph_runtime_inmem import checkpoint as checkpoint_module
from langgraph_runtime_inmem import store as store_module


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for persistence flush")
        time.sleep(0.005)


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    original_stores = _persistence._stores
    original_loop_was_running = (
        _persistence._flush_thread is not None
        and _persistence._flush_thread[1].is_alive()
    )
    _persistence.stop_flush_loop()
    monkeypatch.setattr(_persistence, "_stores", {})
    monkeypatch.setattr(_persistence, "_flush_thread", None)
    monkeypatch.setattr(_persistence, "_flush_interval", 0.01)
    monkeypatch.setattr(_persistence, "DISABLE_FILE_PERSISTENCE", False)
    try:
        yield
    finally:
        _persistence.stop_flush_loop()
        monkeypatch.undo()
        if original_loop_was_running:
            for store_ref in original_stores.values():
                if (persistent_dict := store_ref()) is not None:
                    _persistence.register_persistent_dict(persistent_dict)
                    break


def test_periodic_flush_keeps_a_live_empty_dict_registered(
    tmp_path: Path, isolated_registry
) -> None:
    path = tmp_path / "empty.pckl"
    persistent_dict = PersistentDict(dict, filename=str(path))

    _persistence.register_persistent_dict(persistent_dict)
    _wait_until(path.exists)

    assert _load_pickle(path) == {}
    assert _persistence._stores[str(path)]() is persistent_dict

    persistent_dict["written-after-empty-flush"] = True
    _wait_until(lambda: _load_pickle(path).get("written-after-empty-flush") is True)
    assert _persistence._stores[str(path)]() is persistent_dict

    _persistence.close_persistent_dict(persistent_dict)
    assert str(path) not in _persistence._stores


def test_same_filename_registration_keeps_live_owner_and_allows_replacement(
    tmp_path: Path, isolated_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    path = tmp_path / "shared.pckl"
    owner = PersistentDict(dict, {"source": "owner"}, filename=str(path))
    contender = PersistentDict(dict, {"source": "contender"}, filename=str(path))

    _persistence.register_persistent_dict(owner)
    _persistence.register_persistent_dict(contender)
    assert _persistence._stores[str(path)]() is owner
    _persistence._flush_registered_once()
    assert _load_pickle(path) == {"source": "owner"}

    # A denied contender may be closed, but it cannot write over the owner.
    _persistence.close_persistent_dict(contender)
    assert _persistence._stores[str(path)]() is owner
    assert _load_pickle(path) == {"source": "owner"}

    _persistence.close_persistent_dict(owner)
    contender["source"] = "contender"
    _persistence.register_persistent_dict(contender)
    _persistence.unregister_persistent_dict(owner)
    assert _persistence._stores[str(path)]() is contender
    _persistence._flush_registered_once()
    assert _load_pickle(path) == {"source": "contender"}
    _persistence.close_persistent_dict(contender)


def test_dead_same_filename_owner_is_replaced(
    tmp_path: Path, isolated_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    path = tmp_path / "dead-owner.pckl"
    owner = PersistentDict(dict, filename=str(path))
    _persistence.register_persistent_dict(owner)
    del owner
    gc.collect()
    assert _persistence._stores[str(path)]() is None

    replacement = PersistentDict(dict, filename=str(path))
    _persistence.register_persistent_dict(replacement)
    assert _persistence._stores[str(path)]() is replacement
    _persistence.close_persistent_dict(replacement)


def test_registration_cannot_replace_owner_during_flush(
    tmp_path: Path, isolated_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    entered_sync = threading.Event()
    release_sync = threading.Event()
    registration_finished = threading.Event()

    class BlockingPersistentDict(PersistentDict):
        def sync(self) -> None:
            entered_sync.set()
            assert release_sync.wait(timeout=1)
            super().sync()

    path = tmp_path / "concurrent.pckl"
    owner = BlockingPersistentDict(dict, {"source": "owner"}, filename=str(path))
    contender = PersistentDict(dict, {"source": "contender"}, filename=str(path))
    _persistence.register_persistent_dict(owner)

    flush_thread = threading.Thread(target=_persistence._flush_registered_once)
    flush_thread.start()
    assert entered_sync.wait(timeout=1)

    def register_contender() -> None:
        _persistence.register_persistent_dict(contender)
        registration_finished.set()

    registration_thread = threading.Thread(target=register_contender)
    registration_thread.start()
    assert not registration_finished.wait(timeout=0.05)

    release_sync.set()
    flush_thread.join(timeout=1)
    registration_thread.join(timeout=1)
    assert registration_finished.is_set()
    assert _persistence._stores[str(path)]() is owner
    assert _load_pickle(path) == {"source": "owner"}

    _persistence.close_persistent_dict(owner)
    _persistence.close_persistent_dict(contender)


def test_failed_flush_keeps_store_registered_for_retry(
    tmp_path: Path, isolated_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)

    class FlakyPersistentDict(PersistentDict):
        attempts = 0

        def sync(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("transient flush failure")
            super().sync()

    path = tmp_path / "flaky.pckl"
    persistent_dict = FlakyPersistentDict(
        dict, {"survives": True}, filename=str(path)
    )
    _persistence.register_persistent_dict(persistent_dict)

    _persistence._flush_registered_once()
    assert _persistence._stores[str(path)]() is persistent_dict
    assert not path.exists()

    _persistence._flush_registered_once()
    assert _load_pickle(path) == {"survives": True}
    assert _persistence._stores[str(path)]() is persistent_dict
    _persistence.close_persistent_dict(persistent_dict)


@pytest.mark.asyncio
async def test_transient_checkpointer_view_does_not_own_or_clobber_files(
    tmp_path: Path,
    isolated_registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    monkeypatch.setattr(checkpoint_module, "DISABLE_FILE_PERSISTENCE", False)
    monkeypatch.setattr(checkpoint_module, "MEMORY", None)

    singleton = checkpoint_module.Checkpointer()
    singleton.storage["thread"][""]["sentinel"] = (
        singleton.serde.dumps_typed({"id": "sentinel", "channel_versions": {}}),
        singleton.serde.dumps_typed({}),
        None,
    )
    owners = {
        filename: store_ref()
        for filename, store_ref in _persistence._stores.items()
    }
    _persistence._flush_registered_once()
    assert "sentinel" in _load_pickle(Path(singleton.storage.filename))["thread"][""]
    flush_loop = _persistence._flush_thread

    adapter = checkpoint_module.Checkpointer(
        unpack_hook=lambda _code, _data: None
    )
    assert adapter.storage is singleton.storage
    assert adapter.writes is singleton.writes
    assert adapter.blobs is singleton.blobs
    assert adapter._persistent_dicts == []
    assert {
        filename: store_ref()
        for filename, store_ref in _persistence._stores.items()
    } == owners

    async with adapter:
        pass

    assert "sentinel" in singleton.storage["thread"][""]
    assert _persistence._flush_thread is flush_loop
    assert flush_loop is not None and flush_loop[1].is_alive()
    assert {
        filename: store_ref()
        for filename, store_ref in _persistence._stores.items()
    } == owners

    singleton.storage["thread"][""]["after-adapter-close"] = (
        singleton.serde.dumps_typed(
            {"id": "after-adapter-close", "channel_versions": {}}
        ),
        singleton.serde.dumps_typed({}),
        "sentinel",
    )
    _persistence._flush_registered_once()
    assert "after-adapter-close" in _load_pickle(Path(singleton.storage.filename))[
        "thread"
    ][""]

    singleton._close_persistence()
    checkpoint_module.MEMORY = None


@pytest.mark.asyncio
async def test_store_periodic_flush_persists_data_vectors_and_ttl_mutations(
    tmp_path: Path,
    isolated_registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    data_path = tmp_path / "store.pckl"
    vector_path = tmp_path / "vectors.pckl"
    ttl_path = tmp_path / "ttl.pckl"
    monkeypatch.setattr(store_module, "_STORE_FILE", str(data_path))
    monkeypatch.setattr(store_module, "_VECTOR_FILE", str(vector_path))
    monkeypatch.setattr(store_module, "_TTL_FILE", str(ttl_path))
    now = datetime(2026, 9, 9, tzinfo=UTC)
    monkeypatch.setattr(store_module, "_utcnow", lambda: now)

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    store = store_module.DiskBackedInMemStore(
        index={"dims": 2, "embed": embed, "fields": ["text"]},
        ttl={"default_ttl": 10, "refresh_on_read": True},
    )
    namespace = ("durable",)
    await store.aput(namespace, "item", {"text": "persist me"})
    assert store._vectors[namespace]["item"]["text"]

    _persistence._flush_registered_once()
    assert _load_pickle(data_path)[namespace]["item"].value == {
        "text": "persist me"
    }
    assert _load_pickle(vector_path)[namespace]["item"]["text"]
    original_expiry = _load_pickle(ttl_path)[(namespace, "item")]["expires_at"]

    # A second instance models a fresh process reading the periodic snapshot;
    # no explicit close/final sync of the live owner has happened yet.
    restarted = store_module.DiskBackedInMemStore(
        index={"dims": 2, "embed": embed, "fields": ["text"]},
        ttl={"default_ttl": 10, "refresh_on_read": True},
    )
    restored = await restarted.aget(namespace, "item", refresh_ttl=False)
    assert restored is not None and restored.value == {"text": "persist me"}
    assert restarted._vectors[namespace]["item"]["text"]
    restarted.close()
    assert _persistence._stores[str(data_path)]() is store._data
    assert _persistence._stores[str(vector_path)]() is store._vectors
    assert _persistence._stores[str(ttl_path)]() is store._ttl

    now += timedelta(minutes=5)
    assert await store.aget(namespace, "item") is not None
    _persistence._flush_registered_once()
    assert _load_pickle(ttl_path)[(namespace, "item")]["expires_at"] > original_expiry

    await store.adelete(namespace, "item")
    _persistence._flush_registered_once()
    assert _load_pickle(data_path) == {}
    assert _load_pickle(vector_path) == {}
    assert _load_pickle(ttl_path) == {}
    store.close()
