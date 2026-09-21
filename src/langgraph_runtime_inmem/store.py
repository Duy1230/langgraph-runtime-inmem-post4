import asyncio
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict

import structlog
from langgraph.checkpoint.memory import PersistentDict
from langgraph.store.base import (
    GetOp,
    Item,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memory import InMemoryStore

from langgraph_runtime_inmem import _persistence
from langgraph_runtime_inmem.size_tracking import (
    SizeTracker,
    estimate_pickle_size,
    size_log_fields,
)

_STORE_CONFIG = None
logger = structlog.stdlib.get_logger(__name__)


class StoreTTLRecord(TypedDict):
    """Persistent expiration state for one store item."""

    expires_at: datetime
    ttl_minutes: float


class StoreTTLSweepStats(TypedDict):
    """Size and item counts captured by the most recent Store TTL sweep."""

    deleted_items: int
    deleted_size_bytes: int
    total_before_bytes: int
    deleted_ratio_percent: float


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DiskBackedInMemStore(InMemoryStore):
    supports_ttl = True

    def __init__(
        self,
        *args: Any,
        ttl: TTLConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ttl_config = ttl.copy() if ttl else None
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()
        self._operation_lock = asyncio.Lock()
        self._closed = False
        if not _persistence.DISABLE_FILE_PERSISTENCE:
            self._data = PersistentDict(dict, filename=_STORE_FILE)
            self._vectors = PersistentDict(
                lambda: defaultdict(dict), filename=_VECTOR_FILE
            )
            self._ttl: dict[tuple[tuple[str, ...], str], StoreTTLRecord] = (
                PersistentDict(dict, filename=_TTL_FILE)
            )
            _persistence.register_persistent_dict(self._data)
            _persistence.register_persistent_dict(self._vectors)
            _persistence.register_persistent_dict(self._ttl)
            self._load_data(self._data, which="data")
            self._load_data(self._vectors, which="vectors")
            self._load_data(self._ttl, which="ttl")
        else:
            self._data = defaultdict(dict)
            # [ns][key][path]
            self._vectors = defaultdict(lambda: defaultdict(dict))
            self._ttl = {}
        self._drop_orphaned_ttl_records()
        self._size_tracker = SizeTracker()
        self._rebuild_size_tracking()
        self._last_ttl_sweep_stats: StoreTTLSweepStats = {
            "deleted_items": 0,
            "deleted_size_bytes": 0,
            "total_before_bytes": self.estimated_size_bytes,
            "deleted_ratio_percent": 0.0,
        }

    def _load_data(self, container: PersistentDict, which: str) -> None:
        if not container.filename:
            return
        try:
            container.load()
        except FileNotFoundError:
            # It's okay if the file doesn't exist yet
            pass

        except (EOFError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load store {which} from {container.filename}. "
                "This may be due to changes in the stored data structure. "
                "Consider clearing the local store by running: rm -rf .langgraph_api"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading store {which} from {container.filename}: {e!s}"
            ) from e

    def _drop_orphaned_ttl_records(self) -> None:
        """Discard TTL metadata whose corresponding item no longer exists."""
        for namespace, key in list(self._ttl):
            if key not in self._data.get(namespace, {}):
                self._ttl.pop((namespace, key), None)

    @property
    def estimated_size_bytes(self) -> int:
        """Estimated serialized bytes currently held by Store items and vectors."""
        return self._size_tracker.total_bytes

    @property
    def last_ttl_sweep_stats(self) -> StoreTTLSweepStats:
        """Return a copy of the most recent Store TTL sweep measurements."""
        return self._last_ttl_sweep_stats.copy()

    def _estimate_item_size(self, namespace: tuple[str, ...], key: str) -> int:
        item = self._data.get(namespace, {}).get(key)
        vectors = self._vectors.get(namespace, {}).get(key)
        return estimate_pickle_size((namespace, key, item, vectors))

    def _track_item_size(self, namespace: tuple[str, ...], key: str) -> None:
        item_key = (namespace, key)
        if key not in self._data.get(namespace, {}):
            self._size_tracker.remove(item_key)
            return
        self._size_tracker.set_size(
            item_key,
            self._estimate_item_size(namespace, key),
        )

    def _rebuild_size_tracking(self) -> None:
        """Rebuild estimates once after loading persisted Store data."""
        self._size_tracker.clear()
        for namespace, items in self._data.items():
            for key in items:
                self._track_item_size(namespace, key)

    def _refresh_ttl(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        now: datetime,
    ) -> None:
        record = self._ttl.get((namespace, key))
        if record is None:
            return
        ttl_minutes = float(record["ttl_minutes"])
        self._ttl[(namespace, key)] = {
            "expires_at": now + timedelta(minutes=ttl_minutes),
            "ttl_minutes": ttl_minutes,
        }

    def _prepare_ops(self, ops: Iterable[Op]):
        ops_list = list(ops)
        results, put_ops, search_ops = super()._prepare_ops(ops_list)

        now = _utcnow()
        for index, op in enumerate(ops_list):
            if (
                isinstance(op, GetOp)
                and op.refresh_ttl
                and isinstance(results[index], Item)
            ):
                self._refresh_ttl(op.namespace, op.key, now=now)

        return results, put_ops, search_ops

    def _batch_search(
        self,
        ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
        queryinmem_store: dict[str, list[float]],
        results: list[Result],
    ) -> None:
        super()._batch_search(ops, queryinmem_store, results)

        now = _utcnow()
        for index, (op, _) in ops.items():
            if not op.refresh_ttl:
                continue
            for item in results[index] or []:
                self._refresh_ttl(item.namespace, item.key, now=now)

    def _apply_put_ops(
        self,
        put_ops: dict[tuple[tuple[str, ...], str], PutOp],
    ) -> None:
        now = _utcnow()
        ttl_updates: dict[tuple[tuple[str, ...], str], StoreTTLRecord | None] = {}
        for item_key, op in put_ops.items():
            if op.value is None or op.ttl is None:
                ttl_updates[item_key] = None
            else:
                ttl_minutes = float(op.ttl)
                ttl_updates[item_key] = {
                    "expires_at": now + timedelta(minutes=ttl_minutes),
                    "ttl_minutes": ttl_minutes,
                }

        # Validate every TTL before changing either the item or its sidecar metadata.
        super()._apply_put_ops(put_ops)
        for item_key, record in ttl_updates.items():
            if record is None:
                self._ttl.pop(item_key, None)
            else:
                self._ttl[item_key] = record

        for namespace, key in put_ops:
            self._track_item_size(namespace, key)

        # InMemoryStore uses defaultdict and leaves empty buckets after deletes.
        for namespace, _ in put_ops:
            if not self._data.get(namespace):
                self._data.pop(namespace, None)
            if not self._vectors.get(namespace):
                self._vectors.pop(namespace, None)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Keep search candidate selection, TTL refresh, writes, and sweeps ordered.
        async with self._operation_lock:
            return await super().abatch(ops)

    async def sweep_ttl(self) -> int:
        """Delete expired items and their vectors, returning the item count."""
        sweep_started = time.perf_counter()
        async with self._operation_lock:
            now = _utcnow()
            total_before = self.estimated_size_bytes
            expired = [
                item_key
                for item_key, record in self._ttl.items()
                if record["expires_at"] <= now
            ]

            deleted = 0
            deleted_size = 0
            touched_namespaces: set[tuple[str, ...]] = set()
            for namespace, key in expired:
                touched_namespaces.add(namespace)
                data = self._data.get(namespace)
                if data is not None and key in data:
                    data.pop(key, None)
                    deleted += 1
                    deleted_size += self._size_tracker.remove((namespace, key))
                else:
                    self._size_tracker.remove((namespace, key))
                vectors = self._vectors.get(namespace)
                if vectors is not None:
                    vectors.pop(key, None)
                self._ttl.pop((namespace, key), None)

            for namespace in touched_namespaces:
                if not self._data.get(namespace):
                    self._data.pop(namespace, None)
                if not self._vectors.get(namespace):
                    self._vectors.pop(namespace, None)

            deleted_ratio = (
                round(100 * deleted_size / total_before, 1) if total_before > 0 else 0.0
            )
            self._last_ttl_sweep_stats = {
                "deleted_items": deleted,
                "deleted_size_bytes": deleted_size,
                "total_before_bytes": total_before,
                "deleted_ratio_percent": deleted_ratio,
            }

        logger.info(
            "Store TTL sweep completed",
            deleted_items=deleted,
            **size_log_fields(deleted_size, total_before),
            duration=time.perf_counter() - sweep_started,
        )
        return deleted

    async def start_ttl_sweeper(
        self,
        sweep_interval_minutes: float | None = None,
    ) -> asyncio.Task[None]:
        """Start the standard best-effort background store TTL sweeper."""
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()
        configured_interval = (
            sweep_interval_minutes
            if sweep_interval_minutes is not None
            else self.ttl_config.get("sweep_interval_minutes")
        )
        interval = float(configured_interval if configured_interval is not None else 5)
        if interval <= 0:
            raise ValueError("Store TTL sweep interval must be greater than zero.")
        logger.info(
            f"Starting store TTL sweeper with interval {interval:g} minutes",
            interval_minutes=interval,
        )

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    try:
                        await asyncio.wait_for(
                            self._ttl_stop_event.wait(),
                            timeout=interval * 60,
                        )
                        break
                    except TimeoutError:
                        pass

                    await self.sweep_ttl()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception(
                        "Store TTL sweep iteration failed",
                        exc_info=exc,
                    )

        task = asyncio.create_task(_sweep_loop(), name="ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the background TTL sweeper, waiting up to ``timeout`` seconds."""
        task = self._ttl_sweeper_task
        if task is None or task.done():
            return True

        self._ttl_stop_event.set()
        try:
            if timeout is None:
                await task
            else:
                await asyncio.wait_for(task, timeout=timeout)
        except TimeoutError:
            return False

        self._ttl_sweeper_task = None
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._ttl_sweeper_task
        if task is not None and not task.done():
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # The owning loop is already closed; there is nothing left to stop.
                pass
        if isinstance(self._data, PersistentDict):
            _persistence.close_persistent_dict(self._data)
        if isinstance(self._vectors, PersistentDict):
            _persistence.close_persistent_dict(self._vectors)
        if isinstance(self._ttl, PersistentDict):
            _persistence.close_persistent_dict(self._ttl)


class BatchedStore(AsyncBatchedBaseStore):
    def __init__(self, store: DiskBackedInMemStore) -> None:
        super().__init__()
        self._store = store
        self.supports_ttl = store.supports_ttl
        self.ttl_config = store.ttl_config

    @property
    def estimated_size_bytes(self) -> int:
        return self._store.estimated_size_bytes

    @property
    def last_ttl_sweep_stats(self) -> StoreTTLSweepStats:
        return self._store.last_ttl_sweep_stats

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._store.batch(ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await self._store.abatch(ops)

    async def start_ttl_sweeper(self) -> asyncio.Task[None]:
        return await self._store.start_ttl_sweeper()

    async def sweep_ttl(self) -> int:
        return await self._store.sweep_ttl()

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        return await self._store.stop_ttl_sweeper(timeout=timeout)

    def close(self) -> None:
        self._store.close()


_STORE_FILE = os.path.join(".langgraph_api", "store.pckl")
_VECTOR_FILE = os.path.join(".langgraph_api", "store.vectors.pckl")
_TTL_FILE = os.path.join(".langgraph_api", "store.ttl.pckl")
os.makedirs(".langgraph_api", exist_ok=True)
STORE: DiskBackedInMemStore | None = None
BATCHED_STORE = threading.local()
_STORE_LOCK = threading.Lock()


def set_store_config(config: dict) -> None:
    global _STORE_CONFIG, STORE
    _STORE_CONFIG = config.copy()
    if _STORE_CONFIG.get("path"):
        return
    from langgraph_api.graph import resolve_embeddings

    index_config = _STORE_CONFIG.get("index", {})
    if index_config:
        _STORE_CONFIG["index"]["embed"] = resolve_embeddings(index_config)
        index_config["embed"] = _STORE_CONFIG["index"]["embed"]
    ttl_config = _STORE_CONFIG.get("ttl") or None
    # Re-create the store
    if STORE is not None:
        STORE.close()
    STORE = DiskBackedInMemStore(index=index_config, ttl=ttl_config)


def Store(*args: Any, **kwargs: Any) -> BatchedStore:
    global STORE
    if STORE is None:
        with _STORE_LOCK:
            if STORE is None:
                STORE = DiskBackedInMemStore()
    if not hasattr(BATCHED_STORE, "store") or BATCHED_STORE.store._store is not STORE:
        BATCHED_STORE.store = BatchedStore(STORE)
    return BATCHED_STORE.store
