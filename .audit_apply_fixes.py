#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(relpath: str, old: str, new: str) -> None:
    path = ROOT / relpath
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relpath}: expected exactly one matching block, found {count}. "
            "The repository may no longer match commit "
            "0ba5074566813e0532b2baf254cf24ce8f157a3f."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(relpath: str, marker: str, content: str) -> None:
    path = ROOT / relpath
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n" + content.rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Persistence ownership: a duplicate that was denied ownership must never
#    sync later after the original owner has closed.
# ---------------------------------------------------------------------------

replace_once(
    "src/langgraph_runtime_inmem/_persistence.py",
    '''_flush_interval: float = 10
_registry_lock = threading.RLock()
DISABLE_FILE_PERSISTENCE = (
''',
    '''_flush_interval: float = 10
_registry_lock = threading.RLock()
_PERSISTENCE_OWNER_ATTR = "_langgraph_runtime_persistence_owner"
DISABLE_FILE_PERSISTENCE = (
''',
)

replace_once(
    "src/langgraph_runtime_inmem/_persistence.py",
    '''        if current is not None and current is not d:
            logger.debug(
                "Ignoring duplicate live persistence registration for %s",
                d.filename,
            )
            return

        _stores[d.filename] = weakref.ref(d)
''',
    '''        if current is not None and current is not d:
            # Remember that this exact instance was denied ownership.  Merely
            # observing an empty registry later must not grant it permission to
            # sync stale data over the former owner's final snapshot.
            setattr(d, _PERSISTENCE_OWNER_ATTR, False)
            logger.debug(
                "Ignoring duplicate live persistence registration for %s",
                d.filename,
            )
            return

        _stores[d.filename] = weakref.ref(d)
        setattr(d, _PERSISTENCE_OWNER_ATTR, True)
''',
)

replace_once(
    "src/langgraph_runtime_inmem/_persistence.py",
    '''    with _registry_lock:
        current_ref = _stores.get(d.filename)
        current = current_ref() if current_ref is not None else None
        if current is not None and current is not d:
            # ``d`` was denied ownership at registration time.  Closing it
            # must not perform a final sync over the live owner's file.
            dict.clear(d)
            return
        d.close()
        if current_ref is not None:
            _stores.pop(d.filename, None)
''',
    '''    with _registry_lock:
        ownership = getattr(d, _PERSISTENCE_OWNER_ATTR, None)
        if ownership is False:
            # This instance was explicitly denied ownership earlier.  That
            # decision survives the original owner's close/unregistration.
            # Clear the transient container without ever syncing it.
            dict.clear(d)
            return

        current_ref = _stores.get(d.filename)
        current = current_ref() if current_ref is not None else None
        if current is not None and current is not d:
            # A newer owner is present.  The stale instance must not touch the
            # shared filename even if it used to own it.
            setattr(d, _PERSISTENCE_OWNER_ATTR, False)
            dict.clear(d)
            return

        d.close()
        setattr(d, _PERSISTENCE_OWNER_ATTR, None)
        if current_ref is not None:
            _stores.pop(d.filename, None)
''',
)

# ---------------------------------------------------------------------------
# 2. Resumable stream IDs: make Redis-style IDs monotonic/unique and compare
#    them numerically instead of lexicographically.
# ---------------------------------------------------------------------------

replace_once(
    "src/langgraph_runtime_inmem/inmem_stream.py",
    '''import asyncio
import logging
import time
from collections import defaultdict
''',
    '''import asyncio
import logging
import threading
import time
from collections import defaultdict
''',
)

replace_once(
    "src/langgraph_runtime_inmem/inmem_stream.py",
    '''logger = logging.getLogger(__name__)


def _ensure_uuid(id: str | UUID) -> UUID:
    return UUID(id) if isinstance(id, str) else id


def _generate_ms_seq_id() -> str:
    """Generate a Redis-like millisecond-sequence ID (e.g., '1234567890123-0')"""
    # Get current time in milliseconds
    ms = int(time.time() * 1000)
    # For simplicity, always use sequence 0 since we're not handling high throughput
    return f"{ms}-0"
''',
    '''logger = logging.getLogger(__name__)

_stream_id_lock = threading.Lock()
_last_stream_ms = -1
_last_stream_seq = -1


def _ensure_uuid(id: str | UUID) -> UUID:
    return UUID(id) if isinstance(id, str) else id


def _generate_ms_seq_id() -> str:
    """Generate a process-wide monotonic Redis-style ``millisecond-sequence`` ID."""
    global _last_stream_ms, _last_stream_seq

    current_ms = int(time.time() * 1000)
    with _stream_id_lock:
        # Wall clocks can move backwards.  Preserve monotonic ordering by
        # pinning to the last emitted millisecond and incrementing the sequence.
        ms = max(current_ms, _last_stream_ms)
        if ms == _last_stream_ms:
            _last_stream_seq += 1
        else:
            _last_stream_ms = ms
            _last_stream_seq = 0
        return f"{_last_stream_ms}-{_last_stream_seq}"


def _parse_ms_seq_id(value: str | bytes) -> tuple[int, int]:
    text = value.decode() if isinstance(value, bytes) else value
    ms, separator, seq = text.partition("-")
    if not separator:
        raise ValueError(f"Not a millisecond-sequence stream ID: {text!r}")
    return int(ms), int(seq)
''',
)

replace_once(
    "src/langgraph_runtime_inmem/inmem_stream.py",
    '''        messages = self.message_stores.get(thread_id, {}).get(run_id, ())
        try:
            # Handle ms-seq format (e.g., "1234567890123-0")
            for message in messages:
                if message.id is not None and message.id.decode() > message_id:
                    yield message
        except TypeError:
            # Try integer format if ms-seq fails
            message_idx = int(message_id) + 1
            yield from messages[message_idx:]
''',
    '''        messages = self.message_stores.get(thread_id, {}).get(run_id, ())

        # Backward compatibility for the older index-based resume cursor.
        if "-" not in message_id:
            message_idx = int(message_id) + 1
            yield from messages[message_idx:]
            return

        cursor = _parse_ms_seq_id(message_id)
        for message in messages:
            if message.id is None:
                continue
            try:
                candidate = _parse_ms_seq_id(message.id)
            except (UnicodeDecodeError, ValueError):
                continue
            if candidate > cursor:
                yield message
''',
)

# ---------------------------------------------------------------------------
# 3. Thread TTL validation + maintenance coordination.
# ---------------------------------------------------------------------------

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''import copy
import json
import typing
import uuid
''',
    '''import copy
import json
import math
import typing
import uuid
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''_THREAD_TTL_STORE_KEY = "thread_ttls"


def _thread_ttl_store(conn: InMemConnectionProto) -> dict[str, dict[str, Any]]:
''',
    '''_THREAD_TTL_STORE_KEY = "thread_ttls"
_THREAD_TTL_MAINTENANCE: set[str] = set()


async def _wait_for_thread_ttl_maintenance(thread_id: UUID | str) -> None:
    """Wait until background TTL maintenance releases ``thread_id``.

    Agent Server's in-memory metadata mutations run on the event loop.  Once
    this wait returns, the run/thread mutation paths below perform their final
    metadata changes without another await, preventing a sweeper from claiming
    the same thread in the middle of that critical section.
    """
    thread_key = str(thread_id)
    while thread_key in _THREAD_TTL_MAINTENANCE:
        await asyncio.sleep(0.01)


def _thread_ttl_store(conn: InMemConnectionProto) -> dict[str, dict[str, Any]]:
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''    try:
        ttl_minutes = float(ttl_minutes)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail="Thread TTL must be a number of minutes."
        ) from None
    if ttl_minutes < 0:
        raise HTTPException(
            status_code=422, detail="Thread TTL must be greater than or equal to 0."
        )
    return {"strategy": strategy, "ttl_minutes": ttl_minutes}
''',
    '''    try:
        ttl_minutes = float(ttl_minutes)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail="Thread TTL must be a number of minutes."
        ) from None
    if not math.isfinite(ttl_minutes):
        raise HTTPException(status_code=422, detail="Thread TTL must be finite.")
    if ttl_minutes < 0:
        raise HTTPException(
            status_code=422, detail="Thread TTL must be greater than or equal to 0."
        )
    try:
        # Validate the exact operation used later by the sweeper/info path.
        datetime.now(UTC) + timedelta(minutes=ttl_minutes)
    except (OverflowError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Thread TTL is outside the supported datetime range.",
        ) from None
    return {"strategy": strategy, "ttl_minutes": ttl_minutes}
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''    if "ttl_minutes" in state:
        config = {
            "strategy": state.get("strategy", "delete"),
            "ttl_minutes": state["ttl_minutes"],
        }
    else:
        config = _global_ttl_config()
''',
    '''    if "ttl_minutes" in state:
        config = _normalize_ttl_config(
            {
                "strategy": state.get("strategy", "delete"),
                "ttl_minutes": state["ttl_minutes"],
            }
        )
    else:
        config = _global_ttl_config()
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''        filters = await Threads.handle_event(
            ctx,
            "create",
            Auth.types.ThreadsCreate(
                thread_id=thread_id, metadata=metadata, if_exists=if_exists
            ),
        )
        # Re-fetch in case an auth handler replaced the thread object in the store
        # (e.g. via a loopback patch call, which deep-copies and replaces the element).
        existing_thread = next(
''',
    '''        filters = await Threads.handle_event(
            ctx,
            "create",
            Auth.types.ThreadsCreate(
                thread_id=thread_id, metadata=metadata, if_exists=if_exists
            ),
        )
        await _wait_for_thread_ttl_maintenance(thread_id)
        # Re-fetch in case an auth handler or TTL maintenance replaced/removed
        # the thread while the request was awaiting authorization/maintenance.
        existing_thread = next(
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''        if thread_idx is not None:
            filters = await Threads.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id, metadata=metadata),
            )
            if not filters or _check_filter_match(
                thread_list[thread_idx]["metadata"], filters
            ):
                thread = copy.deepcopy(thread_list[thread_idx])
                thread.setdefault("state_updated_at", thread.get("updated_at"))
                thread["metadata"] = {**thread["metadata"], **metadata}
                thread["updated_at"] = datetime.now(UTC)
                thread_list[thread_idx] = thread
                if ttl is not None:
                    _set_thread_ttl_override(conn, thread_id, ttl)

                async def thread_iterator() -> AsyncIterator[Thread]:
                    yield thread

                return thread_iterator()
''',
    '''        if thread_idx is not None:
            filters = await Threads.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id, metadata=metadata),
            )
            await _wait_for_thread_ttl_maintenance(thread_id)

            # ``await`` above may have allowed TTL deletion or another patch to
            # replace the list element.  Resolve it again instead of using a
            # stale list index.
            thread_idx = next(
                (
                    idx
                    for idx, thread in enumerate(thread_list)
                    if thread["thread_id"] == thread_id
                ),
                None,
            )
            if thread_idx is not None and (
                not filters
                or _check_filter_match(thread_list[thread_idx]["metadata"], filters)
            ):
                thread = copy.deepcopy(thread_list[thread_idx])
                thread.setdefault("state_updated_at", thread.get("updated_at"))
                thread["metadata"] = {**thread["metadata"], **metadata}
                thread["updated_at"] = datetime.now(UTC)
                thread_list[thread_idx] = thread
                if ttl is not None:
                    _set_thread_ttl_override(conn, thread_id, ttl)

                async def thread_iterator() -> AsyncIterator[Thread]:
                    yield thread

                return thread_iterator()
''',
)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''            if assistant_filters and not _check_filter_match(
                assistant.get("metadata", {}), assistant_filters
            ):
                return _empty_generator()

        if existing_thread and filters:
''',
    '''            if assistant_filters and not _check_filter_match(
                assistant.get("metadata", {}), assistant_filters
            ):
                return _empty_generator()

        if thread_id is not None:
            await _wait_for_thread_ttl_maintenance(thread_id)
            # TTL maintenance can delete the thread while auth handlers are
            # running.  Re-resolve after the maintenance gate before the
            # mutation section, which contains no further await for an
            # existing thread.
            existing_thread = next(
                (t for t in conn.store["threads"] if t["thread_id"] == thread_id),
                None,
            )

        if existing_thread and filters:
''',
)

old_sweep = '''        processed = 0
        deleted = 0
        for offset in range(0, len(expired), batch_size):
            batch = expired[offset : offset + batch_size]
            keep_latest_ids = [
                str(thread_id)
                for _, thread_id, ttl_config, _, _ in batch
                if ttl_config["strategy"] == "keep_latest"
            ]
            if keep_latest_ids:
                if checkpointer is None:
                    checkpointer = await _get_checkpointer(conn)
                await checkpointer.aprune(keep_latest_ids, strategy="keep_latest")

            for _, thread_id, ttl_config, ttl_state, updated_at in batch:
                if ttl_config["strategy"] == "delete":
                    try:
                        deleted_iter = await Threads.delete(conn, thread_id)
                        async for _ in deleted_iter:
                            deleted += 1
                            processed += 1
                    except HTTPException as exc:
                        if exc.status_code != 404:
                            raise
                else:
                    # Do not prune the same inactive state every minute.  Any
                    # later thread update advances ``updated_at`` and makes it
                    # eligible again after another full TTL period.
                    ttl_state["last_swept_updated_at"] = updated_at
                    _thread_ttl_store(conn)[str(thread_id)] = ttl_state
                    processed += 1
'''

new_sweep = '''        processed = 0
        deleted = 0
        for offset in range(0, len(expired), batch_size):
            batch = expired[offset : offset + batch_size]
            for _, thread_id, _selected_config, _selected_state, _selected_updated in batch:
                thread_key = str(thread_id)
                if thread_key in _THREAD_TTL_MAINTENANCE:
                    continue

                # Claim the thread before the destructive await.  Runs.put and
                # Threads.put/patch wait on this marker before their final
                # metadata mutation, so a new foreground run cannot appear
                # halfway through TTL deletion/pruning.
                _THREAD_TTL_MAINTENANCE.add(thread_key)
                try:
                    current_thread = next(
                        (
                            thread
                            for thread in conn.store["threads"]
                            if thread["thread_id"] == thread_id
                        ),
                        None,
                    )
                    if current_thread is None:
                        continue

                    # Revalidate all mutable eligibility inputs after earlier
                    # threads in this sweep may have awaited external cleanup.
                    if any(
                        run["thread_id"] == thread_id
                        and run["status"] in {"pending", "running"}
                        for run in conn.store["runs"]
                    ):
                        continue

                    try:
                        ttl_config, ttl_state = _effective_thread_ttl(conn, thread_id)
                    except HTTPException as exc:
                        logger.warning(
                            "Skipping thread with invalid TTL state",
                            thread_id=thread_key,
                            detail=exc.detail,
                        )
                        continue
                    if ttl_config is None:
                        continue

                    updated_at = _as_utc(current_thread["updated_at"])
                    last_swept = ttl_state.get("last_swept_updated_at")
                    if (
                        ttl_config["strategy"] == "keep_latest"
                        and last_swept is not None
                        and _as_utc(last_swept) >= updated_at
                    ):
                        continue

                    expires_at = updated_at + timedelta(
                        minutes=ttl_config["ttl_minutes"]
                    )
                    if expires_at > datetime.now(UTC):
                        continue

                    if ttl_config["strategy"] == "delete":
                        try:
                            deleted_iter = await Threads.delete(conn, thread_id)
                            async for _ in deleted_iter:
                                deleted += 1
                                processed += 1
                        except HTTPException as exc:
                            if exc.status_code != 404:
                                raise
                    else:
                        if checkpointer is None:
                            checkpointer = await _get_checkpointer(conn)
                        await checkpointer.aprune(
                            [thread_key], strategy="keep_latest"
                        )
                        # Do not prune the same inactive state every minute.
                        # Any later thread update advances ``updated_at`` and
                        # makes it eligible again after another full TTL period.
                        ttl_state["last_swept_updated_at"] = updated_at
                        _thread_ttl_store(conn)[thread_key] = ttl_state
                        processed += 1
                finally:
                    _THREAD_TTL_MAINTENANCE.discard(thread_key)
'''
replace_once("src/langgraph_runtime_inmem/ops.py", old_sweep, new_sweep)

replace_once(
    "src/langgraph_runtime_inmem/ops.py",
    '''            ttl_config, ttl_state = _effective_thread_ttl(conn, thread_id)
            if ttl_config is None:
                continue
''',
    '''            try:
                ttl_config, ttl_state = _effective_thread_ttl(conn, thread_id)
            except HTTPException as exc:
                logger.warning(
                    "Skipping thread with invalid TTL state",
                    thread_id=str(thread_id),
                    detail=exc.detail,
                )
                continue
            if ttl_config is None:
                continue
''',
)

# ---------------------------------------------------------------------------
# 4. keep_latest: only preserve ancestry for channels explicitly identified as
#    DeltaChannel state; add a cycle guard.
# ---------------------------------------------------------------------------

replace_once(
    "src/langgraph_runtime_inmem/checkpoint.py",
    '''                latest_entry = checkpoints[latest_id]
                latest_checkpoint = self.serde.loads_typed(latest_entry[0])

                # Missing/empty values need the ancestor write chain.  This is
                # how DeltaChannel stores non-snapshot checkpoints.
                needed_channels = {
                    channel
                    for channel, version in latest_checkpoint.get(
                        "channel_versions", {}
                    ).items()
                    if (
                        (
                            blob := self.blobs.get(
                                (thread_id, checkpoint_ns, channel, version)
                            )
                        )
                        is None
                        or blob[0] == "empty"
                    )
                }

                parent_id = latest_entry[2]
                while parent_id is not None and needed_channels:
                    parent_entry = checkpoints.get(parent_id)
                    if parent_entry is None:
                        break
                    keep_ids.add(parent_id)
''',
    '''                latest_entry = checkpoints[latest_id]
                latest_checkpoint = self.serde.loads_typed(latest_entry[0])
                latest_metadata = self.serde.loads_typed(latest_entry[1])
                delta_counters = (
                    latest_metadata.get("counters_since_delta_snapshot", {})
                    if isinstance(latest_metadata, dict)
                    else {}
                )
                if not isinstance(delta_counters, dict):
                    delta_counters = {}

                # Only DeltaChannel state needs the ancestor write chain.  A
                # generic missing/empty blob without this metadata must not
                # cause unrelated history to be retained indefinitely.
                needed_channels = {
                    channel
                    for channel, version in latest_checkpoint.get(
                        "channel_versions", {}
                    ).items()
                    if channel in delta_counters
                    and (
                        (
                            blob := self.blobs.get(
                                (thread_id, checkpoint_ns, channel, version)
                            )
                        )
                        is None
                        or blob[0] == "empty"
                    )
                }

                parent_id = latest_entry[2]
                visited: set[Any] = set()
                while parent_id is not None and needed_channels:
                    if parent_id in visited:
                        break
                    visited.add(parent_id)
                    parent_entry = checkpoints.get(parent_id)
                    if parent_entry is None:
                        break
                    keep_ids.add(parent_id)
''',
)

# ---------------------------------------------------------------------------
# 5. Background TTL interval validation.
# ---------------------------------------------------------------------------

replace_once(
    "src/langgraph_runtime_inmem/thread_ttl.py",
    '''import asyncio
from typing import Any
''',
    '''import asyncio
import math
from typing import Any
''',
)

replace_once(
    "src/langgraph_runtime_inmem/thread_ttl.py",
    '''    interval_minutes = float(config.get("sweep_interval_minutes", 5))
    # Zero is useful for integration tests but must not create a hot loop.
    interval_seconds = max(interval_minutes * 60, 0.1)
''',
    '''    try:
        interval_minutes = float(config.get("sweep_interval_minutes", 5))
    except (TypeError, ValueError):
        raise ValueError(
            "Thread TTL sweep interval must be a number of minutes."
        ) from None
    if not math.isfinite(interval_minutes) or interval_minutes < 0:
        raise ValueError(
            "Thread TTL sweep interval must be finite and greater than or equal to zero."
        )
    # Zero is useful for integration tests but must not create a hot loop.
    interval_seconds = max(interval_minutes * 60, 0.1)
''',
)

# ---------------------------------------------------------------------------
# Regression tests.
# ---------------------------------------------------------------------------

append_once(
    "tests/test_persistence.py",
    "test_denied_duplicate_cannot_clobber_after_owner_closes",
    r'''
def test_denied_duplicate_cannot_clobber_after_owner_closes(
    tmp_path: Path, isolated_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_persistence, "_flush_interval", 60)
    path = tmp_path / "late-contender.pckl"
    owner = PersistentDict(dict, {"source": "owner-initial"}, filename=str(path))
    contender = PersistentDict(
        dict, {"source": "contender-stale"}, filename=str(path)
    )

    _persistence.register_persistent_dict(owner)
    _persistence.register_persistent_dict(contender)
    owner["source"] = "owner-final"

    _persistence.close_persistent_dict(owner)
    assert _load_pickle(path) == {"source": "owner-final"}

    # The denied contender is closed only after the registry became empty.
    # It still must not acquire write permission retroactively.
    _persistence.close_persistent_dict(contender)
    assert _load_pickle(path) == {"source": "owner-final"}
''',
)

append_once(
    "tests/test_inmem_stream.py",
    "test_resumable_stream_ids_are_unique_and_numerically_ordered",
    r'''
@pytest.mark.asyncio
async def test_resumable_stream_ids_are_unique_and_numerically_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inmem_stream.time, "time", lambda: 1234.567)
    monkeypatch.setattr(inmem_stream, "_last_stream_ms", -1)
    monkeypatch.setattr(inmem_stream, "_last_stream_seq", -1)

    manager = StreamManager()
    thread_id = uuid4()
    run_id = uuid4()

    for index in range(12):
        await manager.put(
            run_id,
            thread_id,
            Message(
                topic=f"run:{run_id}:stream".encode(),
                data=str(index).encode(),
            ),
            resumable=True,
        )

    messages = manager.message_stores[thread_id][run_id]
    ids = [message.id.decode() for message in messages]
    assert ids[0] == "1234567-0"
    assert ids[-1] == "1234567-11"
    assert len(ids) == len(set(ids))

    # Numeric tuple ordering must handle sequence 10/11 correctly.  Plain
    # string comparison would incorrectly place "-10" before "-2".
    restored = list(manager.restore_messages(run_id, thread_id, ids[2]))
    assert [message.data for message in restored] == [
        str(index).encode() for index in range(3, 12)
    ]
''',
)

replace_once(
    "tests/test_thread_ttl.py",
    '''    monkeypatch.setattr(api_config, "THREAD_TTL", None)
    monkeypatch.setattr(api_config, "USE_CUSTOM_CHECKPOINTER", False)
''',
    '''    monkeypatch.setattr(api_config, "THREAD_TTL", None)
    monkeypatch.setattr(api_config, "USE_CUSTOM_CHECKPOINTER", False)
    ops._THREAD_TTL_MAINTENANCE.clear()
''',
)

replace_once(
    "tests/test_thread_ttl.py",
    '''    add_checkpoint(
        saver,
        thread_id,
        "0003",
        parent_id="0002",
        value=["delta-2"],
        materialized=False,
    )
''',
    '''    add_checkpoint(
        saver,
        thread_id,
        "0003",
        parent_id="0002",
        value=["delta-2"],
        materialized=False,
        metadata_extra={"counters_since_delta_snapshot": {"state": (2, 3)}},
    )
''',
)

append_once(
    "tests/test_thread_ttl.py",
    "test_invalid_thread_ttl_values_are_rejected",
    r'''
@pytest.mark.parametrize("ttl", ["nan", "inf", "-inf", 1e308])
def test_invalid_thread_ttl_values_are_rejected(ttl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        ops._normalize_ttl_config({"strategy": "delete", "ttl": ttl})
    assert exc_info.value.status_code == 422


def test_keep_latest_does_not_retain_non_delta_empty_ancestors(saver) -> None:
    thread_id = uuid4()
    add_checkpoint(saver, thread_id, "0001", value="materialized")
    add_checkpoint(
        saver,
        thread_id,
        "0002",
        parent_id="0001",
        value="ordinary-empty",
        materialized=False,
    )

    saver.prune([str(thread_id)], strategy="keep_latest")

    assert set(saver.storage[str(thread_id)][""]) == {"0002"}
    assert (str(thread_id), "", "0001") not in saver.writes


@pytest.mark.asyncio
async def test_ttl_sweep_revalidates_activity_after_await(
    conn, saver, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = StreamManager()
    monkeypatch.setattr(ops, "get_stream_manager", lambda: manager)

    first = await create_thread(
        conn, ttl={"strategy": "delete", "ttl": 1}, age_minutes=3
    )
    second = await create_thread(
        conn, ttl={"strategy": "delete", "ttl": 1}, age_minutes=2
    )

    async def cleanup(thread_id, _conn, run_id=None):
        # Deleting the older first thread yields to external cleanup.  During
        # that await, a foreground run becomes active on the second thread.
        if thread_id == first["thread_id"]:
            conn.store["runs"].append(
                {
                    "run_id": uuid4(),
                    "thread_id": second["thread_id"],
                    "status": "running",
                }
            )
        await asyncio.sleep(0)

    monkeypatch.setattr(ops, "_delete_checkpoints_for_thread", cleanup)

    assert await ops.Threads.sweep_ttl(conn) == (1, 1)
    remaining = {thread["thread_id"] for thread in conn.store["threads"]}
    assert first["thread_id"] not in remaining
    assert second["thread_id"] in remaining


@pytest.mark.asyncio
async def test_run_creation_waits_for_ttl_maintenance(conn) -> None:
    thread = await create_thread(conn)
    assistant_id = uuid4()
    run_id = uuid4()
    conn.store["assistants"].append(
        {
            "assistant_id": assistant_id,
            "graph_id": "graph",
            "config": {},
            "context": {},
            "metadata": {"created_by": "system"},
        }
    )

    thread_key = str(thread["thread_id"])
    ops._THREAD_TTL_MAINTENANCE.add(thread_key)
    task = asyncio.create_task(
        ops.Runs.put(
            conn,
            assistant_id,
            {"config": {}},
            thread_id=thread["thread_id"],
            run_id=run_id,
            metadata={},
            prevent_insert_if_inflight=False,
        )
    )
    await asyncio.sleep(0)
    assert not task.done()

    ops._THREAD_TTL_MAINTENANCE.discard(thread_key)
    iterator = await asyncio.wait_for(task, timeout=1)
    created = await anext(iterator)
    assert created["run_id"] == run_id
''',
)

append_once(
    "tests/test_thread_ttl_loop.py",
    "test_background_loop_rejects_invalid_interval",
    r'''
@pytest.mark.asyncio
@pytest.mark.parametrize("interval", ["nan", "inf", "-1", -1])
async def test_background_loop_rejects_invalid_interval(
    monkeypatch: pytest.MonkeyPatch, interval
) -> None:
    from langgraph_api import config as api_config

    monkeypatch.setattr(
        api_config,
        "THREAD_TTL",
        {
            "strategy": "delete",
            "default_ttl": 1,
            "sweep_interval_minutes": interval,
        },
    )
    with pytest.raises(ValueError, match="finite"):
        await thread_ttl.thread_ttl_sweep_loop()
''',
)

print("Applied audit fixes successfully.")