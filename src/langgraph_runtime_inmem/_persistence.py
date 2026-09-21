"""Periodic flushing for all PersistentDict stores."""

from __future__ import annotations

import functools
import logging
import os
import threading
import weakref

from langgraph.checkpoint.memory import PersistentDict

logger = logging.getLogger(__name__)

_stores: dict[str, weakref.ref[PersistentDict]] = {}
_flush_thread: tuple[threading.Event, threading.Thread] | None = None
_flush_interval: float = 10
_registry_lock = threading.RLock()
_PERSISTENCE_OWNER_ATTR = "_langgraph_runtime_persistence_owner"
DISABLE_FILE_PERSISTENCE = (
    os.getenv("LANGGRAPH_DISABLE_FILE_PERSISTENCE", "false").lower() == "true"
)


def register_persistent_dict(d: PersistentDict) -> None:
    """Register ``d`` for periodic flushing without replacing a live owner.

    A filename has exactly one live owner.  This matters for checkpoint
    adapters: constructing a serializer-specific saver used to create three
    short-lived, empty ``PersistentDict`` instances for the same filenames as
    the singleton saver.  Replacing the registry entries with those weakrefs
    could both disable future flushes and let a transient instance overwrite a
    newer snapshot.
    """
    if DISABLE_FILE_PERSISTENCE:
        return
    global _flush_thread
    with _registry_lock:
        current_ref = _stores.get(d.filename)
        current = current_ref() if current_ref is not None else None
        if current is not None and current is not d:
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
        if _flush_thread is None or not _flush_thread[1].is_alive():
            logger.info("Starting dev persistence flush loop")
            stop_event = threading.Event()
            thread = threading.Thread(
                target=functools.partial(_flush_loop, stop_event), daemon=True
            )
            _flush_thread = (stop_event, thread)
            thread.start()


def unregister_persistent_dict(d: PersistentDict) -> None:
    """Unregister ``d`` if it is still the owner of its filename.

    The identity check prevents a late close/finalizer for an old instance
    from unregistering a newer replacement.
    """
    with _registry_lock:
        current_ref = _stores.get(d.filename)
        if current_ref is not None and current_ref() is d:
            _stores.pop(d.filename, None)


def close_persistent_dict(d: PersistentDict) -> None:
    """Close ``d`` without allowing a denied duplicate to clobber its owner.

    For an owner, keeping the registry lock through ``close`` prevents a new
    owner for the same filename from being registered between the old owner's
    final sync and its removal.  A live non-owner is cleared without syncing.
    """
    with _registry_lock:
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


def stop_flush_loop() -> None:
    """Stop the background flush thread."""
    global _flush_thread
    with _registry_lock:
        flush_thread = _flush_thread
        # Clear the slot before joining so a concurrent registration can start
        # a replacement loop instead of being stranded behind a stopped one.
        _flush_thread = None
        if flush_thread is not None:
            logger.info("Stopping dev persistence flush loop")
            flush_thread[0].set()
    if flush_thread is not None and flush_thread[1] is not threading.current_thread():
        flush_thread[1].join()


def _flush_registered_once() -> None:
    """Flush one stable snapshot of the registry.

    Registration, unregistration, and a filename's sync are serialized by the
    same lock.  A failing store is retained for a later retry and cannot kill
    the background loop.
    """
    with _registry_lock:
        for store_key, store_ref in list(_stores.items()):
            # A prior iteration/callback may have installed another owner.
            if _stores.get(store_key) is not store_ref:
                continue
            store = store_ref()
            if store is None:
                _stores.pop(store_key, None)
                continue
            try:
                # An empty PersistentDict is still live and must be synced: an
                # empty snapshot can represent a real deletion.
                store.sync()
            except Exception:
                logger.exception("Failed to flush persistent store %s", store_key)


def _flush_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(timeout=_flush_interval):
        _flush_registered_once()
    logger.info("dev persistence flush loop exiting")
