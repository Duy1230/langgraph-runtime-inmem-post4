from __future__ import annotations

from uuid import uuid4

import pytest

from langgraph_runtime_inmem import inmem_stream
from langgraph_runtime_inmem.inmem_stream import (
    STREAM_CLOSE_TOPIC,
    Message,
    StreamManager,
)


def _all_maps_empty(manager: StreamManager) -> bool:
    return not any(
        (
            manager.queues,
            manager.control_keys,
            manager.control_queues,
            manager.thread_streams,
            manager.message_stores,
        )
    )


def test_missing_stream_lookups_do_not_allocate_state() -> None:
    manager = StreamManager()
    run_id = uuid4()
    thread_id = uuid4()

    assert manager.get_queues(run_id, thread_id) == []
    assert manager.get_control_queues(run_id, thread_id) == []
    assert manager.get_control_key(run_id, thread_id) is None
    assert list(manager.restore_messages(run_id, thread_id, "0-0")) == []
    assert manager.get_queues_by_thread_id(thread_id) == []
    assert _all_maps_empty(manager)


@pytest.mark.asyncio
async def test_run_delete_cleans_only_target_run_state() -> None:
    from langgraph_api.utils.stream_codec import decode_stream_message

    manager = StreamManager()
    thread_id = uuid4()
    other_thread_id = uuid4()
    target_run_id = uuid4()
    sibling_run_id = uuid4()
    other_run_id = uuid4()

    target_queue = await manager.add_queue(target_run_id, thread_id)
    target_control_queue = await manager.add_control_queue(target_run_id, thread_id)
    sibling_queue = await manager.add_queue(sibling_run_id, thread_id)
    sibling_control_queue = await manager.add_control_queue(sibling_run_id, thread_id)
    await manager.add_queue(other_run_id, other_thread_id)
    thread_queue = await manager.add_thread_stream(thread_id)

    await manager.put(
        target_run_id,
        thread_id,
        Message(topic=f"run:{target_run_id}:stream".encode(), data=b"event"),
        resumable=True,
    )
    await manager.put(
        target_run_id,
        thread_id,
        Message(topic=f"run:{target_run_id}:control".encode(), data=b"interrupt"),
    )
    await manager.put(
        sibling_run_id,
        thread_id,
        Message(topic=f"run:{sibling_run_id}:stream".encode(), data=b"sibling"),
        resumable=True,
    )
    await manager.put(
        sibling_run_id,
        thread_id,
        Message(topic=f"run:{sibling_run_id}:control".encode(), data=b"interrupt"),
    )

    await manager.delete_run(target_run_id, thread_id)

    for mapping in (
        manager.queues,
        manager.control_keys,
        manager.control_queues,
        manager.message_stores,
    ):
        assert target_run_id not in mapping.get(thread_id, {})
    assert manager.queues[thread_id][sibling_run_id] == [sibling_queue]
    assert manager.control_queues[thread_id][sibling_run_id] == [
        sibling_control_queue
    ]
    assert manager.control_keys[thread_id][sibling_run_id].data == b"interrupt"
    assert sibling_run_id in manager.message_stores[thread_id]
    assert other_thread_id in manager.queues
    assert manager.thread_streams[thread_id] == [thread_queue]

    # Existing subscribers are released even though the manager no longer
    # retains their queues.
    target_messages = []
    while not target_queue.empty():
        target_messages.append(target_queue.get_nowait())
    done = decode_stream_message(
        target_messages[-1].data, channel=target_messages[-1].topic
    )
    assert done.event_bytes == b"control"
    assert done.message_bytes == b"done"
    assert target_control_queue.get_nowait().data == b"interrupt"
    assert target_control_queue.get_nowait().data == b"done"


@pytest.mark.asyncio
async def test_active_run_delete_cannot_be_recreated_by_final_publication() -> None:
    manager = StreamManager()
    thread_id = uuid4()
    run_id = uuid4()
    manager.mark_run_active(run_id, thread_id)
    await manager.add_queue(run_id, thread_id)

    await manager.delete_run(run_id, thread_id)
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:stream".encode(), data=b"late"),
        resumable=True,
    )
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:control".encode(), data=b"done"),
    )

    assert not manager.queues
    assert not manager.control_keys
    assert not manager.message_stores
    assert (thread_id, run_id) in manager._pending_run_deletes

    manager.mark_run_inactive(run_id, thread_id)
    assert not manager._active_runs
    assert not manager._pending_run_deletes
    assert _all_maps_empty(manager)


@pytest.mark.asyncio
async def test_thread_delete_cleans_all_owned_state_and_preserves_other_thread() -> None:
    manager = StreamManager()
    target_thread_id = uuid4()
    other_thread_id = uuid4()
    target_runs = (uuid4(), uuid4())
    other_run_id = uuid4()

    for run_id in target_runs:
        await manager.add_queue(run_id, target_thread_id)
        await manager.add_control_queue(run_id, target_thread_id)
        await manager.put(
            run_id,
            target_thread_id,
            Message(topic=f"run:{run_id}:stream".encode(), data=b"event"),
            resumable=True,
        )
        await manager.put(
            run_id,
            target_thread_id,
            Message(topic=f"run:{run_id}:control".encode(), data=b"interrupt"),
        )
    target_thread_queue = await manager.add_thread_stream(target_thread_id)

    other_queue = await manager.add_queue(other_run_id, other_thread_id)
    other_control_queue = await manager.add_control_queue(
        other_run_id, other_thread_id
    )
    other_thread_queue = await manager.add_thread_stream(other_thread_id)
    await manager.put(
        other_run_id,
        other_thread_id,
        Message(topic=f"run:{other_run_id}:stream".encode(), data=b"other"),
        resumable=True,
    )
    await manager.put(
        other_run_id,
        other_thread_id,
        Message(topic=f"run:{other_run_id}:control".encode(), data=b"interrupt"),
    )

    await manager.delete_thread(target_thread_id)

    for mapping in (
        manager.queues,
        manager.control_keys,
        manager.control_queues,
        manager.message_stores,
        manager.thread_streams,
    ):
        assert target_thread_id not in mapping
    assert manager.queues[other_thread_id][other_run_id] == [other_queue]
    assert manager.control_queues[other_thread_id][other_run_id] == [
        other_control_queue
    ]
    assert manager.control_keys[other_thread_id][other_run_id].data == b"interrupt"
    assert manager.thread_streams[other_thread_id] == [other_thread_queue]
    assert other_run_id in manager.message_stores[other_thread_id]
    assert target_thread_queue.get_nowait().topic == STREAM_CLOSE_TOPIC


@pytest.mark.asyncio
async def test_remove_subscriptions_prunes_empty_parent_buckets() -> None:
    manager = StreamManager()
    thread_id = uuid4()
    run_id = uuid4()
    queue = await manager.add_queue(run_id, thread_id)
    control_queue = await manager.add_control_queue(run_id, thread_id)
    thread_queue = await manager.add_thread_stream(thread_id)

    await manager.remove_queue(run_id, thread_id, queue)
    await manager.remove_control_queue(run_id, thread_id, control_queue)
    await manager.remove_thread_stream(thread_id, thread_queue)

    assert _all_maps_empty(manager)


@pytest.mark.asyncio
async def test_stop_stream_signals_and_clears_every_state_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = StreamManager()
    thread_id = uuid4()
    run_id = uuid4()
    queue = await manager.add_queue(run_id, thread_id)
    control_queue = await manager.add_control_queue(run_id, thread_id)
    thread_queue = await manager.add_thread_stream(thread_id)
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:stream".encode(), data=b"event"),
        resumable=True,
    )
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:control".encode(), data=b"interrupt"),
    )
    manager.mark_run_active(run_id, thread_id)
    monkeypatch.setattr(inmem_stream, "stream_manager", manager)

    await inmem_stream.stop_stream()

    assert _all_maps_empty(manager)
    assert not manager._active_runs
    assert not manager._pending_run_deletes
    assert manager._stopped is True
    assert not queue.empty()
    assert not control_queue.empty()
    assert thread_queue.get_nowait().topic == STREAM_CLOSE_TOPIC

    # An old run context completing after global shutdown cannot recreate data.
    await manager.put(
        run_id,
        thread_id,
        Message(topic=f"run:{run_id}:stream".encode(), data=b"late"),
        resumable=True,
    )
    assert _all_maps_empty(manager)

    await inmem_stream.start_stream()
    replacement = inmem_stream.get_stream_manager()
    assert replacement is not manager
    assert replacement._stopped is False
    assert _all_maps_empty(replacement)


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
