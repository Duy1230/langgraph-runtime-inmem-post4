import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID

logger = logging.getLogger(__name__)

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


@dataclass
class Message:
    topic: bytes
    data: bytes
    id: bytes | None = None


class ContextQueue(asyncio.Queue):
    """Queue that supports async context manager protocol"""

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        # Clear the queue
        while not self.empty():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                break


THREADLESS_KEY = "no-thread"
STREAM_CLOSE_TOPIC = b"__langgraph_inmem_stream_closed__"


def _thread_key(thread_id: UUID | str | None) -> UUID | str:
    return THREADLESS_KEY if thread_id is None else _ensure_uuid(thread_id)


class StreamManager:
    def __init__(self):
        self.queues = defaultdict(
            lambda: defaultdict(list)
        )  # Dict[str, List[asyncio.Queue]]
        self.control_keys = defaultdict(lambda: defaultdict())
        self.control_queues = defaultdict(lambda: defaultdict(list))
        self.thread_streams = defaultdict(list)

        self.message_stores = defaultdict(
            lambda: defaultdict(list[Message])
        )  # Dict[str, List[Message]]
        # A pending delete exists only while the corresponding run context is
        # still active.  It prevents that context's final publications from
        # recreating state after deletion; ``mark_run_inactive`` removes it.
        self._active_runs: set[tuple[UUID | str, UUID]] = set()
        self._pending_run_deletes: set[tuple[UUID | str, UUID]] = set()
        self._stopped = False

    def get_queues(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> list[asyncio.Queue]:
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        return self.queues.get(thread_id, {}).get(run_id, [])

    def get_control_queues(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> list[asyncio.Queue]:
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        return self.control_queues.get(thread_id, {}).get(run_id, [])

    def get_control_key(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> Message | None:
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        return self.control_keys.get(thread_id, {}).get(run_id)

    async def put(
        self,
        run_id: UUID | str | None,
        thread_id: UUID | str | None,
        message: Message,
        resumable: bool = False,
    ) -> None:
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        run_key = (thread_id, run_id)
        if self._stopped or run_key in self._pending_run_deletes:
            return

        message.id = _generate_ms_seq_id().encode()
        # For resumable run streams, embed the generated message ID into the frame
        topic = message.topic.decode()
        if resumable:
            self.message_stores[thread_id][run_id].append(message)
        if "control" in topic:
            self.control_keys[thread_id][run_id] = message
            queues = self.control_queues.get(thread_id, {}).get(run_id, ())
        else:
            queues = self.queues.get(thread_id, {}).get(run_id, ())
        coros = [queue.put(message) for queue in queues]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.exception(f"Failed to put message in queue: {result}")

    async def put_thread(
        self,
        thread_id: UUID | str,
        message: Message,
    ) -> None:
        thread_id = _ensure_uuid(thread_id)
        if self._stopped:
            return
        message.id = _generate_ms_seq_id().encode()
        queues = self.thread_streams.get(thread_id, ())
        coros = [queue.put(message) for queue in queues]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.exception(f"Failed to put message in queue: {result}")

    async def add_queue(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> asyncio.Queue:
        run_id = _ensure_uuid(run_id)
        queue = ContextQueue()
        thread_id = _thread_key(thread_id)
        if self._stopped or (thread_id, run_id) in self._pending_run_deletes:
            self._close_queue(queue)
            return queue
        self.queues[thread_id][run_id].append(queue)
        return queue

    async def add_control_queue(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> asyncio.Queue:
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        queue = ContextQueue()
        if self._stopped or (thread_id, run_id) in self._pending_run_deletes:
            self._finish_control_queue(queue, run_id)
            return queue
        self.control_queues[thread_id][run_id].append(queue)
        return queue

    async def add_thread_stream(self, thread_id: UUID | str) -> asyncio.Queue:
        thread_id = _ensure_uuid(thread_id)
        queue = ContextQueue()
        if self._stopped:
            self._close_queue(queue)
            return queue
        self.thread_streams[thread_id].append(queue)
        return queue

    async def remove_queue(
        self, run_id: UUID | str, thread_id: UUID | str | None, queue: asyncio.Queue
    ):
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        queues_by_run = self.queues.get(thread_id)
        if queues_by_run is None or run_id not in queues_by_run:
            return
        try:
            queues_by_run[run_id].remove(queue)
        except ValueError:
            return
        if not queues_by_run[run_id]:
            del queues_by_run[run_id]
        if not queues_by_run:
            self.queues.pop(thread_id, None)

    async def remove_control_queue(
        self, run_id: UUID | str, thread_id: UUID | str | None, queue: asyncio.Queue
    ):
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        queues_by_run = self.control_queues.get(thread_id)
        if queues_by_run is None or run_id not in queues_by_run:
            return
        try:
            queues_by_run[run_id].remove(queue)
        except ValueError:
            return
        if not queues_by_run[run_id]:
            del queues_by_run[run_id]
        if not queues_by_run:
            self.control_queues.pop(thread_id, None)

    async def remove_thread_stream(
        self, thread_id: UUID | str, queue: asyncio.Queue
    ) -> None:
        thread_id = _ensure_uuid(thread_id)
        queues = self.thread_streams.get(thread_id)
        if queues is None:
            return
        try:
            queues.remove(queue)
        except ValueError:
            return
        if not queues:
            self.thread_streams.pop(thread_id, None)

    def restore_messages(
        self, run_id: UUID | str, thread_id: UUID | str | None, message_id: str | None
    ) -> Iterator[Message]:
        """Get a stored message by ID for resumable streams."""
        run_id = _ensure_uuid(run_id)
        thread_id = _thread_key(thread_id)
        if message_id is None:
            return
        messages = self.message_stores.get(thread_id, {}).get(run_id, ())

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

    def get_queues_by_thread_id(self, thread_id: UUID | str) -> list[asyncio.Queue]:
        """Get all queues for a specific thread_id across all runs."""
        all_queues = []
        # Search through all stored queue keys for ones ending with the thread_id
        thread_id = _ensure_uuid(thread_id)
        if thread_id in self.queues:
            for run_id in self.queues[thread_id]:
                all_queues.extend(self.queues[thread_id][run_id])

        return all_queues

    def mark_run_active(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> None:
        run_key = (_thread_key(thread_id), _ensure_uuid(run_id))
        if not self._stopped and run_key not in self._pending_run_deletes:
            self._active_runs.add(run_key)

    def mark_run_inactive(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> None:
        run_key = (_thread_key(thread_id), _ensure_uuid(run_id))
        self._active_runs.discard(run_key)
        if run_key in self._pending_run_deletes:
            self._clear_run_state(*run_key)
            self._pending_run_deletes.discard(run_key)

    @staticmethod
    def _close_queue(queue: asyncio.Queue) -> None:
        message = Message(topic=STREAM_CLOSE_TOPIC, data=b"done")
        message.id = _generate_ms_seq_id().encode()
        try:
            queue.put_nowait(message)
        except (asyncio.QueueFull, RuntimeError):
            pass

    @staticmethod
    def _finish_control_queue(queue: asyncio.Queue, run_id: UUID) -> None:
        message = Message(topic=f"run:{run_id}:control".encode(), data=b"done")
        message.id = _generate_ms_seq_id().encode()
        try:
            queue.put_nowait(message)
        except (asyncio.QueueFull, RuntimeError):
            pass

    @staticmethod
    def _finish_run_queue(queue: asyncio.Queue, run_id: UUID) -> None:
        from langgraph_api.utils.stream_codec import STREAM_CODEC  # noqa: PLC0415

        message = Message(
            topic=f"run:{run_id}:stream".encode(),
            data=STREAM_CODEC.encode("control", b"done"),
        )
        message.id = _generate_ms_seq_id().encode()
        try:
            queue.put_nowait(message)
        except (asyncio.QueueFull, RuntimeError):
            pass

    def _signal_run(self, thread_id: UUID | str, run_id: UUID) -> None:
        for queue in tuple(self.queues.get(thread_id, {}).get(run_id, ())):
            self._finish_run_queue(queue, run_id)
        for queue in tuple(self.control_queues.get(thread_id, {}).get(run_id, ())):
            self._finish_control_queue(queue, run_id)

    def _clear_run_state(self, thread_id: UUID | str, run_id: UUID) -> None:
        for mapping in (
            self.queues,
            self.control_queues,
            self.control_keys,
            self.message_stores,
        ):
            values_by_run = mapping.get(thread_id)
            if values_by_run is None:
                continue
            values_by_run.pop(run_id, None)
            if not values_by_run:
                mapping.pop(thread_id, None)

    async def delete_run(
        self, run_id: UUID | str, thread_id: UUID | str | None
    ) -> None:
        """Release only the stream state owned by one run."""
        run_key = (_thread_key(thread_id), _ensure_uuid(run_id))
        self._signal_run(*run_key)
        if run_key in self._active_runs:
            self._pending_run_deletes.add(run_key)
        self._clear_run_state(*run_key)

    async def delete_thread(self, thread_id: UUID | str) -> None:
        """Release all run and thread-stream state for one thread."""
        thread_id = _ensure_uuid(thread_id)
        run_ids: set[UUID] = {
            run_id
            for mapping in (
                self.queues,
                self.control_queues,
                self.control_keys,
                self.message_stores,
            )
            for run_id in mapping.get(thread_id, {})
        }
        run_ids.update(
            run_id
            for active_thread_id, run_id in self._active_runs
            if active_thread_id == thread_id
        )
        for run_id in run_ids:
            self._signal_run(thread_id, run_id)
            run_key = (thread_id, run_id)
            if run_key in self._active_runs:
                self._pending_run_deletes.add(run_key)
            self._clear_run_state(thread_id, run_id)
        for mapping in (
            self.queues,
            self.control_queues,
            self.control_keys,
            self.message_stores,
        ):
            mapping.pop(thread_id, None)
        for queue in tuple(self.thread_streams.get(thread_id, ())):
            self._close_queue(queue)
        self.thread_streams.pop(thread_id, None)

    async def close(self) -> None:
        """Signal subscribers and drop every resource owned by this manager."""
        self._stopped = True
        run_keys = {
            (thread_id, run_id)
            for mapping in (
                self.queues,
                self.control_queues,
                self.control_keys,
                self.message_stores,
            )
            for thread_id, values_by_run in mapping.items()
            for run_id in values_by_run
        }
        for thread_id, run_id in run_keys:
            self._signal_run(thread_id, run_id)
        for queues in self.thread_streams.values():
            for queue in tuple(queues):
                self._close_queue(queue)
        self.queues.clear()
        self.control_keys.clear()
        self.control_queues.clear()
        self.thread_streams.clear()
        self.message_stores.clear()
        self._active_runs.clear()
        self._pending_run_deletes.clear()


# Global instance
stream_manager = StreamManager()


async def start_stream() -> None:
    """Initialize the queue system.
    In this in-memory implementation, we just need to ensure we have a clean StreamManager instance.
    """
    global stream_manager
    await stream_manager.close()
    stream_manager = StreamManager()


async def stop_stream() -> None:
    """Clean up the queue system.
    Clear all queues and stored control messages."""
    global stream_manager

    await stream_manager.close()


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance."""
    return stream_manager
