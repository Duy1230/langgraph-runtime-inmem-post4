from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from langgraph_runtime_inmem import thread_ttl


@pytest.mark.asyncio
async def test_background_loop_invokes_sweeper(monkeypatch) -> None:
    from langgraph_api import config as api_config

    config = {
        "strategy": "keep_latest",
        "default_ttl": 43200,
        "sweep_interval_minutes": 1,
        "sweep_limit": 7,
    }
    monkeypatch.setattr(api_config, "THREAD_TTL", config)

    connection = object()

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield connection

    calls = []
    log_events = []

    async def fake_sweep(conn, *, limit, batch_size, stats):
        calls.append((conn, limit, batch_size))
        stats.update(
            {
                "deleted_items": 4,
                "deleted_size_bytes": 256,
                "total_before_bytes": 1024,
                "size_tracking_available": True,
            }
        )
        return (1, 0)

    class FakeLogger:
        async def ainfo(self, event, **fields):
            log_events.append((event, fields))

        def exception(self, *args, **kwargs):
            raise AssertionError("The sweep loop should not log an exception")

    sleep_calls = 0

    async def fake_sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        assert seconds == 60
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(thread_ttl, "connect", fake_connect)
    monkeypatch.setattr(thread_ttl.Threads, "sweep_ttl", fake_sweep)
    monkeypatch.setattr(thread_ttl.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(thread_ttl, "logger", FakeLogger())

    with pytest.raises(asyncio.CancelledError):
        await thread_ttl.thread_ttl_sweep_loop()

    assert calls == [(connection, 7, 100)]
    _, fields = next(
        event for event in log_events if event[0] == "Checkpoint TTL sweep completed"
    )
    assert fields["deleted_items"] == 4
    assert fields["deleted_size"] == "256 B"
    assert fields["total_before"] == "1.0 KB"
    assert fields["deleted_ratio"] == "25.0%"
