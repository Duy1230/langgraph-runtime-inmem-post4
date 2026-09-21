from __future__ import annotations

import importlib.util
import json
import asyncio
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]


def test_example_config_uses_postgres_checkpointer() -> None:
    config = json.loads((ROOT / "example" / "langgraph.json").read_text())

    assert config["checkpointer"]["path"] == (
        "./postgres_checkpointer.py:checkpointer"
    )
    assert "langgraph-checkpoint-postgres" in config["dependencies"]
    assert "psycopg[binary,pool]>=3.2" in config["dependencies"]


def test_example_config_uses_postgres_store_without_disabling_ttl() -> None:
    config = json.loads((ROOT / "example" / "langgraph.json").read_text())

    assert config["store"]["path"] == "./postgres_store.py:store"
    assert config["store"]["ttl"] == {
        "refresh_on_read": True,
        "sweep_interval_minutes": 1,
        "default_ttl": 10080,
    }


def test_postgres_checkpointer_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph.checkpoint.postgres.aio")
    from langgraph_runtime_inmem import postgres

    events: list[Any] = []

    class FakeSaver:
        @classmethod
        def from_conn_string(cls, connection_string: str) -> FakeSaver:
            events.append(("connect", connection_string))
            return cls()

        async def __aenter__(self) -> FakeSaver:
            events.append("enter")
            return self

        async def __aexit__(self, *args: object) -> None:
            events.append("exit")

        async def setup(self) -> None:
            events.append("setup")

    postgres_module = ModuleType("langgraph.checkpoint.postgres")
    aio_module = ModuleType("langgraph.checkpoint.postgres.aio")
    aio_module.AsyncPostgresSaver = FakeSaver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", postgres_module)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", aio_module)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test/database")
    monkeypatch.setattr(postgres, "RuntimePostgresSaver", FakeSaver)

    module_path = ROOT / "example" / "postgres_checkpointer.py"
    spec = importlib.util.spec_from_file_location("postgres_checkpointer", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def exercise_checkpointer() -> None:
        async with module.checkpointer() as saver:
            assert isinstance(saver, FakeSaver)
            assert events == [
                ("connect", "postgresql://test/database"),
                "enter",
                "setup",
            ]

    asyncio.run(exercise_checkpointer())
    assert events[-1] == "exit"


def test_postgres_store_lifecycle_keeps_ttl_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langgraph.store.postgres.aio")
    from langgraph_runtime_inmem import postgres

    events: list[Any] = []

    async def precision(store):
        events.append("precision")

    monkeypatch.setattr(postgres, "ensure_store_ttl_precision", precision)

    class FakeStore:
        @classmethod
        def from_conn_string(
            cls, connection_string: str, *, ttl: dict[str, Any], pool_config: dict
        ) -> FakeStore:
            assert pool_config["max_size"] == 8
            events.append(("connect", connection_string, ttl))
            return cls()

        async def __aenter__(self) -> FakeStore:
            events.append("enter")
            return self

        async def __aexit__(self, *args: object) -> None:
            events.append("exit")

        async def setup(self) -> None:
            events.append("setup")

        async def start_ttl_sweeper(self) -> None:
            events.append("start_ttl")

        async def stop_ttl_sweeper(self) -> None:
            events.append("stop_ttl")

    postgres_module = ModuleType("langgraph.store.postgres")
    aio_module = ModuleType("langgraph.store.postgres.aio")
    aio_module.AsyncPostgresStore = FakeStore  # type: ignore[attr-defined]
    monkeypatch.setattr(postgres, "RuntimePostgresStore", FakeStore)
    monkeypatch.setitem(sys.modules, "langgraph.store.postgres", postgres_module)
    monkeypatch.setitem(sys.modules, "langgraph.store.postgres.aio", aio_module)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fallback/database")
    monkeypatch.setenv("STORE_DATABASE_URL", "postgresql://store/database")
    monkeypatch.setenv("STORE_TTL_DEFAULT_MINUTES", "60")
    monkeypatch.setenv("STORE_TTL_REFRESH_ON_READ", "false")
    monkeypatch.setenv("STORE_TTL_SWEEP_INTERVAL_MINUTES", "5")

    module_path = ROOT / "example" / "postgres_store.py"
    spec = importlib.util.spec_from_file_location("postgres_store", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def exercise_store() -> None:
        async with module.store() as postgres_store:
            assert isinstance(postgres_store, FakeStore)
            assert events[-3:] == ["setup", "precision", "start_ttl"]

    asyncio.run(exercise_store())
    assert events == [
        (
            "connect",
            "postgresql://store/database",
            {
                "default_ttl": 60.0,
                "refresh_on_read": False,
                "sweep_interval_minutes": 5.0,
            },
        ),
        "enter",
        "setup",
        "precision",
        "start_ttl",
        "stop_ttl",
        "exit",
    ]
