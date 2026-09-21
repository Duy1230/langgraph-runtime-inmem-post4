"""Regression checks discovered through real post4 user journeys."""

import importlib.util
import operator
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langgraph.channels import DeltaChannel
from langgraph.graph import END, START, StateGraph


def merge_deltas(state, writes):
    return state + [item for batch in writes for item in batch]


class GraphState(TypedDict):
    events: Annotated[list[str], operator.add]
    payload: dict


class DeltaState(TypedDict):
    events: Annotated[list[str], DeltaChannel(merge_deltas, snapshot_frequency=3)]


class LongDeltaState(TypedDict):
    events: Annotated[list[str], DeltaChannel(merge_deltas, snapshot_frequency=1000)]


def example(name):
    path = Path(__file__).parents[2] / "example" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_store_never_loads_old_sidecars(tmp_path):
    folder = tmp_path / ".langgraph_api"
    folder.mkdir()
    sidecar = folder / "store.pckl"
    sidecar.write_bytes(b"intentionally invalid legacy sidecar")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from langgraph_runtime_inmem import store; "
                "assert store.STORE is None; store.set_store_config({'path': 'custom.store'}); "
                "assert store.STORE is None"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sidecar.read_bytes() == b"intentionally invalid legacy sidecar"


def test_shutdown_does_not_initialize_unused_checkpointer(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", (
            "import asyncio; from langgraph_runtime_inmem import checkpoint, database; "
            "assert checkpoint.MEMORY is None; asyncio.run(database.stop_pool()); "
            "assert checkpoint.MEMORY is None"
        )],
        cwd=tmp_path, capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
async def test_postgres_fractional_ttl_survives_refresh(monkeypatch):
    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    monkeypatch.setenv("STORE_DATABASE_URL", url)
    monkeypatch.setenv("STORE_TTL_SWEEP_INTERVAL_MINUTES", "1")
    namespace = ("fractional", str(uuid.uuid4()))
    async with example("postgres_store").store() as store:
        await store.aput(namespace, "key", {"v": 1}, ttl=0.1)
        await store.aget(namespace, "key", refresh_ttl=True)
        async with store._cursor() as cur:
            await cur.execute(
                "SELECT ttl_minutes, extract(epoch from expires_at - now()) AS seconds "
                "FROM store WHERE key = 'key' AND prefix LIKE %s",
                (f"%{namespace[1]}%",),
            )
            row = await cur.fetchone()
        assert row["ttl_minutes"] == pytest.approx(0.1), row
        assert 4 < row["seconds"] <= 6, row
        await store.adelete(namespace, "key")


@pytest.mark.integration
@pytest.mark.parametrize("delta", [False, 3, 1000])
async def test_postgres_prune_and_delete_preserve_survivors(delta):
    from langgraph_runtime_inmem.postgres import RuntimePostgresSaver

    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    builder = StateGraph(
        {False: GraphState, 3: DeltaState, 1000: LongDeltaState}[delta]
    )
    builder.add_node("step", lambda state: {"events": ["node"]})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    async with RuntimePostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        runs = [str(uuid.uuid4()) for _ in range(4)]
        try:
            for i, rid in enumerate(runs[:3]):
                await graph.ainvoke(
                    {
                        "events": [str(i)],
                        **({"payload": {"shared": [1, 2, 3]}} if i == 0 else {}),
                    },
                    {**cfg, "metadata": {"run_id": rid}},
                )
            before = (await graph.aget_state(cfg)).values
            await saver.adelete_for_runs([runs[0]])
            assert (await graph.aget_state(cfg)).values == before
            count_before = len([cp async for cp in saver.alist(cfg)])
            await saver.aprune([tid])
            assert (await graph.aget_state(cfg)).values == before
            count_after = len([cp async for cp in saver.alist(cfg)])
            assert count_after <= count_before
            if not delta:
                assert count_after == 1
            result = await graph.ainvoke(
                {"events": ["after"]}, {**cfg, "metadata": {"run_id": runs[3]}}
            )
            assert result["events"] == before["events"] + ["after", "node"]
            await saver.adelete_for_runs([runs[3]])
            assert (await graph.aget_state(cfg)).values == before
            # The operation is idempotent, and cleanup removes all payload tables.
            await saver.adelete_for_runs([runs[3]])
            await saver.aprune([tid], strategy="delete_all")
            assert not [cp async for cp in saver.alist(cfg)]
            async with saver._cursor() as cur:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    await cur.execute(
                        f"SELECT count(*) AS n FROM {table} WHERE thread_id=%s", (tid,)
                    )
                    assert (await cur.fetchone())["n"] == 0
        finally:
            await saver.adelete_thread(tid)


@pytest.mark.integration
@pytest.mark.parametrize(
    "op, expected",
    [
        ("$gt", [100]),
        ("$gte", [50, 100]),
        ("$lt", [5]),
        ("$lte", [5, 50]),
    ],
)
async def test_store_numeric_filters_ignore_other_json_types(monkeypatch, op, expected):
    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    monkeypatch.setenv("STORE_DATABASE_URL", url)
    namespace = ("numeric", str(uuid.uuid4()))
    async with example("postgres_store").store() as store:
        values = [5, 50, 100, "not-a-number", None, True]
        try:
            for i, value in enumerate(values):
                await store.aput(namespace, str(i), {"score": value})
            found = await store.asearch(namespace, filter={"score": {op: 50}})
            assert sorted(item.value["score"] for item in found) == expected
        finally:
            for i in range(len(values)):
                await store.adelete(namespace, str(i))


@pytest.mark.parametrize(
    "name,value",
    [
        ("STORE_TTL_DEFAULT_MINUTES", "nan"),
        ("STORE_TTL_DEFAULT_MINUTES", "-1"),
        ("STORE_TTL_SWEEP_INTERVAL_MINUTES", "inf"),
        ("STORE_TTL_SWEEP_INTERVAL_MINUTES", "0"),
    ],
)
def test_invalid_ttl_rejected_before_connect(monkeypatch, name, value):
    from langgraph_runtime_inmem.postgres import store_ttl_config

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        store_ttl_config()


@pytest.mark.integration
async def test_prune_is_atomic_on_mid_delete_failure(monkeypatch):
    from langgraph_runtime_inmem.postgres import RuntimePostgresSaver

    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    builder = StateGraph(GraphState)
    builder.add_node("step", lambda state: {"events": ["node"]})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    async with RuntimePostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({"events": ["a"], "payload": {}}, cfg)
        before = [cp async for cp in saver.alist(cfg)]
        original_cursor = saver._cursor

        class FailingCursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def __getattr__(self, name):
                return getattr(self.cursor, name)

            async def executemany(self, query, params):
                if query.startswith("DELETE FROM checkpoints "):
                    raise RuntimeError("injected failure after write deletion")
                return await self.cursor.executemany(query, params)

        @asynccontextmanager
        async def fail_cursor():
            async with original_cursor() as cursor:
                yield FailingCursor(cursor)

        try:
            with monkeypatch.context() as scoped:
                scoped.setattr(saver, "_cursor", fail_cursor)
                with pytest.raises(RuntimeError, match="injected"):
                    await saver.aprune([tid])
            after = [cp async for cp in saver.alist(cfg)]
            assert after == before
            assert (await graph.aget_state(cfg)).values["events"] == ["a", "node"]
        finally:
            await saver.adelete_thread(tid)


@pytest.mark.integration
async def test_real_pgvector_search_update_and_delete():
    from langgraph_runtime_inmem.postgres import RuntimePostgresStore

    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    namespace = ("vector", str(uuid.uuid4()))
    def embed(texts):
        return [[1.0, 0.0] if "cat" in text else [0.0, 1.0] for text in texts]
    async with RuntimePostgresStore.from_conn_string(
        url, index={"dims": 2, "embed": embed, "fields": ["text"]},
    ) as store:
        await store.setup()
        try:
            await store.aput(namespace, "cat", {"text": "cat"})
            await store.aput(namespace, "dog", {"text": "dog"})
            found = await store.asearch(namespace, query="cat", limit=2)
            assert found[0].key == "cat" and found[0].score == pytest.approx(1)
            await store.aput(namespace, "cat", {"text": "dog"})
            found = await store.asearch(namespace, query="cat", limit=2)
            assert all(item.score == pytest.approx(0) for item in found)
        finally:
            await store.adelete(namespace, "cat")
            await store.adelete(namespace, "dog")
        async with store._cursor() as cur:
            await cur.execute("SELECT count(*) AS n FROM store_vectors WHERE prefix LIKE %s", (f"%{namespace[1]}%",))
            assert (await cur.fetchone())["n"] == 0


@pytest.mark.integration
async def test_subgraph_namespaces_survive_pruning():
    from langgraph_runtime_inmem.postgres import RuntimePostgresSaver

    url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL not configured")
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    child = StateGraph(GraphState)
    child.add_node("child", lambda state: {"events": ["child"]})
    child.add_edge(START, "child")
    child.add_edge("child", END)
    parent = StateGraph(GraphState)
    parent.add_node("subgraph", child.compile(checkpointer=True))
    parent.add_edge(START, "subgraph")
    parent.add_edge("subgraph", END)
    async with RuntimePostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        graph = parent.compile(checkpointer=saver)
        try:
            for i in range(3):
                await graph.ainvoke({"events": [str(i)], "payload": {"v": i}}, cfg)
            before = (await graph.aget_state(cfg, subgraphs=True)).values
            namespaces_before = {cp.config["configurable"]["checkpoint_ns"] async for cp in saver.alist(cfg)}
            assert len(namespaces_before) > 1
            await saver.aprune([tid])
            assert (await graph.aget_state(cfg, subgraphs=True)).values == before
            retained = [cp async for cp in saver.alist(cfg)]
            assert len(retained) == len(namespaces_before)
            await graph.ainvoke({"events": ["after"]}, cfg)
            assert "after" in (await graph.aget_state(cfg)).values["events"]
        finally:
            await saver.adelete_thread(tid)
