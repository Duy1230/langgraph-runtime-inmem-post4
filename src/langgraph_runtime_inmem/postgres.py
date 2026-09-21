"""Optional PostgreSQL helpers for the Agent Server's custom persistence hooks."""

import math
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

POOL_CONFIG = {
    "min_size": 1,
    "max_size": 8,
    "timeout": 5,
    "max_waiting": 64,
    "check": AsyncConnectionPool.check_connection,
    "kwargs": {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
        "connect_timeout": 3,
        "options": "-c statement_timeout=15000",
    },
}


class RuntimePostgresStore(AsyncPostgresStore):
    def _get_filter_condition(self, key, op, value):
        symbols = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        if (
            op in symbols
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return (
                (
                    "CASE WHEN jsonb_typeof(value->%s) = 'number' "
                    f"THEN (value->>%s)::numeric END {symbols[op]} %s"
                ),
                [key, key, value],
            )
        return super()._get_filter_condition(key, op, value)


async def ensure_store_ttl_precision(store) -> None:
    """Upstream 3.1.2 stores TTL minutes as INT, which breaks fractional refresh."""
    async with store._cursor() as cur:
        await cur.execute(
            "SELECT atttypid = 'double precision'::regtype AS ready "
            "FROM pg_attribute WHERE attrelid = 'store'::regclass "
            "AND attname = 'ttl_minutes' AND NOT attisdropped"
        )
        row = await cur.fetchone()
        if not row or not row["ready"]:
            await cur.execute(
                "ALTER TABLE store ALTER COLUMN ttl_minutes TYPE DOUBLE PRECISION "
                "USING ttl_minutes::double precision"
            )


class RuntimePostgresSaver(AsyncPostgresSaver):
    """Add run deletion and delta-safe pruning required by langgraph-api 0.13.3."""

    @classmethod
    @asynccontextmanager
    async def from_conn_string(cls, conn_string, *, pipeline=False, serde=None):
        if pipeline:
            async with super().from_conn_string(
                conn_string, pipeline=True, serde=serde
            ) as saver:
                yield saver
            return
        async with AsyncConnectionPool(conn_string, open=False, **POOL_CONFIG) as pool:
            await pool.wait(timeout=5)
            yield cls(pool, serde=serde)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        if run_ids:
            await self._remove_history(run_ids=[str(value) for value in run_ids])

    async def aprune(
        self, thread_ids: Sequence[str], *, strategy="keep_latest"
    ) -> None:
        if strategy not in {"keep_latest", "delete", "delete_all"}:
            raise ValueError(f"Unsupported prune strategy: {strategy!r}")
        if not thread_ids:
            return
        if strategy in {"delete", "delete_all"}:
            for thread_id in thread_ids:
                await self.adelete_thread(str(thread_id))
        else:
            await self._remove_history(thread_ids=[str(value) for value in thread_ids])

    async def _remove_history(self, *, thread_ids=None, run_ids=None) -> None:
        # ponytail: table-level maintenance lock and O(history) metadata per thread;
        # use coordinated per-thread writer locks and paged SQL for large deployments.
        async with self._cursor() as cur, cur.connection.transaction():
            await cur.execute("SET LOCAL lock_timeout = '5s'")
            await cur.execute("SET LOCAL statement_timeout = '15s'")
            await cur.execute(
                "LOCK TABLE checkpoints, checkpoint_blobs, checkpoint_writes "
                "IN SHARE ROW EXCLUSIVE MODE"
            )
            if run_ids is not None:
                await cur.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints "
                    "WHERE metadata->>'run_id' = ANY(%s)",
                    (run_ids,),
                )
                thread_ids = [row["thread_id"] for row in await cur.fetchall()]
            for thread_id in thread_ids:
                await cur.execute(
                    "SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                    "checkpoint->'channel_versions' AS versions, "
                    "metadata->'counters_since_delta_snapshot' AS delta, "
                    "metadata->>'run_id' AS run_id "
                    "FROM checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )
                rows = {
                    (r["checkpoint_ns"], r["checkpoint_id"]): r
                    for r in await cur.fetchall()
                }
                if not rows:
                    continue
                if run_ids is not None:
                    candidates = {
                        key for key, row in rows.items() if row["run_id"] in run_ids
                    }
                else:
                    latest = {}
                    for ns, checkpoint_id in rows:
                        latest[ns] = max(checkpoint_id, latest.get(ns, ""))
                    candidates = set(rows) - set(latest.items())
                await cur.execute(
                    "SELECT checkpoint_ns, channel, version FROM checkpoint_blobs "
                    "WHERE thread_id = %s AND type <> 'empty'",
                    (thread_id,),
                )
                materialized = {
                    (r["checkpoint_ns"], r["channel"], r["version"])
                    for r in await cur.fetchall()
                }
                protected = set()
                for key in rows.keys() - candidates:
                    ns, _ = key
                    row = rows[key]
                    needed = {
                        channel
                        for channel in (row["delta"] or {})
                        if channel in (row["versions"] or {})
                        and (ns, channel, str(row["versions"][channel]))
                        not in materialized
                    }
                    parent = row["parent_checkpoint_id"]
                    visited = set()
                    while parent and needed:
                        parent_key = (ns, parent)
                        if parent_key in visited or parent_key not in rows:
                            break
                        visited.add(parent_key)
                        protected.add(parent_key)
                        row = rows[parent_key]
                        needed = {
                            channel
                            for channel in needed
                            if (ns, channel, str((row["versions"] or {}).get(channel)))
                            not in materialized
                        }
                        parent = row["parent_checkpoint_id"]
                removed = [
                    (thread_id, ns, checkpoint_id)
                    for ns, checkpoint_id in candidates - protected
                ]
                if not removed:
                    continue
                await cur.executemany(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s "
                    "AND checkpoint_ns = %s AND checkpoint_id = %s",
                    removed,
                )
                await cur.executemany(
                    "DELETE FROM checkpoints WHERE thread_id = %s "
                    "AND checkpoint_ns = %s AND checkpoint_id = %s",
                    removed,
                )
                await cur.execute(
                    "DELETE FROM checkpoint_blobs b WHERE b.thread_id = %s "
                    "AND NOT EXISTS (SELECT 1 FROM checkpoints c "
                    "WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns "
                    "AND c.checkpoint->'channel_versions'->>b.channel = b.version)",
                    (thread_id,),
                )


def store_ttl_config():
    ttl = float(os.getenv("STORE_TTL_DEFAULT_MINUTES", "10080"))
    interval = float(os.getenv("STORE_TTL_SWEEP_INTERVAL_MINUTES", "1"))
    if not math.isfinite(ttl) or ttl < 0:
        raise ValueError("STORE_TTL_DEFAULT_MINUTES must be finite and non-negative")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("STORE_TTL_SWEEP_INTERVAL_MINUTES must be finite and positive")
    return {
        "default_ttl": ttl,
        "refresh_on_read": os.getenv("STORE_TTL_REFRESH_ON_READ", "true")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        "sweep_interval_minutes": interval,
    }


@asynccontextmanager
async def store():
    """Packaged custom Store hook; setup requires schema-owner privileges."""
    ttl = store_ttl_config()
    database_url = os.getenv("STORE_DATABASE_URL") or os.environ["DATABASE_URL"]
    async with RuntimePostgresStore.from_conn_string(
        database_url,
        pool_config=POOL_CONFIG,
        ttl=ttl,
    ) as instance:
        await instance.setup()
        await ensure_store_ttl_precision(instance)
        await instance.start_ttl_sweeper()
        try:
            yield instance
        finally:
            await instance.stop_ttl_sweeper()


@asynccontextmanager
async def checkpointer():
    """Packaged custom checkpointer hook, using a reconnecting bounded pool."""
    database_url = os.getenv("CHECKPOINT_DATABASE_URL") or os.environ["DATABASE_URL"]
    async with RuntimePostgresSaver.from_conn_string(database_url) as saver:
        await saver.setup()
        yield saver
