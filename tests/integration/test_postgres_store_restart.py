from __future__ import annotations

import asyncio
import os
import uuid

import pytest


pytestmark = pytest.mark.integration


def test_store_item_survives_connection_restart() -> None:
    aio = pytest.importorskip("langgraph.store.postgres.aio")
    database_url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL is not configured")

    async def exercise_restart() -> None:
        namespace = ("integration-test", str(uuid.uuid4()))
        key = "restart-item"
        value = {"message": "persists across Store lifecycles"}

        async with aio.AsyncPostgresStore.from_conn_string(database_url) as first:
            await first.setup()
            await first.aput(namespace, key, value)

        # A new Store context models an API restart: no Store object is reused.
        async with aio.AsyncPostgresStore.from_conn_string(database_url) as second:
            item = await second.aget(namespace, key)
            assert item is not None
            assert item.value == value
            await second.adelete(namespace, key)

    asyncio.run(exercise_restart())
