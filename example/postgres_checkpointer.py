"""Compatibility export; the PostgreSQL integration is included in the wheel."""

from langgraph_runtime_inmem.postgres import checkpointer

__all__ = ["checkpointer"]
