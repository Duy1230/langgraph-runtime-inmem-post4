from __future__ import annotations

import os

# langgraph-api expects these deployment variables during module import.  The
# in-memory tests never connect to either service.
os.environ.setdefault("REDIS_URI", "redis://localhost:6379")
os.environ.setdefault("DATABASE_URI", "postgres://localhost/test")
