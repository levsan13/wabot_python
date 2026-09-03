"""Async SQLAlchemy engine and session helpers (SQLite through aiosqlite)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base every model inherits from."""


# Module-level singletons, created once by init_engine() at startup.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_sqlite_dir(url: str) -> None:
    """Create the directory holding the .db file when it does not exist yet."""
    marker = "sqlite+aiosqlite:///"
    if not url.startswith(marker):
        return
    raw = url[len(marker) :]
    if raw in (":memory:", "") or raw.startswith(":memory:"):
        return
    Path(raw).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def init_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """Create the global engine (idempotent)."""
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    _ensure_sqlite_dir(database_url)
    _engine = create_async_engine(database_url, echo=echo, pool_pre_ping=True, future=True)

    if database_url.startswith("sqlite"):

        @event.listens_for(_engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - infra
            # WAL lets readers and the writer coexist, which matters because the
            # webhook keeps writing while background workers read history.
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    logger.info("Database configured: %s", database_url)
    return _engine


async def create_all() -> None:
    """Create any table that does not exist yet."""
    if _engine is None:
        raise RuntimeError("init_engine() must be called before create_all()")
    from app.db import models  # noqa: F401  (imports register models on the metadata)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session: commits on success, rolls back on error."""
    if _session_factory is None:
        raise RuntimeError("init_engine() must be called before opening sessions")
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency wrapping `session_scope`."""
    async with session_scope() as session:
        yield session


async def dispose_engine() -> None:
    """Close pooled connections — used at shutdown and between tests."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
