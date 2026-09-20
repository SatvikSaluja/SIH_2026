"""Shared FastAPI dependencies -- split out from main.py specifically so
a second router module (advisory.py) can depend on `get_session`/
`_get_ward_job_or_404` without a circular import (advisory.py imports
this; main.py also imports this; neither imports the other). Moving
these out changes nothing about their behavior or the test-override
pattern (`app.dependency_overrides[get_session] = ...`) -- FastAPI
overrides are keyed by the function object, not by which module
originally defined it, and `main.py` re-imports these names so
`from geocadastra.api.main import get_session` (the existing test
pattern) still works unchanged.
"""
from __future__ import annotations

import os

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from geocadastra.store.schema import WardJob

_DB_URL = os.environ.get("GEOCADASTRA_DB_URL", "postgresql+psycopg://geocadastra:geocadastra@localhost:5432/geocadastra")
_DB_SCHEMA = os.environ.get("GEOCADASTRA_DB_SCHEMA", "public")
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_DB_URL, connect_args={"options": f"-csearch_path={_DB_SCHEMA},public"})
    return _engine


def get_session():
    session = sessionmaker(bind=_get_engine())()
    try:
        yield session
    finally:
        session.close()


def get_db_config() -> tuple[str, str]:
    """`(db_url, schema)` for `run_ward()`'s Celery dispatch -- a SEPARATE
    dependency from `get_session()`, not the same module globals baked
    directly into the `/run` handler, because a test overriding
    `get_session` (the standard FastAPI testing pattern, pointing requests
    at a test schema) must ALSO redirect where dispatched tasks look for
    their own data, or a task opens a session against the real default
    schema while the request's own session was pointed at the test one --
    two different databases silently in play at once. Found exactly this
    way: `/run` without this indirection passed `_DB_URL`/`_DB_SCHEMA`
    straight through even under a test override, and every dispatched
    task then failed with `UndefinedTable` looking for `block_jobs` in a
    schema that was never created.
    """
    return _DB_URL, _DB_SCHEMA


def _get_ward_job_or_404(session: Session, ward_job_id: int) -> WardJob:
    job = session.get(WardJob, ward_job_id)
    if job is None:
        raise HTTPException(404, f"no ward {ward_job_id}")
    return job
