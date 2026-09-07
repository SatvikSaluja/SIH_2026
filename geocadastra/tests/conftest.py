"""Stage 2 DB test fixtures.

Tests run against a real local Postgres+PostGIS (no mocking a spatial
database is worth trusting) in a dedicated schema, isolated per test via a
transaction that's always rolled back -- so the suite never leaves data
behind regardless of pass/fail, and can run repeatedly against the same
persistent database.
"""
import os
from urllib.parse import urlsplit

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from geocadastra.store.schema import Base

TEST_DB_URL = os.environ.get(
    "GEOCADASTRA_TEST_DB_URL", "postgresql+psycopg://geocadastra:geocadastra@localhost:5432/geocadastra"
)
TEST_SCHEMA = "test_stage2"

# This fixture DROPs/CREATEs TEST_SCHEMA every session -- refuse to point
# that at anything that isn't obviously a local throwaway database, since an
# overridden GEOCADASTRA_TEST_DB_URL is the one place an externally-supplied
# value drives a real destructive action (found by review).
_host = urlsplit(TEST_DB_URL).hostname
if _host not in ("localhost", "127.0.0.1", "::1"):
    raise RuntimeError(
        f"GEOCADASTRA_TEST_DB_URL points at host {_host!r}, not localhost -- "
        f"refusing to run tests that DROP SCHEMA {TEST_SCHEMA} CASCADE against it. "
        "Point this at a local/throwaway Postgres instance."
    )


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        TEST_DB_URL, connect_args={"options": f"-csearch_path={TEST_SCHEMA},public"}
    )
    with create_engine(TEST_DB_URL).begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {TEST_SCHEMA}"))
    Base.metadata.create_all(engine)
    yield engine
    with create_engine(TEST_DB_URL).begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE"))
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    connection = db_engine.connect()
    outer_txn = connection.begin()
    session_factory = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")
    session = session_factory()
    yield session
    session.close()
    outer_txn.rollback()
    connection.close()


# every table a real-commit test might touch, in an order TRUNCATE...CASCADE
# handles regardless (CASCADE covers FK order) -- one list shared by both
# fixtures below so a Stage 8 table added to schema.py only needs adding here
_ALL_TABLES = (
    "provenance", "conflicts", "survey_points", "legacy_records", "recorded_parcels",
    "ingested_blocks", "block_jobs", "ward_jobs", "face_boundaries", "faces", "edges", "nodes", "changesets",
)


@pytest.fixture()
def concurrent_sessions(db_engine):
    """Two genuinely independent sessions (separate connections, each free
    to commit on its own) for tests that need real cross-transaction
    concurrency -- `db_session`'s single-shared-transaction isolation can't
    exercise that. A real commit here isn't auto-rolled-back, so the test
    itself is responsible for deleting whatever it created (a `TRUNCATE` of
    every real table, in dependency order, at teardown -- simpler and
    less error-prone than per-row tracking with composite primary keys)."""
    from sqlalchemy.orm import sessionmaker as _sessionmaker

    Session = _sessionmaker(bind=db_engine)
    s1, s2 = Session(), Session()
    yield s1, s2
    s1.rollback()
    s2.rollback()
    for table in _ALL_TABLES:
        s1.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
    s1.commit()
    s1.close()
    s2.close()


@pytest.fixture()
def committed_session(db_engine):
    """One real-commit session (Stage 8: Celery tasks open their OWN
    session per `db_url`/`schema`, separate from whatever session a test
    uses to set up fixtures or assert on results afterward -- a rolled-
    back `db_session` transaction would be invisible to that separate
    connection, exactly the way a real worker process's session can't see
    a test's own uncommitted setup). Same real-commit-needs-real-cleanup
    contract as `concurrent_sessions`."""
    from sqlalchemy.orm import sessionmaker as _sessionmaker

    session = _sessionmaker(bind=db_engine)()
    yield session
    session.rollback()
    for table in _ALL_TABLES:
        session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
    session.commit()
    session.close()
