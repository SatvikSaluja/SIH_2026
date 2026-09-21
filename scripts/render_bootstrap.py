"""Run once before the web service starts (see Dockerfile CMD): enables
PostGIS, creates every table via Base.metadata.create_all (there's no
Alembic migration history in this repo -- conftest.py's own test fixture
bootstraps the exact same way), then applies the two incremental SQL
patches under migrations/. Every step is idempotent (IF NOT EXISTS /
create_all's own "skip what already exists" behavior) so re-running this
on every deploy is safe, not just on the first one.
"""
import os
from pathlib import Path

import psycopg
from sqlalchemy import create_engine, text

from geocadastra.store.schema import Base

db_url = os.environ["GEOCADASTRA_DB_URL"]
schema = os.environ.get("GEOCADASTRA_DB_SCHEMA", "public")
engine = create_engine(db_url, connect_args={"options": f"-csearch_path={schema},public"})

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

Base.metadata.create_all(engine)

# Each migration file is meant to run as one script (its own header says
# "Apply ... with psql -v ON_ERROR_STOP=1", and each wraps itself in its own
# BEGIN;/COMMIT;) -- so it's sent to Postgres as one multi-statement string
# over a raw, autocommit, unparameterized psycopg connection. That's the
# same simple-query-protocol path `psql -f` itself uses, and Postgres's own
# parser handles a DO $$ ... $$; block's internal semicolons correctly.
#
# The previous version pre-split each file on ';' before calling
# SQLAlchemy's text()/execute() -- required, because SQLAlchemy's execute()
# goes through the extended query protocol, which Postgres restricts to one
# statement per call. But splitting on a bare ';' isn't how SQL is
# tokenized: a dollar-quoted body can contain its own semicolons, and
# migrations/0001's own DO $$ block does. Naive splitting cut it apart mid-
# body -- reproduced directly against a fresh schema (not just an
# already-migrated one): "unterminated dollar-quoted string". This would
# have crashed Render's first-ever boot against its brand-new database,
# not just a re-run.
psycopg_dsn = db_url.replace("postgresql+psycopg://", "postgresql://", 1)
migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
with psycopg.connect(psycopg_dsn, options=f"-csearch_path={schema},public", autocommit=True) as conn:
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        conn.execute(sql_file.read_text())

print("render_bootstrap: PostGIS enabled, schema created, patches applied.")
