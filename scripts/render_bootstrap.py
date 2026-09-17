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

from sqlalchemy import create_engine, text

from geocadastra.store.schema import Base

db_url = os.environ["GEOCADASTRA_DB_URL"]
schema = os.environ.get("GEOCADASTRA_DB_SCHEMA", "public")
engine = create_engine(db_url, connect_args={"options": f"-csearch_path={schema},public"})

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

Base.metadata.create_all(engine)

migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
with engine.begin() as conn:
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        # psycopg3's execute() is one-statement-at-a-time; split on ';' and
        # drop BEGIN/COMMIT lines (this call is already inside its own
        # transaction via engine.begin() -- a nested BEGIN is at best a
        # no-op warning, not worth relying on).
        for statement in sql_file.read_text().split(";"):
            statement = statement.strip()
            if not statement or statement.upper() in ("BEGIN", "COMMIT"):
                continue
            conn.execute(text(statement))

print("render_bootstrap: PostGIS enabled, schema created, patches applied.")
