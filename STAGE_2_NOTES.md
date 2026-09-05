# Stage 2 — store, versioning, provenance

## Environment
Real local PostgreSQL 16.15 + PostGIS 3.4.2, installed via `apt` (not Docker --
Docker Desktop's WSL integration wasn't enabled in this environment), enabled
as a systemd service so it survives WSL restarts, data on the standard
persistent `/var/lib/postgresql/16/main`. A dedicated `geocadastra` role/database
own the schema; tests run against a separate `test_stage2` schema in the same
database, isolated per test via a rolled-back transaction. Added `SQLAlchemy`,
`alembic`, `GeoAlchemy2`, and `psycopg[binary]` to the stack -- none are in the
original pinned list, but "SQLAlchemy 2.0 + PostGIS 3.4" can't actually talk to
each other without a driver and a geometry-column bridge; GeoAlchemy2 is the
standard one, not a substitution for anything pinned.

## Built
`geocadastra/store/schema.py`: `Changeset`, `NodeVersion`/`EdgeVersion` (append-
only version rows, PK `(id, version)`, never UPDATEd), `Face`/`FaceBoundary`
(a derived cache -- `FaceBoundary` is a real join table, not a JSON blob,
specifically so "which faces touch this edge" is one indexed lookup), and
`Provenance` (hash-chained evidence log). Three Postgres sequences
(`node_id_seq`/`edge_id_seq`/`face_id_seq`) allocate globally-unique ids across
the whole store, independent of any one block.

`geocadastra/store/changeset.py`: `load_block_graph()` rehydrates a Stage 1
`PlanarGraph` from a block's current (latest-version) rows; `seed_block_graph()`
writes a fresh in-memory graph's nodes/edges/faces as an initial changeset,
remapping the graph's own 0-based ids onto the global sequences first;
`ChangesetContext` is the edit context manager -- loads the block, lets
`move_node()` calls run against Stage 1's own graph, and on a clean exit:
checks no part of the loaded block changed underneath it since load (see
review below), rejects an edit that makes any touched face invalid, persists
new node versions + updated face geometry + a provenance record per affected
edge, and commits -- all in one transaction, or none of it.

`geocadastra/store/provenance.py`: `append_provenance()`/`verify_chain()`,
hash-chained (SHA-256, honestly documented as corruption-evident rather than
tamper-evident against a privileged writer -- see review), serialized against
concurrent appends via a transaction-scoped Postgres advisory lock.

Tests: `test_schema.py`, `test_changeset.py`, `test_provenance.py` -- 32 tests,
including the Stage 2 acceptance criteria (editing a shared boundary node
updates both adjacent parcels consistently in one transaction; replaying the
version log from empty matches a direct latest-version query; the provenance
chain verifies), a GiST-index-not-a-seq-scan test at 20,000 rows, and real
cross-transaction concurrency tests using genuinely separate DB connections
(not just parallel `with` blocks in one transaction). 207 tests total across
all three stages, all pass (`pytest geocadastra/`, ~80s).

## Full multi-angle review (post-build)
Ran the same review workflow as Stages 0/1, focused on what's actually new at
this stage: transactions, concurrency, and a real database. It found several
serious bugs standard single-threaded tests can't surface at all:

- **Node/edge ids weren't actually global.** A fresh in-memory `PlanarGraph`
  always numbers its own nodes/edges/faces starting from 0, and Stage 1 has no
  concept of "block" at all -- so two blocks seeded independently, each using
  the graph's own ids as-is, collided on every table's primary key the moment
  a second block was seeded. Fixed by remapping onto the three new global
  sequences at seed time; everything downstream only ever sees the
  already-global ids rehydrated from the store, so nothing else changed.
- **A silent lost update on the face cache under concurrent edits.**
  Reproduced with two real, genuinely overlapping transactions: both loaded
  the same block, each moved a *different* node on the *same* face, and both
  committed successfully -- the second commit silently overwrote the first
  transaction's already-committed change to that face's cached geometry, with
  zero errors. This is exactly what invariant #6 ("nothing is silently
  resolved") forbids. Fixed with block-level optimistic concurrency: every
  node/edge version loaded is re-checked immediately before writing anything,
  and any change since load aborts the whole changeset with
  `ConcurrentModificationError` instead of overwriting.
- **The provenance hash chain could fork under concurrent legitimate appends**,
  by the identical unlocked-read race -- two concurrent `append_provenance`
  calls both read the same chain tip and both committed, and `verify_chain()`
  then permanently (and falsely) reports tampering with no way to tell that
  apart from a real attack, on a log that's append-only by design. Fixed with
  a transaction-scoped Postgres advisory lock serializing appends.
- **`ChangesetContext.__exit__` didn't roll back when its own commit logic
  raised** (only when the `with`-body did), leaving the SQLAlchemy session in
  a broken (`PendingRollbackError`) state for whatever the caller did next --
  contradicting the class's own documented atomicity guarantee. Fixed by
  wrapping the commit path in its own try/except that always rolls back
  before re-raising.
- **A dangling `FaceBoundary.edge_id`** (referencing a missing edge row)
  poisoned every edit anywhere in the block, not just the corrupted face --
  the `KeyError` only surfaced lazily inside `faces_to_polygons()`, which
  `_persist()` calls over every touched face unconditionally. Fixed with a
  fail-fast consistency check in `load_block_graph()`.
- **No geometry validation at all.** An edit that made a face self-intersecting
  committed silently -- confirmed invalid by both shapely and PostGIS's own
  `ST_IsValid`. Fixed: `_persist()` now checks every touched face's recomputed
  polygon and refuses the whole changeset if any is invalid.
- **The hash chain's "tamper-evident" framing was oversold.** Plain SHA-256
  with no key means anyone with the DB write access this system already
  requires can recompute a self-consistent chain after deliberately editing a
  row -- this detects corruption and concurrency bugs, not a privileged
  insider. Re-documented honestly rather than fixed in code (a keyed MAC needs
  a key this application doesn't hold -- out of scope here, noted for later).
- **Performance**: `_persist()`'s touched-faces computation was
  `O(touched_nodes x edges_in_block)` (confirmed 1195x slower than a fix at
  150k edges/1000 touched nodes) and made one DB round-trip per touched node
  instead of one batched query (36.7x slower at 1000 nodes); `seed_block_graph`
  added every row one at a time instead of bulk-inserting. All three fixed.
  `NodeVersion.geom`/`Face.geom` each had two redundant identical GiST indexes
  (an explicit one plus GeoAlchemy2's own auto-created default) -- removed the
  redundant explicit ones.

Applied, not just reported: a cheap safety guard in `conftest.py` refusing to
run the schema-dropping test fixture against anything but `localhost` (an
overridden `GEOCADASTRA_TEST_DB_URL` was the one place an externally-influenced
value drove a real destructive `DROP SCHEMA ... CASCADE`).

## Deferred (correctly, per Stage 2's own scope)
- **`_latest_versions()`'s `DISTINCT ON` query flips from an index scan to a
  full sequential scan once the requested id-list is ~1.5-2% of the table's
  total distinct ids** -- confirmed at ward-scale (420k rows). Since `nodes`/
  `edges` are shared across the whole store, a single block's load/edit cost
  ends up tracking *ward-wide* row count, not the block's own size, which
  matters directly for the 50k-200k-parcel target. The reviewing agent's own
  fix (a per-id `LATERAL` rewrite) is faster in the regime a real block hits
  but worse once the id-list is large -- a regime-aware fix, not a one-line
  change, so left as a known scaling limitation to revisit rather than a
  rushed partial fix.
- No `blocks` table exists yet -- `block_id` is an opaque caller-supplied
  integer with no FK/authorization check tying it to an actual Stage 1 block.
  Fine for Stage 2's single-trusted-caller scope; a real gap to close before
  a multi-tenant API (Stage 8) lets external callers supply it directly.
- No size/shape validation on the JSONB payload columns
  (`Changeset.affected_entities`, `EdgeVersion.interior`, `Provenance.payload`)
  -- harmless while this module is the only writer, a real gap once a later
  stage's API accepts any of these from an HTTP request.
- alembic is installed and pinned but no migration has been written yet --
  Stage 2's tests create the schema directly via `Base.metadata.create_all()`.
  First real migration should land whenever schema.py next changes after this
  point.
