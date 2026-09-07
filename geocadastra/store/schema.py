"""PostGIS schema for the versioned planar graph store (Stage 2).

Nodes and edges are append-only version rows -- an edit never UPDATEs a
row, it INSERTs a new (id, version) row, so "replay the changeset log from
empty" is literally "re-insert every changeset's rows, in order." Faces are
a derived cache (see changeset.py's face rebuild), never a source of truth,
matching Stage 1's graph model exactly -- this store is just Stage 1's
PlanarGraph, persisted.

`FaceBoundary` is a real join table, not a JSON blob, specifically so
"which faces does this edge touch" is one indexed lookup (needed to update
"both incident faces in the same transaction" when a shared edge changes),
not a JSON-containment scan.
"""
from __future__ import annotations

import datetime as dt

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, Sequence, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SRID = 32643  # this project's working CRS throughout (see geocadastra.core.crs), EPSG:32643


class Base(DeclarativeBase):
    pass


# Global id allocation for nodes/edges/faces: a fresh in-memory PlanarGraph
# (Stage 1) always numbers its own nodes/edges/faces starting from 0, and
# nothing about that model knows or cares about "blocks" -- so two blocks
# seeded independently, each using the graph's own ids as-is, WILL collide
# on these primary keys the moment a second block is seeded (found in
# review). Seeding remaps a fresh graph's local ids onto ids pulled from
# these sequences before anything is written; everything downstream
# (load_block_graph, ChangesetContext) only ever sees the already-global ids
# rehydrated from the store, so the remap is invisible past seed time.
# Registered on Base.metadata so create_all()/drop_all() manage them too.
node_id_seq = Sequence("node_id_seq", metadata=Base.metadata)
edge_id_seq = Sequence("edge_id_seq", metadata=Base.metadata)
face_id_seq = Sequence("face_id_seq", metadata=Base.metadata)

# Stage 8 has exactly the same disease: `synth.generator.generate_ward()`
# numbers a ward's own blocks and parcels starting from 0 EVERY call, with
# no idea a store might already hold a different ward's "block 0"/"parcel
# 0". Found by review, reproduced directly: without these, a second
# `POST /wards/ingest` crashes on a duplicate `RecordedParcel`/`LegacyRecord`
# primary key, and -- more seriously -- `Face.block_id` (no `ward_job_id`
# column at all) silently POOLS two different wards' same-numbered blocks
# into one `load_block_graph()` result, corrupting both. `ingest_synthetic_
# ward()` (jobs/orchestrator.py) remaps every block/parcel/legacy-record id
# onto one of these before writing, exactly the node/edge/face pattern above.
block_id_seq = Sequence("block_id_seq", metadata=Base.metadata)
recorded_parcel_id_seq = Sequence("recorded_parcel_id_seq", metadata=Base.metadata)
legacy_record_id_seq = Sequence("legacy_record_id_seq", metadata=Base.metadata)


class Changeset(Base):
    __tablename__ = "changesets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    author: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    # every entity this changeset touched, e.g. {"nodes": [1, 2], "edges": [7]} --
    # directly queryable/listable from the changeset row itself, not just
    # inferable by joining every versioned table on changeset_id
    affected_entities: Mapped[dict] = mapped_column(JSONB, default=dict)


class NodeVersion(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="POINT", srid=SRID))  # GiST index: geoalchemy2's own default
    changeset_id: Mapped[int] = mapped_column(ForeignKey("changesets.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EdgeVersion(Base):
    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    n0_id: Mapped[int] = mapped_column(BigInteger, index=True)  # a node's stable id, not a pinned version --
    n1_id: Mapped[int] = mapped_column(BigInteger, index=True)  # current position resolves via the node's latest row
    interior: Mapped[list] = mapped_column(JSONB, default=list)  # [[x, y], ...] shape points, n0->n1 order
    changeset_id: Mapped[int] = mapped_column(ForeignKey("changesets.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    block_id: Mapped[int] = mapped_column(BigInteger, index=True)
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="POLYGON", srid=SRID))  # GiST index: geoalchemy2's own default
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    source_changeset_id: Mapped[int] = mapped_column(ForeignKey("changesets.id"))


class FaceBoundary(Base):
    """One row per edge reference in a face's ordered boundary walk."""

    __tablename__ = "face_boundaries"

    face_id: Mapped[int] = mapped_column(ForeignKey("faces.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)  # order within the ring walk
    edge_id: Mapped[int] = mapped_column(BigInteger, index=True)  # stable edge id (not version-pinned)
    forward: Mapped[bool] = mapped_column(Boolean)  # True: traverse edge n0->n1; False: n1->n0


class WardJob(Base):
    """One end-to-end orchestration run (Stage 8): ingest -> per-block
    processing -> certified parcel layer. `status` is the coarse job-level
    view; per-block progress (the actual resumability unit) lives in
    `BlockJob` -- a ward job's own status is derived by the orchestrator
    from its blocks' statuses, not tracked independently, so the two can
    never silently disagree.
    """

    __tablename__ = "ward_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String)  # e.g. "synthetic:seed=7" -- what was ingested
    # the exact synth.generator.WardParams (dataclasses.asdict) used to generate
    # this ward -- needed to faithfully regenerate the SAME ward later (see
    # jobs/orchestrator.py's module docstring on why the raster evidence field is
    # regenerated, not stored). Not just the seed: two different WardParams with
    # the same seed produce two DIFFERENT wards, and regenerating with the wrong
    # params (e.g. silently falling back to WardParams()'s defaults) gives a
    # block whose evidence-field raster window doesn't even overlap the real
    # block's geometry -- found this exact bug the first time this was tested
    # end to end, not assumed safe from reading the code.
    params: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|done|failed
    crs: Mapped[str] = mapped_column(String)
    # co-registration residual (RMS control-point error, metres) from the
    # ingest-time upload alignment -- carried into Stage 5's reliability
    # priors (see api/main.py's co-registration endpoint and jobs/
    # orchestrator.py's use of it): a poorly-aligned upload should widen
    # this ward's own legacy-evidence sigma, not be silently treated as
    # perfectly registered. NULL until co-registration actually runs.
    coreg_residual_m: Mapped[float | None] = mapped_column()
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BlockJob(Base):
    """Per-block processing status within one `WardJob` -- the actual
    resumability mechanism: a killed-and-restarted orchestrator run only
    (re)dispatches blocks not already `done` here, per the Stage 8
    Done-when ("resumes without redoing completed blocks"). `error`
    records the last failure's message for a `failed` block, so a retry
    doesn't need to reproduce the crash to know what to look at.
    """

    __tablename__ = "block_jobs"

    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), primary_key=True)
    block_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|done|failed
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IngestedBlock(Base):
    """One block's own polygon, from Stage 1's `build_blocks()`, persisted
    at ingest time -- `process_block` needs the block's true geometry
    (the boundary `assign_parcels()`/Stage 5's exterior-node projection
    are measured against), not just its id.

    `block_id` is globally unique (from `block_id_seq`), the id every
    other table and every API endpoint addresses this block by.
    `local_block_id` is the synthetic ward's OWN 0-based index (what
    `simulate_evidence_field(ward, local_block_id)` needs, since the
    regenerated `ward` object still numbers its own blocks from 0) --
    kept only so `process_block` can translate back to it; nothing
    outside that one call should ever need it.
    """

    __tablename__ = "ingested_blocks"

    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), primary_key=True)
    block_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    local_block_id: Mapped[int] = mapped_column(BigInteger)
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="POLYGON", srid=SRID))


class RecordedParcel(Base):
    """One ingested recorded-area parcel (Stage 3's capacity constraint
    input) -- the department's known area/style for a plot, independent of
    where the model or legacy layer thinks its boundary sits."""

    __tablename__ = "recorded_parcels"

    # caller-supplied, not server-generated -- MUST come from recorded_parcel_id_seq
    # (see ingest_synthetic_ward()), never the synthetic generator's own ward-local
    # Parcel.id: two different wards' "parcel 0" would collide on this primary key
    # otherwise (found by review, reproduced: a second ward's ingest crashed with
    # IntegrityError on this exact column).
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), index=True)
    block_id: Mapped[int] = mapped_column(BigInteger, index=True)
    area: Mapped[float] = mapped_column()
    style: Mapped[str] = mapped_column(String)
    seed_point: Mapped[object] = mapped_column(Geometry(geometry_type="POINT", srid=SRID))


class LegacyRecord(Base):
    """One ingested legacy-GIS-layer polygon (Stage 5's `legacy_estimate()`
    evidence source) -- kept separate from `RecordedParcel` since the two
    disagree by construction (that disagreement is the whole point of
    fusing them) and a legacy polygon may merge or omit true parcels."""

    __tablename__ = "legacy_records"

    # caller-supplied, from legacy_record_id_seq -- see RecordedParcel.id's own comment
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), index=True)
    block_id: Mapped[int] = mapped_column(BigInteger, index=True)
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="GEOMETRY", srid=SRID))  # Polygon or MultiPolygon


class SurveyPoint(Base):
    """One ground-truth-quality point tied to a specific parcel: either
    ingested up front (`source="synthetic_gt"` / `"rtk_survey"`) or added
    later through the field-verification-attachment endpoint
    (`source="field_verification"`) -- the same shape either way, since
    both are "a surveyor's measurement of where a corner actually is",
    just at different points in the workflow. Stage 6/7's own calibration
    and prioritisation code only ever wants `(x, y, parcel_id)`
    (`synth.generator.GTPoint`'s own shape) regardless of which path added
    the row.
    """

    __tablename__ = "survey_points"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), index=True)
    parcel_id: Mapped[int] = mapped_column(BigInteger, index=True)
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="POINT", srid=SRID))
    source: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PersistedConflict(Base):
    """A Stage 5 `core.conflicts.ConflictRecord`, persisted -- deferred in
    Stage 5 itself ("a real store/API home for them is Stage 8's
    concern", per STAGE_5_NOTES.md) since there was no API to serve them
    from yet. `sources` mirrors the in-memory record's own tuple of
    contributing source names.
    """

    __tablename__ = "conflicts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ward_job_id: Mapped[int] = mapped_column(ForeignKey("ward_jobs.id"), index=True)
    block_id: Mapped[int] = mapped_column(BigInteger, index=True)
    node_id: Mapped[int] = mapped_column(BigInteger, index=True)
    sources: Mapped[list] = mapped_column(JSONB)
    disagreement_m: Mapped[float] = mapped_column()
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="GEOMETRY", srid=SRID))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Provenance(Base):
    """Append-only evidence log, keyed to a specific edge version, hash-chained
    (each row's hash covers its own content plus the previous row's hash) so
    the log is corruption-evident: recompute the chain and compare. See
    provenance.py's module docstring for exactly what this does and doesn't
    guarantee (plain SHA-256, not a keyed MAC)."""

    __tablename__ = "provenance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    edge_id: Mapped[int] = mapped_column(BigInteger, index=True)
    edge_version: Mapped[int] = mapped_column(Integer)
    evidence_type: Mapped[str] = mapped_column(String)  # e.g. "model_inference", "gt_survey_point", "manual_edit"
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    prev_hash: Mapped[str] = mapped_column(String(64))
    this_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
