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
