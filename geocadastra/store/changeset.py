"""Transactional multi-entity edits (Stage 2).

A Changeset is a context manager: everything staged inside it commits
atomically -- one DB transaction, one `changesets` row listing every
entity touched, a new version row for every node that moved, a provenance
record for every edge that move touched -- or, on any exception (including
one raised by this class's own commit logic), nothing at all.

Editing reuses Stage 1's `PlanarGraph` directly rather than re-deriving
face geometry: rehydrate the affected block's current graph from the
store, call the same `move_node()` Stage 1 already has, and persist
whatever changed (the moved node's new version, and every face that
`faces_to_polygons()` now reports differently -- which is every face
touching that node, so "applying a changeset to a shared edge updates both
incident faces" falls out for free instead of being special-cased).

Concurrency: a block is optimistically locked as a whole -- `__enter__`
snapshots the version of every node/edge it loads, and `__exit__` re-checks
every one of them immediately before writing anything. A concurrent commit
to *any* part of the same block between load and commit aborts this
changeset with `ConcurrentModificationError` instead of silently
overwriting the other change (found by review: without this check, two
changesets editing different corners of the same face raced to a silent
lost update on that face's cached geometry, no error at all).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from geoalchemy2.shape import from_shape, to_shape
from shapely.errors import GEOSException
from shapely.geometry import Point
from shapely.validation import explain_validity
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from geocadastra.core.crs import CRSMismatchError
from geocadastra.core.graph import Edge, Face as GraphFace, Node, PlanarGraph, _round_pt
from geocadastra.core.planarize import GRID
from geocadastra.store.provenance import append_provenance
from geocadastra.store.constraints import validate_area_change
from geocadastra.store.schema import (
    SRID,
    Changeset,
    EdgeVersion,
    Face,
    FaceBoundary,
    NodeVersion,
    RecordedParcel,
    IngestedBlock,
    edge_id_seq,
    face_id_seq,
    node_id_seq,
)


class ConcurrentModificationError(RuntimeError):
    """Some part of this block changed in the store after this changeset
    loaded it and before it committed. Nothing was written."""


_OVERLAP_TOL = 1e-4  # m^2 -- see _persist()'s cross-face overlap check


def lock_block(session: Session, block_id: int) -> None:
    """Serialize all writes to a block until the owning transaction ends.

    Include the schema so isolated wards/test stores do not share locks.
    A transaction lock stays on its connection until commit/rollback; no
    session-scoped lock can escape into the connection pool.
    """
    session.execute(text(
        "SELECT pg_advisory_xact_lock(hashtextextended("
        "current_schema() || ':block:' || CAST(:block_id AS text), 0))"
    ), {"block_id": block_id})


def allocate_ids(session: Session, seq, n: int) -> list[int]:
    """`n` fresh values from `seq`, in one round trip. Public (not `_`-
    prefixed) since `jobs/orchestrator.py`'s own `ingest_synthetic_ward()`
    needs the exact same "remap a synthetic object's own restarts-at-0 ids
    onto globally-unique ones before writing" pattern this function
    already existed for -- one shared helper, not a second copy of the
    same `nextval()`-over-`generate_series()` trick.
    """
    if n == 0:
        return []
    return list(session.execute(text(f"SELECT nextval('{seq.name}') FROM generate_series(1, :n)"), {"n": n}).scalars())


def _latest_versions(session: Session, model, ids: list[int]):
    if not ids:
        return []
    stmt = select(model).filter(model.id.in_(ids)).distinct(model.id).order_by(model.id, model.version.desc())
    return session.execute(stmt).scalars().all()


def load_block_graph(session: Session, block_id: int) -> PlanarGraph:
    """Rehydrate a Stage 1 PlanarGraph from this block's current state."""
    faces = session.query(Face).filter_by(block_id=block_id).all()
    face_ids = [f.id for f in faces]
    boundaries = (
        session.query(FaceBoundary)
        .filter(FaceBoundary.face_id.in_(face_ids))
        .order_by(FaceBoundary.face_id, FaceBoundary.ring, FaceBoundary.position)
        .all()
        if face_ids
        else []
    )
    edge_ids = sorted({b.edge_id for b in boundaries})
    edge_rows = _latest_versions(session, EdgeVersion, edge_ids)
    edge_by_id = {e.id: e for e in edge_rows}
    missing = set(edge_ids) - set(edge_by_id)
    if missing:
        # fail fast, here, rather than lazily inside faces_to_polygons() the
        # first time something needs this edge's geometry -- that failure
        # mode poisoned every OTHER face in the block too, not just this one
        raise RuntimeError(
            f"block {block_id}: face_boundaries reference edge id(s) {sorted(missing)} "
            "with no corresponding row in edges -- store is inconsistent"
        )

    node_ids = sorted({e.n0_id for e in edge_rows} | {e.n1_id for e in edge_rows})
    node_rows = _latest_versions(session, NodeVersion, node_ids)

    graph = PlanarGraph(crs=f"EPSG:{SRID}")
    loaded_versions: dict[tuple[str, int], int] = {}
    for n in node_rows:
        pt = to_shape(n.geom)
        graph.nodes[n.id] = Node(n.id, pt.x, pt.y)
        graph._point_to_node[_round_pt((pt.x, pt.y))] = n.id
        loaded_versions[("node", n.id)] = n.version
    graph._next_node = (max(node_ids) + 1) if node_ids else 0

    for e in edge_rows:
        graph.edges[e.id] = Edge(e.id, e.n0_id, e.n1_id, tuple(tuple(c) for c in e.interior))
        loaded_versions[("edge", e.id)] = e.version
    graph._next_edge = (max(edge_ids) + 1) if edge_ids else 0

    by_face = defaultdict(list)
    edge_faces = defaultdict(list)
    for b in boundaries:
        by_face[b.face_id].append(b)
        edge_faces[b.edge_id].append(b.face_id)
    for f in faces:
        walks = defaultdict(list)
        for b in sorted(by_face[f.id], key=lambda b: (b.ring, b.position)):
            walks[b.ring].append((b.edge_id, b.forward))
        graph.faces[f.id] = GraphFace(f.id, tuple(walks[0]),
                                      tuple(tuple(walks[r]) for r in sorted(walks) if r != 0))
        if f.recorded_parcel_id is not None:
            graph.face_parcel_ids[f.id] = f.recorded_parcel_id
    graph._edge_faces = dict(edge_faces)
    graph._loaded_versions = loaded_versions  # Stage-2 bookkeeping for ChangesetContext's conflict check
    return graph


def seed_block_graph(
    session: Session, block_id: int, graph: PlanarGraph, author: str | None = None, description: str = "initial load",
    *, evidence: dict | None = None,
) -> Changeset:
    """Write every node/edge/face in `graph` as version-1 rows, all under
    one changeset -- ingesting a block for the first time is just its first
    changeset, not a separate code path.

    `graph`'s own node/edge/face ids are local to that one in-memory
    PlanarGraph (a fresh graph always numbers from 0) -- they are remapped
    onto globally-unique ids pulled from the store's sequences before
    anything is written, so a second block seeded into the same store never
    collides with the first on primary keys.
    """
    if graph.crs != f"EPSG:{SRID}":
        raise CRSMismatchError(f"store requires EPSG:{SRID}, got {graph.crs}")
    lock_block(session, block_id)
    if session.scalar(select(Face.id).where(Face.block_id == block_id).limit(1)) is not None:
        raise ValueError(f"block {block_id} already has a graph")
    for fid, pid in graph.face_parcel_ids.items():
        record = session.get(RecordedParcel, pid)
        if fid not in graph.faces or record is None or record.block_id != block_id:
            raise ValueError(f"invalid parcel association: face {fid}, parcel {pid}, block {block_id}")
    cs = Changeset(author=author, description=description, affected_entities={})
    session.add(cs)
    session.flush()

    node_ids = list(graph.nodes)
    edge_ids = list(graph.edges)
    face_ids = list(graph.faces)
    node_id_map = dict(zip(node_ids, allocate_ids(session, node_id_seq, len(node_ids))))
    edge_id_map = dict(zip(edge_ids, allocate_ids(session, edge_id_seq, len(edge_ids))))
    face_id_map = dict(zip(face_ids, allocate_ids(session, face_id_seq, len(face_ids))))

    session.bulk_insert_mappings(
        NodeVersion,
        [
            {
                "id": node_id_map[nid],
                "version": 1,
                "geom": from_shape(Point(n.x, n.y), srid=SRID),
                "changeset_id": cs.id,
            }
            for nid, n in graph.nodes.items()
        ],
    )
    session.bulk_insert_mappings(
        EdgeVersion,
        [
            {
                "id": edge_id_map[eid],
                "version": 1,
                "n0_id": node_id_map[e.n0],
                "n1_id": node_id_map[e.n1],
                "interior": [list(c) for c in e.coords],
                "changeset_id": cs.id,
            }
            for eid, e in graph.edges.items()
        ],
    )

    polys = graph.faces_to_polygons()
    session.bulk_insert_mappings(
        Face,
        [
            {
                "id": face_id_map[fid],
                "block_id": block_id,
                "recorded_parcel_id": graph.face_parcel_ids.get(fid),
                "geom": from_shape(polys[fid].geom, srid=SRID),
                "source_changeset_id": cs.id,
            }
            for fid in graph.faces
        ],
    )
    session.bulk_insert_mappings(
        FaceBoundary,
        [
            {"face_id": face_id_map[fid], "ring": ring_id, "position": pos, "edge_id": edge_id_map[eid], "forward": forward}
            for fid, face in graph.faces.items()
            for ring_id, walk in enumerate(face.rings)
            for pos, (eid, forward) in enumerate(walk)
        ],
    )

    cs.affected_entities = {
        "block_id": block_id,
        "nodes": sorted(node_id_map.values()),
        "edges": sorted(edge_id_map.values()),
        "faces": sorted(face_id_map.values()),
        "source_face_ids": {str(fid): new_id for fid, new_id in face_id_map.items()},
        "face_parcel_ids": {str(face_id_map[fid]): pid for fid, pid in graph.face_parcel_ids.items()},
        "graph_seed": {
            "crs": graph.crs,
            "nodes": [[node_id_map[nid], n.x, n.y] for nid, n in graph.nodes.items()],
            "edges": [[edge_id_map[eid], node_id_map[e.n0], node_id_map[e.n1], list(e.coords)]
                      for eid, e in graph.edges.items()],
            "faces": [[face_id_map[fid], [[[edge_id_map[eid], forward] for eid, forward in ring]
                                         for ring in face.rings]] for fid, face in graph.faces.items()],
        },
    }
    for edge_id in edge_id_map.values():
        append_provenance(
            session, edge_id=edge_id, edge_version=1, evidence_type="initial_load",
            payload={"changeset_id": cs.id, "block_id": block_id, "description": description, "evidence": evidence or {}},
        )
    session.flush()
    return cs


class ChangesetContext:
    """`with ChangesetContext(session, block_id=...) as cs: cs.move_node(...)`

    Everything staged commits in one transaction on a clean `__exit__`;
    any exception -- from the `with` body, or from this class's own commit
    logic -- rolls the whole thing back, leaves the session reusable, and
    nothing is written. With commit=False, this context owns a savepoint;
    the caller owns the encompassing transaction and its final commit.
    """

    def __init__(self, session: Session, block_id: int, author: str | None = None, description: str | None = None, *, commit: bool = True):
        self.session = session
        self.commit = commit
        self._savepoint = None
        self.block_id = block_id
        self.author = author
        self.description = description
        self.graph: PlanarGraph | None = None
        self._touched_nodes: set[int] = set()
        self._identity_faces: set[int] = set()
        self._node_evidence: dict[int, tuple[str, dict]] = {}
        self._changeset: Changeset | None = None

    def __enter__(self) -> "ChangesetContext":
        if not self.commit:
            self._savepoint = self.session.begin_nested()
        try:
            self.graph = load_block_graph(self.session, self.block_id)
            self._original_polygons = self.graph.faces_to_polygons()
            self._loaded_parcel_ids = dict(self.graph.face_parcel_ids)
            self._changeset = Changeset(author=self.author, description=self.description, affected_entities={})
            self.session.add(self._changeset)
            self.session.flush()
            return self
        except Exception:
            self._rollback()
            raise

    def _rollback(self):
        if self._savepoint is not None:
            self._savepoint.rollback()
        else:
            self.session.rollback()

    def move_node(
        self, node_id: int, x: float, y: float, evidence_type: str | None = None, detail: dict | None = None
    ) -> None:
        """`evidence_type`/`detail` override the default "manual_edit"
        provenance recorded for edges touching this node (e.g. Stage 5's
        fusion passes evidence_type="fusion" with which sources/
        reliabilities produced the new position) -- so provenance says
        *why* geometry moved, not just that it did."""
        self.graph.move_node(node_id, x, y)
        self._touched_nodes.add(node_id)
        if evidence_type is not None:
            self._node_evidence[node_id] = (evidence_type, detail or {})

    def associate_faces(self, assignments: dict[int, int]) -> None:
        """Explicit human resolution for a complete block, including old stores.

        Never infer historical identity from a centroid. Every supplied record
        must belong to this block; one record may own multiple face components.
        """
        if set(assignments) != set(self.graph.faces):
            raise ValueError("parcel associations must name every face in the block exactly once")
        for pid in set(assignments.values()):
            record = self.session.get(RecordedParcel, pid)
            if record is None or record.block_id != self.block_id:
                raise ValueError(f"parcel {pid} does not belong to block {self.block_id}")
        self._identity_faces.update(fid for fid, pid in assignments.items()
                                    if self.graph.face_parcel_ids.get(fid) != pid)
        self.graph.face_parcel_ids = dict(assignments)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._rollback()
            return False
        try:
            # Acquire BEFORE the optimistic check, and hold through commit.
            lock_block(self.session, self.block_id)
            self._check_no_concurrent_modification()
            touched_faces = self._persist()
            self._changeset.affected_entities = {
                "block_id": self.block_id,
                "nodes": sorted(self._touched_nodes), "faces": sorted(touched_faces),
                "node_positions": {str(nid): [self.graph.nodes[nid].x, self.graph.nodes[nid].y]
                                   for nid in self._touched_nodes},
                "face_parcel_ids": {str(fid): self.graph.face_parcel_ids[fid] for fid in self._identity_faces},
            }
            self.session.flush()
            if self._savepoint is not None:
                self._savepoint.commit()
            else:
                self.session.commit()
        except Exception:
            self._rollback()
            raise
        return False

    def _check_no_concurrent_modification(self) -> None:
        current_faces = set(self.session.scalars(select(Face.id).where(Face.block_id == self.block_id)))
        if current_faces != set(self.graph.faces):
            raise ConcurrentModificationError(f"block {self.block_id}: faces changed since load")
        current_associations = dict(self.session.execute(
            select(Face.id, Face.recorded_parcel_id).where(Face.block_id == self.block_id,
                                                         Face.recorded_parcel_id.is_not(None))).all())
        if current_associations != self._loaded_parcel_ids:
            raise ConcurrentModificationError(f"block {self.block_id}: parcel associations changed since load")
        loaded: dict[tuple[str, int], int] = getattr(self.graph, "_loaded_versions", {})
        node_ids = [i for (kind, i) in loaded if kind == "node"]
        edge_ids = [i for (kind, i) in loaded if kind == "edge"]
        for row in _latest_versions(self.session, NodeVersion, node_ids):
            if loaded.get(("node", row.id)) != row.version:
                raise ConcurrentModificationError(
                    f"block {self.block_id}: node {row.id} changed since this changeset was opened"
                )
        for row in _latest_versions(self.session, EdgeVersion, edge_ids):
            if loaded.get(("edge", row.id)) != row.version:
                raise ConcurrentModificationError(
                    f"block {self.block_id}: edge {row.id} changed since this changeset was opened"
                )

    def _persist(self) -> set[int]:
        cs_id = self._changeset.id

        node_to_edges: dict[int, list[int]] = defaultdict(list)
        for edge in self.graph.edges.values():
            node_to_edges[edge.n0].append(edge.id)
            node_to_edges[edge.n1].append(edge.id)

        touched_faces: set[int] = set()
        affected_edges: set[int] = set()
        for node_id in self._touched_nodes:
            for edge_id in node_to_edges.get(node_id, ()):
                affected_edges.add(edge_id)
                touched_faces.update(fid for fid in self.graph._edge_faces.get(edge_id, ()) if fid in self.graph.faces)
        touched_faces.update(self._identity_faces)
        for fid in self._identity_faces:
            affected_edges.update(eid for eid, _ in self.graph.faces[fid].all_edges)

        new_polys = self.graph.faces_to_polygons()
        for face_id in touched_faces:
            poly = new_polys[face_id].geom
            if not poly.is_valid:
                raise ValueError(
                    f"block {self.block_id}: this edit makes face {face_id} invalid "
                    f"({explain_validity(poly)}) -- refusing to persist"
                )
        # Individual validity (above) catches a face self-intersecting, but
        # NOT two different faces overlapping each other -- a real failure
        # mode this check alone missed: several nodes moved independently
        # (e.g. a Stage 5 fusion pass touching many nodes in one changeset)
        # can each keep their own face simple while pushing it across a
        # face it doesn't even share a node with. Found via the Stage 5
        # topology-invariant acceptance test, which is exactly the
        # "classic bug" a multi-node edit is supposed to catch. `_OVERLAP_TOL`
        # is a tiny absolute area (~1cm^2) -- far above GEOS float noise at
        # this project's real (UTM) coordinate scale, far below any overlap
        # from an actual crossed edge.
        #
        # `grid_size=GRID`: a raw overlay op without an explicit precision
        # grid can throw shapely.errors.GEOSException ("side location
        # conflict") on real, individually-valid inputs, not just return
        # an imprecise number -- this project's own established GEOS
        # lesson (see geocadastra/core/planarize.py). Found by review:
        # `grid_size` alone does not eliminate this on real synthetic
        # data (a near-zero-area internal gap between two parcels -- a
        # separate, pre-existing Stage 0 generator artifact, see
        # STAGE_5_NOTES.md -- is numerically pathological enough to still
        # trip GEOS's overlay robustness). A GEOS exception here means
        # the SAME thing an actual detected overlap does -- this edit is
        # not safe to trust -- so it's caught and refused the same way,
        # rather than left to crash the whole changeset. A crash is
        # strictly worse than the safe refusal this check exists to
        # produce.
        for face_id in touched_faces:
            poly = new_polys[face_id].geom
            for other_id, other in new_polys.items():
                if other_id == face_id:
                    continue
                try:
                    overlap = poly.intersection(other.geom, grid_size=GRID).area
                except GEOSException as e:
                    raise ValueError(
                        f"block {self.block_id}: this edit makes face {face_id} vs face {other_id} "
                        f"numerically unsafe to check ({e}) -- refusing to persist"
                    ) from e
                if overlap > _OVERLAP_TOL:
                    raise ValueError(
                        f"block {self.block_id}: this edit makes face {face_id} overlap face {other_id} "
                        f"by {overlap:.4f} m^2 -- refusing to persist"
                    )

        validate_area_change(self.session, self.block_id, self.graph,
                             self._original_polygons, new_polys, touched_faces)

        current_node_versions = {
            row.id: row.version for row in _latest_versions(self.session, NodeVersion, list(self._touched_nodes))
        }
        node_rows = [
            {
                "id": node_id,
                "version": current_node_versions.get(node_id, 0) + 1,
                "geom": from_shape(Point(self.graph.nodes[node_id].x, self.graph.nodes[node_id].y), srid=SRID),
                "changeset_id": cs_id,
            }
            for node_id in self._touched_nodes
        ]
        if node_rows:
            self.session.bulk_insert_mappings(NodeVersion, node_rows)

        for edge_id in affected_edges:
            edge_version = self.graph._loaded_versions.get(("edge", edge_id), 1)
            edge = self.graph.edges[edge_id]
            # an override on either endpoint node wins over the "manual_edit"
            # default; if both endpoints were touched by different callers
            # with different overrides (rare -- a fusion pass and a manual
            # edit in the same changeset), the lower node id's override wins,
            # deterministically, rather than depending on dict iteration order
            override = self._node_evidence.get(min(edge.n0, edge.n1)) or self._node_evidence.get(max(edge.n0, edge.n1))
            evidence_type, extra = override if override else ("manual_edit", {})
            if self._identity_faces:
                extra = {**extra, "face_parcel_ids": {str(fid): self.graph.face_parcel_ids[fid]
                         for fid in self._identity_faces if fid in self.graph.faces_of_edge(edge_id)}}
                if not override:
                    evidence_type = "parcel_identity"
            append_provenance(
                self.session,
                edge_id=edge_id,
                edge_version=edge_version,
                evidence_type=evidence_type,
                payload={"changeset_id": cs_id, "moved_nodes": sorted(self._touched_nodes), **extra},
            )

        for face_id in touched_faces:
            row = self.session.get(Face, face_id)
            row.geom = from_shape(new_polys[face_id].geom, srid=SRID)
            row.recorded_parcel_id = self.graph.face_parcel_ids.get(face_id)
            row.source_changeset_id = cs_id
        return touched_faces


@dataclass(frozen=True)
class FusionApplyResult:
    applied: list  # node_id -- fused position was safe and is now persisted
    conflicts: list  # ConflictRecord -- fusion's own source-vs-source disagreements, passed through unchanged
    topology_refused: list  # dict{"node_id", "sources", "reason"} -- see apply_fusion()


def apply_fusion(session: Session, block_id: int, fuse_result, author: str | None = None, *, commit: bool = True) -> FusionApplyResult:
    """Apply a Stage 5 `fuse_block()` result's moves ONE AT A TIME, each in
    its own changeset -- deliberately not one giant transaction for the
    whole block's worth of moves. With commit=False, each changeset uses
    a savepoint in the caller's transaction, so failed moves are isolated
    and the caller can publish or roll back the block atomically.

    Found by review (via the Stage 5 topology-invariant acceptance test):
    a "nearest point on legacy/model evidence" estimate has no awareness
    of nearby, unrelated faces, so some individual fused positions can be
    topologically unsafe even alone -- independent of how far they moved
    the node (a near-zero move can cross a nearby thin sliver; a large
    move along open space can be perfectly safe). Bundling every node's
    move into one all-or-nothing changeset meant a single unsafe candidate
    silently blocked every OTHER, perfectly safe improvement in the same
    block too -- the exact opposite of "where evidence is strong it
    dominates." Applying independently means 31 good moves are not held
    hostage by 21 bad ones in the same block.

    A move `ChangesetContext` refuses (would break planarity) is not
    silently dropped: it's recorded in `.topology_refused`, consistent
    with "nothing is silently resolved" -- fused evidence that disagrees
    with the *existing topology* too much to trust is conceptually the
    same kind of event as two sources disagreeing with each other, just
    caught procedurally rather than by comparing sigmas up front.
    """
    applied: list = []
    topology_refused: list = []
    for node_id, fused in fuse_result.moved.items():
        try:
            with ChangesetContext(session, block_id=block_id, author=author, description="fusion", commit=commit) as cs:
                cs.move_node(node_id, fused.x, fused.y, evidence_type="fusion", detail={"sources": list(fused.sources)})
            applied.append(node_id)
        except (ValueError, ConcurrentModificationError) as e:
            # ConcurrentModificationError is a RuntimeError, not a ValueError --
            # `except ValueError` alone let it propagate out of this whole
            # function, aborting every already-applied result and every
            # not-yet-attempted move for one ordinary concurrent edit landing
            # mid-batch (found by review: reproduced directly with a second
            # session committing a change to the same node between this
            # changeset's load and commit). It's the same kind of event as a
            # topology refusal -- this specific move is no longer safe to
            # trust -- so it's recorded the same way, not left to crash the
            # batch. A generic RuntimeError (e.g. load_block_graph's "store is
            # inconsistent" guard) is NOT caught here and still propagates --
            # that signals real data corruption, not a recoverable race, and
            # must not be silently folded into "this one move was refused".
            topology_refused.append({"node_id": node_id, "sources": fused.sources, "reason": str(e)})
    return FusionApplyResult(applied=applied, conflicts=list(fuse_result.conflicts), topology_refused=topology_refused)


__all__ = [
    "ChangesetContext",
    "ConcurrentModificationError",
    "FusionApplyResult",
    "apply_fusion",
    "load_block_graph",
    "lock_block",
    "seed_block_graph",
]
