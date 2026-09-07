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
from shapely.geometry import Point
from shapely.validation import explain_validity
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from geocadastra.core.graph import Edge, Face as GraphFace, Node, PlanarGraph, _round_pt
from geocadastra.store.provenance import append_provenance
from geocadastra.store.schema import (
    SRID,
    Changeset,
    EdgeVersion,
    Face,
    FaceBoundary,
    NodeVersion,
    edge_id_seq,
    face_id_seq,
    node_id_seq,
)


class ConcurrentModificationError(RuntimeError):
    """Some part of this block changed in the store after this changeset
    loaded it and before it committed. Nothing was written."""


_OVERLAP_TOL = 1e-4  # m^2 -- see _persist()'s cross-face overlap check


def _allocate_ids(session: Session, seq, n: int) -> list[int]:
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
        .order_by(FaceBoundary.face_id, FaceBoundary.position)
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
        entries = sorted(by_face[f.id], key=lambda b: b.position)
        graph.faces[f.id] = GraphFace(f.id, tuple((b.edge_id, b.forward) for b in entries))
    graph._edge_faces = dict(edge_faces)
    graph._loaded_versions = loaded_versions  # Stage-2 bookkeeping for ChangesetContext's conflict check
    return graph


def seed_block_graph(
    session: Session, block_id: int, graph: PlanarGraph, author: str | None = None, description: str = "initial load"
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
    cs = Changeset(author=author, description=description, affected_entities={})
    session.add(cs)
    session.flush()

    node_ids = list(graph.nodes)
    edge_ids = list(graph.edges)
    face_ids = list(graph.faces)
    node_id_map = dict(zip(node_ids, _allocate_ids(session, node_id_seq, len(node_ids))))
    edge_id_map = dict(zip(edge_ids, _allocate_ids(session, edge_id_seq, len(edge_ids))))
    face_id_map = dict(zip(face_ids, _allocate_ids(session, face_id_seq, len(face_ids))))

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
                "geom": from_shape(polys[fid].geom, srid=SRID),
                "source_changeset_id": cs.id,
            }
            for fid in graph.faces
        ],
    )
    session.bulk_insert_mappings(
        FaceBoundary,
        [
            {"face_id": face_id_map[fid], "position": pos, "edge_id": edge_id_map[eid], "forward": forward}
            for fid, face in graph.faces.items()
            for pos, (eid, forward) in enumerate(face.boundary)
        ],
    )

    cs.affected_entities = {
        "nodes": sorted(node_id_map.values()),
        "edges": sorted(edge_id_map.values()),
        "faces": sorted(face_id_map.values()),
    }
    session.flush()
    return cs


class ChangesetContext:
    """`with ChangesetContext(session, block_id=...) as cs: cs.move_node(...)`

    Everything staged commits in one transaction on a clean `__exit__`;
    any exception -- from the `with` body, or from this class's own commit
    logic -- rolls the whole thing back, leaves the session reusable, and
    nothing is written.
    """

    def __init__(self, session: Session, block_id: int, author: str | None = None, description: str | None = None):
        self.session = session
        self.block_id = block_id
        self.author = author
        self.description = description
        self.graph: PlanarGraph | None = None
        self._touched_nodes: set[int] = set()
        self._node_evidence: dict[int, tuple[str, dict]] = {}
        self._changeset: Changeset | None = None

    def __enter__(self) -> "ChangesetContext":
        self.graph = load_block_graph(self.session, self.block_id)
        self._changeset = Changeset(author=self.author, description=self.description, affected_entities={})
        self.session.add(self._changeset)
        self.session.flush()  # assigns an id, without committing
        return self

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

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.session.rollback()
            return False
        try:
            self._check_no_concurrent_modification()
            touched_faces = self._persist()
            self._changeset.affected_entities = {
                "nodes": sorted(self._touched_nodes),
                "faces": sorted(touched_faces),
            }
            self.session.commit()
        except Exception:
            # a failure in OUR OWN commit logic (a conflict, an invalid
            # polygon, a DB error) must roll back just as surely as a
            # with-body exception does, or the session is left broken
            # (SQLAlchemy's PendingRollbackError) for whatever the caller
            # does next -- found by review.
            self.session.rollback()
            raise
        return False

    def _check_no_concurrent_modification(self) -> None:
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
        for face_id in touched_faces:
            poly = new_polys[face_id].geom
            for other_id, other in new_polys.items():
                if other_id == face_id:
                    continue
                overlap = poly.intersection(other.geom).area
                if overlap > _OVERLAP_TOL:
                    raise ValueError(
                        f"block {self.block_id}: this edit makes face {face_id} overlap face {other_id} "
                        f"by {overlap:.4f} m^2 -- refusing to persist"
                    )

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
            row.source_changeset_id = cs_id
        return touched_faces


@dataclass(frozen=True)
class FusionApplyResult:
    applied: list  # node_id -- fused position was safe and is now persisted
    conflicts: list  # ConflictRecord -- fusion's own source-vs-source disagreements, passed through unchanged
    topology_refused: list  # dict{"node_id", "sources", "reason"} -- see apply_fusion()


def apply_fusion(session: Session, block_id: int, fuse_result, author: str | None = None) -> FusionApplyResult:
    """Apply a Stage 5 `fuse_block()` result's moves ONE AT A TIME, each in
    its own changeset -- deliberately not one giant transaction for the
    whole block's worth of moves.

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
            with ChangesetContext(session, block_id=block_id, author=author, description="fusion") as cs:
                cs.move_node(node_id, fused.x, fused.y, evidence_type="fusion", detail={"sources": list(fused.sources)})
            applied.append(node_id)
        except ValueError as e:
            topology_refused.append({"node_id": node_id, "sources": fused.sources, "reason": str(e)})
    return FusionApplyResult(applied=applied, conflicts=list(fuse_result.conflicts), topology_refused=topology_refused)


__all__ = [
    "ChangesetContext",
    "ConcurrentModificationError",
    "FusionApplyResult",
    "apply_fusion",
    "load_block_graph",
    "seed_block_graph",
]
