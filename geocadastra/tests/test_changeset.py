"""Stage 2 acceptance: a synthetic ward's parcel graph is seeded into the
store, an edit to a shared boundary node updates both adjacent parcels
consistently in one transaction, and replaying the versioned log from
empty reconstructs the same current state a direct "latest version" query
gives."""
import copy
import threading
import time

import pytest
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, Polygon, box
from sqlalchemy import select, text

from geocadastra.core.crs import Geom
from geocadastra.core.fusion import FusedPosition, FuseBlockResult
from geocadastra.core.graph import OUTER, build_graph
from geocadastra.synth.generator import generate_ward
from geocadastra.store.changeset import (
    ChangesetContext,
    ConcurrentModificationError,
    apply_fusion,
    load_block_graph,
    seed_block_graph,
)
from geocadastra.store.provenance import append_provenance, verify_chain
from geocadastra.store.schema import SRID, Changeset, EdgeVersion, Face, FaceBoundary, NodeVersion, Provenance

CRS = "EPSG:32643"


def _seed_a_multiparcel_block(session, seed=1):
    """Find a block with >=2 parcels (so there's a genuine shared boundary
    to edit) in a synthetic ward, seed its parcel graph into the store, and
    return the graph *reloaded from the store* -- seeding remaps the
    in-memory graph's own local (0-based) ids onto globally-unique stored
    ids, so callers must work with the reloaded graph's ids, not the
    pre-seed graph's."""
    for s in range(seed, seed + 20):
        ward = generate_ward(seed=s)
        by_block = {}
        for p in ward.parcels:
            by_block.setdefault(p.block_id, []).append(p)
        block_id, parcels = max(by_block.items(), key=lambda kv: len(kv[1]))
        if len(parcels) >= 2:
            local_graph = build_graph([Geom(p.polygon, CRS) for p in parcels], CRS)
            seed_block_graph(session, block_id=block_id, graph=local_graph, description="initial load")
            session.commit()
            return block_id, load_block_graph(session, block_id)
    raise AssertionError("could not find a multi-parcel block across 20 seeds")


def _any_face_overlap(polys, tol=1e-4) -> bool:
    ids = list(polys)
    return any(
        polys[ids[i]].geom.intersection(polys[ids[j]].geom).area > tol
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    )


def _safe_nudge(graph, node_id, magnitude=0.05, max_tries=20):
    """A small, *verified* displacement -- toward this node's own incident
    faces' combined centroid (a direction unlikely to cross into unrelated
    territory), backing off by half each try until the result actually has
    no cross-face overlap. Found by review (via Stage 5's stricter
    topology-invariant check): this synthetic generator's parcels are
    close/thin enough in places that even a small move in an arbitrary
    direction -- including, for some nodes, *toward the centroid itself*
    -- can create a real overlap with a face the moved node isn't even
    incident to. A direction heuristic alone isn't a guarantee; checking
    the actual result is (see ChangesetContext._persist()'s own matching
    check, which this mirrors so the test verifies what production
    enforces)."""
    incident_faces = {
        fid for e in graph.edges.values() if node_id in (e.n0, e.n1) for fid in graph.faces_of_edge(e.id) if fid != OUTER
    }
    polys = graph.faces_to_polygons()
    cx = sum(polys[fid].geom.centroid.x for fid in incident_faces) / len(incident_faces)
    cy = sum(polys[fid].geom.centroid.y for fid in incident_faces) / len(incident_faces)
    n = graph.nodes[node_id]
    dx, dy = cx - n.x, cy - n.y
    norm = (dx**2 + dy**2) ** 0.5
    if norm < 1e-9:
        dx, dy = 1.0, 0.0  # degenerate: node already sits at its own incident-faces' centroid
    else:
        dx, dy = dx / norm, dy / norm

    for attempt in range(max_tries):
        step = magnitude / (2**attempt)
        trial = copy.deepcopy(graph)
        trial.move_node(node_id, n.x + dx * step, n.y + dy * step)
        if not _any_face_overlap(trial.faces_to_polygons()):
            return dx * step, dy * step
    raise AssertionError(f"no safe nudge found for node {node_id} after {max_tries} tries")


def test_editing_a_shared_node_updates_both_adjacent_faces_consistently(db_session):
    block_id, graph = _seed_a_multiparcel_block(db_session)

    shared_edge_id = next(eid for eid, faces in graph._edge_faces.items() if len(faces) == 2)
    face_a_id, face_b_id = graph._edge_faces[shared_edge_id]
    node_id = graph.edges[shared_edge_id].n0
    old = graph.nodes[node_id]

    before_a = to_shape(db_session.get(Face, face_a_id).geom)
    before_b = to_shape(db_session.get(Face, face_b_id).geom)

    dx, dy = _safe_nudge(graph, node_id)
    with ChangesetContext(db_session, block_id=block_id, description="move shared node") as cs:
        cs.move_node(node_id, old.x + dx, old.y + dy)

    after_a = to_shape(db_session.get(Face, face_a_id).geom)
    after_b = to_shape(db_session.get(Face, face_b_id).geom)

    assert not before_a.equals_exact(after_a, 1e-9), "face A did not change"
    assert not before_b.equals_exact(after_b, 1e-9), "face B did not change"
    # consistent, not just "both changed": they must still share an exact boundary, no gap/overlap
    assert after_a.intersection(after_b).area < 1e-9
    shared_len_before = before_a.intersection(before_b).length
    shared_len_after = after_a.intersection(after_b).length
    assert shared_len_after > 0
    assert abs(shared_len_after - shared_len_before) > 1e-9  # the shared edge itself actually moved

    # the changeset row lists exactly what it touched
    cs_row = db_session.get(Changeset, cs._changeset.id)
    assert node_id in cs_row.affected_entities["nodes"]
    assert {face_a_id, face_b_id} <= set(cs_row.affected_entities["faces"])

    # and the node's version history is append-only: both versions exist
    versions = db_session.query(NodeVersion).filter_by(id=node_id).order_by(NodeVersion.version).all()
    assert [v.version for v in versions] == [1, 2]


def test_a_failed_changeset_leaves_the_store_completely_unchanged(db_session):
    block_id, graph = _seed_a_multiparcel_block(db_session, seed=30)
    node_id = next(iter(graph.nodes))
    before = db_session.query(NodeVersion).count()
    before_changesets = db_session.query(Changeset).count()

    with pytest.raises(KeyError):
        with ChangesetContext(db_session, block_id=block_id) as cs:
            cs.move_node(node_id, 0.0, 0.0)
            cs.move_node(999_999_999, 0.0, 0.0)  # unknown node -> PlanarGraph.move_node raises KeyError

    assert db_session.query(NodeVersion).count() == before
    assert db_session.query(Changeset).count() == before_changesets
    # the node's position is exactly what it was before the aborted edit
    current = load_block_graph(db_session, block_id)
    assert (current.nodes[node_id].x, current.nodes[node_id].y) == (graph.nodes[node_id].x, graph.nodes[node_id].y)


def test_replaying_the_version_log_from_empty_matches_the_latest_version_query(db_session):
    """"Replay from empty": walk every node's version rows in creation
    order, folding into a dict starting from nothing, rather than querying
    "latest version" directly -- an independent path to the same answer."""
    block_id, graph = _seed_a_multiparcel_block(db_session, seed=50)
    node_id = next(iter(graph.nodes))
    n = graph.nodes[node_id]
    with ChangesetContext(db_session, block_id=block_id) as cs:
        cs.move_node(node_id, n.x + 2.0, n.y + 3.0)
    other_node_id = next(nid for nid in graph.nodes if nid != node_id)
    n2 = graph.nodes[other_node_id]
    with ChangesetContext(db_session, block_id=block_id) as cs:
        cs.move_node(other_node_id, n2.x - 1.0, n2.y - 1.0)

    all_versions = (
        db_session.query(NodeVersion)
        .filter(NodeVersion.id.in_(graph.nodes.keys()))
        .order_by(NodeVersion.changeset_id, NodeVersion.id, NodeVersion.version)
        .all()
    )
    replayed: dict[int, tuple[float, float]] = {}
    for row in all_versions:
        pt = to_shape(row.geom)
        replayed[row.id] = (pt.x, pt.y)  # later versions simply overwrite earlier ones, in log order

    current = load_block_graph(db_session, block_id)
    direct = {nid: (node.x, node.y) for nid, node in current.nodes.items()}
    assert replayed == direct


def test_seeding_records_every_node_edge_and_face_under_one_changeset(db_session):
    block_id, graph = _seed_a_multiparcel_block(db_session, seed=70)
    cs = db_session.query(Changeset).order_by(Changeset.id.desc()).first()
    assert set(cs.affected_entities["nodes"]) == set(graph.nodes.keys())
    assert set(cs.affected_entities["edges"]) == set(graph.edges.keys())
    assert set(cs.affected_entities["faces"]) == set(graph.faces.keys())
    assert db_session.query(EdgeVersion).filter(EdgeVersion.id.in_(graph.edges.keys())).count() == len(graph.edges)


def test_editing_a_node_records_provenance_for_every_affected_edge(db_session):
    block_id, graph = _seed_a_multiparcel_block(db_session, seed=90)
    node_id = next(iter(graph.nodes))
    n = graph.nodes[node_id]
    affected_edges = {e.id for e in graph.edges.values() if node_id in (e.n0, e.n1)}
    assert affected_edges

    dx, dy = _safe_nudge(graph, node_id)
    with ChangesetContext(db_session, block_id=block_id) as cs:
        cs.move_node(node_id, n.x + dx, n.y + dy)

    rows = db_session.query(Provenance).filter(Provenance.edge_id.in_(affected_edges)).all()
    assert {r.edge_id for r in rows} == affected_edges
    for r in rows:
        assert r.evidence_type == "manual_edit"
        assert r.payload["moved_nodes"] == [node_id]
    assert verify_chain(db_session) is True


def test_move_node_evidence_type_override_is_recorded_in_provenance(db_session):
    """Stage 5: a caller (fusion) that knows *why* a node moved can say so,
    instead of every edit being recorded as the generic default."""
    block_id, graph = _seed_a_multiparcel_block(db_session, seed=91)
    node_id = next(iter(graph.nodes))
    n = graph.nodes[node_id]
    affected_edges = {e.id for e in graph.edges.values() if node_id in (e.n0, e.n1)}

    dx, dy = _safe_nudge(graph, node_id)
    with ChangesetContext(db_session, block_id=block_id) as cs:
        cs.move_node(node_id, n.x + dx, n.y + dy, evidence_type="fusion", detail={"sources": ("legacy", "gt")})

    rows = db_session.query(Provenance).filter(Provenance.edge_id.in_(affected_edges)).all()
    assert rows
    for r in rows:
        assert r.evidence_type == "fusion"
        assert r.payload["sources"] == ["legacy", "gt"]  # JSON round-trips the tuple as a list
    assert verify_chain(db_session) is True


def test_an_edit_that_makes_two_unrelated_faces_overlap_is_refused(db_session):
    """Regression, deterministic (no dependency on a synthetic seed
    happening to be geometrically "unlucky"): a face's own polygon can
    stay perfectly simple while still growing across a DIFFERENT face it
    shares no node with -- the classic multi-face topology bug individual
    per-face is_valid() checks can't catch. Three boxes: p1 and p2 share
    an edge at x=2; p3 sits above both with a gap, touching neither."""
    p1 = box(0, 0, 2, 2)
    p2 = box(2, 0, 4, 2)
    p3 = box(0, 2.5, 4, 4)
    local_graph = build_graph([Geom(p1, CRS), Geom(p2, CRS), Geom(p3, CRS)], CRS)
    node_id = next(nid for nid, n in local_graph.nodes.items() if (n.x, n.y) == (2.0, 2.0))

    seed_block_graph(db_session, block_id=500, graph=local_graph, description="seed")
    db_session.commit()
    graph = load_block_graph(db_session, 500)
    node_id = next(nid for nid, n in graph.nodes.items() if (n.x, n.y) == (2.0, 2.0))

    with pytest.raises(ValueError, match="overlap"):
        with ChangesetContext(db_session, block_id=500) as cs:
            cs.move_node(node_id, 2.0, 3.2)  # pushes p1/p2's shared corner up into p3, which shares no node with it

    # refused cleanly -- nothing written, session still usable
    assert db_session.query(NodeVersion).filter_by(id=node_id).count() == 1  # only the seeded version-1 row
    db_session.execute(select(NodeVersion).limit(1)).all()


def test_apply_fusion_refuses_one_bad_move_without_blocking_the_rest(db_session):
    """Same deterministic scenario as the test above, but through
    apply_fusion(): one unsafe candidate must not hold a completely
    unrelated, safe candidate hostage in the same block -- the whole
    reason apply_fusion() persists each move in its own changeset rather
    than one all-or-nothing transaction for the block (found by review:
    see its docstring in store/changeset.py)."""
    p1 = box(0, 0, 2, 2)
    p2 = box(2, 0, 4, 2)
    p3 = box(0, 2.5, 4, 4)
    local_graph = build_graph([Geom(p1, CRS), Geom(p2, CRS), Geom(p3, CRS)], CRS)
    seed_block_graph(db_session, block_id=501, graph=local_graph, description="seed")
    db_session.commit()
    graph = load_block_graph(db_session, 501)

    bad_node = next(nid for nid, n in graph.nodes.items() if (n.x, n.y) == (2.0, 2.0))
    safe_node = next(nid for nid, n in graph.nodes.items() if (n.x, n.y) == (0.0, 0.0))

    fake_result = FuseBlockResult(
        moved={
            bad_node: FusedPosition(x=2.0, y=3.2, sigma=1.0, sources=("test",)),  # into p3 -- unsafe
            safe_node: FusedPosition(x=0.01, y=0.01, sigma=1.0, sources=("test",)),  # tiny, safe nudge
        },
        conflicts=[],
        unchanged=[],
    )

    result = apply_fusion(db_session, 501, fake_result)

    assert result.applied == [safe_node]
    assert len(result.topology_refused) == 1
    assert result.topology_refused[0]["node_id"] == bad_node
    assert "overlap" in result.topology_refused[0]["reason"]

    final = load_block_graph(db_session, 501)
    assert (final.nodes[safe_node].x, final.nodes[safe_node].y) == (0.01, 0.01)
    assert (final.nodes[bad_node].x, final.nodes[bad_node].y) == (2.0, 2.0)  # left exactly alone


class TestTwoBlocksAndDataIntegrity:
    def test_two_independently_seeded_blocks_do_not_collide_on_ids(self, db_session):
        """Regression: a fresh in-memory PlanarGraph always numbers its own
        nodes/edges/faces starting from 0, so two blocks seeded with their
        own graph's ids as-is collided on every table's primary key the
        moment a second block was seeded."""
        ward = generate_ward(seed=100)
        by_block = {}
        for p in ward.parcels:
            by_block.setdefault(p.block_id, []).append(p)
        two_blocks = [bid for bid, ps in by_block.items() if len(ps) >= 1][:2]
        assert len(two_blocks) == 2

        graphs = {}
        for block_id in two_blocks:
            local_graph = build_graph([Geom(p.polygon, CRS) for p in by_block[block_id]], CRS)
            seed_block_graph(db_session, block_id=block_id, graph=local_graph, description="seed")
            db_session.commit()
            graphs[block_id] = load_block_graph(db_session, block_id)

        # both blocks are independently correct and their id sets don't overlap
        ids_a = set(graphs[two_blocks[0]].nodes) | set(graphs[two_blocks[0]].edges)
        ids_b = set(graphs[two_blocks[1]].nodes) | set(graphs[two_blocks[1]].edges)
        assert not (ids_a & ids_b)
        for block_id in two_blocks:
            recovered_area = sum(p.area for p in graphs[block_id].faces_to_polygons().values())
            true_area = sum(p.polygon.area for p in by_block[block_id])
            assert recovered_area == pytest.approx(true_area, rel=1e-6)

    def test_dangling_face_boundary_edge_reference_fails_fast(self, db_session):
        """Regression: a FaceBoundary row referencing a nonexistent edge used
        to load silently, then raise a confusing KeyError deep inside
        faces_to_polygons() the moment ANY face in the block was touched --
        poisoning every edit in the block, not just the broken face."""
        cs = Changeset(description="manual")
        db_session.add(cs)
        db_session.flush()
        db_session.add(Face(id=1, block_id=999, geom=from_shape(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), srid=SRID), source_changeset_id=cs.id))
        db_session.add(FaceBoundary(face_id=1, position=0, edge_id=12345, forward=True))  # no such edge
        db_session.commit()

        with pytest.raises(RuntimeError, match="inconsistent"):
            load_block_graph(db_session, 999)

    def test_moving_a_node_to_create_an_invalid_polygon_is_rejected(self, db_session):
        """No geometry validation used to exist at all: an edit that made a
        face self-intersecting committed silently. Uses a hand-built quad
        (not a synthetic ward) for a deterministic bowtie: moving corner B
        far enough above the A-B-C-D ring's C-D edge makes segment A-B'
        cross segment C-D."""
        quad = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # A, B, C, D
        graph = build_graph([Geom(quad, CRS)], CRS)
        block_id = 12345
        seed_block_graph(db_session, block_id=block_id, graph=graph, description="seed")
        db_session.commit()

        reloaded = load_block_graph(db_session, block_id)
        node_b_id = next(nid for nid, n in reloaded.nodes.items() if (n.x, n.y) == (10.0, 0.0))

        before_count = db_session.query(NodeVersion).count()
        with pytest.raises(ValueError, match="invalid"):
            with ChangesetContext(db_session, block_id=block_id) as cs:
                cs.move_node(node_b_id, 5.0, 20.0)  # crosses the opposite (C-D) edge
        assert db_session.query(NodeVersion).count() == before_count
        # session still usable
        db_session.execute(select(NodeVersion).limit(1)).all()


class TestConcurrency:
    def test_concurrent_edits_to_the_same_block_conflict_instead_of_silently_losing_one(self, concurrent_sessions):
        """Regression: two changesets editing different nodes on the same
        block/face raced to a silent lost update on the shared face's
        cached geometry, with no error at all."""
        s1, s2 = concurrent_sessions
        ward = generate_ward(seed=120)
        by_block = {}
        for p in ward.parcels:
            by_block.setdefault(p.block_id, []).append(p)
        block_id, parcels = max(by_block.items(), key=lambda kv: len(kv[1]))
        local_graph = build_graph([Geom(p.polygon, CRS) for p in parcels], CRS)
        seed_block_graph(s1, block_id=block_id, graph=local_graph, description="seed")
        s1.commit()

        baseline = load_block_graph(s1, block_id)
        node_ids = list(baseline.nodes)
        assert len(node_ids) >= 2
        node_a, node_b = node_ids[0], node_ids[1]

        cs1 = ChangesetContext(s1, block_id=block_id)
        cs1.__enter__()
        cs2 = ChangesetContext(s2, block_id=block_id)
        cs2.__enter__()  # both load the same pre-edit snapshot

        a = baseline.nodes[node_a]
        dx, dy = _safe_nudge(baseline, node_a)
        cs1.move_node(node_a, a.x + dx, a.y + dy)
        cs1.__exit__(None, None, None)  # succeeds

        b = baseline.nodes[node_b]
        cs2.move_node(node_b, b.x - 1.0, b.y - 1.0)
        with pytest.raises(ConcurrentModificationError):
            cs2.__exit__(None, None, None)

        # s2 is still usable after the failure (not left in a broken-transaction state)
        s2.execute(select(NodeVersion).limit(1)).all()

        final = load_block_graph(s1, block_id)
        assert (final.nodes[node_a].x, final.nodes[node_a].y) == (a.x + dx, a.y + dy)
        assert (final.nodes[node_b].x, final.nodes[node_b].y) == (b.x, b.y)  # s2's edit did not apply

    def test_apply_fusion_records_a_concurrent_modification_as_topology_refused_not_a_crash(
        self, concurrent_sessions, monkeypatch
    ):
        """Regression: apply_fusion()'s per-move try/except caught only
        ValueError, but ConcurrentModificationError is a RuntimeError -- an
        ordinary concurrent edit landing mid-batch crashed the WHOLE
        apply_fusion() call uncaught instead of being recorded like any
        other refused move, losing every already-applied result too
        (found by review, reproduced by injecting a real second session's
        commit into the exact load-to-commit window of one move)."""
        s1, s2 = concurrent_sessions
        p1, p2 = box(0, 0, 2, 2), box(2, 0, 4, 2)
        local_graph = build_graph([Geom(p1, CRS), Geom(p2, CRS)], CRS)
        seed_block_graph(s1, block_id=700, graph=local_graph, description="seed")
        s1.commit()

        graph = load_block_graph(s1, 700)
        node_a = next(nid for nid, n in graph.nodes.items() if (n.x, n.y) == (2.0, 0.0))

        # Inject a real concurrent commit to node_a from s2 exactly inside
        # apply_fusion's own load-to-commit window for that node, by having
        # it fire the first time ChangesetContext.move_node() is called --
        # after __enter__ (load) but before __exit__ (check + commit).
        original_move_node = ChangesetContext.move_node
        injected = {"done": False}

        def move_node_with_concurrent_write(self, *args, **kwargs):
            if not injected["done"]:
                injected["done"] = True
                with ChangesetContext(s2, block_id=700) as cs2:
                    cs2.move_node(node_a, 2.0, 0.05)
            return original_move_node(self, *args, **kwargs)

        monkeypatch.setattr(ChangesetContext, "move_node", move_node_with_concurrent_write)

        fake_result = FuseBlockResult(
            moved={node_a: FusedPosition(x=2.0, y=0.02, sigma=1.0, sources=("test",))},
            conflicts=[],
            unchanged=[],
        )
        result = apply_fusion(s1, 700, fake_result)  # must not raise

        assert result.applied == []
        assert len(result.topology_refused) == 1
        assert result.topology_refused[0]["node_id"] == node_a
        assert "changed since" in result.topology_refused[0]["reason"]

    def test_concurrent_provenance_appends_are_serialized_not_forked(self, concurrent_sessions):
        """Regression: two concurrent append_provenance calls both read the
        same chain tip and both committed successfully, forking the chain --
        verify_chain() then permanently (and falsely) reports tampering."""
        s1, s2 = concurrent_sessions
        events = []

        def append_and_commit(session, tag, delay_before_commit):
            append_provenance(session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"who": tag})
            events.append(f"{tag}-locked")
            time.sleep(delay_before_commit)
            session.commit()
            events.append(f"{tag}-committed")

        t1 = threading.Thread(target=append_and_commit, args=(s1, "first", 0.3))
        t1.start()
        time.sleep(0.05)  # let t1 acquire the advisory lock first
        t2 = threading.Thread(target=append_and_commit, args=(s2, "second", 0.0))
        t2.start()
        t1.join()
        t2.join()

        # t2 was blocked on the lock until t1 committed -- not run interleaved
        assert events.index("first-committed") < events.index("second-locked")

        rows = s1.execute(select(Provenance).order_by(Provenance.id)).scalars().all()
        assert len(rows) == 2
        assert rows[1].prev_hash == rows[0].this_hash  # linear, not forked
        assert verify_chain(s1) is True
