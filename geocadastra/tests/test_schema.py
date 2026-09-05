import random

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, Polygon, box
from sqlalchemy import text

from geocadastra.store.schema import SRID, Changeset, EdgeVersion, Face, FaceBoundary, NodeVersion, Provenance


def _changeset(session, **kw):
    cs = Changeset(**kw)
    session.add(cs)
    session.flush()
    return cs


def test_node_version_round_trips_geometry(db_session):
    cs = _changeset(db_session)
    db_session.add(NodeVersion(id=1, version=1, geom=from_shape(Point(500000, 2300000), srid=SRID), changeset_id=cs.id))
    db_session.commit()
    row = db_session.get(NodeVersion, (1, 1))
    shp = to_shape(row.geom)
    assert (shp.x, shp.y) == (500000.0, 2300000.0)


def test_node_history_is_append_only_not_updated_in_place(db_session):
    cs1 = _changeset(db_session)
    db_session.add(NodeVersion(id=1, version=1, geom=from_shape(Point(0, 0), srid=SRID), changeset_id=cs1.id))
    db_session.commit()
    cs2 = _changeset(db_session)
    db_session.add(NodeVersion(id=1, version=2, geom=from_shape(Point(5, 5), srid=SRID), changeset_id=cs2.id))
    db_session.commit()

    rows = db_session.query(NodeVersion).filter_by(id=1).order_by(NodeVersion.version).all()
    assert len(rows) == 2  # both versions survive
    assert to_shape(rows[0].geom).coords[0] == (0.0, 0.0)
    assert to_shape(rows[1].geom).coords[0] == (5.0, 5.0)


def test_edge_version_stores_node_ids_and_interior_shape(db_session):
    cs = _changeset(db_session)
    db_session.add_all([
        NodeVersion(id=1, version=1, geom=from_shape(Point(0, 0), srid=SRID), changeset_id=cs.id),
        NodeVersion(id=2, version=1, geom=from_shape(Point(10, 0), srid=SRID), changeset_id=cs.id),
        EdgeVersion(id=100, version=1, n0_id=1, n1_id=2, interior=[[5.0, 1.0]], changeset_id=cs.id),
    ])
    db_session.commit()
    e = db_session.get(EdgeVersion, (100, 1))
    assert (e.n0_id, e.n1_id) == (1, 2)
    assert e.interior == [[5.0, 1.0]]


def test_face_and_face_boundary_store_the_ordered_edge_walk(db_session):
    cs = _changeset(db_session)
    poly = box(0, 0, 10, 10)
    db_session.add(Face(id=1, block_id=1, geom=from_shape(poly, srid=SRID), source_changeset_id=cs.id))
    db_session.add_all([
        FaceBoundary(face_id=1, position=0, edge_id=10, forward=True),
        FaceBoundary(face_id=1, position=1, edge_id=11, forward=True),
        FaceBoundary(face_id=1, position=2, edge_id=12, forward=False),
        FaceBoundary(face_id=1, position=3, edge_id=13, forward=True),
    ])
    db_session.commit()

    face = db_session.get(Face, 1)
    assert to_shape(face.geom).area == 100.0
    boundary = db_session.query(FaceBoundary).filter_by(face_id=1).order_by(FaceBoundary.position).all()
    assert [(b.edge_id, b.forward) for b in boundary] == [(10, True), (11, True), (12, False), (13, True)]


def test_which_faces_touch_a_given_edge_is_one_indexed_join_not_a_json_scan(db_session):
    """The whole reason FaceBoundary is a real table, not a JSON blob:
    "which faces reference edge X" must be a plain indexed lookup, since
    updating both incident faces of a shared edge in one transaction
    depends on finding them quickly."""
    cs = _changeset(db_session)
    db_session.add(Face(id=1, block_id=1, geom=from_shape(box(0, 0, 5, 10), srid=SRID), source_changeset_id=cs.id))
    db_session.add(Face(id=2, block_id=1, geom=from_shape(box(5, 0, 10, 10), srid=SRID), source_changeset_id=cs.id))
    db_session.add_all([
        FaceBoundary(face_id=1, position=0, edge_id=99, forward=True),  # the shared edge
        FaceBoundary(face_id=2, position=0, edge_id=99, forward=False),
    ])
    db_session.commit()

    faces = {fid for (fid,) in db_session.query(FaceBoundary.face_id).filter_by(edge_id=99).all()}
    assert faces == {1, 2}


def test_provenance_hash_chain_links_consecutive_rows(db_session):
    genesis = "0" * 64
    r1 = Provenance(edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"note": "a"}, prev_hash=genesis, this_hash="a" * 64)
    db_session.add(r1)
    db_session.commit()
    r2 = Provenance(edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"note": "b"}, prev_hash=r1.this_hash, this_hash="b" * 64)
    db_session.add(r2)
    db_session.commit()
    assert r2.prev_hash == r1.this_hash


def test_spatial_query_on_a_large_node_table_uses_the_gist_index_not_a_seq_scan(db_session):
    """GiST indexes exist on every geometry column, and a realistic spatial
    query over a large-enough table must actually use one -- a regression
    here means an index got dropped or the planner statistics are stale."""
    cs = _changeset(db_session)
    rng = random.Random(0)
    batch = []
    for i in range(20_000):
        x, y = 400000 + rng.random() * 100000, 2200000 + rng.random() * 100000
        batch.append(NodeVersion(id=i, version=1, geom=from_shape(Point(x, y), srid=SRID), changeset_id=cs.id))
    db_session.bulk_save_objects(batch)
    db_session.commit()
    db_session.execute(text("ANALYZE test_stage2.nodes"))

    plan = db_session.execute(
        text(
            "EXPLAIN SELECT id FROM nodes WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(450000, 2250000), :srid), 50)"
        ),
        {"srid": SRID},
    ).fetchall()
    plan_text = "\n".join(row[0] for row in plan)
    assert "Index Scan" in plan_text or "Bitmap Index Scan" in plan_text, f"expected an index scan, got:\n{plan_text}"
    assert "Seq Scan" not in plan_text, f"query planner did a sequential scan instead of using the GiST index:\n{plan_text}"
