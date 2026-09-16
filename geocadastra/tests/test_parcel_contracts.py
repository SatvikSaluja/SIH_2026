"""Identity, interior rings and recorded-area contracts through real storage."""
import pytest
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, Polygon, box
from sqlalchemy import select

from geocadastra.api import main as api
from geocadastra.core.crs import Geom
from geocadastra.core.graph import build_graph
from geocadastra.core.priority import face_edge_ids
from geocadastra.core.transport import parcels_to_graph
from geocadastra.store.changeset import ChangesetContext, allocate_ids, load_block_graph, seed_block_graph
from geocadastra.store.constraints import parcel_area_report
from geocadastra.store.schema import Face, FaceBoundary, IngestedBlock, RecordedParcel, SRID, recorded_parcel_id_seq
from geocadastra.tests.test_review_regressions import ingest

CRS = f"EPSG:{SRID}"


def contract_block(session, areas=(50,50)):
    job, bid, pid = ingest(session)
    session.get(IngestedBlock,(job.id,bid)).geom = from_shape(box(0,0,10,10),srid=SRID)
    first = session.get(RecordedParcel, pid)
    first.area = areas[0]
    pid2 = allocate_ids(session, recorded_parcel_id_seq, 1)[0]
    session.add(RecordedParcel(id=pid2, ward_job_id=job.id, block_id=bid, area=areas[1],
                              style="formal", seed_point=from_shape(Point(7,5),srid=SRID)))
    session.flush()
    graph = parcels_to_graph({pid: Geom(box(0,0,5,10), CRS), pid2: Geom(box(5,0,10,10), CRS)}, Geom(box(0,0,10,10), CRS))
    seed_block_graph(session, bid, graph)
    session.commit()
    return job, bid, pid, pid2


def test_hole_and_island_roundtrip_share_edges_and_edit_atomically(db_session):
    donut = Polygon(box(0,0,10,10).exterior.coords, [box(4,4,6,6).exterior.coords])
    graph = build_graph([Geom(donut,CRS), Geom(box(4,4,6,6),CRS)], CRS)
    seed_block_graph(db_session, 999, graph)
    db_session.commit()
    graph = load_block_graph(db_session,999)
    surrounding = next(fid for fid, f in graph.faces.items() if f.holes)
    assert len(face_edge_ids(graph)[surrounding]) == 8
    assert db_session.scalars(select(FaceBoundary).where(FaceBoundary.ring == 1)).all()
    assert graph.face_polygon(surrounding).equals(donut)
    for eid, _ in graph.faces[surrounding].holes[0]:
        assert len(set(graph.faces_of_edge(eid))) == 2
        assert -1 not in graph.faces_of_edge(eid)
    node = next(n for n in graph.nodes.values() if (n.x,n.y)==(4,4))
    with ChangesetContext(db_session,999) as cs:
        cs.move_node(node.id,4.1,4)
    graph = load_block_graph(db_session,999)
    assert sum(g.area for g in graph.faces_to_polygons().values()) == pytest.approx(100)
    assert all(g.geom.is_valid for g in graph.faces_to_polygons().values())
    for row in db_session.scalars(select(Face)):
        assert to_shape(row.geom).equals(graph.face_polygon(row.id))


def test_record_identity_survives_remapping_and_api_exposes_it(committed_session):
    job,bid,pid,pid2 = contract_block(committed_session)
    graph = load_block_graph(committed_session,bid)
    assert set(graph.face_parcel_ids.values()) == {pid,pid2}
    assert parcel_area_report(committed_session,bid,graph)["constraints_satisfied"]
    response = api.parcels(job.id,-1,-1,11,11,committed_session)
    assert {f["parcel_id"] for f in response["parcels"]} == {pid,pid2}
    result = api.analytics(job.id,committed_session)
    assert result["formal"]["area_error_relative"]["p50"] == 0


def test_area_violating_move_rolls_back_but_balanced_move_commits(committed_session):
    _,bid,pid,pid2 = contract_block(committed_session)
    graph = load_block_graph(committed_session,bid)
    bottom = next(n for n in graph.nodes.values() if (n.x,n.y)==(5,0))
    top = next(n for n in graph.nodes.values() if (n.x,n.y)==(5,10))
    with pytest.raises(ValueError,match="recorded-area"):
        with ChangesetContext(committed_session,bid) as cs:
            cs.move_node(bottom.id,5.2,0)
    assert load_block_graph(committed_session,bid).nodes[bottom.id].x == 5
    with ChangesetContext(committed_session,bid) as cs:
        cs.move_node(bottom.id,5.2,0)
        cs.move_node(top.id,4.8,10)
    assert parcel_area_report(committed_session,bid,load_block_graph(committed_session,bid))["constraints_satisfied"]


def test_existing_area_discrepancy_can_improve_but_cannot_worsen(committed_session):
    _,bid,_,_ = contract_block(committed_session,(55,45))
    graph = load_block_graph(committed_session,bid)
    nodes = [n for n in graph.nodes.values() if n.x==5]
    with ChangesetContext(committed_session,bid) as cs:
        for n in nodes:
            cs.move_node(n.id,5.2,n.y)
    report = parcel_area_report(committed_session,bid,load_block_graph(committed_session,bid))
    assert sorted(abs(r["error_m2"]) for r in report["parcels"]) == pytest.approx([3,3])
    assert not report["constraints_satisfied"]
    with pytest.raises(ValueError,match="recorded-area"):
        with ChangesetContext(committed_session,bid) as cs:
            for n in nodes:
                cs.move_node(n.id,5.1,n.y)


def test_field_verification_rejects_wrong_parcel_within_same_block(committed_session):
    job,bid,_,pid2 = contract_block(committed_session)
    graph = load_block_graph(committed_session,bid)
    node = next(n for n in graph.nodes.values() if (n.x,n.y)==(0,0))
    with pytest.raises(api.HTTPException) as error:
        api.field_verification(job.id,api.FieldVerificationRequest(block_id=bid,node_id=node.id,parcel_id=pid2,x=0,y=0),committed_session)
    assert error.value.status_code == 422
    assert "not incident" in error.value.detail


def test_ambiguous_identity_is_reported_not_guessed():
    graph = parcels_to_graph({10:Geom(box(0,0,10,10),CRS),20:Geom(box(0,0,10,10),CRS)},Geom(box(0,0,10,10),CRS))
    assert not graph.face_parcel_ids
    assert graph.identity_conflicts[0]["candidate_parcel_ids"] == [10,20]


def test_explicit_association_resolves_old_faces_and_records_provenance(committed_session):
    from geocadastra.store.schema import Provenance
    job,bid,pid = ingest(committed_session)
    graph = build_graph([Geom(box(0,0,20,20),CRS)],CRS)
    seed_block_graph(committed_session,bid,graph)
    committed_session.commit()
    graph = load_block_graph(committed_session,bid)
    assert not parcel_area_report(committed_session,bid,graph)["constraints_satisfied"]
    result = api.parcel_associations(job.id,api.ParcelAssociationsRequest(block_id=bid,assignments={fid:pid for fid in graph.faces},author="surveyor"),committed_session)
    assert result["constraints_satisfied"]
    assert committed_session.scalars(select(Provenance).where(Provenance.evidence_type=="parcel_identity")).all()


def test_identity_revision_invalidates_an_open_geometry_changeset(concurrent_sessions):
    from geocadastra.store.changeset import ConcurrentModificationError
    a,b = concurrent_sessions
    _,bid,pid,pid2 = contract_block(a)
    stale = ChangesetContext(b,bid)
    stale.__enter__()
    graph = load_block_graph(a,bid)
    with ChangesetContext(a,bid) as cs:
        cs.associate_faces({fid:pid2 if parcel==pid else pid for fid,parcel in graph.face_parcel_ids.items()})
    with pytest.raises(ConcurrentModificationError,match="associations"):
        stale.__exit__(None,None,None)


def test_identity_migration_preserves_old_rings_and_does_not_guess_records(db_engine):
    import uuid
    from pathlib import Path
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from geocadastra.store.schema import Base
    from geocadastra.tests.conftest import TEST_DB_URL
    name = "identity_migration_" + uuid.uuid4().hex[:10]
    with db_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{name}"'))
    engine = create_engine(TEST_DB_URL,connect_args={"options":f"-csearch_path={name},public"})
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            job,bid,pid = ingest(session)
            graph = build_graph([Geom(box(0,0,20,20),CRS)],CRS)
            graph.face_parcel_ids = {0:pid}
            seed_block_graph(session,bid,graph)
            session.commit()
        with engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE faces DROP COLUMN recorded_parcel_id")
            connection.exec_driver_sql("ALTER TABLE recorded_parcels DROP COLUMN area_tolerance_m2")
            connection.exec_driver_sql("ALTER TABLE face_boundaries DROP CONSTRAINT face_boundaries_pkey")
            connection.exec_driver_sql("ALTER TABLE face_boundaries DROP COLUMN ring")
            connection.exec_driver_sql("ALTER TABLE face_boundaries ADD PRIMARY KEY(face_id,position)")
        sql = (Path(__file__).resolve().parents[2]/"migrations/0002_parcel_identity_and_rings.sql").read_text()
        for _ in range(2):
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                connection.exec_driver_sql(sql)
        with Session(engine) as session:
            restored = load_block_graph(session,bid)
            assert len(restored.faces)==1
            assert not restored.face_parcel_ids
            assert next(iter(restored.faces_to_polygons().values())).area==400
            assert session.get(RecordedParcel,pid).area_tolerance_m2==.01
    finally:
        engine.dispose()
        with db_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


def test_changeset_replay_works_without_any_live_graph_rows(committed_session):
    from sqlalchemy import delete
    from geocadastra.store.replay import replay_block_graph
    from geocadastra.store.schema import EdgeVersion,NodeVersion
    _,bid,_,_ = contract_block(committed_session)
    graph = load_block_graph(committed_session,bid)
    bottom = next(n for n in graph.nodes.values() if (n.x,n.y)==(5,0))
    top = next(n for n in graph.nodes.values() if (n.x,n.y)==(5,10))
    with ChangesetContext(committed_session,bid) as cs:
        cs.move_node(bottom.id,5.2,0)
        cs.move_node(top.id,4.8,10)
    expected = load_block_graph(committed_session,bid)
    committed_session.execute(delete(FaceBoundary))
    committed_session.execute(delete(Face))
    committed_session.execute(delete(EdgeVersion))
    committed_session.execute(delete(NodeVersion))
    committed_session.commit()
    restored = replay_block_graph(committed_session,bid)
    assert restored.face_parcel_ids == expected.face_parcel_ids
    for fid,poly in expected.faces_to_polygons().items():
        assert restored.face_polygon(fid).equals_exact(poly.geom,0)


def test_equal_area_translation_cannot_move_parcels_outside_the_block(committed_session):
    _,bid,_,_ = contract_block(committed_session)
    graph = load_block_graph(committed_session,bid)
    with pytest.raises(ValueError,match="block coverage"):
        with ChangesetContext(committed_session,bid) as cs:
            for n in graph.nodes.values():
                cs.move_node(n.id,n.x+1,n.y)
