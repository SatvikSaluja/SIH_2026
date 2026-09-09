"""Failure-window regressions from the September whole-codebase review."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from geoalchemy2.shape import from_shape, to_shape
from shapely import set_precision
from shapely.geometry import Point, Polygon, box
from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session

from geocadastra.api import main as api
from geocadastra.core.conformal import calibrate, certified_band
from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.evaluate import topology_validity_rate
from geocadastra.core.fusion import FusedPosition, FuseBlockResult, model_estimate
from geocadastra.core.graph import build_graph
from geocadastra.core.planarize import GRID
from geocadastra.core.priority import CostModel, face_uncertainty, priority_score, simulate_survey
from geocadastra.jobs import orchestrator as jobs
from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.heads import sdf_nll_loss
from geocadastra.models.infer import run_tiled_inference
from geocadastra.store import changeset as store
from geocadastra.store.provenance import verify_chain
from geocadastra.store.schema import (
    SRID, BlockJob, Changeset, Coregistration, Face, IngestedBlock, LegacyRecord,
    NodeVersion, PersistedConflict, Provenance, RecordedParcel, SurveyPoint, block_id_seq, recorded_parcel_id_seq,
)
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

CRS = f"EPSG:{SRID}"


def ingest(session):
    ward = generate_ward(WardParams(width=20, height=20, gsd=1, n_arterial_h=0,
        n_arterial_v=0, minor_spacing=100, style_weights={"institutional": 1},
        institutional_single_prob=1), seed=1)
    job = jobs.ingest_synthetic_ward(session, ward, 1)
    block_id = session.scalar(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id))
    parcel_id = session.scalar(select(RecordedParcel.id).where(RecordedParcel.ward_job_id == job.id))
    return job, block_id, parcel_id


def process(job, block_id):
    return jobs.process_block(TEST_DB_URL, TEST_SCHEMA, job.id, block_id, job.source, job.params)


def seed_square(session, block_id=999):
    store.seed_block_graph(session, block_id, build_graph([Geom(box(0, 0, 10, 10), CRS)], CRS))
    session.commit()
    return store.load_block_graph(session, block_id)


def test_simultaneous_commits_serialize_check_and_preserve_face_cache(concurrent_sessions, monkeypatch):
    a, b = concurrent_sessions
    graph = seed_square(a)
    contexts = [store.ChangesetContext(a, 999), store.ChangesetContext(b, 999)]
    ids = list(graph.nodes)[:2]
    for cs, nid in zip(contexts, ids):
        cs.__enter__()
        n = cs.graph.nodes[nid]
        cs.move_node(nid, n.x, n.y + .1)
    barrier = threading.Barrier(2)
    original_lock = store.lock_block

    def simultaneous_lock(session, block_id):
        barrier.wait(timeout=10)
        original_lock(session, block_id)

    monkeypatch.setattr(store, "lock_block", simultaneous_lock)

    def commit(cs):
        try:
            cs.__exit__(None, None, None)
            return "committed"
        except store.ConcurrentModificationError:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(commit, contexts))
    assert sorted(outcomes) == ["committed", "conflict"]
    a.expire_all()
    final = store.load_block_graph(a, 999)
    for fid, polygon in final.faces_to_polygons().items():
        assert to_shape(a.get(Face, fid).geom).equals(polygon.geom)
    assert verify_chain(a)


def test_parallel_deliveries_seed_once_and_release_transaction_lock(committed_session, monkeypatch):
    job, bid, _ = ingest(committed_session)
    args = (TEST_DB_URL, TEST_SCHEMA, job.id, bid, job.source, job.params)
    task = jobs.process_block._get_current_object()  # bind Celery before worker threads start
    barrier = threading.Barrier(2)
    original = jobs.assign_parcels
    calls = []

    def assignment(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(jobs, "assign_parcels", assignment)

    def worker(_):
        barrier.wait(timeout=10)
        return task(*args)

    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(worker, range(2)))
    assert sorted(outcomes) == ["already done", "done"]
    assert len(calls) == 1
    assert committed_session.scalar(select(func.count()).select_from(Face).where(Face.block_id == bid)) == 1
    # Another backend can acquire the exact same key after completion.
    committed_session.rollback()
    assert committed_session.scalar(text("SELECT pg_try_advisory_xact_lock(hashtextextended(current_schema() || ':block:' || CAST(:bid AS text), 0))"), {"bid": bid})
    committed_session.rollback()


def test_failure_after_fusion_rolls_back_the_whole_block_and_retry_succeeds(committed_session, monkeypatch):
    job, bid, _ = ingest(committed_session)
    original = jobs.apply_fusion

    def crash(*args, **kwargs):
        result = original(*args, **kwargs)
        assert kwargs["commit"] is False
        raise RuntimeError("crash after staged fusion")

    monkeypatch.setattr(jobs, "apply_fusion", crash)
    with pytest.raises(RuntimeError, match="crash"):
        process(job, bid)
    committed_session.expire_all()
    assert committed_session.get(BlockJob, (job.id, bid)).status == "failed"
    assert committed_session.scalar(select(func.count()).select_from(Face)) == 0
    assert committed_session.scalar(select(func.count()).select_from(NodeVersion)) == 0
    assert committed_session.scalar(select(func.count()).select_from(Provenance)) == 0
    monkeypatch.setattr(jobs, "apply_fusion", original)
    assert process(job, bid) == "done"
    assert committed_session.scalar(select(func.count()).select_from(Face)) == 1


def test_fusion_savepoint_refusal_keeps_parent_seed_and_safe_move(db_session):
    graph = build_graph([Geom(box(0, 0, 10, 10), CRS)], CRS)
    store.seed_block_graph(db_session, 999, graph)
    graph = store.load_block_graph(db_session, 999)
    nodes = list(graph.nodes.values())
    bad, target, good = nodes[:3]
    result = store.apply_fusion(db_session, 999, FuseBlockResult(moved={
        bad.id: FusedPosition(target.x, target.y, 1, ("test",)),
        good.id: FusedPosition(good.x, good.y + .1, 1, ("test",)),
    }, conflicts=[], unchanged=[]), commit=False)
    assert result.applied == [good.id]
    assert len(result.topology_refused) == 1
    db_session.commit()
    assert len(store.load_block_graph(db_session, 999).faces) == 1
    assert store.load_block_graph(db_session, 999).nodes[good.id].y == pytest.approx(good.y + .1)


@pytest.mark.parametrize("n,alpha", [(1, .1), (8, .1), (20, 0)])
def test_insufficient_conformal_samples_are_unbounded(n, alpha):
    bands = calibrate([("formal", float(i)) for i in range(n)], alpha)
    assert math.isinf(certified_band(bands, "formal", .5))


def test_coregistration_applies_original_transform_and_records_history(committed_session):
    job, bid, _ = ingest(committed_session)
    record = committed_session.scalar(select(LegacyRecord).where(LegacyRecord.ward_job_id == job.id))
    original = to_shape(record.geom)
    request = api.CoRegisterRequest(control_points=[(0,0,10,20), (1,0,11,20), (0,1,10,21)])
    for _ in range(2):
        api.coregister(job.id, request, committed_session)
        committed_session.refresh(record)
        assert to_shape(record.geom).centroid.x == pytest.approx(original.centroid.x + 10)
        assert to_shape(record.geom).centroid.y == pytest.approx(original.centroid.y + 20)
    assert to_shape(record.original_geom).equals(original)
    committed_session.refresh(job)
    assert job.coreg_transform[0][2] == pytest.approx(10)
    history = committed_session.scalars(select(Coregistration)).all()
    assert len(history) == 2
    assert all(committed_session.get(Changeset, row.changeset_id) for row in history)


def test_worker_uses_legacy_lines_and_keeps_all_conflict_types(committed_session, monkeypatch):
    job, bid, pid = ingest(committed_session)
    parcel = committed_session.get(RecordedParcel, pid)
    parcel.area *= 2
    committed_session.commit()
    original = jobs.fuse_block

    def fuse(graph, node_ids, **kwargs):
        assert kwargs["legacy_boundary"].geom_type in ("LineString", "MultiLineString")
        result = original(graph, node_ids, **kwargs)
        nodes = list(graph.nodes.values())
        result.moved[nodes[0].id] = FusedPosition(nodes[1].x, nodes[1].y, 1, ("legacy",))
        return result

    monkeypatch.setattr(jobs, "fuse_block", fuse)
    process(job, bid)
    rows = committed_session.scalars(select(PersistedConflict)).all()
    assert {r.kind for r in rows} >= {"area_sum_mismatch", "recorded_area_not_matched", "topology_refused"}
    assert all(r.detail for r in rows if r.kind != "source_disagreement")
    response = api.conflicts(job.id, committed_session)
    assert {c["kind"] for c in response["conflicts"]} >= {"area_sum_mismatch", "topology_refused"}


def test_unknown_edges_require_survey_and_do_not_make_nan_priorities():
    graph = build_graph([Geom(box(0,0,10,10), CRS)], CRS)
    assert math.isinf(face_uncertainty(graph, {})[0])
    edge_ids = tuple(graph.edges)
    curve = simulate_survey([[0]], {0:set()}, {}, {0:edge_ids}, {0:(5,5)}, (0,0), .1, CostModel(10,10))
    assert curve[0].certified_fraction == 0
    assert curve[-1].certified_fraction == 1
    assert curve[-1].hours > 0
    assert priority_score({0:{1}, 1:{0}}, {0:0, 1:math.inf}) == {0:0, 1:math.inf}


def test_store_rejects_wrong_crs_before_writing_and_seeds_provenance(db_session):
    with pytest.raises(CRSMismatchError):
        store.seed_block_graph(db_session, 999, build_graph([Geom(box(0,0,1,1), "EPSG:4326")], "EPSG:4326"))
    assert db_session.scalar(select(func.count()).select_from(Changeset)) == 0
    graph = seed_square(db_session)
    rows = db_session.scalars(select(Provenance)).all()
    assert {r.edge_id for r in rows} == set(graph.edges)
    assert all(r.evidence_type == "initial_load" for r in rows)
    assert verify_chain(db_session)


def test_field_attachment_failure_rolls_back_geometry(committed_session):
    job, bid, pid = ingest(committed_session)
    process(job, bid)
    graph = store.load_block_graph(committed_session, bid)
    node = next(iter(graph.nodes.values()))

    def fail_survey(session, flush_context, instances):
        if any(isinstance(obj, SurveyPoint) and obj.source == "field_verification" for obj in session.new):
            raise RuntimeError("survey insert failed")

    event.listen(committed_session, "before_flush", fail_survey)
    try:
        with pytest.raises(RuntimeError, match="survey insert"):
            api.field_verification(job.id, api.FieldVerificationRequest(block_id=bid, node_id=node.id,
                parcel_id=pid, x=node.x+.01, y=node.y), committed_session)
    finally:
        event.remove(committed_session, "before_flush", fail_survey)
    committed_session.expire_all()
    assert store.load_block_graph(committed_session, bid).nodes[node.id].x == node.x


def test_async_status_finishes_and_field_parcel_ownership_is_checked(committed_session, monkeypatch):
    job, bid, _ = ingest(committed_session)
    monkeypatch.setattr(jobs.process_block, "delay", lambda *a, **kw: None)
    jobs.run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    assert api.status(job.id, committed_session)["status"] == "running"
    process(job, bid)
    assert api.status(job.id, committed_session)["status"] == "done"
    node = next(iter(store.load_block_graph(committed_session, bid).nodes.values()))
    other, _, other_pid = ingest(committed_session)
    for pid in (987654321, other_pid):
        with pytest.raises(HTTPException) as error:
            api.field_verification(job.id, api.FieldVerificationRequest(block_id=bid, node_id=node.id,
                parcel_id=pid, x=node.x, y=node.y), committed_session)
        assert error.value.status_code == 422


def test_persisted_move_uses_geos_grid_and_refuses_node_collision(db_session):
    graph = seed_square(db_session)
    node, other = list(graph.nodes.values())[:2]
    with store.ChangesetContext(db_session, 999) as cs:
        cs.move_node(node.id, node.x+.012345, node.y)
    final = store.load_block_graph(db_session, 999).nodes[node.id]
    assert final.x == set_precision(Point(node.x+.012345, node.y), GRID).x
    with pytest.raises(ValueError, match="merge"):
        with store.ChangesetContext(db_session, 999) as cs:
            cs.move_node(node.id, other.x+.0001, other.y)


def test_analytics_uses_held_out_residuals_and_actual_stratum_quantiles(committed_session):
    job, bid, pid = ingest(committed_session)
    # Supply geometry directly: three zero-error held-out points in one block,
    # one 100m error in another; a 1000m fusion control must be excluded.
    store.seed_block_graph(committed_session, bid, build_graph([Geom(box(0,0,10,10), CRS)], CRS))
    bid2 = store.allocate_ids(committed_session, block_id_seq, 1)[0]
    pid2 = store.allocate_ids(committed_session, recorded_parcel_id_seq, 1)[0]
    committed_session.add(IngestedBlock(ward_job_id=job.id, block_id=bid2, local_block_id=1, geom=from_shape(box(200,0,210,10),srid=SRID)))
    committed_session.add(RecordedParcel(id=pid2, ward_job_id=job.id, block_id=bid2, area=100, style="institutional", seed_point=from_shape(Point(205,5),srid=SRID)))
    store.seed_block_graph(committed_session, bid2, build_graph([Geom(box(200,0,210,10), CRS)], CRS))
    committed_session.flush()
    # Existing synthetic points are made fusion controls in this controlled fixture.
    for row in committed_session.scalars(select(SurveyPoint)):
        row.purpose = "fusion"
    for p, x, y, purpose in [(pid,0,0,"evaluation"),(pid,0,5,"evaluation"),(pid,10,10,"evaluation"),(pid2,310,5,"evaluation"),(pid,1010,5,"fusion")]:
        committed_session.add(SurveyPoint(ward_job_id=job.id, parcel_id=p, geom=from_shape(Point(x,y),srid=SRID), source="test", purpose=purpose))
    committed_session.commit()
    result = api.analytics(job.id, committed_session)["institutional"]
    assert result["n_gt_points"] == 4
    assert result["boundary_position_error_p50"] == 0
    assert result["boundary_position_error_p90"] == pytest.approx(70)
    assert result["fusion_control_residual"]["n"] >= 1


def test_worker_never_uses_evaluation_points_for_fusion(committed_session, monkeypatch):
    job, bid, pid = ingest(committed_session)
    held_out = (12345.,54321.)
    committed_session.add(SurveyPoint(ward_job_id=job.id, parcel_id=pid, geom=from_shape(Point(*held_out),srid=SRID), source="test", purpose="evaluation"))
    committed_session.commit()
    original = jobs.fuse_block
    observed = []

    def fuse(*args, **kwargs):
        observed.append(kwargs["gt_points"])
        return original(*args, **kwargs)

    monkeypatch.setattr(jobs, "fuse_block", fuse)
    process(job, bid)
    assert observed and held_out not in observed[0]


def test_head_loss_tiling_and_fusion_agree_on_effective_variance():
    from rasterio.transform import from_origin
    model = MultiTaskNet(stem_width=16, decoder_width=16).eval()
    with torch.no_grad():
        model.sdf_head.net[-1].weight.zero_()
        model.sdf_head.net[-1].bias.copy_(torch.tensor([0.,20.]))
    logvar = 10*math.tanh(2)
    outputs = run_tiled_inference(model, np.zeros((3,32,32),np.float32), np.zeros((1,32,32),np.float32), tile_size=32, overlap=0)
    assert np.allclose(outputs["log_var"], logvar)
    estimate = model_estimate((5,5), outputs["sdf"], outputs["log_var"], from_origin(0,32,1,1))
    assert estimate.sigma == pytest.approx(math.exp(logvar/2), rel=1e-5)
    loss = sdf_nll_loss(torch.zeros(1),torch.tensor([logvar]),torch.ones(1))
    assert loss.item() == pytest.approx(.5*math.exp(-logvar)+.5*logvar)


def test_topology_evaluator_counts_invalid_faces_without_overlaying_them():
    graph = build_graph([Geom(Polygon([(0,0),(2,2),(0,2),(2,0)]),CRS), Geom(box(.2,.2,1,1),CRS)],CRS)
    result = topology_validity_rate(graph)
    assert result["n_invalid"] == 1
    assert result["n_unchecked_pairs"] == 1
    assert result["rate"] < 1


def test_schema_migration_upgrades_old_rows_and_is_repeatable(db_engine):
    import uuid
    from pathlib import Path
    from sqlalchemy import create_engine
    from geocadastra.store.schema import Base

    name = "migration_" + uuid.uuid4().hex[:12]
    with db_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{name}"'))
    engine = create_engine(TEST_DB_URL, connect_args={"options": f"-csearch_path={name},public"})
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            job, bid, pid = ingest(session)
            ward_id = job.id
        # Strip the additions to simulate the old schema with real ingested rows.
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE coregistrations")
            conn.exec_driver_sql("ALTER TABLE ward_jobs DROP COLUMN coreg_transform")
            conn.exec_driver_sql("ALTER TABLE legacy_records DROP COLUMN original_geom")
            conn.exec_driver_sql("ALTER TABLE survey_points DROP CONSTRAINT fk_survey_parcel_ward")
            conn.exec_driver_sql("ALTER TABLE survey_points DROP COLUMN purpose")
            conn.exec_driver_sql("ALTER TABLE recorded_parcels DROP CONSTRAINT uq_recorded_parcel_ward_id")
            conn.exec_driver_sql("ALTER TABLE conflicts DROP COLUMN kind, DROP COLUMN detail")
            conn.exec_driver_sql("ALTER TABLE conflicts ALTER COLUMN disagreement_m SET NOT NULL")
        sql = (Path(__file__).resolve().parents[2] / "migrations/0001_review_integrity.sql").read_text()
        for _ in range(2):
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.exec_driver_sql(sql)
        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(RecordedParcel)) == 1
            for point in session.scalars(select(SurveyPoint)):
                assert point.purpose == "fusion"  # old observations are never retroactively held out
            legacy = session.scalar(select(LegacyRecord))
            assert to_shape(legacy.original_geom).equals(to_shape(legacy.geom))
            api.coregister(ward_id, api.CoRegisterRequest(control_points=[(0,0,1,2),(1,0,2,2),(0,1,1,3)]),session)
            assert session.scalar(select(func.count()).select_from(Coregistration)) == 1
    finally:
        engine.dispose()
        with db_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))
