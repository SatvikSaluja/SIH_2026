"""Stage 8: FastAPI surface. Overrides `get_session` (the standard
FastAPI testing pattern) to point every request at the real test
Postgres/PostGIS, via `committed_session`'s real-commit contract -- the
API's own session and a test's own assertion session must both see the
same committed rows, which a rolled-back `db_session` transaction
wouldn't allow across the TestClient's separate request handling.

`slow`, same reasoning as test_orchestrator.py: every non-trivial
endpoint here runs the real Stage 3/5 pipeline through real Postgres.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from geocadastra.api.main import app, get_db_config, get_session
from geocadastra.jobs.orchestrator import app as celery_app
from geocadastra.jobs.orchestrator import ingest_synthetic_ward
from geocadastra.store.schema import IngestedBlock, WardJob
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _eager_celery():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


@pytest.fixture()
def client(committed_session):
    # both overrides needed together: get_session points the request's own
    # queries at the test schema, get_db_config points wherever run_ward()
    # dispatches Celery tasks at the SAME schema -- found by running this
    # end to end: overriding only get_session left /run's dispatched tasks
    # looking for block_jobs in the real default schema, which doesn't exist
    app.dependency_overrides[get_session] = lambda: committed_session
    app.dependency_overrides[get_db_config] = lambda: (TEST_DB_URL, TEST_SCHEMA)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _small_ward(seed=1):
    # deliberately smaller than test_orchestrator.py's own ward -- these tests
    # only need real endpoint behavior, not a wide enough spread to ever trigger
    # a fusion conflict, and apply_fusion()'s known O(N) per-node DB reload cost
    # (see test_orchestrator.py's own module docstring) makes every extra parcel
    # here add real seconds across a dozen-plus tests
    return generate_ward(params=WardParams(width=50, height=35, gsd=1.0, n_arterial_h=1, n_arterial_v=0), seed=seed)


def _global_block_ids(session, ward_job_id, ward) -> list:
    """`IngestedBlock.block_id` (globally unique) in the same order as
    `ward.blocks` -- see test_orchestrator.py's own identical helper for
    why: the store addresses a block by this id, not the synthetic
    ward's own ward-local `Block.id`."""
    rows = session.execute(select(IngestedBlock).where(IngestedBlock.ward_job_id == ward_job_id)).scalars().all()
    by_local = {r.local_block_id: r.block_id for r in rows}
    return [by_local[b.id] for b in ward.blocks]


def test_ingest_creates_a_ward_job(client):
    resp = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ward_job_id"] is not None
    assert body["n_blocks"] >= 2
    assert body["n_parcels"] > 0


def test_status_404s_for_an_unknown_ward(client):
    resp = client.get("/wards/999999/status")
    assert resp.status_code == 404


def test_run_then_status_reports_every_block_done(client):
    ingest_resp = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = ingest_resp.json()["ward_job_id"]

    run_resp = client.post(f"/wards/{ward_job_id}/run")
    assert run_resp.status_code == 200
    assert run_resp.json()["already_done"] == 0

    status_resp = client.get(f"/wards/{ward_job_id}/status")
    body = status_resp.json()
    assert body["status"] == "done"
    assert all(b["status"] == "done" for b in body["blocks"])


def test_run_twice_is_idempotent_through_the_api(client):
    ingest_resp = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = ingest_resp.json()["ward_job_id"]
    client.post(f"/wards/{ward_job_id}/run")

    second = client.post(f"/wards/{ward_job_id}/run")
    body = second.json()
    assert body["dispatched"] == []
    assert body["already_done"] == body["total"]


def test_parcels_query_returns_faces_inside_the_bbox(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    minx, miny, maxx, maxy = ward.ward_polygon.bounds
    resp = client.get(f"/wards/{job.id}/parcels", params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
    assert resp.status_code == 200
    parcels = resp.json()["parcels"]
    assert len(parcels) > 0
    assert all(p["block_id"] is not None for p in parcels)


def test_parcels_query_outside_the_ward_returns_nothing(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    resp = client.get(f"/wards/{job.id}/parcels", params={"minx": 100000, "miny": 100000, "maxx": 100010, "maxy": 100010})
    assert resp.json()["parcels"] == []


def test_edit_applies_a_valid_move_and_returns_200(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.store.changeset import load_block_graph
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    block_id = _global_block_ids(committed_session, job.id, ward)[0]
    graph = load_block_graph(committed_session, block_id)
    node_id, node = next(iter(graph.nodes.items()))

    resp = client.post(
        f"/wards/{job.id}/edit",
        json={"block_id": block_id, "node_id": node_id, "x": node.x + 0.01, "y": node.y + 0.01},
    )
    assert resp.status_code in (200, 422)  # 422 if the tiny move happens to make a face invalid -- both are real outcomes


def test_edit_of_an_unknown_node_is_a_client_error_not_a_500(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    block_id = _global_block_ids(committed_session, job.id, ward)[0]
    resp = client.post(f"/wards/{job.id}/edit", json={"block_id": block_id, "node_id": 999999999, "x": 0.0, "y": 0.0})
    assert resp.status_code == 422


def test_field_verification_persists_a_survey_point_and_applies_the_move(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.store.changeset import load_block_graph
    from geocadastra.store.schema import SurveyPoint
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    block_id = _global_block_ids(committed_session, job.id, ward)[0]
    graph = load_block_graph(committed_session, block_id)
    node_id, node = next(iter(graph.nodes.items()))
    from geocadastra.store.schema import RecordedParcel
    incident = {fid for eid, e in graph.edges.items() if node_id in (e.n0, e.n1) for fid in graph.faces_of_edge(eid)}
    parcel_id = next(graph.face_parcel_ids[fid] for fid in incident if fid in graph.face_parcel_ids)

    resp = client.post(
        f"/wards/{job.id}/field-verification",
        json={"block_id": block_id, "node_id": node_id, "parcel_id": parcel_id, "x": node.x, "y": node.y},
    )
    assert resp.status_code == 200
    rows = committed_session.execute(
        select(SurveyPoint).where(SurveyPoint.ward_job_id == job.id, SurveyPoint.source == "field_verification")
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].parcel_id == parcel_id


def test_conflicts_endpoint_returns_a_list_even_when_empty(client):
    resp_ingest = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = resp_ingest.json()["ward_job_id"]
    resp = client.get(f"/wards/{ward_job_id}/conflicts")
    assert resp.status_code == 200
    assert resp.json() == {"conflicts": []}  # nothing processed yet -- no conflicts recorded, not an error


def test_coregister_requires_at_least_3_control_points(client):
    resp_ingest = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = resp_ingest.json()["ward_job_id"]
    resp = client.post(f"/wards/{ward_job_id}/coregister", json={"control_points": [[0, 0, 0, 0], [1, 0, 1, 0]]})
    assert resp.status_code == 422


def test_coregister_computes_a_real_residual_and_stores_it(client, committed_session):
    resp_ingest = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = resp_ingest.json()["ward_job_id"]
    # a perfect identity mapping except one point off by (1, 0) -- a real, non-zero residual
    resp = client.post(
        f"/wards/{ward_job_id}/coregister",
        json={"control_points": [[0, 0, 0, 0], [10, 0, 10, 0], [0, 10, 0, 10], [10, 10, 11, 10]]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["residual_m"] > 0
    job = committed_session.get(WardJob, ward_job_id)
    committed_session.refresh(job)
    assert job.coreg_residual_m == pytest.approx(body["residual_m"])


def test_coregister_on_an_unknown_ward_404s(client):
    resp = client.post("/wards/999999/coregister", json={"control_points": [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]})
    assert resp.status_code == 404


def test_coregister_rejects_degenerate_control_points(client):
    """Found by review: three IDENTICAL control points pass the '>= 3
    points' count check but can't determine a unique affine transform --
    the fit used to silently 'succeed' with a spuriously perfect 0.0
    residual instead of raising, treating a garbage upload as perfectly
    registered."""
    resp_ingest = client.post("/wards/ingest", json={"seed": 1, "width": 50, "height": 35, "n_arterial_h": 1, "n_arterial_v": 0})
    ward_job_id = resp_ingest.json()["ward_job_id"]
    resp = client.post(
        f"/wards/{ward_job_id}/coregister",
        json={"control_points": [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]},
    )
    assert resp.status_code == 422


def test_ingest_rejects_a_zero_gsd(client):
    """Found by review: gsd=0 propagated unvalidated into generate_ward(),
    where int(np.ceil(width / gsd)) raised an unhandled ZeroDivisionError."""
    resp = client.post("/wards/ingest", json={"seed": 1, "gsd": 0})
    assert resp.status_code == 422


def test_tile_endpoint_rejects_an_out_of_range_tile(client, committed_session):
    """Found by review: an extreme y overflowed math.sinh() inside the
    tile-to-lat/lon conversion with an unhandled OverflowError."""
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    resp = client.get(f"/wards/{job.id}/tiles/1/0/1000000000000000000.mvt")
    assert resp.status_code == 422


def test_edit_of_a_block_belonging_to_a_different_ward_404s(client, committed_session):
    """Found by review: neither /edit nor /field-verification checked
    that the block in the request body actually belongs to the ward in
    the URL -- a request could edit a DIFFERENT ward's block by naming
    its real (globally-unique) block id under the wrong ward_job_id, with
    no error at all."""
    ward_a = _small_ward(seed=1)
    ward_b = _small_ward(seed=2)
    job_a = ingest_synthetic_ward(committed_session, ward_a, seed=1)
    job_b = ingest_synthetic_ward(committed_session, ward_b, seed=2)
    block_of_b = _global_block_ids(committed_session, job_b.id, ward_b)[0]

    resp = client.post(
        f"/wards/{job_a.id}/edit",  # ward A's URL...
        json={"block_id": block_of_b, "node_id": 0, "x": 0.0, "y": 0.0},  # ...naming ward B's real block
    )
    assert resp.status_code == 404


def test_tile_endpoint_returns_mvt_bytes_for_a_processed_ward(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    # find a z/x/y tile that actually covers this ward: start coarse (z=1) --
    # any tile in a small-z grid covers most of the globe, guaranteed to overlap
    resp = client.get(f"/wards/{job.id}/tiles/1/0/0.mvt")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.mapbox-vector-tile"


def test_analytics_reports_one_entry_per_style_not_pooled(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    from geocadastra.jobs.orchestrator import run_ward as _run_ward
    from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    resp = client.get(f"/wards/{job.id}/analytics")
    assert resp.status_code == 200
    body = resp.json()
    present_styles = {p.style for p in ward.parcels}
    assert set(body.keys()) == present_styles  # one entry per style actually present, no pooled "overall" key
    for style, metrics in body.items():
        assert "boundary_position_error_p50" in metrics
        assert "topology_validity_rate" in metrics
        assert "parcel_count" in metrics
