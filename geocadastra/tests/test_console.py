"""Stage 8: the bhoomi-ai console compatibility surface (api/console.py).

These endpoints were an Express server (artifacts/api-server/src/routes/
bhoomi.ts) until they moved here, and that server had no tests at all --
which is how it went unnoticed that production never routed through it, so
every route it implemented 404'd once deployed. The port was verified
against the running Express implementation response-by-response before
this file existed; these tests are what keeps the behaviour pinned now
that the comparison target is gone.

Same fixture contract as test_api.py, for the same reason: the request's
own session and the test's assertion session must see the same committed
rows, and `run_ward`'s dispatched tasks must land in the test schema too.
"""
import pytest
from fastapi.testclient import TestClient

from geocadastra.api.main import app, get_db_config, get_session
from geocadastra.jobs.orchestrator import app as celery_app
from geocadastra.jobs.orchestrator import ingest_synthetic_ward
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
    app.dependency_overrides[get_session] = lambda: committed_session
    app.dependency_overrides[get_db_config] = lambda: (TEST_DB_URL, TEST_SCHEMA)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _ingest(session, seed=1):
    ward = generate_ward(
        params=WardParams(width=50, height=35, gsd=1.0, n_arterial_h=1, n_arterial_v=0), seed=seed)
    job = ingest_synthetic_ward(session, ward, seed=seed)
    session.commit()
    return job


# ------------------------------------------------------------- discovery --


def test_regions_lists_each_ward_with_its_console_id(client, committed_session):
    job = _ingest(committed_session)
    body = client.get("/regions").json()
    entry = next(r for r in body if r["id"] == f"ward-{job.id}")
    assert entry["name"] == f"Ward {job.id} ({job.source})"
    # every ward this backend can build today is synthetic (see /wards/ingest)
    assert entry["type"] == "Synthetic ward"
    assert entry["status"] == "pending"


def test_dashboard_leaves_unbacked_metrics_null(client, committed_session):
    _ingest(committed_session)
    body = client.get("/dashboard").json()
    # not zero, not a plausible number: this backend computes none of these
    assert body["topologyErrors"] is None
    assert body["encroachments"] is None
    assert body["accuracyScore"] is None
    assert body["parcelsExtracted"] == 0  # nothing processed yet -- a real count
    assert body["activeRun"] is None
    assert body["recentActivity"]


def test_dashboard_counts_only_completed_area(client, committed_session):
    _ingest(committed_session)
    # ward is `pending`, so its area must not count toward processed area
    assert client.get("/dashboard").json()["areaProcessedSqKm"] == 0


# --------------------------------------------------------------- parcels --


def test_parcels_are_empty_before_processing(client, committed_session):
    job = _ingest(committed_session)
    assert client.get(f"/parcels?regionId=ward-{job.id}").json() == []


def test_parcels_carry_real_area_and_null_unknowns(client, committed_session):
    job = _ingest(committed_session)
    client.post(f"/wards/{job.id}/run")
    parcels = client.get(f"/parcels?regionId=ward-{job.id}").json()
    assert parcels, "processing a ward should produce faces"
    for parcel in parcels:
        assert parcel["areaSqM"] > 0
        assert parcel["regionId"] == f"ward-{job.id}"
        assert parcel["status"] in {"matched", "unmatched"}
        # the store has no owner and no per-parcel confidence; a placeholder
        # here would be indistinguishable from a measured one
        assert parcel["ownership"] is None
        assert parcel["confidence"] is None
        assert parcel["coordinateSystem"] == "LOCAL_METRES"


def test_parcels_status_filter_is_a_real_subset(client, committed_session):
    job = _ingest(committed_session)
    client.post(f"/wards/{job.id}/run")
    everything = client.get(f"/parcels?regionId=ward-{job.id}").json()
    matched = client.get(f"/parcels?regionId=ward-{job.id}&status=matched").json()
    assert len(matched) <= len(everything)
    assert all(p["status"] == "matched" for p in matched)


def test_single_parcel_lookup_round_trips(client, committed_session):
    job = _ingest(committed_session)
    client.post(f"/wards/{job.id}/run")
    first = client.get(f"/parcels?regionId=ward-{job.id}").json()[0]
    assert client.get(f"/parcels/{first['id']}").json() == first


def test_unknown_parcel_404s(client, committed_session):
    _ingest(committed_session)
    assert client.get("/parcels/99999999").status_code == 404


def test_parcel_editing_is_refused_not_faked(client, committed_session):
    _ingest(committed_session)
    resp = client.patch("/parcels/1", json={"status": "matched"})
    assert resp.status_code == 501
    assert "node-edit" in resp.json()["error"]


# ------------------------------------------ region id validation (no guessing) --


def test_a_malformed_region_id_is_a_client_error(client):
    assert client.get("/parcels?regionId=not-a-ward").status_code == 400


def test_an_unknown_region_id_404s_rather_than_returning_empty(client, committed_session):
    _ingest(committed_session)
    # the failure mode this guards: an empty list reads as "this ward has
    # nothing in it", which is a different fact from "no such ward"
    assert client.get("/parcels?regionId=ward-9999999").status_code == 404


# ------------------------------------------------------------ processing --


def test_processing_runs_report_real_block_progress(client, committed_session):
    job = _ingest(committed_session)
    before = next(r for r in client.get("/processing/runs").json() if r["regionId"] == f"ward-{job.id}")
    assert before["progress"] == 0
    client.post(f"/wards/{job.id}/run")
    after = next(r for r in client.get("/processing/runs").json() if r["regionId"] == f"ward-{job.id}")
    assert after["progress"] == 100
    assert after["status"] == "done"


def test_run_rejects_a_dataset_that_is_not_the_wards_own(client, committed_session):
    job = _ingest(committed_session)
    resp = client.post("/processing/runs", json={"regionId": f"ward-{job.id}", "dataset": "some-other-source"})
    # 409, never "generate a matching ward" -- a mismatch is a conflict to
    # report, not a cue to manufacture data behind the caller
    assert resp.status_code == 409


def test_run_accepts_the_wards_own_dataset(client, committed_session):
    job = _ingest(committed_session)
    resp = client.post("/processing/runs", json={"regionId": f"ward-{job.id}", "dataset": job.source})
    assert resp.status_code == 202
    assert resp.json()["regionId"] == f"ward-{job.id}"


def test_synthetic_creation_is_explicit(client):
    resp = client.post("/processing/synthetic", json={"seed": 3})
    assert resp.status_code == 201
    body = resp.json()
    assert body["n_blocks"] >= 2 and body["n_parcels"] > 0


# ---------------------------------------------------- topology and export --


def test_topology_scan_reports_its_own_limitations(client, committed_session):
    job = _ingest(committed_session)
    client.post(f"/wards/{job.id}/run")
    body = client.post("/topology/scan", json={"regionId": f"ward-{job.id}"}).json()
    assert body["checks"] == ["polygon_validity", "positive_area_overlap"]
    assert body["limitations"], "a scan must say what it does NOT certify"


def test_topology_fix_is_refused_not_faked(client, committed_session):
    _ingest(committed_session)
    resp = client.post("/topology/fix", json={"regionId": "ward-1"})
    assert resp.status_code == 501


def test_changes_is_honestly_empty(client):
    # change detection does not exist in this backend; [] is the true answer
    assert client.get("/changes").json() == []


def test_geojson_export_refuses_local_coordinates(client, committed_session):
    job = _ingest(committed_session)
    client.post(f"/wards/{job.id}/run")
    resp = client.post("/exports", json={"regionId": f"ward-{job.id}", "format": "GeoJSON"})
    # synthetic wards live in a local metric frame with no geographic
    # position -- emitting them as WGS84 GeoJSON would place them off Africa
    assert resp.status_code == 422
    assert "local coordinates" in resp.json()["detail"]


def test_unsupported_export_format_is_refused(client, committed_session):
    job = _ingest(committed_session)
    resp = client.post("/exports", json={"regionId": f"ward-{job.id}", "format": "Shapefile"})
    assert resp.status_code == 422
