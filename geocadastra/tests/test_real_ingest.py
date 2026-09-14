"""Ingesting data the project did not generate.

Everything here is validation at a trust boundary: this is the one place
data the project did not produce gets in, and each refusal exists because
the failure it prevents is silent further downstream.
"""
import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box, mapping

from geocadastra.core.crs import CRSMismatchError
from geocadastra.jobs.ingest import (ingest_ward, parcels_from_geojson,
                                     read_raster_triple)

CRS = "EPSG:32643"
TRANSFORM = Affine(0.5, 0.0, 0.0, 0.0, -0.5, 60.0)  # 0.5 m pixels, north-up


def _write(path, array, *, transform=TRANSFORM, crs=CRS, dtype=None):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=dtype or array.dtype.name,
                       crs=crs, transform=transform) as dst:
        dst.write(array)
    return path


@pytest.fixture
def imagery(tmp_path):
    """A 100 m x 60 m ward at 0.5 m: 120 rows, 200 cols."""
    height, width = 120, 200
    rng = np.random.default_rng(0)
    ortho = rng.integers(0, 255, size=(3, height, width), dtype=np.uint8)
    dtm = np.full((height, width), 100.0, dtype=np.float32)
    dsm = dtm + 3.0
    return (_write(tmp_path / "ortho.tif", ortho),
            _write(tmp_path / "dsm.tif", dsm),
            _write(tmp_path / "dtm.tif", dtm))


@pytest.fixture
def ward_inputs(imagery):
    block = box(0, 0, 40, 40)
    parcels = [{"block": 0, "area": 800.0, "style": "formal", "seed_point": (10, 20),
                "area_tolerance_m2": 1.0},
               {"block": 0, "area": 800.0, "style": "formal", "seed_point": (30, 20),
                "area_tolerance_m2": 1.0}]
    return [block], parcels, imagery


def test_reads_a_consistent_raster_triple(imagery):
    ortho, dsm, dtm, transform, crs = read_raster_triple(*imagery)
    assert ortho.shape == (120, 200, 3) and ortho.dtype == np.uint8
    assert dsm.shape == dtm.shape == (120, 200)
    assert crs == CRS
    assert np.allclose(dsm - dtm, 3.0)
    assert tuple(transform)[:6] == pytest.approx(tuple(TRANSFORM)[:6])


def test_dsm_on_a_different_grid_is_refused(tmp_path, imagery):
    ortho, _, dtm = imagery
    shifted = _write(tmp_path / "shifted.tif", np.full((120, 200), 103.0, dtype=np.float32),
                     transform=Affine(0.5, 0.0, 25.0, 0.0, -0.5, 60.0))
    with pytest.raises(ValueError, match="georeferencing differs"):
        read_raster_triple(ortho, shifted, dtm)


def test_dsm_of_a_different_size_is_refused(tmp_path, imagery):
    ortho, _, dtm = imagery
    small = _write(tmp_path / "small.tif", np.full((60, 100), 103.0, dtype=np.float32))
    with pytest.raises(ValueError, match="ortho is"):
        read_raster_triple(ortho, small, dtm)


def test_a_foreign_projection_is_refused_rather_than_assumed(tmp_path):
    """Silently treating degrees as metres is the failure this prevents."""
    array = np.zeros((3, 10, 10), dtype=np.uint8)
    wgs84 = _write(tmp_path / "wgs84.tif", array, crs="EPSG:4326",
                   transform=Affine(0.001, 0, 77.0, 0, -0.001, 28.0))
    surface = _write(tmp_path / "s.tif", np.zeros((10, 10), dtype=np.float32), crs="EPSG:4326",
                     transform=Affine(0.001, 0, 77.0, 0, -0.001, 28.0))
    with pytest.raises(CRSMismatchError, match="reproject before ingest"):
        read_raster_triple(wgs84, surface, surface)


def test_ingests_a_ward_with_no_generator_involved(committed_session, ward_inputs):
    from sqlalchemy import select
    from geocadastra.store.rasters import has_ward_rasters
    from geocadastra.store.schema import IngestedBlock, RecordedParcel

    blocks, parcels, (ortho, dsm, dtm) = ward_inputs
    job = ingest_ward(committed_session, source="upload:test", blocks=blocks, parcels=parcels,
                      ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)
    committed_session.commit()

    assert job.source == "upload:test"
    assert job.params == {}  # nothing to regenerate from, and nothing pretending otherwise
    assert has_ward_rasters(committed_session, job.id)
    stored = committed_session.scalars(
        select(RecordedParcel).where(RecordedParcel.ward_job_id == job.id)).all()
    assert len(stored) == 2
    assert {r.area for r in stored} == {800.0}
    block_row = committed_session.scalar(
        select(IngestedBlock).where(IngestedBlock.ward_job_id == job.id))
    assert block_row.local_block_id == 0


def test_two_uploads_do_not_collide_on_ids(committed_session, ward_inputs):
    from sqlalchemy import select
    from geocadastra.store.schema import RecordedParcel

    blocks, parcels, (ortho, dsm, dtm) = ward_inputs
    first = ingest_ward(committed_session, source="upload:a", blocks=blocks, parcels=parcels,
                        ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)
    second = ingest_ward(committed_session, source="upload:b", blocks=blocks, parcels=parcels,
                         ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)
    committed_session.commit()
    ids = committed_session.scalars(select(RecordedParcel.id).where(
        RecordedParcel.ward_job_id.in_([first.id, second.id]))).all()
    assert len(ids) == len(set(ids)) == 4


@pytest.mark.parametrize("mutate, message", [
    (lambda b, p: (b, [{**p[0], "seed_point": (100, 100)}, p[1]]), "outside its own block"),
    (lambda b, p: (b, [{**p[0], "area": -5.0}, p[1]]), "non-positive"),
    (lambda b, p: (b, [{**p[0], "block": 7}, p[1]]), "does not exist"),
    (lambda b, p: (b, [{k: v for k, v in p[0].items() if k != "area"}, p[1]]), "missing 'area'"),
    (lambda b, p: ([box(0, 0, 40, 40), box(20, 20, 60, 60)], p), "overlap"),
    (lambda b, p: (b, [{**p[0], "area": 5000.0}, p[1]]), "outside the"),
])
def test_unrecoverable_input_is_refused_at_the_boundary(committed_session, ward_inputs, mutate, message):
    blocks, parcels, (ortho, dsm, dtm) = ward_inputs
    blocks, parcels = mutate(blocks, parcels)
    with pytest.raises(ValueError, match=message):
        ingest_ward(committed_session, source="upload:bad", blocks=blocks, parcels=parcels,
                    ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)


def test_imagery_must_cover_every_block(committed_session, ward_inputs):
    blocks, parcels, (ortho, dsm, dtm) = ward_inputs
    far = box(500, 500, 540, 540)
    parcels = [{**p, "seed_point": (510 + 10 * i, 520)} for i, p in enumerate(parcels)]
    with pytest.raises(ValueError, match="does not cover"):
        ingest_ward(committed_session, source="upload:uncovered", blocks=[far], parcels=parcels,
                    ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)


def test_parcels_from_geojson_uses_an_interior_seed_point(tmp_path):
    """A centroid can fall outside an L-shaped parcel -- and therefore inside
    a neighbour, where it would anchor the capacity solver on the wrong land."""
    from shapely.geometry import Polygon

    ell = Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)])
    assert not ell.covers(ell.centroid), "test needs a shape whose centroid escapes it"
    path = tmp_path / "parcels.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(ell), "properties": {"area": 500.0, "style": "informal"}}]}))

    records = parcels_from_geojson(path, [box(0, 0, 40, 40)])
    assert len(records) == 1 and records[0]["style"] == "informal"
    assert ell.covers(__import__("shapely.geometry", fromlist=["Point"]).Point(*records[0]["seed_point"]))


def test_an_uploaded_ward_processes_without_ever_regenerating(committed_session, ward_inputs, tmp_path, monkeypatch):
    """The point of the whole stage: a ward with no seed reaches `done`.

    `_regenerate_ward` raises on a non-synthetic source, so if the worker
    still reached for the generator this would fail rather than pass quietly.
    """
    import torch
    from sqlalchemy import select
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.models.backbone import MultiTaskNet
    from geocadastra.store.schema import IngestedBlock, Provenance
    from geocadastra.tests.test_review_regressions import TEST_DB_URL, TEST_SCHEMA

    blocks, parcels, (ortho, dsm, dtm) = ward_inputs
    job = ingest_ward(committed_session, source="upload:test", blocks=blocks, parcels=parcels,
                      ortho_path=ortho, dsm_path=dsm, dtm_path=dtm)
    committed_session.commit()
    block_id = committed_session.scalar(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id))

    weights = tmp_path / "w.pt"
    torch.save({"model": MultiTaskNet().state_dict()}, weights)
    monkeypatch.setenv("GEOCADASTRA_MODEL_WEIGHTS", str(weights))
    jobs._load_model.cache_clear()
    try:
        assert jobs.process_block(TEST_DB_URL, TEST_SCHEMA, job.id, block_id,
                                  job.source, job.params) == "done"
        evidence = committed_session.scalars(select(Provenance).where(
            Provenance.evidence_type == "initial_load")).all()[0].payload["evidence"]
        assert evidence["evidence_source"] == "model"
        assert evidence["pixels"] == "stored"
    finally:
        jobs._load_model.cache_clear()
