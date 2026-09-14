"""Ward imagery must be durable, not regenerated from a seed.

`process_block` reconstructed its rasters by re-running the synthetic
generator. Real imagery has no seed to regenerate from, so these tests pin
the stored-raster path: what is written, what comes back, and that reading a
block's window off disk agrees with cropping it from an in-memory ward.
"""
import numpy as np
import pytest
from affine import Affine

from geocadastra.core.crs import CRSMismatchError
from geocadastra.store.rasters import (has_ward_rasters, read_block_window,
                                       store_ward_rasters)
from geocadastra.store.schema import WardRaster
from geocadastra.synth.generator import WardParams, generate_ward

CRS = "EPSG:32643"


@pytest.fixture
def raster_root(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOCADASTRA_RASTER_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def ward():
    return generate_ward(params=WardParams(width=100, height=60, n_arterial_h=1), seed=1)


def _ward_job(session):
    from geocadastra.store.schema import WardJob
    job = WardJob(source="test", params={}, status="pending", crs=CRS)
    session.add(job)
    session.flush()
    return job


def test_round_trip_preserves_pixels(committed_session, raster_root, ward):
    job = _ward_job(committed_session)
    store_ward_rasters(committed_session, job.id, ortho=ward.ortho, dsm=ward.dsm,
                       dtm=ward.dtm, transform=ward.transform, crs=ward.crs)
    committed_session.commit()
    assert has_ward_rasters(committed_session, job.id)

    minx, miny, maxx, maxy = ward.ward_polygon.bounds
    rgb, ndsm, transform = read_block_window(committed_session, job.id, (minx, miny, maxx, maxy))
    assert rgb.shape[0] == 3 and rgb.dtype == np.float32
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert rgb.shape[1:] == ndsm.shape
    # ndsm is DSM - DTM, so a rendered building must stand above zero somewhere
    assert ndsm.max() > 0.0


def test_stored_window_matches_an_in_memory_crop(committed_session, raster_root, ward):
    """The refactor must not have moved any pixels: reading a block's window
    off disk has to agree with cropping the same block out of the ward."""
    from geocadastra.models.infer import crop_ward_block

    job = _ward_job(committed_session)
    store_ward_rasters(committed_session, job.id, ortho=ward.ortho, dsm=ward.dsm,
                       dtm=ward.dtm, transform=ward.transform, crs=ward.crs)
    committed_session.commit()

    block = ward.blocks[0]
    memory_rgb, memory_ndsm, memory_transform = crop_ward_block(ward, block.id)
    disk_rgb, disk_ndsm, disk_transform = read_block_window(
        committed_session, job.id, block.polygon.bounds)

    assert disk_rgb.shape == memory_rgb.shape, "window rounding disagrees between the two paths"
    assert tuple(disk_transform)[:6] == pytest.approx(tuple(memory_transform)[:6])
    np.testing.assert_allclose(disk_rgb, memory_rgb, rtol=0, atol=1 / 255)
    np.testing.assert_allclose(disk_ndsm, memory_ndsm, rtol=0, atol=1e-5)


def test_window_is_clamped_and_never_empty(committed_session, raster_root, ward):
    job = _ward_job(committed_session)
    store_ward_rasters(committed_session, job.id, ortho=ward.ortho, dsm=ward.dsm,
                       dtm=ward.dtm, transform=ward.transform, crs=ward.crs)
    committed_session.commit()
    # entirely outside the raster, and a degenerate zero-area request
    for bounds in [(-500, -500, -400, -400), (1.0, 1.0, 1.0, 1.0)]:
        rgb, ndsm, _ = read_block_window(committed_session, job.id, bounds)
        assert rgb.shape[1] >= 1 and rgb.shape[2] >= 1
        assert ndsm.size >= 1


def test_mismatched_grids_are_refused(committed_session, raster_root, ward):
    job = _ward_job(committed_session)
    with pytest.raises(ValueError, match="share one grid"):
        store_ward_rasters(committed_session, job.id, ortho=ward.ortho,
                           dsm=ward.dsm[:-1], dtm=ward.dtm,
                           transform=ward.transform, crs=ward.crs)


def test_foreign_crs_is_refused(committed_session, raster_root, ward):
    job = _ward_job(committed_session)
    with pytest.raises(CRSMismatchError):
        store_ward_rasters(committed_session, job.id, ortho=ward.ortho, dsm=ward.dsm,
                           dtm=ward.dtm, transform=ward.transform, crs="EPSG:4326")


def test_missing_rasters_report_rather_than_guess(committed_session, raster_root):
    job = _ward_job(committed_session)
    committed_session.commit()
    assert not has_ward_rasters(committed_session, job.id)
    with pytest.raises(LookupError, match="no stored"):
        read_block_window(committed_session, job.id, (0, 0, 10, 10))


def test_ingest_stores_the_wards_rasters(committed_session, raster_root):
    from sqlalchemy import select
    from geocadastra.tests.test_review_regressions import ingest

    job, _, _ = ingest(committed_session)
    rows = {r.kind: r for r in committed_session.scalars(
        select(WardRaster).where(WardRaster.ward_job_id == job.id))}
    assert set(rows) == {"ortho", "dsm", "dtm"}
    for row in rows.values():
        assert row.crs == CRS
        assert row.width > 0 and row.height > 0
        assert len(row.transform) == 6
        assert Affine(*row.transform).a > 0


def test_worker_reads_stored_pixels_rather_than_regenerating(committed_session, raster_root, tmp_path, monkeypatch):
    """The point of storing them: the model path must not need the seed."""
    import torch
    from sqlalchemy import select
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.models.backbone import MultiTaskNet
    from geocadastra.store.schema import Provenance
    from geocadastra.tests.test_review_regressions import ingest, process

    job, block_id, _ = ingest(committed_session)
    weights = tmp_path / "w.pt"
    torch.save({"model": MultiTaskNet().state_dict()}, weights)
    monkeypatch.setenv("GEOCADASTRA_MODEL_WEIGHTS", str(weights))
    jobs._load_model.cache_clear()
    try:
        assert process(job, block_id) == "done"
        evidence = committed_session.scalars(
            select(Provenance).where(Provenance.evidence_type == "initial_load")).all()[0].payload["evidence"]
        assert evidence["evidence_source"] == "model"
        assert evidence["pixels"] == "stored"
    finally:
        jobs._load_model.cache_clear()
