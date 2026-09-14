"""The model must actually participate in block processing.

Stages 0-8 were built with a simulated boundary-evidence field, so every
green acceptance test proved the geometry engine worked without proving the
network contributed anything. These tests run the real `MultiTaskNet` over a
real rendered ward and assert the worker consumes its output.
"""
import numpy as np
import pytest
import torch

from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.infer import block_evidence_from_model, sdf_to_evidence
from geocadastra.synth.generator import WardParams, generate_ward, simulate_evidence_field


@pytest.fixture(scope="module")
def ward():
    return generate_ward(params=WardParams(width=100, height=60, n_arterial_h=1), seed=1)


def test_model_evidence_matches_the_simulated_field_contract(ward):
    """Same (array, transform) shape contract Stage 3 already consumes."""
    model = MultiTaskNet()
    field, transform = block_evidence_from_model(model, ward, ward.blocks[0].id, tile_size=32, overlap=8)
    simulated, _ = simulate_evidence_field(ward, ward.blocks[0].id)
    assert field.ndim == simulated.ndim == 2
    assert np.isfinite(field).all()
    assert 0.0 <= field.min() and field.max() <= 1.0  # evidence is a decayed distance, never a raw SDF
    assert transform.a > 0 and transform.e < 0  # north-up, same convention as the ward raster


def test_evidence_window_covers_the_block_it_claims_to(ward):
    """A transform disagreeing with its array puts parcels in the wrong place."""
    block = ward.blocks[0]
    minx, miny, maxx, maxy = block.polygon.bounds
    field, transform = block_evidence_from_model(MultiTaskNet(), ward, block.id, tile_size=32, overlap=8)
    height, width = field.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    gsd = abs(transform.a)
    assert left <= minx + gsd and top >= maxy - gsd
    assert right >= maxx - gsd and bottom <= miny + gsd


def test_a_trained_boundary_response_changes_the_evidence_field(ward):
    """Not a tautology: two different weight sets must give different evidence.

    An untrained net still returns a field, so shape assertions alone cannot
    tell a wired model from a discarded one.
    """
    block_id = ward.blocks[0].id
    torch.manual_seed(0)
    first, _ = block_evidence_from_model(MultiTaskNet(), ward, block_id, tile_size=32, overlap=8)
    torch.manual_seed(7)
    second, _ = block_evidence_from_model(MultiTaskNet(), ward, block_id, tile_size=32, overlap=8)
    assert not np.allclose(first, second, rtol=0, atol=1e-4)


def test_sdf_to_evidence_peaks_on_the_predicted_boundary():
    evidence = sdf_to_evidence(np.array([[0.0, 1.5, 10.0]], dtype=np.float32))
    assert evidence[0, 0] == pytest.approx(1.0)
    assert evidence[0, 1] > evidence[0, 2]


def _block_inputs(session, job, block_id):
    """The arguments `_block_evidence` takes, from the store."""
    from geoalchemy2.shape import to_shape
    from geocadastra.core.crs import Geom
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.store.schema import IngestedBlock

    row = session.get(IngestedBlock, (job.id, block_id))
    return (jobs._regenerate_ward(job.source, job.params),
            Geom(to_shape(row.geom), job.crs), row.local_block_id)


def test_worker_defaults_to_simulation_and_records_which_source_ran(committed_session, monkeypatch):
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.tests.test_review_regressions import ingest

    monkeypatch.delenv("GEOCADASTRA_MODEL_WEIGHTS", raising=False)
    job, block_id, _ = ingest(committed_session)
    regenerated, block_geom, local_id = _block_inputs(committed_session, job, block_id)
    field, _, provenance = jobs._block_evidence(committed_session, job.id, block_geom, lambda: regenerated, local_id)
    assert provenance == {"evidence_source": "simulated"}
    expected, _ = simulate_evidence_field(regenerated, local_id)
    assert np.array_equal(field, expected)


def test_worker_uses_configured_weights_and_stamps_their_hash(committed_session, monkeypatch, tmp_path):
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.tests.test_review_regressions import ingest

    job, block_id, _ = ingest(committed_session)
    committed_session.commit()
    weights = tmp_path / "w.pt"
    torch.save({"model": MultiTaskNet().state_dict()}, weights)
    monkeypatch.setenv("GEOCADASTRA_MODEL_WEIGHTS", str(weights))
    jobs._load_model.cache_clear()
    try:
        regenerated, block_geom, local_id = _block_inputs(committed_session, job, block_id)
        field, _, provenance = jobs._block_evidence(
            committed_session, job.id, block_geom, lambda: regenerated, local_id)
        assert provenance["evidence_source"] == "model"
        assert len(provenance["weights_sha256"]) == 16
        # ingest stored the rasters, so the model must read them rather than
        # depend on the ward being regenerable from its seed
        assert provenance["pixels"] == "stored"
        simulated, _ = simulate_evidence_field(regenerated, local_id)
        assert field.shape != simulated.shape or not np.allclose(field, simulated)
    finally:
        jobs._load_model.cache_clear()


def test_loader_accepts_a_checkpoint_train_py_actually_wrote(tmp_path, ward):
    """Regression: the loader first looked for "model_state"/"model_kwargs",
    which `train.py` never writes. A hand-rolled checkpoint in a test hid it;
    only a real one from the training entry point catches a key mismatch."""
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.models.train import TrainConfig, train

    # TrainConfig's default ward size, not a smaller one: the backbone
    # downsamples a 32x32 raster to 1x1, and BatchNorm cannot train on that
    # ("Expected more than 1 value per channel when training").
    config = TrainConfig(n_wards=1, epochs=1, device="cpu")
    checkpoint = tmp_path / "ckpt.pt"
    train(config, checkpoint_path=checkpoint)
    assert checkpoint.exists(), "training did not write the checkpoint this test loads"

    jobs._load_model.cache_clear()
    try:
        model, digest = jobs._load_model(str(checkpoint))
        assert len(digest) == 16
        field, _ = block_evidence_from_model(model, ward, ward.blocks[0].id, tile_size=32, overlap=8)
        assert np.isfinite(field).all()
    finally:
        jobs._load_model.cache_clear()


def test_worker_processes_a_block_end_to_end_on_model_evidence(committed_session, monkeypatch, tmp_path):
    """The claim this whole file exists to defend: a block reaches `done`
    with its geometry derived from a network's prediction, and the stored
    provenance names the weights that produced it."""
    from sqlalchemy import select
    from geocadastra.jobs import orchestrator as jobs
    from geocadastra.store.schema import Provenance
    from geocadastra.tests.test_review_regressions import ingest, process

    job, block_id, _ = ingest(committed_session)
    weights = tmp_path / "w.pt"
    torch.save({"model_state": MultiTaskNet().state_dict()}, weights)
    monkeypatch.setenv("GEOCADASTRA_MODEL_WEIGHTS", str(weights))
    jobs._load_model.cache_clear()
    try:
        assert process(job, block_id) == "done"
        evidence = committed_session.scalars(
            select(Provenance).where(Provenance.evidence_type == "initial_load")).all()[0].payload["evidence"]
        assert evidence["evidence_source"] == "model"
        assert evidence["weights_sha256"]
    finally:
        jobs._load_model.cache_clear()
