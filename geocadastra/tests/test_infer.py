import numpy as np
import torch

from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.infer import run_tiled_inference, sdf_to_evidence


def test_sdf_to_evidence_is_high_near_zero_and_decays_with_distance():
    sdf = np.array([0.0, 0.5, 1.5, 5.0, 20.0], dtype=np.float32)
    evidence = sdf_to_evidence(sdf)
    assert evidence[0] == 1.0
    assert (np.diff(evidence) < 0).all()  # strictly decreasing as |sdf| grows
    assert 0.0 <= evidence.min() and evidence.max() <= 1.0


def test_tiled_inference_matches_output_shape_and_produces_all_fields():
    torch.manual_seed(0)
    model = MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16)
    rgb = np.random.default_rng(0).uniform(0, 1, size=(3, 80, 100)).astype(np.float32)
    ndsm = np.random.default_rng(1).uniform(0, 5, size=(1, 80, 100)).astype(np.float32)

    result = run_tiled_inference(model, rgb, ndsm, tile_size=48, overlap=16, n_landuse_classes=4)
    for key in ("sdf", "log_var", "road", "building", "evidence"):
        assert result[key].shape == (80, 100)
    assert result["landuse"].shape == (4, 80, 100)
    # road/building/landuse are probabilities -- must be in [0,1] and landuse must sum to ~1 per pixel
    assert (result["road"] >= 0).all() and (result["road"] <= 1).all()
    assert (result["building"] >= 0).all() and (result["building"] <= 1).all()
    assert np.allclose(result["landuse"].sum(axis=0), 1.0, atol=1e-4)
    assert (result["evidence"] >= 0).all() and (result["evidence"] <= 1).all()


def test_tiled_inference_handles_an_image_smaller_than_one_tile():
    torch.manual_seed(0)
    model = MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16)
    rgb = np.random.default_rng(0).uniform(0, 1, size=(3, 30, 40)).astype(np.float32)
    ndsm = np.random.default_rng(1).uniform(0, 5, size=(1, 30, 40)).astype(np.float32)
    result = run_tiled_inference(model, rgb, ndsm, tile_size=64, overlap=16)
    assert result["sdf"].shape == (30, 40)


def test_more_overlap_reduces_the_seam_discontinuity_between_tiles():
    """The whole point of blending with a tapered window instead of an
    unweighted average of non-overlapping tiles: more overlap should smooth
    out the seam, not leave it just as sharp."""
    torch.manual_seed(0)
    model = MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16)
    rgb = np.random.default_rng(0).uniform(0, 1, size=(3, 96, 96)).astype(np.float32)
    ndsm = np.random.default_rng(1).uniform(0, 5, size=(1, 96, 96)).astype(np.float32)

    def seam_roughness(overlap):
        result = run_tiled_inference(model, rgb, ndsm, tile_size=48, overlap=overlap)
        sdf = result["sdf"]
        # second derivative magnitude near the expected tile boundary (x=48-ish) vs interior
        col = 48
        return float(np.abs(np.diff(sdf[:, col - 2 : col + 3], axis=1, n=2)).mean())

    rough_no_overlap = seam_roughness(0)
    rough_with_overlap = seam_roughness(24)
    assert rough_with_overlap <= rough_no_overlap
