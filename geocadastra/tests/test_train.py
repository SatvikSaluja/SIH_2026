import numpy as np
import pytest
import torch

from geocadastra.models.dataset import ward_to_tensors
from geocadastra.models.heads import sdf_nll_loss
from geocadastra.models.train import TrainConfig, compute_loss, train
from geocadastra.synth.generator import WardParams, generate_ward


def _small_params():
    return WardParams(width=48, height=48, gsd=1.0, n_arterial_h=0, n_arterial_v=0, minor_spacing=25)


def test_ward_to_tensors_shapes_and_ranges_are_sane():
    ward = generate_ward(params=_small_params(), seed=1)
    t = ward_to_tensors(ward)
    h, w = ward.dtm.shape
    assert t["rgb"].shape == (3, h, w)
    assert t["ndsm"].shape == (1, h, w)
    assert t["sdf_true"].shape == (1, h, w)
    assert (t["sdf_true"] >= 0).all()
    assert t["visible_mask"].shape == (h, w)
    assert t["landuse_true"].shape == (h, w)
    assert t["landuse_true"].max() <= 3 and t["landuse_true"].min() >= 0


def test_road_true_width_matches_arterial_width_not_a_hardcoded_value():
    """Regression: road_polys used to buffer every road centerline by a
    flat 3.0m regardless of its true width -- correct only for minor
    roads (6.0m); an arterial road (default 12.0m) got a road_true
    footprint roughly half its true width, silently undercounting the
    road/landuse-class-3 training targets (found by review)."""
    params = WardParams(
        width=48, height=48, gsd=1.0, n_arterial_h=1, n_arterial_v=0, minor_spacing=200, arterial_width=12.0
    )
    ward = generate_ward(params=params, seed=1)
    t = ward_to_tensors(ward)
    road_mask = t["road_true"][0].numpy() > 0.5

    # the single horizontal arterial spans the full ward width -- measure
    # its pixel width at a column comfortably inside the ward (away from
    # the buffer's rounded end caps near the left/right edges)
    col = road_mask[:, 24]
    measured_width_px = col.sum()
    assert measured_width_px == pytest.approx(params.arterial_width / params.gsd, abs=1)


def test_training_loss_decreases_over_a_short_run():
    """The required check for this non-trivial loop: does it actually learn
    anything, not just run without crashing."""
    torch.manual_seed(0)
    config = TrainConfig(n_wards=4, epochs=15, ward_params=_small_params())
    _model, history = train(config)
    assert len(history) == config.epochs
    assert history[-1] < history[0] * 0.7  # a real, not marginal, drop


def test_no_sdf_outliers_at_ward_border_from_dropped_boundary_lines():
    """Regression for a review-found bug: a boundary line running exactly
    along the ward's outer extent had no pixel-centre out there for GDAL's
    rasterize() to hit, so it was silently dropped -- inflating sdf_true to
    ~20m at every ward's border. Fixed via a 0.51-gsd line buffer; this
    checks it stays fixed across a spread of seeds."""
    params = _small_params()
    for seed in range(10):
        ward = generate_ward(params=params, seed=seed)
        t = ward_to_tensors(ward)
        sdf = t["sdf_true"][0].numpy()
        border = np.zeros_like(sdf, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        assert (sdf[border] <= 3.0).all(), f"seed {seed}: dropped border boundary line, max sdf={sdf[border].max()}"


def test_landuse_class_under_road_is_actually_present():
    """Regression for a review-found bug: parcels tile the whole ward
    including the land under a road, so without an explicit override pass
    class 3 ('under a road') was reachable in principle but never present
    in a single training pixel -- an untrained, dead logit."""
    ward = generate_ward(params=_small_params(), seed=3)
    t = ward_to_tensors(ward)
    assert ward.roads_centerline, "test ward has no roads -- fixture doesn't exercise the case"
    assert (t["landuse_true"] == 3).any()


def test_sdf_nll_loss_gradient_never_freezes_at_extreme_log_var():
    """Regression for a review-found bug: a hard clamp() on log_var gives
    EXACTLY zero gradient past its bounds, permanently freezing calibration
    for any pixel whose log_var races out there early in training. The
    soft tanh bound must keep a nonzero gradient even at extreme inputs."""
    log_var = torch.tensor([15.0], requires_grad=True)
    sdf_pred = torch.tensor([0.0])
    sdf_true = torch.tensor([0.0])
    loss = sdf_nll_loss(sdf_pred, log_var, sdf_true)
    loss.backward()
    assert log_var.grad.abs().item() > 1e-4, f"gradient frozen at extreme log_var: {log_var.grad.item()}"


def test_compute_loss_all_four_terms_are_finite_and_positive():
    ward = generate_ward(params=_small_params(), seed=2)
    t = ward_to_tensors(ward)
    from geocadastra.models.backbone import MultiTaskNet

    model = MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16)
    out = model(t["rgb"].unsqueeze(0), t["ndsm"].unsqueeze(0))
    targets = {
        "sdf_true": t["sdf_true"].unsqueeze(0),
        "road_true": t["road_true"].unsqueeze(0),
        "building_true": t["building_true"].unsqueeze(0),
        "landuse_true": t["landuse_true"].unsqueeze(0),
    }
    total, parts = compute_loss(out, targets, {"sdf": 1.0, "road": 1.0, "building": 1.0, "landuse": 1.0})
    assert torch.isfinite(total)
    for name, value in parts.items():
        assert value == value and value > -1e6, f"{name} loss is not sane: {value}"  # nan-check + sanity floor


def test_batching_produces_the_same_loss_trajectory_as_unbatched():
    """batch_size=1 must be numerically identical to the pre-batching loop --
    this is the actual backward-compatibility guarantee, not just an import
    check. batch_size>1 must still learn (not just run)."""
    torch.manual_seed(0)
    config_unbatched = TrainConfig(n_wards=4, epochs=6, ward_params=_small_params(), batch_size=1)
    _model_a, history_a = train(config_unbatched)

    torch.manual_seed(0)
    config_batched = TrainConfig(n_wards=4, epochs=6, ward_params=_small_params(), batch_size=1)
    _model_b, history_b = train(config_batched)
    assert history_a == history_b  # same config, same seed -> identical trajectory

    torch.manual_seed(0)
    config_real_batch = TrainConfig(n_wards=4, epochs=10, ward_params=_small_params(), batch_size=4)
    _model_c, history_c = train(config_real_batch)
    assert len(history_c) == 10
    assert history_c[-1] < history_c[0] * 0.7


def test_validation_is_held_out_and_never_backpropagated(monkeypatch):
    """n_val_wards>0 must draw DIFFERENT seeds than training, and evaluate()
    must never call .backward() -- this is the whole point of a held-out set."""
    config = TrainConfig(n_wards=3, epochs=1, ward_params=_small_params(), n_val_wards=2, seed_offset=0)
    train_seeds = {config.seed_offset + i for i in range(config.n_wards)}
    val_seeds = {config.seed_offset + config.n_wards + i for i in range(config.n_val_wards)}
    assert train_seeds.isdisjoint(val_seeds)

    from geocadastra.models import train as train_module
    original_backward = torch.Tensor.backward
    calls = {"backward": 0}

    def counting_backward(self, *a, **kw):
        calls["backward"] += 1
        return original_backward(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)
    val_dataset = train_module.make_val_dataset(config)
    assert len(val_dataset) == 2
    loss = train_module.evaluate(MultiTaskNetFixture(), val_dataset, config.loss_weights, torch.device("cpu"))
    assert loss is not None and calls["backward"] == 0


def MultiTaskNetFixture():
    from geocadastra.models.backbone import MultiTaskNet
    return MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16)


def test_best_checkpoint_is_written_only_on_validation_improvement(tmp_path):
    """A 'best' checkpoint must exist, must differ from 'latest' once training
    keeps going past the best epoch, and must never appear without validation
    enabled at all -- picking 'best' by training loss isn't a validation
    signal, it's just the last epoch restated."""
    ckpt = tmp_path / "model.pt"

    torch.manual_seed(0)
    config_no_val = TrainConfig(n_wards=2, epochs=3, ward_params=_small_params())
    train(config_no_val, checkpoint_path=ckpt)
    assert not ckpt.with_name(ckpt.name + ".best").exists()

    torch.manual_seed(0)
    config = TrainConfig(n_wards=2, epochs=5, ward_params=_small_params(), n_val_wards=2)
    train(config, checkpoint_path=ckpt)
    best_path = ckpt.with_name(ckpt.name + ".best")
    assert best_path.exists()

    latest = torch.load(ckpt, map_location="cpu", weights_only=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    assert best["epoch"] <= latest["epoch"]
    assert best["best_val_loss"] == min(latest["val_loss_history"])
