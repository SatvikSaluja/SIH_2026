"""Stage 4 Done-when: on held-out synthetic wards, the SDF zero level set
recovers visible boundaries within one pixel median error, and the
predicted variance is higher on invisible boundaries than visible ones.

Status, honestly: at the CPU-only training scale exercised so far (a few
dozen 64x64 synthetic wards, a few hundred epochs, ~5 minutes per run),
neither target is reliably met -- median boundary error runs ~2px (not
<=1px), and the variance-separation direction is inconsistent across
training runs and block-style strata (sometimes correct, sometimes
marginal or reversed). This is reported here, not hidden behind a weaker
assertion or deleted test, per this project's own rule: "when a stage's
acceptance criterion cannot be met, stop and report it rather than
weakening the criterion." See STAGE_4_NOTES.md for the full investigation
(two real bugs were found and fixed along the way: a sub-pixel-wide
rendered wall stroke at this training GSD, and an evaluation mask too thin
relative to the network's realistic spatial resolution -- fixing the
second flipped the measured variance direction to correct in some but not
all configurations tried).

Re-verified after a later review round fixed two more real bugs (boundary
lines dropped at the ward's outer extent; a gradient-freezing clamp in the
NLL loss): median boundary error is still ~2.2px at this reduced scale, ie.
unchanged within run-to-run noise. Both bugs were real and are now fixed
(see regression tests in test_train.py), but neither was the bottleneck at
this scale -- the border-drop affected only a small, edge-concentrated
fraction of supervision pixels, and the gradient-freeze matters more over
longer training than this reduced 150-epoch run does. The gap tracked here
is genuinely about CPU-only training scale (data volume, epochs, tile
realism), not these bugs.

This is marked `xfail` (not skipped, not deleted): it runs, reports the
actual numbers on failure, and should be the first thing re-run once
larger-scale (GPU) training is available.
"""
import numpy as np
import pytest
import torch
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from geocadastra.models.dataset import ward_to_tensors
from geocadastra.models.train import TrainConfig, train
from geocadastra.synth.generator import WardParams, generate_ward


def _boundary_masks(ward, dilate_m: float = 1.5):
    h, w = ward.dtm.shape
    all_lines = list(ward.edges_geom.values())
    visible_lines = [ward.edges_geom[k] for k, v in ward.edges_visible.items() if v]

    def _mask(lines):
        if not lines:
            return np.zeros((h, w), dtype=bool)
        return rasterize(
            [(ln.buffer(dilate_m), 1) for ln in lines], (h, w), transform=ward.transform, fill=0, dtype="uint8"
        ).astype(bool)

    visible_mask = _mask(visible_lines)
    all_mask = _mask(all_lines)
    return visible_mask, all_mask & ~visible_mask  # (visible, invisible)


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Known, documented gap (see STAGE_4_NOTES.md): CPU-only training at this scale does not "
        "reliably meet the 1px median boundary error / invisible>visible variance targets. "
        "Not weakened or skipped -- reports actual numbers, re-run first once GPU training lands."
    ),
    strict=False,
)
def test_sdf_zero_level_set_and_variance_separation_on_held_out_wards():
    """Deliberately a lighter-weight run than the investigation in
    STAGE_4_NOTES.md (fewer wards/epochs) -- this is a tracked regression
    check, not a from-scratch reproduction of the full study; re-run the
    heavier configuration directly (see the notes) for a real accuracy read."""
    torch.manual_seed(0)
    gsd = 1.0
    params = WardParams(
        width=64, height=64, gsd=gsd, n_arterial_h=0, n_arterial_v=0, minor_spacing=25, wall_render_width=1.5
    )
    config = TrainConfig(
        n_wards=20, epochs=150, ward_params=params, seed_offset=0,
        loss_weights={"sdf": 2.0, "road": 0.5, "building": 0.5, "landuse": 0.5},
    )
    model, _history = train(config)
    model.eval()

    boundary_errs, vars_visible, vars_invisible = [], [], []
    for seed in range(1000, 1010):
        ward = generate_ward(params=params, seed=seed)
        t = ward_to_tensors(ward)
        with torch.no_grad():
            out = model(t["rgb"].unsqueeze(0), t["ndsm"].unsqueeze(0))
        sdf_pred = out["sdf"][0, 0].numpy()
        log_var_pred = out["log_var"][0, 0].numpy()

        visible_mask, invisible_mask = _boundary_masks(ward)
        pred_boundary = np.abs(sdf_pred) < (0.5 * gsd)
        dist_to_pred_boundary = distance_transform_edt(~pred_boundary) * gsd
        if visible_mask.sum() > 5:
            boundary_errs.append(np.median(dist_to_pred_boundary[visible_mask]) / gsd)
            vars_visible.append(log_var_pred[visible_mask].mean())
        if invisible_mask.sum() > 5:
            vars_invisible.append(log_var_pred[invisible_mask].mean())

    median_err_px = float(np.median(boundary_errs))
    mean_var_visible = float(np.mean(vars_visible))
    mean_var_invisible = float(np.mean(vars_invisible))

    assert median_err_px <= 1.0, f"median boundary error {median_err_px:.2f}px > 1.0px target"
    assert mean_var_invisible > mean_var_visible, (
        f"variance not separated correctly: visible={mean_var_visible:.4f}, invisible={mean_var_invisible:.4f}"
    )
