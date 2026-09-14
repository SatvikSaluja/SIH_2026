"""Focused checks for the real-imagery masking math -- the part that decides
which pixels are trustworthy supervision, not just which files exist."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

spec = importlib.util.spec_from_file_location('prepare_real', Path(__file__).parents[1] / 'prepare_real.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_distance_is_zero_on_the_boundary_and_grows_away_from_it():
    boundary = np.zeros((10, 10), dtype=bool)
    boundary[5, :] = True  # one horizontal boundary line at row 5
    valid = np.ones_like(boundary)
    ndsm = np.zeros_like(boundary, dtype=float)
    distance, mask = module.compute_distance_and_mask(boundary, valid, ndsm, gsd_m=0.5)
    assert distance[5, 3] == 0.0
    assert distance[0, 3] == 5 * 0.5  # 5 pixels away, 0.5 m/px
    assert distance[9, 3] == 4 * 0.5


def test_pixels_near_unknown_coverage_are_excluded_when_a_closer_boundary_could_be_hiding():
    """A pixel far from any LABELED boundary but right next to unmapped
    territory must be masked out -- the true nearest boundary could be just
    past the edge of what's known, closer than the labelled one."""
    boundary = np.zeros((20, 20), dtype=bool)
    boundary[0, 0] = True  # one labelled boundary, far corner
    valid = np.ones_like(boundary)
    valid[:, 15:] = False  # unmapped strip starts at column 15
    ndsm = np.zeros_like(boundary, dtype=float)
    distance, mask = module.compute_distance_and_mask(boundary, valid, ndsm, gsd_m=1.0)

    # Right at the edge of known coverage (col 14): the unmapped region is
    # essentially adjacent, almost certainly closer than the labelled corner
    # boundary far away -- must be excluded.
    assert not mask[10, 14]
    # Deep in known territory, close to the labelled boundary, far from the
    # unmapped strip: trustworthy, must be included.
    assert mask[1, 1]


def test_nonfinite_ndsm_pixels_are_never_valid_even_inside_mapped_coverage():
    boundary = np.zeros((10, 10), dtype=bool)
    boundary[0, 0] = True
    valid = np.ones_like(boundary)
    ndsm = np.zeros((10, 10))
    ndsm[3, 3] = np.nan
    _distance, mask = module.compute_distance_and_mask(boundary, valid, ndsm, gsd_m=1.0)
    assert not mask[3, 3]


def test_matches_the_original_unrefactored_formula_directly():
    """The refactor must be behavior-preserving -- re-derive the exact
    pre-refactor expression independently and compare."""
    rng = np.random.default_rng(0)
    boundary = rng.random((30, 25)) > 0.9
    valid = rng.random((30, 25)) > 0.15
    ndsm = rng.normal(size=(30, 25))
    ndsm[rng.random((30, 25)) > 0.97] = np.nan

    distance, mask = module.compute_distance_and_mask(boundary, valid, ndsm, gsd_m=0.3)

    expected_distance = distance_transform_edt(~boundary.astype(bool)) * .3
    known = valid.astype(bool) & np.isfinite(ndsm)
    expected_unknown_distance = distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1] * .3
    expected_mask = known & (expected_unknown_distance > expected_distance)

    np.testing.assert_array_equal(distance, expected_distance)
    np.testing.assert_array_equal(mask, expected_mask)
