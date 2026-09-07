"""Stage 6: conformal calibration -- certified accuracy bands for fused
boundary predictions, calibrated against held-out GT survey points.

Nonconformity score = |true - predicted| / predicted sigma (a standardized
residual): this is what makes stratifying by settlement type meaningful --
comparing RAW displacement across formal/informal blocks would just
reflect fusion's own uncertainty being systematically larger for informal
blocks, not miscalibration. Mondrian conformal (a separate quantile per
stratum) is what makes coverage hold WITHIN each stratum, per the doc:
"coverage must hold within informal settlements specifically, not just on
average."
"""
import math

import numpy as np
import pytest

from geocadastra.core.conformal import (
    CalibratedBands,
    calibrate,
    calibration_examples,
    certified_band,
    empirical_coverage,
    nonconformity_score,
)
from geocadastra.synth.generator import GTPoint


def test_nonconformity_score_is_displacement_over_sigma():
    # true at (0,0), predicted at (3,4) -- displacement 5.0
    score = nonconformity_score((0.0, 0.0), (3.0, 4.0), sigma=2.0)
    assert score == pytest.approx(2.5)


def test_nonconformity_score_exact_match_is_zero():
    assert nonconformity_score((1.0, 1.0), (1.0, 1.0), sigma=1.0) == 0.0


def test_calibrate_picks_the_standard_conformal_quantile():
    """The textbook example: 9 calibration scores [1..9], alpha=0.1 (90%
    nominal coverage) -> the ceil((n+1)*(1-alpha))-th order statistic =
    ceil(10*0.9) = 9th smallest = 9."""
    points = [("formal", float(x)) for x in range(1, 10)]
    bands = calibrate(points, alpha=0.1)
    assert bands.quantile_by_stratum["formal"] == pytest.approx(9.0)
    assert bands.alpha == 0.1


def test_calibrate_is_mondrian_separate_quantile_per_stratum():
    """Coverage must hold WITHIN each stratum -- informal's own (noisier)
    scores must not be diluted by formal's (tighter) ones, or vice versa."""
    points = [("formal", 1.0), ("formal", 2.0)] + [("informal", 10.0), ("informal", 20.0)]
    bands = calibrate(points, alpha=0.5)  # n=2 per stratum, alpha=0.5 -> k=ceil(3*0.5)=2 -> max
    assert bands.quantile_by_stratum["formal"] == pytest.approx(2.0)
    assert bands.quantile_by_stratum["informal"] == pytest.approx(20.0)


def test_calibrate_with_a_single_calibration_point_uses_it_as_the_quantile():
    """Degenerate but sane: too little data to do anything but be maximally
    conservative (use the one point available)."""
    bands = calibrate([("formal", 7.0)], alpha=0.1)
    assert bands.quantile_by_stratum["formal"] == pytest.approx(7.0)


def test_calibrate_rejects_an_empty_calibration_set():
    with pytest.raises(ValueError):
        calibrate([], alpha=0.1)


def test_calibrate_rejects_an_empty_calibration_generator():
    """`if not calibration_points` alone is a no-op for a generator (always
    truthy, no __len__/__bool__) -- must actually materialize it first."""
    with pytest.raises(ValueError):
        calibrate(iter(()), alpha=0.1)


def test_calibrate_picks_the_quantile_in_the_non_clamped_regime():
    """Every other quantile test lands exactly on k==n (the clamp) or n==1
    (trivial). n=19, alpha=0.1 -> k=ceil(20*0.9)=18, strictly between 1 and
    n=19 -- exercises the real index math, not just its boundary."""
    points = [("formal", float(x)) for x in range(1, 20)]  # scores 1..19
    bands = calibrate(points, alpha=0.1)
    assert bands.quantile_by_stratum["formal"] == pytest.approx(18.0)


@pytest.mark.parametrize("alpha", [0.01, 0.99])
def test_calibrate_handles_alpha_near_the_domain_edges(alpha):
    points = [("formal", float(x)) for x in range(1, 101)]  # scores 1..100
    bands = calibrate(points, alpha=alpha)
    k = min(math.ceil(101 * (1 - alpha)), 100)
    assert bands.quantile_by_stratum["formal"] == pytest.approx(float(k))


@pytest.mark.parametrize("alpha", [1.0, 1.5, -0.1])
def test_calibrate_rejects_an_out_of_domain_alpha(alpha):
    """alpha>=1 previously wrapped scores[k-1] via silent negative
    indexing instead of raising -- e.g. alpha=1.5 on 9 calibration points
    used to silently return scores[-6] as if it were a real quantile."""
    with pytest.raises(ValueError):
        calibrate([("formal", float(x)) for x in range(1, 10)], alpha=alpha)


def test_certified_band_scales_the_quantile_by_this_predictions_own_sigma():
    bands = CalibratedBands(quantile_by_stratum={"formal": 2.0}, alpha=0.1)
    assert certified_band(bands, "formal", sigma=3.0) == pytest.approx(6.0)


def test_certified_band_raises_for_an_unseen_stratum():
    """Nothing is silently resolved: a stratum with no calibration data at
    all must not silently fall back to some other stratum's band."""
    bands = CalibratedBands(quantile_by_stratum={"formal": 2.0}, alpha=0.1)
    with pytest.raises(KeyError):
        certified_band(bands, "informal", sigma=1.0)


def test_empirical_coverage_at_the_calibration_quantile_covers_by_construction():
    """The defining conformal-prediction guarantee, directly exercised:
    calibrate on one batch of exchangeable scores, evaluate coverage on a
    held-out batch drawn from the SAME distribution -- coverage should
    land close to the nominal (1-alpha), not exactly (finite-sample), but
    within a wide, well-understood tolerance for this sample size."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    # true=0, predicted=N(0,1)*sigma, sigma=1 for every point -- so scores
    # are just |N(0,1)|, a known distribution to calibrate against
    all_scores = np.abs(rng.normal(0, 1, size=2000))
    cal, test = all_scores[:1000], all_scores[1000:]
    bands = calibrate([("formal", s) for s in cal], alpha=alpha)
    test_points = [("formal", (0.0, 0.0), (s, 0.0), 1.0) for s in test]
    coverage = empirical_coverage(test_points, bands)
    assert coverage["formal"] == pytest.approx(1 - alpha, abs=0.05)


def test_calibration_examples_pairs_gt_truth_with_a_callers_prediction():
    gt_points = [GTPoint(x=1.0, y=2.0, parcel_id=10), GTPoint(x=5.0, y=5.0, parcel_id=20)]
    parcel_style_by_id = {10: "formal", 20: "informal"}

    def predict_fn(gt):
        return (gt.x + 1.0, gt.y), 0.5  # a trivial "always off by 1m in x" predictor

    examples = calibration_examples(gt_points, parcel_style_by_id, predict_fn)
    assert examples == [
        ("formal", (1.0, 2.0), (2.0, 2.0), 0.5),
        ("informal", (5.0, 5.0), (6.0, 5.0), 0.5),
    ]


def test_calibration_examples_skips_a_gt_point_the_predictor_has_no_opinion_on():
    """A GT point with no nearby non-GT evidence (e.g. no legacy record
    close enough) must be skipped, not fabricated into a fake example."""
    gt_points = [GTPoint(x=1.0, y=2.0, parcel_id=10), GTPoint(x=5.0, y=5.0, parcel_id=20)]
    parcel_style_by_id = {10: "formal", 20: "informal"}

    def predict_fn(gt):
        return None if gt.parcel_id == 20 else ((gt.x, gt.y), 1.0)

    examples = calibration_examples(gt_points, parcel_style_by_id, predict_fn)
    assert len(examples) == 1
    assert examples[0][0] == "formal"


def test_empirical_coverage_reports_per_stratum_not_averaged():
    """Doc: "If informal-settlement coverage is materially below nominal
    while the overall number looks fine, the stratification is doing its
    job... report it, do not average it away." -- confirm the function's
    OWN return shape can't average strata together even by accident."""
    bands = CalibratedBands(quantile_by_stratum={"formal": 1.0, "informal": 1.0}, alpha=0.1)
    # formal: both points covered; informal: neither -- if these were
    # averaged, the result would hide the informal failure entirely
    test_points = [
        ("formal", (0.0, 0.0), (0.5, 0.0), 1.0),
        ("formal", (0.0, 0.0), (0.5, 0.0), 1.0),
        ("informal", (0.0, 0.0), (5.0, 0.0), 1.0),
        ("informal", (0.0, 0.0), (5.0, 0.0), 1.0),
    ]
    coverage = empirical_coverage(test_points, bands)
    assert coverage["formal"] == pytest.approx(1.0)
    assert coverage["informal"] == pytest.approx(0.0)
