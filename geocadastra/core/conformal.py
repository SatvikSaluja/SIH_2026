"""Conformal calibration (Stage 6): certified accuracy bands for fused
boundary predictions, calibrated against held-out GT survey points.

Every boundary is supposed to carry a certified accuracy: a band such that
the true line falls inside it with a stated, empirically verified
probability. That guarantee comes from split conformal prediction, not
from trusting a model's own predicted sigma at face value -- a predicted
sigma can be systematically wrong (too tight, too loose), and conformal
calibration corrects for that using held-out GT displacement, without
needing to know WHY the sigma was wrong.

Mondrian (stratified) conformal, not one global quantile: the doc is
explicit that coverage must hold WITHIN each settlement type, not just on
average -- a model that's well-calibrated for formal blocks and badly
under-confident for informal ones would look fine on a single pooled
number while silently failing exactly the parcels this project exists to
help. `calibrate()` computes one quantile per stratum; `empirical_
coverage()` reports one coverage number per stratum, never averaged.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedBands:
    quantile_by_stratum: dict  # stratum (e.g. settlement style) -> nonconformity quantile
    alpha: float  # miscoverage rate this was calibrated for (nominal coverage = 1 - alpha)


def nonconformity_score(true_xy: tuple, pred_xy: tuple, sigma: float) -> float:
    """|true - predicted| / sigma -- a standardized residual. Dividing by
    the model's OWN predicted sigma is what makes scores comparable across
    points the model is more or less confident about, and across strata
    with systematically different noise levels (informal legacy records
    are noisier than formal ones by construction -- see fusion.py's own
    DEFAULT_SIGMA_LEGACY_BY_STYLE) -- comparing raw displacement across
    strata would just measure that noise difference, not miscalibration.
    """
    dx, dy = true_xy[0] - pred_xy[0], true_xy[1] - pred_xy[1]
    return (dx**2 + dy**2) ** 0.5 / sigma


def calibrate(calibration_points, alpha: float = 0.1) -> CalibratedBands:
    """`calibration_points`: iterable of `(stratum, nonconformity_score)`.

    Mondrian split conformal: a SEPARATE (1-alpha) quantile per stratum
    (doc: "stratified by settlement type... because coverage must hold
    within informal settlements specifically, not just on average").
    Within each stratum, the quantile is the standard finite-sample-
    correct choice -- the `ceil((n+1)*(1-alpha))`-th smallest score (not
    a naive `numpy.quantile`, which under-covers for small n) -- clamped
    to the largest available score when a stratum has too few points to
    reach that far exactly, the maximally-conservative fallback rather
    than fabricating a number past what the data supports.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    calibration_points = list(calibration_points)  # also guards an empty iterator/generator, not just an empty list
    if not calibration_points:
        raise ValueError("calibrate() needs at least one calibration point")
    by_stratum = defaultdict(list)
    for stratum, score in calibration_points:
        by_stratum[stratum].append(score)
    quantile_by_stratum = {}
    for stratum, scores in by_stratum.items():
        scores = sorted(scores)
        n = len(scores)
        k = min(math.ceil((n + 1) * (1 - alpha)), n)
        quantile_by_stratum[stratum] = scores[k - 1]
    return CalibratedBands(quantile_by_stratum=quantile_by_stratum, alpha=alpha)


def certified_band(bands: CalibratedBands, stratum: str, sigma: float) -> float:
    """The certified radius around a prediction with this `sigma`, in this
    `stratum`: the true position is expected to fall within this radius
    with probability >= 1 - `bands.alpha`, per the calibration guarantee.

    Raises `KeyError` (not a silent fallback to some other stratum's
    quantile) if `stratum` had no calibration data at all -- "nothing is
    silently resolved" applies here too: a certified band for a stratum
    this calibration never measured is not a real certification.
    """
    return bands.quantile_by_stratum[stratum] * sigma


def calibration_examples(gt_points, parcel_style_by_id: dict, predict_fn) -> list:
    """Turn a ward's GT survey points into `(stratum, true_xy, pred_xy,
    sigma)` tuples ready for `calibrate()`/`empirical_coverage()`.

    `gt_points`: iterable of objects with `.x`, `.y`, `.parcel_id` (e.g.
    `synth.generator.GTPoint`). `parcel_style_by_id`: parcel id -> style,
    the stratum for Mondrian calibration. `predict_fn(gt_point) ->
    (pred_xy, sigma) | None`: the caller's own prediction for where this
    GT point's true position should be -- deliberately a caller-supplied
    function, not this module calling into `core/fusion.py` itself,
    because it is the CALLER's responsibility to make sure `predict_fn`
    never uses this (or any) GT point as its own input evidence. Using
    the same point as both the model's input and the calibration target
    would trivially inflate apparent accuracy -- e.g. `core/fusion.py`'s
    `legacy_estimate()` (which never looks at GT points at all) is a
    natural leakage-free choice.

    Output tuples are ready for `empirical_coverage()` directly, but NOT
    for `calibrate()` as-is -- `calibrate()` takes `(stratum, score)`
    pairs, so run each `(true_xy, pred_xy, sigma)` through
    `nonconformity_score()` first (see `tests/test_stage6_acceptance.py`
    for the conversion).

    A GT point `predict_fn` has no opinion about (returns `None` -- e.g.
    no legacy record close enough) is skipped, not fabricated into a
    fake example.
    """
    examples = []
    for gt in gt_points:
        result = predict_fn(gt)
        if result is None:
            continue
        pred_xy, sigma = result
        examples.append((parcel_style_by_id[gt.parcel_id], (gt.x, gt.y), pred_xy, sigma))
    return examples


def empirical_coverage(test_points, bands: CalibratedBands) -> dict:
    """`test_points`: iterable of `(stratum, true_xy, pred_xy, sigma)`.

    Per stratum, the fraction of held-out points whose true position
    falls within its own certified band -- reported per stratum, never
    pooled into one overall number (doc: "If informal-settlement coverage
    is materially below nominal while the overall number looks fine, the
    stratification is doing its job... report it, do not average it
    away.").
    """
    by_stratum = defaultdict(list)
    for stratum, true_xy, pred_xy, sigma in test_points:
        band = certified_band(bands, stratum, sigma)
        dx, dy = true_xy[0] - pred_xy[0], true_xy[1] - pred_xy[1]
        displacement = (dx**2 + dy**2) ** 0.5
        by_stratum[stratum].append(displacement <= band)
    return {stratum: sum(hits) / len(hits) for stratum, hits in by_stratum.items()}
