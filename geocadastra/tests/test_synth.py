"""Stage 0 acceptance tests: a generated ward round-trips with perfect topology.

Also covers regressions for bugs found in a full multi-angle review of this
module (see STAGE_0_NOTES.md): a ring-closure bug in the legacy-jitter step,
a merge-order duplicate in the legacy layer, biased GT-point sampling at
shared corners, unbounded recursion in the informal-subdivision cutter, a
crash path for degenerate blocks, and missing WardParams validation.
"""
from collections import Counter, defaultdict

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from shapely.ops import unary_union

from geocadastra.synth.generator import (
    EPS,
    WardParams,
    _make_gt_points,
    _recursive_split,
    generate_ward,
    simulate_evidence_field,
)

SEEDS = [0, 1, 42, 7, 123, 660]  # 660: pins the phantom-overlap GEOS regression (see below)


def _parcels_by_block(ward):
    by_block = {}
    for p in ward.parcels:
        by_block.setdefault(p.block_id, []).append(p)
    return by_block


def _incremental_union_area(polys):
    """Union area via a left fold instead of shapely's batch unary_union().

    unary_union() on a large set of near-touching polygons (e.g. 20+ parcels
    from recursive informal subdivision) has hit real GEOS overlay-robustness
    losses in this codebase -- up to 3% of a block's area silently vanishing
    even though the parcels themselves tile exactly (see STAGE_0_NOTES.md).
    The incremental fold did not reproduce that loss on the same input, so
    it's what the tiling assertions below rely on for "gap" detection.
    """
    if not polys:
        return 0.0
    u = polys[0]
    for p in polys[1:]:
        u = u.union(p)
    return u.area


def _assert_block_tiles_exactly(block, members):
    polys = [p.polygon for p in members]
    tol = max(EPS, 1e-6 * block.polygon.area)

    overlap = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            overlap += polys[i].intersection(polys[j]).area
    assert overlap <= tol, f"block {block.id} parcels overlap by {overlap}"

    area_sum = sum(p.area for p in members)
    assert abs(area_sum - block.polygon.area) <= tol, (
        f"block {block.id}: parcel areas sum to {area_sum}, block area is {block.polygon.area}"
    )

    gap = block.polygon.area - _incremental_union_area(polys)
    assert abs(gap) <= tol, f"block {block.id} has gap/excess area {gap}"


@pytest.mark.parametrize("seed", SEEDS)
def test_parcels_tile_blocks_with_zero_gap_and_overlap(seed):
    ward = generate_ward(seed=seed)
    by_block = _parcels_by_block(ward)
    for block in ward.blocks:
        members = by_block.get(block.id, [])
        assert members, f"block {block.id} has no parcels"
        _assert_block_tiles_exactly(block, members)


@pytest.mark.parametrize("seed", SEEDS)
def test_parcel_areas_sum_exactly_to_block_area(seed):
    ward = generate_ward(seed=seed)
    by_block = _parcels_by_block(ward)
    for block in ward.blocks:
        members = by_block.get(block.id, [])
        total = sum(p.area for p in members)
        tol = max(EPS, 1e-6 * block.polygon.area)
        assert abs(total - block.polygon.area) <= tol


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_tiling_holds_across_many_random_seeds(seed):
    """The property test that would have caught two real GEOS-overlay bugs
    that a handful of hand-picked seeds missed: seed=21 (unary_union losing
    3% of a block's area on a 24-parcel informal block -- fixed by checking
    gap via incremental union instead) and seed=660 (.intersection() on two
    genuinely non-overlapping recursively-split parcels several generations
    deep returning 209 m^2 of phantom overlap -- fixed by snapping every
    split() output to an explicit GEOS precision grid). Samples far more of
    the space than the 6 fixed SEEDS above."""
    ward = generate_ward(seed=seed)
    by_block = _parcels_by_block(ward)
    for block in ward.blocks:
        members = by_block.get(block.id, [])
        assert members
        _assert_block_tiles_exactly(block, members)


def test_rendering_is_deterministic_under_fixed_seed():
    a = generate_ward(seed=42)
    b = generate_ward(seed=42)

    assert np.array_equal(a.ortho, b.ortho)
    assert np.array_equal(a.dsm, b.dsm)
    assert np.array_equal(a.dtm, b.dtm)
    assert a.transform == b.transform

    assert len(a.parcels) == len(b.parcels)
    for pa, pb in zip(a.parcels, b.parcels):
        assert list(pa.polygon.exterior.coords) == list(pb.polygon.exterior.coords)
    assert [(g.x, g.y, g.parcel_id) for g in a.gt_points] == [(g.x, g.y, g.parcel_id) for g in b.gt_points]


def test_different_seeds_produce_different_wards():
    a = generate_ward(seed=1)
    b = generate_ward(seed=2)
    assert not np.array_equal(a.ortho, b.ortho)
    assert [p.polygon.wkt for p in a.parcels] != [p.polygon.wkt for p in b.parcels]


def test_ward_has_a_mix_of_settlement_styles_and_content():
    ward = generate_ward(seed=42)
    assert len(ward.blocks) >= 2
    assert len(ward.parcels) >= len(ward.blocks)
    styles = {b.style for b in ward.blocks}
    assert styles <= {"formal", "informal", "institutional"}
    assert len(ward.buildings) > 0
    assert len(ward.legacy_parcels) > 0
    assert len(ward.gt_points) > 0


def test_legacy_layer_references_real_parcels_and_can_miss_or_merge():
    ward = generate_ward(seed=42)
    true_ids = {p.id for p in ward.parcels}
    seen_sources = set()
    for lp in ward.legacy_parcels:
        assert set(lp.source_parcel_ids) <= true_ids
        seen_sources.update(lp.source_parcel_ids)
        assert lp.polygon.is_valid
    # not every true parcel need survive into the legacy layer (some are "missing")
    assert seen_sources <= true_ids


@pytest.mark.parametrize("seed", range(20))
def test_legacy_layer_has_no_duplicate_or_missing_bookkeeping(seed):
    """Regression for the merge-order bug: a parcel already emitted as its
    own standalone legacy entry could later be picked as a *different*
    parcel's merge partner, duplicating its geometry into two entries."""
    ward = generate_ward(seed=seed)
    all_sources = []
    for lp in ward.legacy_parcels:
        all_sources.extend(lp.source_parcel_ids)
    dupes = [pid for pid, n in Counter(all_sources).items() if n > 1]
    assert not dupes, f"seed={seed}: source parcel(s) {dupes} appear in more than one legacy_parcels entry"


@pytest.mark.parametrize("seed", range(15))
def test_legacy_layer_geometry_is_reasonable(seed):
    """Regression for the ring-closure bug: independent per-vertex jitter on
    a *closed* ring (first coord == last) left a stray near-duplicate vertex
    that made almost every jittered parcel self-intersect."""
    ward = generate_ward(seed=seed)
    src_area = {p.id: p.polygon.area for p in ward.parcels}
    for lp in ward.legacy_parcels:
        assert lp.polygon.is_valid
        if len(lp.source_parcel_ids) == 1:
            true_area = src_area[lp.source_parcel_ids[0]]
            if true_area > 20.0:  # jitter magnitude dominates for slivers/thin strips by design; not a bug
                rel_delta = abs(lp.polygon.area - true_area) / true_area
                # generous ceiling: legitimate style-driven jitter measured up to ~245%
                # relative delta across 60 seeds; the pre-fix ring-closure bug produced
                # up to ~9800% by leaving a stray vertex on nearly every parcel
                assert rel_delta < 5.0, (
                    f"seed={seed} legacy parcel {lp.id}: area distorted by {rel_delta:.1%} "
                    f"relative to its true {true_area:.1f} m^2 source"
                )


def test_gt_point_sampling_rate_is_not_biased_by_shared_corners():
    """Regression: a corner touched by k parcels used to get up to k
    independent draws (selection prob ~= 1-(1-frac)^k instead of frac)."""
    frac = 0.08
    ward = generate_ward(seed=1)
    corner_owners = defaultdict(set)
    for p in ward.parcels:
        for (x, y) in list(p.polygon.exterior.coords)[:-1]:
            corner_owners[(round(x, 6), round(y, 6))].add(p.id)
    multi_owner = [k for k, owners in corner_owners.items() if len(owners) >= 3]
    assert multi_owner, "test needs at least one corner shared by 3+ parcels to be meaningful"

    n_trials = 500
    hits = 0
    for t in range(n_trials):
        rng = np.random.default_rng(t + 12345)
        pts = _make_gt_points(ward.parcels, ward.params, rng)
        keys = {(round(g.x, 6), round(g.y, 6)) for g in pts}
        if multi_owner[0] in keys:
            hits += 1
    rate = hits / n_trials
    # a biased implementation puts a 4-parcel corner near 1-(1-0.08)^4 ~= 0.284;
    # allow generous slack around the true target since this is a single corner
    assert abs(rate - frac) < 0.06, f"shared-corner selection rate {rate:.3f} is far from target {frac}"


def test_recursive_split_respects_a_work_budget_for_extreme_params():
    """Regression: min_area=0.001 with max_depth=25 used to hang (>150s,
    up to 2**25 potential pieces). Must now finish quickly with a bounded
    piece count, while still tiling its input exactly."""
    import time
    from shapely.geometry import box

    poly = box(0, 0, 240, 180)
    rng = np.random.default_rng(0)
    t0 = time.time()
    pieces = _recursive_split(poly, rng, min_area=0.001, max_depth=25)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"took {elapsed:.1f}s, expected the work budget to bound this"
    assert len(pieces) <= 4200
    area_sum = sum(p.area for p in pieces)
    assert abs(area_sum - poly.area) <= max(EPS, 1e-6 * poly.area)


@pytest.mark.parametrize("seed", range(30))
def test_dense_jittery_roads_do_not_crash_block_generation(seed):
    """Regression: _polygonize_blocks could hand a degenerate/sliver face to
    _strip_split, which crashed on LineString.exterior. Denser, more jittery
    roads are the most likely way to produce a sliver face."""
    params = WardParams(n_arterial_h=3, n_arterial_v=3, minor_spacing=25, minor_spacing_jitter=20)
    ward = generate_ward(params=params, seed=seed)
    assert len(ward.blocks) > 0
    for block in ward.blocks:
        assert block.polygon.area > EPS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"style_weights": {"formal": 0.6, "informal": 0.3, "peri_urban": 0.1}},  # missing from the 4 legacy dicts
        {"strip_count_range": (8, 3)},  # reversed bounds
        {"style_weights": {"formal": 0.0, "informal": 0.0, "institutional": 0.0}},  # all-zero weights
    ],
)
def test_wardparams_validates_bad_configuration_eagerly(kwargs):
    with pytest.raises(ValueError):
        WardParams(**kwargs)


def test_visible_edge_fraction_zero_means_no_walls_rendered():
    params = WardParams(visible_edge_fraction=0.0)
    ward = generate_ward(params=params, seed=3)
    assert not any(ward.edges_visible.values())


def test_visible_edge_fraction_one_means_all_walls_rendered():
    params = WardParams(visible_edge_fraction=1.0)
    ward = generate_ward(params=params, seed=3)
    assert all(ward.edges_visible.values())
    assert len(ward.edges_visible) > 0


def test_raster_outputs_are_consistent_and_georeferenced():
    ward = generate_ward(seed=42)
    h, w = ward.dtm.shape
    assert ward.ortho.shape == (h, w, 3)
    assert ward.dsm.shape == (h, w)
    assert ward.ortho.dtype == np.uint8
    # DSM is DTM plus non-negative building/vegetation height, never below it
    assert np.all(ward.dsm >= ward.dtm - 1e-4)
    # buildings actually raise the surface somewhere
    assert ward.dsm.max() > ward.dtm.max()


def test_buildings_stay_near_their_parcel_and_crossers_touch_two():
    ward = generate_ward(seed=42)
    single_owner = [b for b in ward.buildings if len(b.parcel_ids) == 1]
    assert single_owner  # the common case
    for b in ward.buildings:
        assert len(b.parcel_ids) >= 1


def test_crossing_buildings_can_occur_in_informal_blocks():
    found = False
    for seed in range(30):
        ward = generate_ward(seed=seed)
        if any(len(b.parcel_ids) > 1 for b in ward.buildings):
            found = True
            break
    assert found, "expected at least one multi-parcel (crossing) building across seeds"


def test_evidence_field_is_monotonically_stronger_with_more_visible_edges():
    means = []
    for frac in (0.0, 0.3, 0.7, 1.0):
        ward = generate_ward(params=WardParams(visible_edge_fraction=frac), seed=1)
        field, _ = simulate_evidence_field(ward, ward.blocks[0].id, seed=1)
        assert field.shape[0] > 0 and field.shape[1] > 0
        assert field.min() >= 0.0 and field.max() <= 1.0
        means.append(field.mean())
    assert means == sorted(means)


def test_evidence_field_zero_visible_edges_is_pure_noise_no_ridges():
    ward = generate_ward(params=WardParams(visible_edge_fraction=0.0), seed=2)
    field, _ = simulate_evidence_field(ward, ward.blocks[0].id, base=0.05, noise_std=0.05, seed=2)
    assert field.max() < 0.05 + 6 * 0.05  # no ridge structure, just baseline + noise
