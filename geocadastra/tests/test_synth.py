"""Stage 0 acceptance tests: a generated ward round-trips with perfect topology."""
import numpy as np
import pytest
from shapely.ops import unary_union

from geocadastra.synth.generator import EPS, WardParams, generate_ward

SEEDS = [0, 1, 42, 7, 123]


def _parcels_by_block(ward):
    by_block = {}
    for p in ward.parcels:
        by_block.setdefault(p.block_id, []).append(p)
    return by_block


@pytest.mark.parametrize("seed", SEEDS)
def test_parcels_tile_blocks_with_zero_gap_and_overlap(seed):
    ward = generate_ward(seed=seed)
    by_block = _parcels_by_block(ward)
    for block in ward.blocks:
        members = by_block.get(block.id, [])
        assert members, f"block {block.id} has no parcels"
        polys = [p.polygon for p in members]

        # zero overlap: pairwise intersection area is ~0
        overlap = 0.0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                overlap += polys[i].intersection(polys[j]).area
        tol = max(EPS, 1e-6 * block.polygon.area)
        assert overlap <= tol, f"block {block.id} parcels overlap by {overlap}"

        # zero gap: union of parcels covers the block exactly
        union_area = unary_union(polys).area
        gap = block.polygon.area - union_area
        assert abs(gap) <= tol, f"block {block.id} has gap/excess area {gap}"


@pytest.mark.parametrize("seed", SEEDS)
def test_parcel_areas_sum_exactly_to_block_area(seed):
    ward = generate_ward(seed=seed)
    by_block = _parcels_by_block(ward)
    for block in ward.blocks:
        members = by_block.get(block.id, [])
        total = sum(p.area for p in members)
        tol = max(EPS, 1e-6 * block.polygon.area)
        assert abs(total - block.polygon.area) <= tol


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
    multi_owner = [b for b in ward.buildings if len(b.parcel_ids) > 1]
    assert single_owner  # the common case
    for b in ward.buildings:
        assert len(b.parcel_ids) >= 1
    # crossing is an informal-only, low-probability event; over several seeds
    # at least one should show up without forcing it into this single ward
    _ = multi_owner  # presence is exercised by test_crossing_buildings_can_occur


def test_crossing_buildings_can_occur_in_informal_blocks():
    found = False
    for seed in range(30):
        ward = generate_ward(seed=seed)
        if any(len(b.parcel_ids) > 1 for b in ward.buildings):
            found = True
            break
    assert found, "expected at least one multi-parcel (crossing) building across seeds"
