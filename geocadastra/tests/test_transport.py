"""Stage 3 tests: capacity-constrained parcel assignment.

The headline test (`test_constrained_solver_beats_evidence_only_...`) is the
whole thesis of this stage: the constrained solver must beat evidence-only
segmentation, and the gap must widen as visible-edge fraction drops. If the
curves converge, something is wrong.
"""
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.transport import (
    Conflict,
    assign_parcels,
    assign_parcels_evidence_only,
    parcels_to_graph,
)
from geocadastra.synth.generator import WardParams, generate_ward, simulate_evidence_field

CRS = "EPSG:32643"


def _geom(shapely_geom):
    return Geom(shapely_geom, CRS)


def _strip_evidence(width, height, gsd, boundary_x, strength=0.95, decay=0.5, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out_h, out_w = int(height / gsd), int(width / gsd)
    transform = from_origin(0, height, gsd, gsd)
    xs = np.linspace(gsd / 2, width - gsd / 2, out_w)
    ridge = np.exp(-np.abs(xs - boundary_x) / decay)
    field = np.tile(ridge, (out_h, 1)).astype(np.float32) * strength + (1 - strength) * 0.05
    field = np.clip(field + rng.normal(0, noise, size=field.shape), 0, 1).astype(np.float32)
    return field, transform


def test_strong_evidence_at_the_true_boundary_recovers_it_closely():
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [80.0, 120.0]
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)
    assert result.conflicts == []
    assert result.parcel_polygons[0].geom.area == pytest.approx(80.0, abs=5.0)
    assert result.parcel_polygons[1].geom.area == pytest.approx(120.0, abs=5.0)


def test_zero_evidence_still_matches_recorded_areas_via_the_constraint_alone():
    """The core thesis, minimal case: with NO evidence anywhere (a fully
    invisible boundary), the area constraint alone places the split at
    (almost) exactly the recorded position."""
    block = box(0, 0, 20, 10)
    rng = np.random.default_rng(1)
    field = np.clip(0.05 + rng.normal(0, 0.02, size=(50, 100)), 0, 1).astype(np.float32)
    transform = from_origin(0, 10, 0.2, 0.2)
    areas = [80.0, 120.0]
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)
    assert result.parcel_polygons[0].geom.area == pytest.approx(80.0, abs=5.0)
    assert result.parcel_polygons[1].geom.area == pytest.approx(120.0, abs=5.0)


def test_labelling_exactly_tiles_the_block_with_no_gap_or_overlap():
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [80.0, 120.0]
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)
    polys = [g.geom for g in result.parcel_polygons.values()]
    total = sum(p.area for p in polys)
    assert total == pytest.approx(block.area, rel=1e-6)
    overlap = polys[0].intersection(polys[1]).area
    assert overlap < 1e-6


def test_output_areas_match_recorded_areas_within_one_superpixel():
    block = box(0, 0, 20, 10)
    n_segments = 400
    superpixel_area = block.area / n_segments
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [80.0, 120.0]
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=n_segments)
    assert result.conflicts == []
    for i, target in enumerate(areas):
        assert abs(result.parcel_polygons[i].geom.area - target) <= superpixel_area * 3  # a handful of boundary superpixels


def test_area_sum_mismatch_is_rescaled_and_recorded_as_a_conflict():
    block = box(0, 0, 20, 10)  # true area 200
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [40.0, 60.0]  # sums to 100, half of the true 200
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)
    assert any(c.kind == "area_sum_mismatch" for c in result.conflicts)
    total = sum(g.geom.area for g in result.parcel_polygons.values())
    assert total == pytest.approx(block.area, rel=1e-6)
    # rescaled proportionally: parcel 1 should still be ~1.5x parcel 0
    ratio = result.parcel_polygons[1].geom.area / result.parcel_polygons[0].geom.area
    assert ratio == pytest.approx(1.5, rel=0.15)


def test_a_parcel_missing_its_recorded_area_is_uncapacitated_not_dropped():
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [80.0, None]  # parcel 1 has no recorded area
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)
    assert 0 in result.parcel_polygons and 1 in result.parcel_polygons
    assert result.parcel_polygons[0].geom.area == pytest.approx(80.0, abs=5.0)
    # parcel 1 gets whatever's left over, roughly the remaining ~120
    assert result.parcel_polygons[1].geom.area == pytest.approx(120.0, abs=15.0)


def test_no_legacy_record_at_all_falls_back_to_watershed():
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    result = assign_parcels(_geom(block), [None, None], [(4.0, 5.0), (14.0, 5.0)], field, transform, n_segments=300)
    assert any(c.kind == "no_legacy_record_for_block" for c in result.conflicts)
    assert len(result.parcel_polygons) >= 1
    total = sum(g.geom.area for g in result.parcel_polygons.values())
    assert total <= block.area + 1e-6


def test_evidence_only_baseline_ignores_area_and_follows_nearest_seed():
    """With a seed placed off-centre and strong (but slightly offset)
    evidence, the evidence-only baseline should NOT be forced to the
    recorded 80/120 split -- unlike assign_parcels, nothing here enforces it."""
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=10.0)  # evidence ridge at the true midline
    seeds = [(4.0, 5.0), (16.0, 5.0)]  # asymmetric seeds
    result = assign_parcels_evidence_only(_geom(block), seeds, field, transform, n_segments=300)
    total = sum(g.geom.area for g in result.parcel_polygons.values())
    assert total == pytest.approx(block.area, rel=1e-6)
    # roughly a 100/100 split (evidence ridge at the midline), NOT 80/120
    assert result.parcel_polygons[0].geom.area == pytest.approx(100.0, abs=10.0)


def test_parcels_to_graph_completes_the_pipeline_into_a_real_planar_graph():
    """The doc's own step 5: extracted boundaries -> planarize -> build_graph.
    Not just "polygons that happen to tile" -- a real PlanarGraph with the
    shared boundary as one edge referenced by both faces."""
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    areas = [80.0, 120.0]
    seeds = [(4.0, 5.0), (14.0, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=300)

    graph = parcels_to_graph(result.parcel_polygons, _geom(block))
    assert len(graph.faces) == 2
    polys = graph.faces_to_polygons()
    total = sum(p.area for p in polys.values())
    assert total == pytest.approx(block.area, rel=1e-6)

    from geocadastra.core.graph import OUTER

    shared = [e for e in graph.edges.values() if OUTER not in graph.faces_of_edge(e.id)]
    assert shared, "expected at least one interior edge shared by both parcel faces"
    for e in graph.edges.values():
        assert len(graph.faces_of_edge(e.id)) == 2


# --------------------------------------------------------------------------
# regression tests for the review's findings
# --------------------------------------------------------------------------

def test_out_of_range_evidence_is_clipped_not_left_to_poison_dijkstra():
    """Regression: an unclipped value outside [0,1] (e.g. a raw model logit)
    became a negative graph edge weight -- and since the graph is undirected,
    one negative edge is already a negative cycle, sending dijkstra into
    unbounded memory growth instead of raising or clamping."""
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    field = field.copy()
    field[10:13, 30:33] = -3.0  # a small patch, plain finite, out of [0,1]
    # must complete quickly and without error -- the old behaviour was to hang/OOM
    result = assign_parcels(_geom(block), [80.0, 120.0], [(4.0, 5.0), (14.0, 5.0)], field, transform, n_segments=200)
    assert sum(g.geom.area for g in result.parcel_polygons.values()) == pytest.approx(block.area, rel=1e-6)


def test_nan_evidence_raises_a_clear_error_not_a_silent_wrong_answer():
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    field = field.copy()
    field[5, 5] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf"):
        assign_parcels(_geom(block), [80.0, 120.0], [(4.0, 5.0), (14.0, 5.0)], field, transform, n_segments=200)


def test_seed_a_few_pixels_north_or_west_of_the_raster_does_not_crash():
    """Regression: _seed_to_superpixel's window bounds were floor-clamped on
    one end only, so plain numpy slicing silently wrapped a negative stop
    index, reaching a crash a few lines later instead of this function's own
    clean fallback. Realistic: a legacy centroid jittered a few pixels
    outside a tightly-cropped block raster is the expected case, not a
    pathological one."""
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    # seed 1 sits 0.6m (3px at gsd=0.2) above the raster's north edge (y=10)
    seeds = [(4.0, 5.0), (12.0, 10.6)]
    result = assign_parcels(_geom(block), [80.0, 120.0], seeds, field, transform, n_segments=300)
    assert sum(g.geom.area for g in result.parcel_polygons.values()) == pytest.approx(block.area, rel=1e-6)
    # and west of the raster (x=0)
    seeds2 = [(-0.6, 5.0), (14.0, 5.0)]
    result2 = assign_parcels(_geom(block), [80.0, 120.0], seeds2, field, transform, n_segments=300)
    assert sum(g.geom.area for g in result2.parcel_polygons.values()) == pytest.approx(block.area, rel=1e-6)


def test_a_seed_far_outside_the_block_still_raises_the_clean_error():
    """The floor-clamp fix must not turn a genuinely-wrong seed (not just a
    few pixels of jitter) into a silently-accepted, nonsensical assignment."""
    from geocadastra.core.transport import _seed_to_superpixel
    from skimage.segmentation import slic

    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0)
    from rasterio.features import rasterize

    mask = rasterize([(block, 1)], field.shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    labels = slic(field, n_segments=300, compactness=5.0, mask=mask, channel_axis=None, start_label=0)
    with pytest.raises(ValueError, match="no reachable superpixel"):
        _seed_to_superpixel((4000.0, 5.0), transform, labels)  # 4km away, not a few pixels of jitter


def test_two_unhandled_parcels_competing_for_leftover_can_starve_one_and_it_is_reported():
    """Regression: when 2+ unhandled parcels compete for the same leftover
    area, one could end up with zero superpixels (its seed effectively
    "inside" territory already claimed by the capacitated parcel) with no
    conflict recorded at all."""
    block = box(0, 0, 20, 10)
    field, transform = _strip_evidence(20, 10, 0.2, boundary_x=8.0, strength=0.0)  # no evidence to help either
    # parcel 0 capacitated with nearly the whole block; parcels 1 and 2 unhandled
    # (missing area), with parcel 2's seed buried deep in parcel 0's territory
    areas = [190.0, None, None]
    seeds = [(2.0, 5.0), (19.0, 5.0), (0.2, 5.0)]
    result = assign_parcels(_geom(block), areas, seeds, field, transform, n_segments=400)
    starved = [c for c in result.conflicts if c.kind == "leftover_parcel_got_no_area"]
    if 2 not in result.parcel_polygons:
        assert starved and 2 in starved[0].detail["parcel_indices"]
    # whichever happened, nothing was silently dropped without a conflict
    if not starved:
        assert 1 in result.parcel_polygons and 2 in result.parcel_polygons


def test_recorded_area_not_matched_conflict_fires_when_a_parcel_is_shortchanged():
    """Direct unit test of the post-solve safety net itself, independent of
    which upstream mechanism would cause the shortfall."""
    from geocadastra.core.transport import Conflict as _C
    from geocadastra.core.transport import _verify_recorded_areas
    import numpy as _np

    sp_ids = _np.array([0, 1, 2, 3])
    sp_areas = _np.array([10.0, 10.0, 10.0, 10.0])
    sp_to_parcel = {0: 0, 1: 0, 2: 0, 3: 0}  # parcel 1 got nothing at all
    parcel_areas = [40.0, 40.0]
    conflicts: list = []
    _verify_recorded_areas(parcel_areas, sp_to_parcel, sp_ids, sp_areas, conflicts)
    assert any(c.kind == "recorded_area_not_matched" for c in conflicts)
    detail = next(c for c in conflicts if c.kind == "recorded_area_not_matched").detail
    assert detail["parcels"][0]["parcel_index"] == 1
    assert detail["parcels"][0]["assigned"] == 0.0


def test_recorded_area_not_matched_does_not_fire_within_tolerance():
    from geocadastra.core.transport import _verify_recorded_areas
    import numpy as _np

    sp_ids = _np.array([0, 1, 2, 3])
    sp_areas = _np.array([10.0, 10.0, 10.0, 10.0])
    sp_to_parcel = {0: 0, 1: 0, 2: 1, 3: 1}
    parcel_areas = [20.0, 20.0]
    conflicts: list = []
    _verify_recorded_areas(parcel_areas, sp_to_parcel, sp_ids, sp_areas, conflicts)
    assert conflicts == []


def test_parcels_to_graph_wraps_the_hole_error_with_actionable_context():
    """Direct unit test of the improved error message: build_graph()'s
    generic NotImplementedError is re-raised naming what likely caused it,
    not left as a bare "some face has a hole"."""
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    donut_parcel = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[hole])
    island_parcel = Polygon(hole)
    parcel_polygons = {0: _geom(donut_parcel), 1: _geom(island_parcel)}
    with pytest.raises(NotImplementedError, match="island"):
        parcels_to_graph(parcel_polygons, _geom(outer))


def test_parcels_to_graph_rejects_a_parcel_in_the_wrong_crs():
    """Regression: parcels_to_graph() used to silently unwrap each
    parcel's Geom without ever checking its declared CRS matched the
    block's -- exactly the "silently coerce" pattern Geom/
    CRSMismatchError exist to prevent everywhere else in this project."""
    block = box(0, 0, 20, 10)
    parcel_polygons = {0: Geom(box(0, 0, 10, 10), "EPSG:4326")}  # lon/lat degrees, not UTM metres
    with pytest.raises(CRSMismatchError):
        parcels_to_graph(parcel_polygons, _geom(block))


# --------------------------------------------------------------------------
# the headline test: the whole thesis of Stage 3
# --------------------------------------------------------------------------

def _clean_blocks(fraction, n_blocks, base_seed, min_parcels=4, max_parcels=14):
    """Blocks where every parcel has a clean 1:1 legacy correspondence (no
    merges/missing), so each has both a recorded area and a legacy-centroid
    seed -- the well-behaved case this headline test targets. Capped at
    `max_parcels` too: this measures the method's central claim, not its
    behavior on the hardest many-rival-neighbor blocks, which is a real but
    separate scaling question."""
    found = []
    s = base_seed
    while len(found) < n_blocks and s < base_seed + 400:
        ward = generate_ward(
            params=WardParams(
                visible_edge_fraction=fraction,
                legacy_p_missing={"formal": 0.0, "informal": 0.0, "institutional": 0.0},
                legacy_p_merge={"formal": 0.0, "informal": 0.0, "institutional": 0.0},
                legacy_jitter_std={"formal": 0.05, "informal": 0.05, "institutional": 0.05},
            ),
            seed=s,
        )
        s += 1
        by_block = {}
        for p in ward.parcels:
            by_block.setdefault(p.block_id, []).append(p)
        legacy_by_source = {lp.source_parcel_ids[0]: lp for lp in ward.legacy_parcels if len(lp.source_parcel_ids) == 1}
        for block_id, parcels in by_block.items():
            if min_parcels <= len(parcels) <= max_parcels and all(p.id in legacy_by_source for p in parcels):
                found.append((ward, next(b for b in ward.blocks if b.id == block_id), parcels, legacy_by_source))
                break
    assert len(found) == n_blocks, f"only found {len(found)}/{n_blocks} clean blocks for fraction={fraction}"
    return found


def _boundary_error(assigned_polys, true_parcels, block_area):
    total_symdiff = 0.0
    for pos, p in enumerate(true_parcels):
        assigned = assigned_polys.get(pos)
        total_symdiff += p.polygon.area if assigned is None else assigned.geom.symmetric_difference(p.polygon).area
    return total_symdiff / block_area


def test_constrained_solver_beats_evidence_only_and_the_gap_widens_as_evidence_drops():
    """The whole thesis, measured properly: averaged over several blocks per
    fraction, not asserted on any single noisy instance -- a specific block
    with many rival neighbors and almost no evidence can occasionally let
    evidence-only get lucky even where the constrained solver recovers each
    parcel's area almost exactly (checked separately, per-instance, by the
    unit tests above); the aggregate trend is the actual claim."""
    fractions = [0.9, 0.5, 0.15]
    n_blocks_per_fraction = 8
    avg_errors_constrained, avg_errors_evidence_only = [], []

    for frac_idx, frac in enumerate(fractions):
        blocks = _clean_blocks(frac, n_blocks_per_fraction, base_seed=frac_idx * 1000 + 1)
        errs_c, errs_e = [], []
        for ward, block, parcels, legacy_by_source in blocks:
            field, transform = simulate_evidence_field(ward, block.id, gsd=0.3, seed=1)
            areas = [p.area for p in parcels]
            seeds_xy = [legacy_by_source[p.id].polygon.centroid.coords[0] for p in parcels]

            constrained = assign_parcels(_geom(block.polygon), areas, seeds_xy, field, transform, n_segments=1200)
            evidence_only = assign_parcels_evidence_only(_geom(block.polygon), seeds_xy, field, transform, n_segments=1200)

            errs_c.append(_boundary_error(constrained.parcel_polygons, parcels, block.polygon.area))
            errs_e.append(_boundary_error(evidence_only.parcel_polygons, parcels, block.polygon.area))

        avg_errors_constrained.append(float(np.mean(errs_c)))
        avg_errors_evidence_only.append(float(np.mean(errs_e)))

    for frac, ec, ee in zip(fractions, avg_errors_constrained, avg_errors_evidence_only):
        assert ec <= ee, f"fraction={frac}: constrained ({ec:.4f}) did not beat evidence-only ({ee:.4f}) on average"

    gaps = [ee - ec for ec, ee in zip(avg_errors_constrained, avg_errors_evidence_only)]
    assert gaps[-1] > gaps[0], (
        f"gap did not widen as visible_edge_fraction dropped: fractions={fractions} gaps={gaps}"
    )
