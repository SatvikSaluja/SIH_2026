"""Stage 5: fusion in distance-field-adjacent space (per source, a
correspondence-free local lookup: nearest point on legacy linework,
nearest low-SDF pixel from the model, nearest GT point within capture
radius) followed by an inverse-variance-weighted combine. Never matches
vertex-to-vertex between independently-vertexed polygons.
"""
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Point, box

from geocadastra.core.crs import Geom
from geocadastra.core.fusion import (
    ConflictRecord,
    SourceEstimate,
    _is_exterior_node,
    fuse_estimates,
    fuse_node,
    gt_estimate,
    legacy_estimate,
    model_estimate,
)
from geocadastra.core.graph import build_graph


def test_fuse_estimates_closed_form_inverse_variance_weighted():
    # two equally-confident estimates straddling the origin -> exactly the midpoint
    a = SourceEstimate("legacy", 0.0, 0.0, 1.0)
    b = SourceEstimate("gt", 10.0, 0.0, 1.0)
    fused = fuse_estimates([a, b])
    assert fused.x == pytest.approx(5.0)
    assert fused.y == pytest.approx(0.0)
    # fused sigma is smaller than either input's -- combining evidence always increases confidence
    assert fused.sigma < 1.0


def test_fuse_estimates_low_sigma_source_dominates():
    confident = SourceEstimate("gt", 1.0, 0.0, 0.01)  # near-exact
    vague = SourceEstimate("legacy", 100.0, 0.0, 10.0)  # far away, very unsure
    fused = fuse_estimates([confident, vague])
    assert fused.x == pytest.approx(1.0, abs=0.5)  # pulled almost exactly to the confident source


def test_fuse_estimates_single_source_returns_it_unchanged():
    only = SourceEstimate("model", 3.0, 4.0, 0.7)
    fused = fuse_estimates([only])
    assert (fused.x, fused.y, fused.sigma) == (3.0, 4.0, 0.7)


def test_fuse_estimates_requires_at_least_one():
    with pytest.raises(ValueError):
        fuse_estimates([])


def test_legacy_estimate_uses_nearest_point_on_boundary_not_nearest_vertex():
    # a long edge from (0,0) to (100,0): nearest VERTEX to (50, 1) is far
    # (either endpoint), but nearest POINT on the line is directly below --
    # confirms this doesn't do vertex-to-vertex correspondence.
    boundary = LineString([(0, 0), (100, 0)])
    est = legacy_estimate((50.0, 1.0), boundary, style="formal")
    assert est is not None
    assert est.x == pytest.approx(50.0)
    assert est.y == pytest.approx(0.0)


def test_legacy_estimate_sigma_depends_on_block_style():
    boundary = LineString([(0, 0), (100, 0)])
    formal = legacy_estimate((50.0, 1.0), boundary, style="formal")
    informal = legacy_estimate((50.0, 1.0), boundary, style="informal")
    assert formal.sigma < informal.sigma  # doc: informal legacy records are less reliable


def test_legacy_estimate_none_when_no_boundary_given():
    assert legacy_estimate((50.0, 1.0), None, style="formal") is None


def test_gt_estimate_used_within_capture_radius_ignored_outside_it():
    gt_points = [(10.0, 10.0)]
    near = gt_estimate((11.0, 10.0), gt_points, capture_radius=3.0)  # 1m away
    far = gt_estimate((20.0, 10.0), gt_points, capture_radius=3.0)  # 10m away
    assert near is not None
    assert near.x == pytest.approx(10.0) and near.y == pytest.approx(10.0)
    assert far is None  # doc: "carries no information outside it" -- not a huge-sigma estimate, no estimate at all


def test_gt_estimate_sigma_is_very_small():
    est = gt_estimate((10.5, 10.0), [(10.0, 10.0)], capture_radius=3.0)
    assert est.sigma < 0.2  # doc: "near-certain"


def _flat_sdf_raster(shape=(40, 40), boundary_row=20):
    """A synthetic SDF raster with a known, exact minimum: a horizontal
    boundary at pixel row `boundary_row`, distance in pixels (gsd=1) to it."""
    rows = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    sdf = np.abs(rows - boundary_row).astype(np.float32)
    log_var = np.full(shape, -2.0, dtype=np.float32)  # low variance = confident, everywhere
    transform = Affine.translation(0, 0) @ Affine.scale(1.0, -1.0)  # 1 px = 1 m, origin at world (0,0)
    return sdf, log_var, transform


def test_model_estimate_finds_the_local_sdf_minimum():
    sdf, log_var, transform = _flat_sdf_raster()
    # query near row 20 (the known boundary) but offset a few pixels off in x/y
    est = model_estimate((5.5, -18.5), sdf, log_var, transform, search_radius_px=10, max_sdf_px=5.0)
    assert est is not None
    assert est.y == pytest.approx(-20.5, abs=1e-6)  # exact pixel-centre of the boundary row


def test_model_estimate_none_when_nothing_nearby_within_max_sdf():
    sdf, log_var, transform = _flat_sdf_raster()
    # far from the boundary (row 20) and a tight search window -- no
    # in-window pixel is close enough to a predicted boundary
    est = model_estimate((5.5, -0.5), sdf, log_var, transform, search_radius_px=2, max_sdf_px=1.0)
    assert est is None


def test_fuse_node_no_sources_is_a_noop_returns_none():
    result = fuse_node((5.0, 5.0), face_ids=(1, 2))
    assert result is None


def test_fuse_node_single_source_fuses_to_it_directly():
    boundary = LineString([(0, 0), (100, 0)])
    result = fuse_node((50.0, 1.0), face_ids=(1, 2), legacy_boundary=boundary, style="formal")
    assert not isinstance(result, ConflictRecord)
    assert result.y == pytest.approx(0.0, abs=1e-6)


def test_fuse_node_agreeing_sources_fuse_not_conflict():
    boundary = LineString([(0, 0), (100, 0)])
    gt_points = [(50.0, 0.2)]  # agrees closely with the legacy line
    result = fuse_node(
        (50.0, 1.0), face_ids=(1, 2), legacy_boundary=boundary, style="formal", gt_points=gt_points, tolerance=3.0
    )
    assert not isinstance(result, ConflictRecord)


def test_fuse_node_disagreeing_sources_yield_conflict_not_average():
    boundary = LineString([(0, 0), (100, 0)])  # says y=0
    gt_points = [(50.0, 20.0)]  # says y=20 -- wildly disagrees
    result = fuse_node(
        (50.0, 1.0),
        face_ids=(3, 4),
        node_id=9,
        legacy_boundary=boundary,
        style="formal",
        gt_points=gt_points,
        capture_radius=25.0,
        tolerance=3.0,
        crs="EPSG:32643",
    )
    assert isinstance(result, ConflictRecord)
    assert result.node_id == 9
    assert result.face_ids == (3, 4)
    assert set(result.sources) == {"legacy", "gt"}
    assert result.disagreement_m == pytest.approx(20.0, abs=0.1)


def test_is_exterior_node_true_for_a_node_on_the_blocks_own_perimeter():
    # p1=[0,4]x[0,4], p2=[4,8]x[0,4] -- shares the x=4 edge. Every node
    # here (T-junctions and real corners alike) sits on the block's own
    # perimeter -- there is no genuinely interior node in a two-box block.
    crs = "EPSG:32643"
    graph = build_graph([Geom(box(0, 0, 4, 4), crs), Geom(box(4, 0, 8, 4), crs)], crs)
    assert all(_is_exterior_node(graph, nid) for nid in graph.nodes)


def test_is_exterior_node_false_for_a_genuinely_interior_node():
    # a 2x2 grid of boxes: the centre point (4,4), shared by all four, is
    # strictly inside the block -- none of its 4 incident edges touch OUTER.
    crs = "EPSG:32643"
    graph = build_graph(
        [Geom(box(0, 0, 4, 4), crs), Geom(box(4, 0, 8, 4), crs), Geom(box(0, 4, 4, 8), crs), Geom(box(4, 4, 8, 8), crs)],
        crs,
    )
    interior_node = next(nid for nid, n in graph.nodes.items() if (n.x, n.y) == (4.0, 4.0))
    assert not _is_exterior_node(graph, interior_node)
    others = [nid for nid in graph.nodes if nid != interior_node]
    assert all(_is_exterior_node(graph, nid) for nid in others)


def test_fuse_node_conflict_without_a_crs_raises_rather_than_faking_one():
    # invariant: never let a geometry cross a module boundary without a
    # declared CRS -- a conflict's geometry must not silently get a fake one
    boundary = LineString([(0, 0), (100, 0)])
    gt_points = [(50.0, 20.0)]
    with pytest.raises(ValueError):
        fuse_node(
            (50.0, 1.0),
            face_ids=(3, 4),
            legacy_boundary=boundary,
            style="formal",
            gt_points=gt_points,
            capture_radius=25.0,
            tolerance=3.0,
        )
