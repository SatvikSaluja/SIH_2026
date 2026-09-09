"""Evaluation harness metrics -- stratified reporting is the whole point
(doc: "a mean number across formal and informal blocks hides the only
failure mode that matters"), so every function here takes one already-
filtered population and returns one number for it, never pools strata
internally.
"""
import pytest
from shapely.geometry import box

from geocadastra.core.crs import Geom
from geocadastra.core.evaluate import (
    area_error_distribution,
    boundary_position_error,
    parcel_count_error,
    topology_validity_rate,
)
from geocadastra.core.graph import build_graph
from geocadastra.synth.generator import GTPoint

CRS = "EPSG:32643"


def _two_square_graph():
    return build_graph([Geom(box(0, 0, 4, 4), CRS), Geom(box(4, 0, 8, 4), CRS)], CRS)


def test_boundary_position_error_is_zero_for_a_gt_point_exactly_on_an_edge():
    graph = _two_square_graph()
    result = boundary_position_error([GTPoint(x=4.0, y=2.0, parcel_id=0)], graph)  # on the shared edge
    assert result["p50"] == pytest.approx(0.0)
    assert result["n"] == 1


def test_boundary_position_error_measures_distance_to_nearest_edge_not_nearest_vertex():
    graph = _two_square_graph()
    # (4, 2) is on the shared edge (x=4) but 2m from the nearest VERTEX (4,0)/(4,4)
    result = boundary_position_error([(2.0, 4.0)], graph)  # on the top edge, y=4
    assert result["p50"] == pytest.approx(0.0)


def test_boundary_position_error_reports_p50_and_p90_across_points():
    graph = _two_square_graph()
    points = [(4.0, 2.0), (4.0, 2.0), (5.0, 2.0)]  # two exact on the shared edge, one 1m off it
    result = boundary_position_error(points, graph)
    assert result["p50"] == pytest.approx(0.0)
    assert result["p90"] == pytest.approx(0.8)  # 90th percentile of [0, 0, 1.0]


def test_boundary_position_error_with_no_gt_points_returns_none_not_zero():
    graph = _two_square_graph()
    result = boundary_position_error([], graph)
    assert result == {"p50": None, "p90": None, "n": 0}


def test_topology_validity_rate_is_1_for_two_clean_non_overlapping_faces():
    graph = _two_square_graph()
    result = topology_validity_rate(graph)
    assert result["rate"] == pytest.approx(1.0)
    assert result["n_faces"] == 2
    assert result["n_invalid"] == 0
    assert result["n_overlapping_pairs"] == 0


def test_topology_validity_rate_with_no_faces_returns_none_not_zero():
    from geocadastra.core.graph import PlanarGraph

    result = topology_validity_rate(PlanarGraph(crs=CRS))
    assert result == {"rate": None, "n_faces": 0, "n_invalid": 0, "n_overlapping_pairs": 0, "n_unchecked_pairs": 0}


def test_topology_validity_rate_detects_a_real_overlap():
    """Fabricate a PlanarGraph with two genuinely overlapping faces directly
    via the graph's own low-level node/edge/face API -- build_graph() itself
    would never produce this (it's designed to make it structurally
    impossible), so this is the only way to exercise the detector at all."""
    from geocadastra.core.graph import Face, PlanarGraph

    g = PlanarGraph(crs=CRS)
    coords1 = [(0, 0), (4, 0), (4, 4), (0, 4)]
    coords2 = [(2, 2), (6, 2), (6, 6), (2, 6)]  # overlaps coords1 by a 2x2 square
    n1 = [g._get_or_add_node(p) for p in coords1]
    e1 = [g._add_edge(n1[i], n1[(i + 1) % 4], []) for i in range(4)]
    n2 = [g._get_or_add_node(p) for p in coords2]
    e2 = [g._add_edge(n2[i], n2[(i + 1) % 4], []) for i in range(4)]
    g.faces[0] = Face(0, tuple((eid, True) for eid in e1))
    g.faces[1] = Face(1, tuple((eid, True) for eid in e2))

    result = topology_validity_rate(g)
    assert result["n_overlapping_pairs"] == 1
    assert result["rate"] == pytest.approx(0.0)  # both faces are in the one bad pair


def test_parcel_count_error_reports_over_and_under_segmentation_separately():
    result = parcel_count_error(predicted_count=12, recorded_count=10)
    assert result == {"predicted": 12, "recorded": 10, "over_segmentation": 2, "under_segmentation": 0}
    result2 = parcel_count_error(predicted_count=7, recorded_count=10)
    assert result2 == {"predicted": 7, "recorded": 10, "over_segmentation": 0, "under_segmentation": 3}


def test_area_error_distribution_uses_relative_not_absolute_error():
    # parcel A: 100 vs 110 (10% off); parcel B: 10 vs 11 (also 10% off) --
    # a relative metric treats these the same; an absolute one wouldn't
    result = area_error_distribution(predicted_areas=[110.0, 11.0], recorded_areas=[100.0, 10.0])
    assert result["p50"] == pytest.approx(0.1)
    assert result["p90"] == pytest.approx(0.1)


def test_area_error_distribution_with_no_parcels_returns_none_not_zero():
    assert area_error_distribution([], []) == {"p50": None, "p90": None, "n": 0}
