"""Stage 7: prioritisation and the headline metric.

The headline number for this project is surveyor-hours to certify 90% of
a ward, not IoU. These tests exercise the pieces that build toward that
curve: parcel adjacency from shared edges (reusing Stage 1's own
`PlanarGraph`), uncertainty weighted by centrality (a parcel anchoring
many uncertain neighbours should outrank an equally-uncertain isolated
one -- the doc's own explicit example), spatial clustering, cluster
ranking by value/cost, and the survey simulation itself.
"""
import numpy as np
import pytest
from shapely.geometry import box

from geocadastra.core.crs import Geom
from geocadastra.core.graph import build_graph
from geocadastra.core.priority import (
    CostModel,
    SurveyPoint,
    build_parcel_adjacency,
    build_priority_order,
    cluster_parcels,
    face_centroids,
    face_edge_ids,
    face_uncertainty,
    hours_to_reach,
    lowest_confidence_order,
    priority_score,
    random_order,
    rank_clusters,
    simulate_survey,
)

CRS = "EPSG:32643"


def _two_square_graph():
    # two 4x4 squares sharing the edge x=4 -- one interior adjacency
    return build_graph([Geom(box(0, 0, 4, 4), CRS), Geom(box(4, 0, 8, 4), CRS)], CRS)


def _three_in_a_row_graph():
    # three 4x4 squares in a row -- middle one adjacent to both ends, ends not adjacent to each other
    return build_graph(
        [Geom(box(0, 0, 4, 4), CRS), Geom(box(4, 0, 8, 4), CRS), Geom(box(8, 0, 12, 4), CRS)], CRS
    )


def test_build_parcel_adjacency_links_faces_sharing_an_edge():
    graph = _two_square_graph()
    adjacency = build_parcel_adjacency(graph)
    assert adjacency[0] == {1}
    assert adjacency[1] == {0}


def test_build_parcel_adjacency_excludes_the_outer_face_and_non_adjacent_faces():
    graph = _three_in_a_row_graph()
    adjacency = build_parcel_adjacency(graph)
    assert adjacency[0] == {1}  # not adjacent to face 2 -- no shared edge
    assert adjacency[1] == {0, 2}
    assert adjacency[2] == {1}
    assert all(-1 not in neighbours for neighbours in adjacency.values())  # OUTER never a "neighbour"


def test_build_parcel_adjacency_includes_isolated_faces_with_an_empty_set():
    graph = build_graph([Geom(box(0, 0, 4, 4), CRS)], CRS)
    adjacency = build_parcel_adjacency(graph)
    assert adjacency == {0: set()}


def test_face_uncertainty_takes_the_worst_boundary_edge_not_the_average():
    graph = _two_square_graph()
    face = graph.faces[0]
    edge_ids = [eid for eid, _ in face.boundary]
    edge_uncertainty = {eid: 1.0 for eid in edge_ids}
    edge_uncertainty[edge_ids[0]] = 9.0  # one bad edge among several good ones
    result = face_uncertainty(graph, edge_uncertainty)
    assert result[0] == pytest.approx(9.0)


def test_face_uncertainty_is_zero_for_a_face_with_no_measured_edges():
    graph = _two_square_graph()
    assert face_uncertainty(graph, {}) == {0: 0.0, 1: 0.0}


def test_priority_score_ranks_a_hub_above_an_equally_uncertain_isolated_parcel():
    """The doc's own explicit example: resolving one parcel that anchors
    many uncertain neighbours ranks above an isolated uncertain parcel."""
    adjacency = {
        "isolated": set(),
        "hub": {"n1", "n2", "n3"},
        "n1": {"hub"},
        "n2": {"hub"},
        "n3": {"hub"},
    }
    # isolated and hub share the exact same OWN uncertainty
    uncertainty = {"isolated": 5.0, "hub": 5.0, "n1": 8.0, "n2": 8.0, "n3": 8.0}
    scores = priority_score(adjacency, uncertainty)
    assert scores["hub"] > scores["isolated"]


def test_priority_score_formula_is_uncertainty_times_one_plus_neighbour_sum():
    adjacency = {"a": {"b", "c"}, "b": {"a"}, "c": {"a"}}
    uncertainty = {"a": 2.0, "b": 3.0, "c": 4.0}
    scores = priority_score(adjacency, uncertainty)
    assert scores["a"] == pytest.approx(2.0 * (1 + 3.0 + 4.0))


def test_cluster_parcels_groups_by_single_linkage_within_max_distance():
    centroids = {1: (0, 0), 2: (5, 0), 3: (10, 0), 4: (100, 100)}
    # 1-2 close, 2-3 close (chain), 4 far from everything
    clusters = cluster_parcels([1, 2, 3, 4], centroids, max_distance=6.0)
    clusters_as_sets = {frozenset(c) for c in clusters}
    assert frozenset({1, 2, 3}) in clusters_as_sets  # transitively merged via the 2-3 link
    assert frozenset({4}) in clusters_as_sets


def test_cluster_parcels_keeps_far_apart_parcels_separate():
    centroids = {1: (0, 0), 2: (1000, 0)}
    clusters = cluster_parcels([1, 2], centroids, max_distance=5.0)
    assert {frozenset(c) for c in clusters} == {frozenset({1}), frozenset({2})}


def test_rank_clusters_prefers_higher_value_over_cost():
    centroids = {1: (0, 0), 2: (0, 0), 3: (1000, 0)}
    scores = {1: 10.0, 2: 10.0, 3: 10.0}
    cost_model = CostModel(minutes_per_parcel=1.0, travel_speed_m_per_min=100.0)
    clusters = [[3], [1, 2]]  # far single parcel vs. a nearby pair with double the value
    ranked = rank_clusters(clusters, scores, centroids, depot=(0, 0), cost_model=cost_model)
    assert ranked[0] == [1, 2]


def test_random_order_is_a_permutation_of_singleton_groups():
    rng = np.random.default_rng(0)
    order = random_order([1, 2, 3, 4], rng)
    assert sorted(g[0] for g in order) == [1, 2, 3, 4]
    assert all(len(g) == 1 for g in order)


def test_lowest_confidence_order_sorts_most_uncertain_first():
    order = lowest_confidence_order({1: 2.0, 2: 9.0, 3: 5.0})
    assert order == [[2], [3], [1]]


def test_simulate_survey_prices_travel_and_fixed_time_correctly():
    # depot at origin, one parcel 100m away; speed 100 m/min, 5 min/parcel
    cost_model = CostModel(minutes_per_parcel=5.0, travel_speed_m_per_min=100.0)
    curve = simulate_survey(
        order=[[1]],
        adjacency={1: set(), 2: set()},
        edge_uncertainty={101: 10.0, 102: 0.0},  # parcel 2's edge already certified
        face_edges={1: (101,), 2: (102,)},
        centroids={1: (100, 0)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    assert curve[0] == SurveyPoint(hours=0.0, certified_fraction=0.5)  # only parcel 2 starts certified
    # travel 100m @ 100 m/min = 1 min, + 5 min on-site = 6 min = 0.1 hour
    assert curve[1].hours == pytest.approx(0.1)
    assert curve[1].certified_fraction == pytest.approx(1.0)


def test_simulate_survey_visiting_an_already_certified_parcel_is_idempotent():
    cost_model = CostModel(minutes_per_parcel=1.0, travel_speed_m_per_min=100.0)
    curve = simulate_survey(
        order=[[1]],
        adjacency={1: set()},
        edge_uncertainty={101: 0.0},  # already certified
        face_edges={1: (101,)},
        centroids={1: (0, 0)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    assert curve[0].certified_fraction == pytest.approx(1.0)
    assert curve[-1].certified_fraction == pytest.approx(1.0)


def test_simulate_survey_of_an_empty_ward_returns_no_curve():
    cost_model = CostModel(minutes_per_parcel=1.0, travel_speed_m_per_min=100.0)
    assert simulate_survey([], {}, {}, {}, {}, (0, 0), 1.0, cost_model) == []


def test_simulate_survey_treats_an_unmeasured_edge_as_zero_not_a_crash():
    """Same rule face_uncertainty() applies: an edge with no certified band
    yet contributes nothing, rather than KeyError-ing -- found by review:
    edge_uncertainty[eid] (no .get()) crashed on any face with an edge
    missing from edge_uncertainty, even before the survey loop started."""
    cost_model = CostModel(minutes_per_parcel=5.0, travel_speed_m_per_min=100.0)
    curve = simulate_survey(
        order=[[1]],
        adjacency={1: set()},
        edge_uncertainty={},  # edge 101 has no certified band at all
        face_edges={1: (101,)},
        centroids={1: (0, 0)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    assert curve[0].certified_fraction == pytest.approx(1.0)  # unmeasured -> treated as 0.0 -> already certified


def test_simulate_survey_prices_travel_sequentially_within_a_multi_member_group():
    """Travel must be priced hop-by-hop between actual visited stops, not
    once to the group's centroid with every other member free -- that bug
    let a single large cluster collapse to near-zero internal travel cost
    regardless of how spread out its members actually were."""
    cost_model = CostModel(minutes_per_parcel=0.0, travel_speed_m_per_min=100.0)
    curve = simulate_survey(
        order=[["a", "b"]],
        adjacency={"a": set(), "b": set()},
        edge_uncertainty={1: 9.0, 2: 9.0},
        face_edges={"a": (1,), "b": (2,)},
        centroids={"a": (100, 0), "b": (100, 100)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    # depot->a = 100m, a->b = 100m -> 200m total @ 100 m/min = 2 min = 2/60 hour
    # (NOT depot->group-centroid (100,50), which would be a shorter ~111.8m)
    assert curve[-1].hours == pytest.approx(200 / 100 / 60)


def test_simulate_survey_visiting_a_hub_certifies_its_neighbours_for_free():
    """The real payoff behind centrality-weighted scoring: verifying a
    parcel resolves its shared edges, which can certify an unvisited
    neighbour too if that shared edge was the neighbour's only problem."""
    cost_model = CostModel(minutes_per_parcel=1.0, travel_speed_m_per_min=1000.0)
    # hub shares edge 12 with n1 and edge 13 with n2; each neighbour's OWN
    # remaining edge (101/102) is already fine, so resolving edge 12/13
    # alone should certify both neighbours without a separate visit
    curve = simulate_survey(
        order=[["hub"]],
        adjacency={"hub": {"n1", "n2"}, "n1": {"hub"}, "n2": {"hub"}},
        edge_uncertainty={12: 9.0, 13: 9.0, 101: 0.0, 102: 0.0},
        face_edges={"hub": (12, 13), "n1": (12, 101), "n2": (13, 102)},
        centroids={"hub": (0, 0)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    assert curve[0].certified_fraction == pytest.approx(0.0)  # nothing certified yet
    assert curve[-1].certified_fraction == pytest.approx(1.0)  # hub visit certified all three
    # only one parcel's worth of on-site time was spent (n1/n2 were free)
    assert curve[-1].hours * 60 == pytest.approx(cost_model.minutes_per_parcel, abs=1e-6)


def test_simulate_survey_does_not_certify_a_neighbour_with_another_bad_edge():
    """Resolving the shared edge is not enough if the neighbour has a
    DIFFERENT bad edge of its own -- it still needs its own visit."""
    cost_model = CostModel(minutes_per_parcel=1.0, travel_speed_m_per_min=1000.0)
    curve = simulate_survey(
        order=[["hub"], ["n1"]],
        adjacency={"hub": {"n1"}, "n1": {"hub"}},
        edge_uncertainty={12: 9.0, 101: 9.0},  # n1's own outer edge 101 is ALSO bad
        face_edges={"hub": (12,), "n1": (12, 101)},
        centroids={"hub": (0, 0), "n1": (0, 0)},
        depot=(0, 0),
        tolerance=1.0,
        cost_model=cost_model,
    )
    assert curve[-1].certified_fraction == pytest.approx(1.0)
    # both parcels needed an actual visit -- 2 parcels' worth of on-site time
    assert curve[-1].hours * 60 == pytest.approx(2 * cost_model.minutes_per_parcel, abs=1e-6)


def test_hours_to_reach_finds_the_first_crossing():
    curve = [SurveyPoint(0.0, 0.5), SurveyPoint(1.0, 0.8), SurveyPoint(2.0, 0.95)]
    assert hours_to_reach(curve, 0.9) == pytest.approx(2.0)
    assert hours_to_reach(curve, 0.5) == pytest.approx(0.0)


def test_hours_to_reach_returns_none_when_the_target_is_never_reached():
    curve = [SurveyPoint(0.0, 0.1), SurveyPoint(1.0, 0.2)]
    assert hours_to_reach(curve, 0.9) is None


def test_face_centroids_matches_each_faces_polygon_centroid():
    graph = _two_square_graph()
    centroids = face_centroids(graph)
    assert centroids[0] == pytest.approx((2.0, 2.0))
    assert centroids[1] == pytest.approx((6.0, 2.0))


def test_build_priority_order_covers_every_parcel_above_tolerance_exactly_once():
    graph = _three_in_a_row_graph()
    edge_uncertainty = {eid: float(eid + 1) for eid in graph.edges}  # every edge measured, distinct values
    adjacency = build_parcel_adjacency(graph)
    uncertainty = face_uncertainty(graph, edge_uncertainty)
    centroids = face_centroids(graph)
    cost_model = CostModel(minutes_per_parcel=5.0, travel_speed_m_per_min=80.0)
    order = build_priority_order(
        adjacency, uncertainty, centroids, depot=(0, 0), cost_model=cost_model, cluster_distance=10.0, tolerance=0.0
    )
    visited = [pid for group in order for pid in group]
    assert sorted(visited) == sorted(graph.faces.keys())  # tolerance=0.0 -> every face needs a visit
    assert len(visited) == len(set(visited))  # no parcel surveyed twice


def test_build_priority_order_excludes_parcels_already_inside_tolerance():
    graph = _three_in_a_row_graph()
    edge_uncertainty = {eid: float(eid + 1) for eid in graph.edges}
    adjacency = build_parcel_adjacency(graph)
    uncertainty = face_uncertainty(graph, edge_uncertainty)
    centroids = face_centroids(graph)
    cost_model = CostModel(minutes_per_parcel=5.0, travel_speed_m_per_min=80.0)
    high_tolerance = max(uncertainty.values()) + 1.0  # every parcel already "certified"
    order = build_priority_order(
        adjacency, uncertainty, centroids, depot=(0, 0), cost_model=cost_model, cluster_distance=10.0,
        tolerance=high_tolerance,
    )
    assert order == []
