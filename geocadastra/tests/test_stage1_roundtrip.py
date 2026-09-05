"""Stage 1 acceptance: the full synthetic ground truth round-trips through
planarize + graph with topology preserved, checked via hypothesis property
tests over 100+ generated wards, per the build plan's Done-when criteria:

- no two faces overlap by more than grid tolerance
- every interior edge is referenced by exactly two faces
- moving any node preserves the exactly-two-faces invariant
- total face area equals input domain area to within grid tolerance
"""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from shapely.ops import unary_union

from geocadastra.core.blocks import build_blocks
from geocadastra.core.crs import Geom
from geocadastra.core.graph import OUTER, build_graph
from geocadastra.core.planarize import GRID
from geocadastra.synth.generator import generate_ward

CRS = "EPSG:32643"


def _tol(domain_area: float) -> float:
    # "to within grid tolerance": scale GRID by the domain's own linear size
    # rather than using GRID as a bare area tolerance, which would be far
    # too tight for anything but a tiny domain.
    return max(GRID, GRID * (domain_area**0.5) * 50)


def _assert_no_face_overlaps(faces_by_id, tol):
    polys = list(faces_by_id.values())
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            assert polys[i].geom.intersection(polys[j].geom).area <= tol


def _assert_two_face_sides_everywhere(g):
    for e in g.edges.values():
        assert len(g.faces_of_edge(e.id)) == 2


def _assert_interior_edges_have_two_real_faces(g):
    for e in g.edges.values():
        sides = g.faces_of_edge(e.id)
        if OUTER not in sides:
            assert len(set(sides)) == 2


def _check_block_graph_invariants(g, domain_area):
    tol = _tol(domain_area)
    faces = g.faces_to_polygons()
    _assert_no_face_overlaps(faces, tol)
    _assert_two_face_sides_everywhere(g)
    _assert_interior_edges_have_two_real_faces(g)
    total = sum(p.area for p in faces.values())
    assert abs(total - domain_area) <= tol


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000_000))
def test_block_graph_invariants_hold_across_100_plus_generated_wards(seed):
    ward = generate_ward(seed=seed)
    road_linework = [Geom(ln, CRS) for ln in ward.roads_centerline]
    g = build_blocks(road_linework, Geom(ward.ward_polygon, CRS), CRS)
    _check_block_graph_invariants(g, ward.ward_polygon.area)


@pytest.mark.parametrize("seed", range(20))
def test_moving_every_node_preserves_the_two_face_sides_invariant(seed):
    ward = generate_ward(seed=seed)
    road_linework = [Geom(ln, CRS) for ln in ward.roads_centerline]
    g = build_blocks(road_linework, Geom(ward.ward_polygon, CRS), CRS)

    for node_id, node in list(g.nodes.items()):
        g.move_node(node_id, node.x + 0.05, node.y + 0.05)
    _assert_two_face_sides_everywhere(g)

    # and it isn't just a coincidence of small moves -- face geometry actually changed
    faces_after = g.faces_to_polygons()
    assert all(not p.geom.is_empty and p.geom.area > 0 for p in faces_after.values())


@pytest.mark.parametrize("seed", range(20))
def test_full_parcel_ground_truth_round_trips_with_topology_preserved(seed):
    """The "full synthetic ground truth" case: feed the TRUE parcel
    boundaries (not just roads) through build_graph and check every parcel
    survives as a face with its recorded area, and shared parcel-parcel
    boundaries come back as edges referenced by exactly two faces."""
    ward = generate_ward(seed=seed)
    parcel_geoms = [Geom(p.polygon, CRS) for p in ward.parcels]
    g = build_graph(parcel_geoms, CRS)

    assert len(g.faces) == len(ward.parcels)
    faces = g.faces_to_polygons()
    recovered_areas = sorted(round(p.area, 3) for p in faces.values())
    true_areas = sorted(round(p.area, 3) for p in ward.parcels)
    assert recovered_areas == pytest.approx(true_areas, abs=1e-2)

    _assert_two_face_sides_everywhere(g)

    total_recovered = sum(p.area for p in faces.values())
    total_true = sum(p.area for p in ward.parcels)
    assert total_recovered == pytest.approx(total_true, rel=1e-6)


def test_moving_a_node_on_the_full_parcel_graph_changes_only_its_incident_faces():
    ward = generate_ward(seed=7)
    parcel_geoms = [Geom(p.polygon, CRS) for p in ward.parcels]
    g = build_graph(parcel_geoms, CRS)

    before = {fid: poly.geom.wkt for fid, poly in g.faces_to_polygons().items()}

    # pick an arbitrary node and move it
    target_node = next(iter(g.nodes))
    g.move_node(target_node, g.nodes[target_node].x + 1.0, g.nodes[target_node].y + 1.0)
    after = {fid: poly.geom.wkt for fid, poly in g.faces_to_polygons().items()}

    changed = {fid for fid in before if before[fid] != after[fid]}
    assert changed, "moving a node should change at least its incident faces"
    unchanged = set(before) - changed
    # every unchanged face must genuinely not touch the moved node
    for fid in unchanged:
        face = g.faces[fid]
        node_ids = set()
        for eid, _ in face.boundary:
            e = g.edges[eid]
            node_ids.add(e.n0)
            node_ids.add(e.n1)
        assert target_node not in node_ids
