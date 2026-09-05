import pytest
from shapely.geometry import LineString, box

from geocadastra.core.blocks import build_blocks
from geocadastra.core.crs import Geom
from geocadastra.core.graph import OUTER
from geocadastra.synth.generator import WardParams, generate_ward

CRS = "EPSG:32643"


def test_no_internal_roads_gives_one_block_bordered_entirely_by_outer():
    ward = Geom(box(0, 0, 10, 10), CRS)
    g = build_blocks([], ward, CRS)
    assert len(g.faces) == 1
    for e in g.edges.values():
        assert OUTER in g.faces_of_edge(e.id)


def test_one_internal_road_splits_ward_into_two_blocks():
    ward = Geom(box(0, 0, 10, 10), CRS)
    road = [Geom(LineString([(5, 0), (5, 10)]), CRS)]
    g = build_blocks(road, ward, CRS)
    assert len(g.faces) == 2
    polys = g.faces_to_polygons()
    total_area = sum(p.area for p in polys.values())
    assert total_area == pytest.approx(100.0, abs=1e-6)

    divider = [e for e in g.edges.values() if {round(c[0]) for c in g.edge_linestring(e.id).coords} == {5}]
    assert len(divider) == 1
    assert OUTER not in g.faces_of_edge(divider[0].id)


def test_a_road_hugging_the_boundary_does_not_crash_or_leave_a_degenerate_sliver_block():
    """Exercises the degenerate-face filter (mirrors a Stage 0 hardening fix
    that Stage 1's version of this same representative-point filter had
    initially dropped): a road running a fraction of a grid cell from the
    ward edge must not produce a zero/near-zero-area phantom block."""
    ward = Geom(box(0, 0, 10, 10), CRS)
    road = [Geom(LineString([(0, 1e-4), (10, 1e-4)]), CRS)]
    g = build_blocks(road, ward, CRS)
    for f in g.faces_to_polygons().values():
        assert f.area > 1e-6
    assert sum(f.area for f in g.faces_to_polygons().values()) == pytest.approx(100.0, abs=1e-3)


def test_road_outside_the_ward_boundary_is_ignored():
    ward = Geom(box(0, 0, 10, 10), CRS)
    stray = [Geom(LineString([(20, 0), (20, 10)]), CRS)]
    g = build_blocks(stray, ward, CRS)
    assert len(g.faces) == 1
    assert g.faces_to_polygons()[0].area == pytest.approx(100.0)


@pytest.mark.parametrize("seed", [0, 1, 5, 42])
def test_matches_the_synthetic_generator_own_block_partition(seed):
    """Cross-check against Stage 0's own (independently written, already
    tested) road->block partitioning: same road network in, same block
    count and total area out."""
    ward = generate_ward(params=WardParams(n_arterial_h=2, n_arterial_v=2), seed=seed)
    road_linework = [Geom(ln, CRS) for ln in ward.roads_centerline]
    ward_geom = Geom(ward.ward_polygon, CRS)

    g = build_blocks(road_linework, ward_geom, CRS)

    assert len(g.faces) == len(ward.blocks)
    total_area = sum(p.area for p in g.faces_to_polygons().values())
    assert total_area == pytest.approx(ward.ward_polygon.area, rel=1e-6)


def test_every_block_edge_has_exactly_two_face_sides():
    ward = generate_ward(params=WardParams(n_arterial_h=1, n_arterial_v=1), seed=3)
    road_linework = [Geom(ln, CRS) for ln in ward.roads_centerline]
    g = build_blocks(road_linework, Geom(ward.ward_polygon, CRS), CRS)
    for e in g.edges.values():
        assert len(g.faces_of_edge(e.id)) == 2
