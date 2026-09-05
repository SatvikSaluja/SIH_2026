import pytest
from shapely.geometry import Polygon, box

from geocadastra.core.crs import Geom
from geocadastra.core.graph import OUTER, PlanarGraph, build_graph
from geocadastra.core.planarize import planarize


def _two_square_graph():
    """West [0,0]-[5,10] and east [5,0]-[10,10], sharing the x=5 divider."""
    west = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    east = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    return build_graph([Geom(west, "EPSG:32643"), Geom(east, "EPSG:32643")], "EPSG:32643")


def test_two_squares_produce_six_nodes_seven_edges_two_faces():
    g = _two_square_graph()
    assert len(g.nodes) == 6
    assert len(g.edges) == 7
    assert len(g.faces) == 2


def test_faces_to_polygons_reconstructs_the_originals_exactly():
    west = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    east = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    g = build_graph([Geom(west, "EPSG:32643"), Geom(east, "EPSG:32643")], "EPSG:32643")
    polys = g.faces_to_polygons()
    assert len(polys) == 2
    areas = sorted(p.area for p in polys.values())
    assert areas == pytest.approx([50.0, 50.0])
    union = list(polys.values())[0].union(list(polys.values())[1])
    assert union.area == pytest.approx(100.0)
    assert union.geom.equals(box(0, 0, 10, 10))


def test_shared_edge_is_referenced_by_exactly_the_two_real_faces():
    g = _two_square_graph()
    # find the divider edge: the one whose linestring is the vertical segment at x=5
    divider = [e for e in g.edges.values() if e.n0 is not None and _is_vertical_at(g, e, 5)]
    assert len(divider) == 1
    faces = g.faces_of_edge(divider[0].id)
    assert OUTER not in faces
    assert len(set(faces)) == 2


def _is_vertical_at(g, edge, x):
    ls = g.edge_linestring(edge.id)
    xs = {round(c[0], 6) for c in ls.coords}
    return xs == {x}


def test_boundary_edges_reference_exactly_one_real_face_and_the_outer_sentinel():
    g = _two_square_graph()
    divider_edges = {e.id for e in g.edges.values() if _is_vertical_at(g, e, 5)}
    for e in g.edges.values():
        faces = g.faces_of_edge(e.id)
        assert len(faces) == 2
        if e.id in divider_edges:
            assert OUTER not in faces
        else:
            assert OUTER in faces


def test_every_edge_is_referenced_by_exactly_two_face_sides():
    """The graph-wide invariant the build plan asks for, phrased to include
    the outer sentinel face so it holds with no exception for boundary edges."""
    g = _two_square_graph()
    for e in g.edges.values():
        assert len(g.faces_of_edge(e.id)) == 2


def test_moving_a_shared_node_changes_every_incident_face():
    g = _two_square_graph()
    node_at_5_0 = [n for n in g.nodes.values() if (n.x, n.y) == (5.0, 0.0)][0]
    before = {fid: poly.geom.wkt for fid, poly in g.faces_to_polygons().items()}

    g.move_node(node_at_5_0.id, 5.5, 0.0)

    after = {fid: poly.geom.wkt for fid, poly in g.faces_to_polygons().items()}
    assert before.keys() == after.keys()
    for fid in before:
        assert before[fid] != after[fid], f"face {fid} did not change after moving its node"


def test_moving_a_node_preserves_the_two_face_sides_invariant():
    g = _two_square_graph()
    node_at_5_0 = [n for n in g.nodes.values() if (n.x, n.y) == (5.0, 0.0)][0]
    g.move_node(node_at_5_0.id, 5.5, 0.5)
    for e in g.edges.values():
        assert len(g.faces_of_edge(e.id)) == 2


def test_build_graph_from_planarize_output_round_trips():
    linework = [
        Geom(box(0, 0, 10, 10).boundary, "EPSG:32643"),
    ]
    # add an internal divider as its own line
    from shapely.geometry import LineString

    linework.append(Geom(LineString([(5, 0), (5, 10)]), "EPSG:32643"))
    faces = planarize(linework)
    g = build_graph(faces, "EPSG:32643")
    assert len(g.faces) == 2
    total_area = sum(p.area for p in g.faces_to_polygons().values())
    assert total_area == pytest.approx(100.0, abs=1e-6)


def test_four_way_junction_node_is_shared_by_all_four_quadrant_faces():
    """A true degree->=4 junction (four blocks meeting at one point), not
    just the degree-3 T-junctions the other fixtures happen to exercise."""
    quads = [
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),
        Polygon([(0, 5), (5, 5), (5, 10), (0, 10)]),
    ]
    g = build_graph([Geom(q, "EPSG:32643") for q in quads], "EPSG:32643")
    center = next(n for n in g.nodes.values() if (n.x, n.y) == (5.0, 5.0))
    touching_faces = set()
    for fid, face in g.faces.items():
        node_ids = {g.edges[eid].n0 for eid, _ in face.boundary} | {g.edges[eid].n1 for eid, _ in face.boundary}
        if center.id in node_ids:
            touching_faces.add(fid)
    assert touching_faces == {0, 1, 2, 3}


def test_single_face_has_no_shared_edges():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    g = build_graph([Geom(square, "EPSG:32643")], "EPSG:32643")
    assert len(g.faces) == 1
    assert len(g.nodes) == 4
    assert len(g.edges) == 4
    for e in g.edges.values():
        assert OUTER in g.faces_of_edge(e.id)


def test_move_node_rejects_an_unknown_node_id():
    g = _two_square_graph()
    with pytest.raises(KeyError):
        g.move_node(9999, 1.0, 1.0)
    assert len(g.nodes) == 6  # no phantom node inserted


def test_move_node_updates_the_point_index_so_a_stale_lookup_does_not_misroute():
    """Regression: move_node left the old-coordinate->node_id index entry in
    place, so re-deriving "the node at this exact old point" would silently
    return the now-relocated node instead of correctly finding nothing."""
    g = _two_square_graph()
    node = next(n for n in g.nodes.values() if (n.x, n.y) == (0.0, 0.0))
    g.move_node(node.id, 50.0, 50.0)
    assert g._get_or_add_node((0.0, 0.0)) != node.id
    assert g._get_or_add_node((50.0, 50.0)) == node.id


def test_build_graph_rejects_the_same_face_passed_twice():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    with pytest.raises(ValueError):
        build_graph([Geom(square, "EPSG:32643"), Geom(square, "EPSG:32643")], "EPSG:32643")


def test_collinear_tol_controls_whether_a_realistically_jittered_point_dissolves():
    """Regression: the old fixed eps=1e-7 normalized-angle test was 100x-
    20000x too tight at realistic grid-snap-noise levels, so genuinely
    redundant shape points almost never dissolved in practice."""
    p_mid_jittered = (5.0, 0.0005)  # 0.5mm perpendicular offset -- realistic 1mm-grid noise
    poly = Polygon([(0.0, 0.0), *[p_mid_jittered], (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    loose = build_graph([Geom(poly, "EPSG:32643")], "EPSG:32643", collinear_tol=1e-3)
    assert len(loose.nodes) == 4  # dissolved: tolerant of grid-snap-scale noise

    tight = build_graph([Geom(poly, "EPSG:32643")], "EPSG:32643", collinear_tol=1e-9)
    assert len(tight.nodes) == 5  # kept: tol smaller than the jitter, so it's a "real" corner


def test_a_face_with_a_hole_fails_loudly_instead_of_silently_wrong():
    donut = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]])
    with pytest.raises(NotImplementedError):
        build_graph([Geom(donut, "EPSG:32643")], "EPSG:32643")


def test_degree_two_points_are_dissolved_into_one_multi_vertex_edge():
    # a single face whose ring has an extra collinear-ish vertex partway
    # along one side -- that vertex has degree 2 and should not become a
    # graph node; it should end up as an interior coordinate of one edge.
    square = Polygon([(0, 0), (10, 0), (10, 5), (10, 10), (0, 10)])
    g = build_graph([Geom(square, "EPSG:32643")], "EPSG:32643")
    assert len(g.nodes) == 4  # not 5 -- (10,5) is degree-2, dissolved
    assert len(g.edges) == 4
    right_edge = [e for e in g.edges.values() if _is_vertical_at(g, e, 10)][0]
    assert len(right_edge.coords) == 1  # the dissolved (10,5) survives as interior shape
