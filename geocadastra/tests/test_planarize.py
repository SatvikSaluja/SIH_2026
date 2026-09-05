import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.planarize import GRID, planarize


def _lines(coords_list, crs="EPSG:32643"):
    return [Geom(LineString(c), crs) for c in coords_list]


def test_empty_input_gives_empty_output():
    assert planarize([]) == []


def test_simple_grid_polygonizes_into_two_clean_rectangles():
    linework = _lines([
        [(0, 0), (10, 0)],
        [(10, 0), (10, 10)],
        [(10, 10), (0, 10)],
        [(0, 10), (0, 0)],
        [(5, 0), (5, 10)],
    ])
    faces = planarize(linework)
    assert len(faces) == 2
    areas = sorted(f.area for f in faces)
    assert areas == pytest.approx([50.0, 50.0])
    for f in faces:
        assert f.crs == "EPSG:32643"
        assert isinstance(f.geom, Polygon)

    # exact tiling: faces don't overlap and cover the domain
    union_area = faces[0].geom.union(faces[1].geom).area
    assert union_area == pytest.approx(100.0)
    assert faces[0].geom.intersection(faces[1].geom).area < GRID


def test_mismatched_crs_in_input_linework_raises():
    a = Geom(LineString([(0, 0), (10, 0)]), "EPSG:32643")
    b = Geom(LineString([(10, 0), (10, 10)]), "EPSG:4326")
    with pytest.raises(CRSMismatchError):
        planarize([a, b])


def test_deterministic_for_identical_input():
    linework = _lines([
        [(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)],
        [(3, 0), (3, 10)], [(7, 0), (7, 10)],
    ])
    a = planarize(linework)
    b = planarize(linework)
    assert [f.geom.wkt for f in a] == [f.geom.wkt for f in b]


def test_adaptive_presnap_merges_near_coincident_vertices_in_a_sparse_region():
    # four arms meeting near a common centre point in a very sparse (40-unit)
    # layout, each off from true centre by ~0.1-0.2m. Without pre-snapping
    # this is a mess of slivers/gaps at the centre instead of 4 clean
    # quadrants; the adaptive tolerance (proportional to local spacing, here
    # ~20-40m) should comfortably merge offsets this small.
    linework = _lines([
        [(0, 20), (19.85, 20.0)],
        [(20.15, 19.9), (40, 20)],
        [(20, 0), (20.0, 19.95)],
        [(20.05, 20.1), (20, 40)],
        [(0, 0), (40, 0)], [(40, 0), (40, 40)], [(40, 40), (0, 40)], [(0, 40), (0, 0)],
    ])
    faces = planarize(linework)
    assert len(faces) == 4
    areas = sorted(f.area for f in faces)
    assert areas == pytest.approx([400.0] * 4, abs=2.0)


def test_grid_snap_makes_two_nearly_identical_wards_agree_on_shared_boundary():
    """The core promise of the fixed-precision stage: coordinates that differ
    only by float noise (e.g. from two different construction paths) collapse
    to the identical grid point, so downstream code can match edges by exact
    coordinate equality."""
    eps = GRID / 10
    a = _lines([[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)], [(5, 0), (5, 10)]])
    b = _lines([[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)],
                [(5 + eps, 0), (5 - eps, 10)]])
    fa = planarize(a)
    fb = planarize(b)
    wkt_a = sorted(f.geom.wkt for f in fa)
    wkt_b = sorted(f.geom.wkt for f in fb)
    assert wkt_a == wkt_b


def test_presnap_does_not_chain_merge_points_beyond_their_own_tolerance():
    """Regression: single-linkage clustering used to fuse A and C onto one
    point through an intermediate B, even though A-C exceeded every
    involved point's own tolerance. Confirmed in review to move a vertex
    over 20m on realistic recursively-split input."""
    from geocadastra.core.planarize import _adaptive_presnap

    line_a = LineString([(0.0, 0.0), (0.0, -5.0)])
    line_b = LineString([(0.9, 0.0), (0.9, 5.0)])
    line_c = LineString([(1.8, 0.0), (1.8, -5.0)])
    out = _adaptive_presnap([line_a, line_b, line_c])
    a0, c0 = out[0].coords[0], out[2].coords[0]
    dist = ((a0[0] - c0[0]) ** 2 + (a0[1] - c0[1]) ** 2) ** 0.5
    assert dist > 1.0, f"A and C collapsed onto (nearly) the same point: {a0} vs {c0}"


def test_presnap_dedup_is_not_fooled_by_utm_scale_coordinates():
    """Regression: the duplicate-point dedup used plain np.allclose, whose
    default rtol scales the effective tolerance with coordinate magnitude --
    at this project's real target CRS (~2.3M-metre UTM northings) that
    inflated a 1mm tolerance to ~23m, silently dropping real vertices."""
    from geocadastra.core.planarize import _adaptive_presnap

    base = 2_300_000.0  # realistic EPSG:32643 magnitude
    ln = LineString([(base, 0), (base + 0.03, 0), (base + 2.03, 0)])
    out = _adaptive_presnap([ln])
    xs = [c[0] for c in out[0].coords]
    assert len(xs) == 2, f"expected the near pair merged and the far point kept, got {xs}"
    assert xs[-1] == pytest.approx(base + 2.03, abs=1e-6)


def test_multilinestring_input_is_flattened_not_crashed_on():
    from shapely.geometry import MultiLineString

    ward = Geom(box(0, 0, 20, 20), "EPSG:32643")
    divided_road = Geom(MultiLineString([[(5, 0), (5, 9)], [(5, 11), (5, 20)]]), "EPSG:32643")
    faces = planarize([divided_road], fixed=[Geom(ward.geom.boundary, "EPSG:32643")])
    assert len(faces) == 2


def test_empty_geometry_in_linework_is_ignored_not_crashed_on():
    a = Geom(LineString([(0, 0), (10, 0)]), "EPSG:32643")
    b = Geom(LineString([(10, 0), (10, 10)]), "EPSG:32643")
    empty = Geom(LineString(), "EPSG:32643")
    # should not raise
    planarize([a, b, empty])


def test_fixed_linework_never_moves_even_when_ordinary_linework_is_near_it():
    """Regression: adaptive pre-snap used to treat the ward boundary as
    ordinary movable linework, letting a nearby-but-unrelated road vertex
    drag an authoritative corner off its true position."""
    ward_line = Geom(box(0, 0, 10, 10).boundary, "EPSG:32643")
    road = Geom(LineString([(10.008, 0.008), (10.008, 5)]), "EPSG:32643")
    faces = planarize([road], fixed=[ward_line])
    all_coords = {tuple(round(v, 6) for v in c) for f in faces for c in f.geom.exterior.coords}
    assert (10.0, 0.0) in all_coords


def test_disjoint_squares_produce_two_separate_faces_not_merged():
    linework = _lines([
        [(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)],
        [(100, 100), (110, 100)], [(110, 100), (110, 110)], [(110, 110), (100, 110)], [(100, 110), (100, 100)],
    ])
    faces = planarize(linework)
    assert len(faces) == 2
    assert sorted(f.area for f in faces) == pytest.approx([100.0, 100.0])
