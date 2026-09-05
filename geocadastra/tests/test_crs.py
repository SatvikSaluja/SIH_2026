import pytest
from shapely.geometry import Point, Polygon, box

from geocadastra.core.crs import CRSMismatchError, Geom, everest_kalianpur_to_wgs84_utm, get_transformer


def test_geom_carries_its_crs():
    g = Geom(Point(1, 2), "EPSG:32643")
    assert g.crs == "EPSG:32643"
    assert g.geom.x == 1
    assert g.geom.y == 2


def test_same_crs_ops_work():
    a = Geom(box(0, 0, 10, 10), "EPSG:32643")
    b = Geom(box(5, 5, 15, 15), "EPSG:32643")

    inter = a.intersection(b)
    assert inter.crs == "EPSG:32643"
    assert inter.geom.area == 25.0

    union = a.union(b)
    assert union.crs == "EPSG:32643"
    assert union.geom.area == 175.0  # 100 + 100 - 25 overlap

    diff = a.difference(b)
    assert diff.crs == "EPSG:32643"
    assert diff.geom.area == 75.0  # a minus the overlap


@pytest.mark.parametrize("op", ["intersection", "union", "difference", "distance"])
def test_mismatched_crs_raises_instead_of_silently_coercing(op):
    a = Geom(box(0, 0, 10, 10), "EPSG:32643")
    b = Geom(box(5, 5, 15, 15), "EPSG:4326")
    with pytest.raises(CRSMismatchError):
        getattr(a, op)(b)


def test_transform_to_reprojects_and_updates_crs():
    # a point in WGS84 lon/lat near Nagpur, India (falls in UTM zone 43N / EPSG:32643)
    g = Geom(Point(79.08, 21.14), "EPSG:4326")
    reprojected = g.transform_to("EPSG:32643")
    assert reprojected.crs == "EPSG:32643"
    # UTM 43N easting/northing should be large positive metres, not lon/lat-scale
    assert reprojected.geom.x > 100_000
    assert reprojected.geom.y > 1_000_000


def test_transform_to_same_crs_is_a_cheap_noop():
    g = Geom(Point(1, 2), "EPSG:32643")
    same = g.transform_to("EPSG:32643")
    assert same.geom.x == 1 and same.geom.y == 2
    assert same.crs == "EPSG:32643"


def test_transformer_is_cached():
    t1 = get_transformer("EPSG:4326", "EPSG:32643")
    t2 = get_transformer("EPSG:4326", "EPSG:32643")
    assert t1 is t2


def test_everest_kalianpur_stub_documents_the_gap_instead_of_faking_it():
    with pytest.raises(NotImplementedError, match="(?i)geoid|orthometric|zone"):
        everest_kalianpur_to_wgs84_utm("EPSG:24379")


def test_geom_fields_cannot_be_reassigned():
    g = Geom(Point(1, 2), "EPSG:32643")
    with pytest.raises(Exception):
        g.crs = "EPSG:4326"


def test_geom_equality_and_hash_are_by_value_not_identity():
    """Two separately-built Geoms wrapping coordinate-equal geometries are
    `==` and hash-equal (ordinary frozen-dataclass behaviour, inherited from
    shapely's own value-based geometry equality) -- documented here because
    nothing in this codebase currently uses a Geom as a dict/set key, and
    anything that starts doing so should mean to collapse coincidentally-
    equal-but-logically-distinct geometries, not assume identity semantics."""
    a = Geom(Point(1, 2), "EPSG:32643")
    b = Geom(Point(1, 2), "EPSG:32643")
    assert a is not b
    assert a == b
    assert len({a, b}) == 1
