"""Map serialization and deterministic checks; all measurements use stored metres."""
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform
from shapely.strtree import STRtree
from shapely.validation import explain_validity


def map_feature(face, geometry, *, crs, synthetic):
    displayed = geometry if synthetic else transform(Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform, geometry)
    return {'type': 'Feature', 'id': str(face.id), 'geometry': mapping(displayed),
            'properties': {'face_id': face.id, 'block_id': face.block_id,
                           'parcel_id': face.recorded_parcel_id, 'area_m2': geometry.area,
                           'updated_at': face.computed_at.isoformat()}}


def validate_faces(features):
    """Detect invalid polygons and positive-area overlaps, not invented risk scores."""
    issues = []
    valid = []
    for face, geom in features:
        if geom.is_empty or not geom.is_valid:
            issues.append({'kind': 'invalid_geometry', 'face_ids': [face.id],
                           'block_id': face.block_id, 'detail': explain_validity(geom)})
        else:
            valid.append((face, geom))
    tree = STRtree([g for _, g in valid])
    for i, (face, geom) in enumerate(valid):
        for j in tree.query(geom, predicate='intersects'):
            if j <= i:
                continue
            other, other_geom = valid[j]
            area = geom.intersection(other_geom).area
            if area > 1e-8:
                issues.append({'kind': 'overlap', 'face_ids': [face.id, other.id],
                               'block_id': face.block_id, 'area_m2': area,
                               'detail': 'Positive-area intersection between stored parcel faces'})
    return issues
