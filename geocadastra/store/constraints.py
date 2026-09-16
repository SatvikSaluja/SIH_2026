"""Recorded-area accounting by durable parcel identity, never face order.

An existing approximation may be improved incrementally, but edits must not
worsen its area discrepancy or move a compliant parcel outside tolerance.
Passing this check is geometric readiness, not boundary certification.
"""
from collections import defaultdict
import math

from sqlalchemy import select
from geoalchemy2.shape import to_shape
from shapely import union_all
from shapely.errors import GEOSException
from geocadastra.core.planarize import GRID

from geocadastra.store.schema import RecordedParcel, IngestedBlock

BLOCK_AREA_TOLERANCE_M2 = 0.01


def block_coverage_error(session, block_id, polygons):
    block = session.scalar(select(IngestedBlock).where(IngestedBlock.block_id == block_id))
    if block is None:
        return None
    union = union_all([g.geom for g in polygons.values()], grid_size=GRID)
    return union.symmetric_difference(to_shape(block.geom), grid_size=GRID).area


def parcel_area_report(session, block_id, graph):
    records = session.scalars(select(RecordedParcel).where(RecordedParcel.block_id == block_id)).all()
    areas, face_ids = defaultdict(float), defaultdict(list)
    for fid, polygon in graph.faces_to_polygons().items():
        pid = graph.face_parcel_ids.get(fid)
        if pid is not None:
            areas[pid] += polygon.area
            face_ids[pid].append(fid)
    rows = []
    for record in records:
        error = areas[record.id] - record.area
        rows.append({"parcel_id": record.id, "style": record.style,
                     "recorded_area_m2": record.area, "predicted_area_m2": areas[record.id],
                     "error_m2": error, "tolerance_m2": record.area_tolerance_m2,
                     "face_ids": sorted(face_ids[record.id]),
                     "within_tolerance": bool(face_ids[record.id]) and abs(error) <= record.area_tolerance_m2})
    unknown = sorted(set(graph.faces) - set(graph.face_parcel_ids))
    try:
        coverage_error = block_coverage_error(session, block_id, graph.faces_to_polygons())
    except GEOSException:
        coverage_error = None  # an indeterminate overlay is never ready
    return {"parcels": rows, "unassigned_face_ids": unknown,
            "block_coverage_error_m2": coverage_error,
            "block_coverage_tolerance_m2": BLOCK_AREA_TOLERANCE_M2,
            "constraints_satisfied": bool(rows) and not unknown and all(r["within_tolerance"] for r in rows)
                                     and coverage_error is not None and coverage_error <= BLOCK_AREA_TOLERANCE_M2}


def validate_area_change(session, block_id, graph, before_polygons, after_polygons, touched_faces):
    records = {r.id: r for r in session.scalars(select(RecordedParcel).where(RecordedParcel.block_id == block_id))}
    if not records:
        return  # geometry-only stores have no recorded-area contract
    try:
        old_coverage = block_coverage_error(session, block_id, before_polygons)
        new_coverage = block_coverage_error(session, block_id, after_polygons)
    except GEOSException as e:
        raise ValueError("block coverage cannot be verified for this edit") from e
    if old_coverage is not None and new_coverage > max(old_coverage, BLOCK_AREA_TOLERANCE_M2) + 1e-8:
        raise ValueError(f"block coverage constraint refused edit ({old_coverage:.6f} -> {new_coverage:.6f} m²)")
    if any(fid not in graph.face_parcel_ids for fid in touched_faces):
        raise ValueError("edit touches an unassigned face; resolve parcel identity first")
    touched_parcels = {graph.face_parcel_ids[fid] for fid in touched_faces}
    for pid in touched_parcels:
        record = records[pid]
        ids = [fid for fid, parcel_id in graph.face_parcel_ids.items() if parcel_id == pid]
        old_area = sum(before_polygons[fid].area for fid in ids)
        new_area = sum(after_polygons[fid].area for fid in ids)
        old_error, new_error = abs(old_area - record.area), abs(new_area - record.area)
        tolerance = record.area_tolerance_m2
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"parcel {pid}: invalid recorded-area tolerance")
        # No implicit 5% or superpixel allowance. Allow recovery from an
        # already flagged initial discrepancy, not its further deterioration.
        if new_error > tolerance and new_error > old_error + 1e-8:
            raise ValueError(f"parcel {pid}: recorded-area constraint refused edit "
                             f"(error {old_error:.6f} -> {new_error:.6f} m²; tolerance {tolerance} m²)")
