"""Read-side ward views shared by main.py's own ward endpoints and
console.py's compatibility surface.

Same split deps.py already makes for the request dependencies, and for the
same reason: console.py needs exactly the facts `/wards`, `/wards/{id}/
geometry` and `/wards/{id}/topology` already compute, and having it import
them FROM main.py while main.py imports console.py to mount its router is a
true circular import. Both import this instead; neither imports the other.

The stronger reason is correctness, not import mechanics: a ward's parcel
count and area are derived, not stored (see WardJob's own docstring on why
its status is derived too). Computing them a second time in the console
layer is exactly the "two places that can silently disagree" this project
keeps refusing elsewhere -- there is one query behind each fact here.
"""
from __future__ import annotations

from geoalchemy2.shape import to_shape
from sqlalchemy import func, select

from geocadastra.api.geometry import map_feature, validate_faces
from geocadastra.jobs.orchestrator import ward_status
from geocadastra.store.changeset import load_block_graph
from geocadastra.store.schema import BlockJob, Face, IngestedBlock, WardJob


def ward_faces(session, ward_job_id):
    """Every face in a ward, via its blocks -- `Face` has no ward column."""
    return (
        select(Face)
        .join(IngestedBlock, IngestedBlock.block_id == Face.block_id)
        .where(IngestedBlock.ward_job_id == ward_job_id)
    )


def ward_summaries(session) -> list[dict]:
    """One row per ward job, with block/parcel counts and area read from
    the same live state `/wards/{id}/status` reports, not from a cache."""
    jobs = session.execute(select(WardJob).order_by(WardJob.id)).scalars().all()
    summaries = []
    for job in jobs:
        state, blocks = ward_status(session, job.id)
        n_parcels = session.execute(
            select(func.count()).select_from(Face)
            .join(IngestedBlock, IngestedBlock.block_id == Face.block_id)
            .where(IngestedBlock.ward_job_id == job.id)
        ).scalar_one()
        area_sqkm = float(session.execute(
            select(func.coalesce(func.sum(func.ST_Area(IngestedBlock.geom)), 0))
            .where(IngestedBlock.ward_job_id == job.id)
        ).scalar_one()) / 1e6
        # WardJob has no updated_at of its own. Its blocks do, and they are
        # what actually changes as processing runs, so the latest of those is
        # the ward's real last activity -- not created_at relabelled.
        last_block_change = session.execute(
            select(func.max(BlockJob.updated_at)).where(BlockJob.ward_job_id == job.id)
        ).scalar_one()
        summaries.append({
            "updated_at": (last_block_change or job.created_at).isoformat(),
            "ward_job_id": job.id, "source": job.source, "status": state,
            "n_blocks": len(blocks), "n_parcels": int(n_parcels),
            "completed_blocks": sum(b["status"] == "done" for b in blocks),
            "crs": job.crs, "synthetic": job.source.startswith("synthetic:"),
            "area_sqkm": area_sqkm, "created_at": job.created_at.isoformat(),
        })
    return summaries


def is_synthetic(job: WardJob) -> bool:
    return job.source.startswith("synthetic:")


def coordinate_system(job: WardJob) -> str:
    """Synthetic wards are generated in a local metric frame that has no
    geographic position; only a real ingest carries a CRS worth projecting
    to WGS84. Callers that export geographic formats must check this."""
    return "LOCAL_METRES" if is_synthetic(job) else "EPSG:4326"


def ward_feature_collection(session, job: WardJob, *, offset: int = 0, limit: int | None = None) -> dict:
    """Face geometry as GeoJSON. `limit=None` means the whole ward in one
    pass -- for in-process callers that would otherwise page through their
    own HTTP endpoint just to reassemble what one query already returns."""
    synthetic = is_synthetic(job)
    query = ward_faces(session, job.id).order_by(Face.id).offset(offset)
    rows = session.execute(query.limit(limit + 1) if limit is not None else query).scalars().all()
    page = rows[:limit] if limit is not None else rows
    return {
        "type": "FeatureCollection",
        "coordinate_system": coordinate_system(job),
        "source_crs": job.crs,
        "synthetic": synthetic,
        "next_offset": offset + limit if limit is not None and len(rows) > limit else None,
        "features": [map_feature(f, to_shape(f.geom), crs=job.crs, synthetic=synthetic) for f in page],
    }


def ward_topology_report(session, job: WardJob) -> dict:
    """Polygon validity and positive-area overlap only -- the report is
    explicit that gaps, sliver classification, survey accuracy and source
    conflicts are NOT certified by these checks."""
    state, blocks = ward_status(session, job.id)
    rows = session.execute(ward_faces(session, job.id).order_by(Face.id)).scalars().all()
    issues = validate_faces([(f, to_shape(f.geom)) for f in rows])
    nodes = []
    for block in blocks:
        if block["status"] == "done":
            graph = load_block_graph(session, block["block_id"])
            nodes.extend({"block_id": block["block_id"], "node_id": n.id, "x": n.x, "y": n.y}
                         for n in graph.nodes.values())
    return {
        "ward_job_id": job.id,
        "status": "not_scanned" if state != "done" or not rows else "attention" if issues else "valid_for_checks",
        "checked_faces": len(rows),
        "checks": ["polygon_validity", "positive_area_overlap"],
        "limitations": ["Gaps, sliver classification, survey accuracy and source conflicts are not certified by these checks."],
        "issues": issues,
        "nodes": nodes,
        "source_crs": job.crs,
        "synthetic": is_synthetic(job),
    }
