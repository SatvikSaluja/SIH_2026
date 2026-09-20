"""Compatibility surface for the bhoomi-ai console: the region / parcel /
processing-run vocabulary that UI was built against, served directly from
the ward / block / face model this backend actually stores.

This is a translation layer, and it is deliberately the ONLY one. It used
to live in artifacts/api-server (Express, routes/bhoomi.ts) and reach this
API over HTTP. That arrangement had a failure nobody would notice until
deploy: the production Netlify config sends /api/* straight here, so every
route that existed only in Express -- /dashboard, /regions, /parcels,
/changes, /processing/runs, /exports -- 404s in production while working
perfectly in development. Reproduced before porting: seven routes, 200
through Express, 404 direct. Moving the translation here makes the
deployed path the tested path.

Two rules it keeps, both inherited from the endpoints it reads:

- Nothing is invented. Fields the ward model genuinely has no answer for
  (a parcel's owner, a confidence score, an accuracy grade) are null, not
  a plausible-looking number. The UI already renders null as "Not
  measured".
- Nothing is silently resolved. An unknown region id is a 400/404, not an
  empty list that reads as "this ward has nothing in it".
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from geocadastra.api.deps import get_db_config, get_session
from geocadastra.api.wardviews import (
    coordinate_system,
    ward_feature_collection,
    ward_summaries,
    ward_topology_report,
)
from geocadastra.jobs.orchestrator import ingest_synthetic_ward, run_ward
from geocadastra.store.schema import WardJob
from geocadastra.synth.generator import WardParams, generate_ward

router = APIRouter(tags=["console compatibility"])

_REGION_ID = re.compile(r"^ward-[1-9]\d*$")


def _region_id(summary: dict) -> str:
    return f"ward-{summary['ward_job_id']}"


def _region_name(summary: dict) -> str:
    return f"Ward {summary['ward_job_id']} ({summary['source']})"


def _progress(summary: dict) -> int:
    """Real completed-block ratio. Never a timer: a run that is stuck at
    one block must keep reporting that block, not drift toward 100%."""
    return int(100 * summary["completed_blocks"] / summary["n_blocks"]) if summary["n_blocks"] else 0


def _run(summary: dict) -> dict:
    return {
        "id": f"run-{summary['ward_job_id']}",
        "name": _region_name(summary),
        "regionId": _region_id(summary),
        "regionName": _region_name(summary),
        "status": summary["status"],
        "progress": _progress(summary),
        "currentStep": summary["status"],
        "steps": ["ingest", "process"],
        "startedAt": summary["created_at"],
    }


def _summary_for_region(session: Session, region_id: str) -> dict:
    if not isinstance(region_id, str) or not _REGION_ID.match(region_id):
        raise HTTPException(400, "Select a ward ID from the region list")
    for summary in ward_summaries(session):
        if _region_id(summary) == region_id:
            return summary
    raise HTTPException(404, "Ward not found")


def _parcels_of(session: Session, summary: dict) -> list[dict]:
    """Faces as the console's parcel records. `ownership` and `confidence`
    stay null: the store has no owner and no per-parcel confidence, and a
    placeholder there would be indistinguishable from a real one."""
    job = session.get(WardJob, summary["ward_job_id"])
    collection = ward_feature_collection(session, job)
    region_id, region_name = _region_id(summary), _region_name(summary)
    parcels = []
    for feature in collection["features"]:
        props = feature["properties"]
        parcels.append({
            "id": str(props["face_id"]),
            "ulpin": f"GC-{props['parcel_id'] if props['parcel_id'] is not None else props['face_id']}",
            "regionId": region_id,
            "regionName": region_name,
            "areaSqM": props["area_m2"],
            "ownership": None,
            "confidence": None,
            "status": "unmatched" if props["parcel_id"] is None else "matched",
            "geometry": feature["geometry"],
            "coordinateSystem": collection["coordinate_system"],
            "updatedAt": props["updated_at"],
        })
    return parcels


# ------------------------------------------------------------- dashboard --


@router.get("/dashboard")
def dashboard(session: Session = Depends(get_session)):
    """Headline counts only where a count genuinely exists. Topology
    errors, encroachments and an accuracy grade are null on purpose:
    topology is a per-ward scan (see /topology/scan), encroachment
    detection does not exist in this backend at all, and accuracy is a
    stratified report (/wards/{id}/analytics), never one number."""
    summaries = ward_summaries(session)
    active = next((s for s in summaries if s["status"] == "running"), None)
    recent = sorted(summaries, key=lambda s: s["created_at"], reverse=True)[:6]
    return {
        "areaProcessedSqKm": sum(s["area_sqkm"] for s in summaries if s["status"] == "done"),
        "parcelsExtracted": sum(s["n_parcels"] for s in summaries),
        "topologyErrors": None,
        "encroachments": None,
        "accuracyScore": None,
        "activeRun": _run(active) if active else None,
        "recentActivity": [{
            "id": _region_id(s),
            "title": f"{_region_name(s)} ingested",
            "detail": f"Current status: {s['status']}",
            "time": s["created_at"],
            "tone": "info",
        } for s in recent],
    }


@router.get("/regions")
def regions(session: Session = Depends(get_session)):
    """processedSqKm/parcelCount/updatedAt are required by the published
    contract (lib/api-spec/openapi.yaml, which generates the frontend's
    types) and neither this endpoint nor the Express one it replaced ever
    returned them -- so the generated client promised three fields that
    never arrived. Caught by test_api_contract.py. processedSqKm counts
    only completed wards, matching /dashboard's own areaProcessedSqKm."""
    return [{
        "id": _region_id(s),
        "name": _region_name(s),
        "type": "Synthetic ward" if s["synthetic"] else "Georeferenced ward",
        "processedSqKm": s["area_sqkm"] if s["status"] == "done" else 0.0,
        "parcelCount": s["n_parcels"],
        "updatedAt": s["updated_at"],
        "areaSqKm": s["area_sqkm"],
        "status": s["status"],
    } for s in ward_summaries(session)]


# --------------------------------------------------------------- parcels --


@router.get("/parcels")
def parcels(
    regionId: str | None = Query(None),
    status: str | None = Query(None),
    session: Session = Depends(get_session),
):
    summaries = [_summary_for_region(session, regionId)] if regionId else ward_summaries(session)
    out = [p for s in summaries for p in _parcels_of(session, s)]
    return [p for p in out if p["status"] == status] if status else out


# Path template is {id}, not {parcel_id}: the published contract spells it
# {id}, and OpenAPI compares path templates literally, so a different
# parameter name reads as a different path even though both route the
# same request. test_api_contract.py checks that they agree.
@router.get("/parcels/{id}")
def parcel(id: str, session: Session = Depends(get_session)):
    for summary in ward_summaries(session):
        for candidate in _parcels_of(session, summary):
            if candidate["id"] == id:
                return candidate
    raise HTTPException(404, "Parcel not found")


@router.patch("/parcels/{id}")
def update_parcel(id: str):
    """501, not a stub that pretends to save. A face's parcel association
    changes through the recorded node-edit/association workflow, which
    writes a changeset; a direct status poke would leave no provenance."""
    return Response(status_code=501, content='{"error":"Parcel status editing is unavailable. Use the recorded ward node-edit workflow."}',
                    media_type="application/json")


# ------------------------------------------------------------ processing --


class RunRequest(BaseModel, extra="forbid"):
    regionId: str
    dataset: str | None = None


class SyntheticRequest(BaseModel, extra="forbid"):
    seed: int
    width: float = Field(50.0, gt=0, le=500)
    height: float = Field(35.0, gt=0, le=500)


@router.get("/processing/runs")
def processing_runs(session: Session = Depends(get_session)):
    return [_run(s) for s in ward_summaries(session)]


@router.post("/processing/runs", status_code=202)
def start_processing_run(
    req: RunRequest,
    session: Session = Depends(get_session),
    db_config: tuple = Depends(get_db_config),
):
    """Runs an EXISTING ward's stored inputs. A ward already identifies
    what it was built from, so a mismatched `dataset` is a conflict to
    report, never a cue to generate a fresh one behind the caller."""
    summary = _summary_for_region(session, req.regionId)
    if req.dataset is not None and req.dataset != summary["source"]:
        raise HTTPException(409, "Dataset does not match the selected ward source")
    db_url, schema = db_config
    run_ward(session, db_url, schema, summary["ward_job_id"])
    return _run(_summary_for_region(session, req.regionId))


@router.post("/processing/synthetic", status_code=201)
def create_synthetic(req: SyntheticRequest, session: Session = Depends(get_session)):
    """Explicitly asked-for test geometry, behind its own endpoint and its
    own button -- never a silent fallback when a real ingest fails."""
    params = WardParams(width=req.width, height=req.height, n_arterial_h=1, n_arterial_v=0)
    ward = generate_ward(params=params, seed=req.seed)
    job = ingest_synthetic_ward(session, ward, seed=req.seed)
    return {"ward_job_id": job.id, "n_blocks": len(ward.blocks), "n_parcels": len(ward.parcels)}


# -------------------------------------------------------------- topology --


class TopologyScanRequest(BaseModel):
    regionId: str


@router.post("/topology/scan")
def topology_scan(req: TopologyScanRequest, session: Session = Depends(get_session)):
    summary = _summary_for_region(session, req.regionId)
    return ward_topology_report(session, session.get(WardJob, summary["ward_job_id"]))


@router.post("/topology/fix")
def topology_fix():
    """No automatic repair. Every geometry change in this system goes
    through a recorded edit; a silent bulk "fix" would rewrite surveyed
    boundaries with no author and no changeset behind them."""
    return Response(status_code=501, content='{"error":"Automatic repair is unavailable. Inspect the detected faces and apply a recorded node edit."}',
                    media_type="application/json")


# --------------------------------------------------- changes and exports --


@router.get("/changes")
def changes():
    """Empty, honestly. Temporal change detection is not implemented in
    this backend -- returning [] is the truthful answer, and the UI
    renders it as "no detections", which is correct."""
    return []


class ExportRequest(BaseModel):
    regionId: str
    format: str


@router.post("/exports", status_code=201)
def create_export(req: ExportRequest, session: Session = Depends(get_session)):
    if req.format != "GeoJSON":
        raise HTTPException(422, f"{req.format} export is not implemented; only GeoJSON is produced from stored geometry")
    summary = _summary_for_region(session, req.regionId)
    job = session.get(WardJob, summary["ward_job_id"])
    if coordinate_system(job) != "EPSG:4326":
        raise HTTPException(422, "Synthetic local coordinates cannot be exported as geographic GeoJSON")
    collection = ward_feature_collection(session, job)
    region_id = _region_id(summary)
    return {
        "id": f"export-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        "format": "GeoJSON",
        "regionId": region_id,
        "status": "ready",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "fileName": f"{region_id}.geojson",
        "geojson": {"type": "FeatureCollection", "features": collection["features"]},
    }
