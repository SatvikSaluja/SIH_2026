"""FastAPI surface (Stage 8): ingest and co-registration, inference job
submission and polling, spatially indexed parcel query, conflict records,
human edit via changeset, field verification attachment, export, and the
analytics endpoints.

A DB session per request comes from `get_session`, itself built from
`GEOCADASTRA_DB_URL`/`GEOCADASTRA_DB_SCHEMA` env vars (defaults matching
this project's own local Postgres convention) -- tests override the
FastAPI dependency directly (`app.dependency_overrides[get_session]`)
rather than relying on those env vars, the standard FastAPI testing
pattern, and exactly how `jobs/orchestrator.py`'s own tests avoid
depending on environment state too.

Every endpoint that touches an existing `WardJob` 404s cleanly if it
doesn't exist -- "nothing is silently resolved" applies here too: a
request against an unknown ward is a client error, not an empty result
that looks like "this ward has no data yet".
"""
from __future__ import annotations

from geocadastra.store.constraints import parcel_area_report

import os

from fastapi import Depends, FastAPI, HTTPException, Response
from geoalchemy2.shape import from_shape, to_shape
from pydantic import BaseModel, Field
from shapely.affinity import affine_transform
from shapely.geometry import Point, box
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from geocadastra.core.crs import Geom
from geocadastra.core.evaluate import boundary_position_residuals, summarize_errors, parcel_count_error, topology_validity_rate
from geocadastra.jobs.orchestrator import ingest_synthetic_ward, run_ward, ward_status
from geocadastra.store.changeset import ChangesetContext, ConcurrentModificationError, load_block_graph, lock_block
from geocadastra.store.schema import (
    SRID,
    BlockJob,
    Changeset,
    Coregistration,
    Face,
    IngestedBlock,
    LegacyRecord,
    PersistedConflict,
    RecordedParcel,
    SurveyPoint,
    WardJob,
)
from geocadastra.synth.generator import WardParams, generate_ward

app = FastAPI(title="GeoCadastra")

_DB_URL = os.environ.get("GEOCADASTRA_DB_URL", "postgresql+psycopg://geocadastra:geocadastra@localhost:5432/geocadastra")
_DB_SCHEMA = os.environ.get("GEOCADASTRA_DB_SCHEMA", "public")
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_DB_URL, connect_args={"options": f"-csearch_path={_DB_SCHEMA},public"})
    return _engine


def get_session():
    session = sessionmaker(bind=_get_engine())()
    try:
        yield session
    finally:
        session.close()


def get_db_config() -> tuple[str, str]:
    """`(db_url, schema)` for `run_ward()`'s Celery dispatch -- a SEPARATE
    dependency from `get_session()`, not the same module globals baked
    directly into the `/run` handler, because a test overriding
    `get_session` (the standard FastAPI testing pattern, pointing requests
    at a test schema) must ALSO redirect where dispatched tasks look for
    their own data, or a task opens a session against the real default
    schema while the request's own session was pointed at the test one --
    two different databases silently in play at once. Found exactly this
    way: `/run` without this indirection passed `_DB_URL`/`_DB_SCHEMA`
    straight through even under a test override, and every dispatched
    task then failed with `UndefinedTable` looking for `block_jobs` in a
    schema that was never created.
    """
    return _DB_URL, _DB_SCHEMA


def _get_ward_job_or_404(session: Session, ward_job_id: int) -> WardJob:
    job = session.get(WardJob, ward_job_id)
    if job is None:
        raise HTTPException(404, f"no ward {ward_job_id}")
    return job


# ---------------------------------------------------------------- ingest --


class IngestRequest(BaseModel):
    seed: int
    width: float = Field(240.0, gt=0)
    height: float = Field(180.0, gt=0)
    # gt=0: gsd=0 propagates unvalidated into generate_ward(), where
    # int(np.ceil(width / gsd)) raises an unhandled ZeroDivisionError ->
    # 500 (found by review, reproduced directly) instead of a clean 422
    gsd: float = Field(0.5, gt=0)
    n_arterial_h: int = 1
    n_arterial_v: int = 1


@app.post("/wards/ingest")
def ingest(req: IngestRequest, session: Session = Depends(get_session)):
    """Real drone imagery doesn't exist yet in this project -- ingest is
    always synthetic for now (doc: "everything must work on synthetic
    first"). A real ingest endpoint would accept an uploaded raster/vector
    bundle instead of `seed`; the persisted shape downstream (`WardJob`,
    `RecordedParcel`, etc.) is exactly what a real one would produce too,
    so nothing else in this API depends on the source being synthetic.
    """
    params = WardParams(
        width=req.width, height=req.height, gsd=req.gsd, n_arterial_h=req.n_arterial_h, n_arterial_v=req.n_arterial_v
    )
    ward = generate_ward(params=params, seed=req.seed)
    job = ingest_synthetic_ward(session, ward, seed=req.seed)
    return {"ward_job_id": job.id, "n_blocks": len(ward.blocks), "n_parcels": len(ward.parcels)}


# -------------------------------------------------------- co-registration --


class CoRegisterRequest(BaseModel):
    # [(src_x, src_y, dst_x, dst_y), ...] -- control points pairing the upload's
    # own coordinates against this ward's already-established reference frame
    control_points: list[tuple[float, float, float, float]]


@app.post("/wards/{ward_job_id}/coregister")
def coregister(ward_job_id: int, req: CoRegisterRequest, session: Session = Depends(get_session)):
    """Estimate a best-fit affine transform from `control_points` (least
    squares) and record its RMS residual on the ward -- carried forward
    into every block's fusion sigma (`jobs/orchestrator.py`), per the doc:
    "co-registration estimates and applies a transform, and carries the
    residual forward into the Stage 5 reliability priors. It does NOT
    reject the upload." No threshold here rejects a badly-aligned upload;
    a large residual just means every block processed after this call
    trusts its legacy evidence less, which is the honest response to
    "the alignment isn't great" -- refusing the data outright is not.

    That "never reject" promise is about the DATA (a genuinely bad but
    well-posed alignment) -- it doesn't extend to control points so
    degenerate a transform can't even be computed (`_fit_affine()`'s own
    `ValueError`, e.g. every point identical): there's no honest residual
    to report for that, only a client mistake in the request itself.
    """
    _get_ward_job_or_404(session, ward_job_id)
    if len(req.control_points) < 3:
        raise HTTPException(422, "need at least 3 control points to fit an affine transform")
    try:
        transform, residual = _fit_affine(req.control_points)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    # All workers/editors use the same block locks. Lock in a stable order
    # before changing the evidence they read; repeat registrations always
    # transform the original upload, never the already transformed geometry.
    block_ids = session.scalars(select(IngestedBlock.block_id).where(
        IngestedBlock.ward_job_id == ward_job_id).order_by(IngestedBlock.block_id)).all()
    for block_id in block_ids:
        lock_block(session, block_id)
    job = session.get(WardJob, ward_job_id, populate_existing=True)
    records = session.scalars(select(LegacyRecord).where(LegacyRecord.ward_job_id == ward_job_id)
                              .execution_options(populate_existing=True)).all()
    a, b, c = transform[0]
    d, e, f = transform[1]
    cs = Changeset(description="co-registration", affected_entities={"legacy_records": [r.id for r in records]})
    session.add(cs)
    session.flush()
    for record in records:
        if record.original_geom is None:
            record.original_geom = record.geom
        record.geom = from_shape(affine_transform(to_shape(record.original_geom), [a, b, d, e, c, f]), srid=SRID)
    job.coreg_transform = transform
    job.coreg_residual_m = residual
    session.add(Coregistration(ward_job_id=ward_job_id, changeset_id=cs.id, transform=transform,
                              control_points=[list(p) for p in req.control_points], residual_m=residual))
    for block_job in session.scalars(select(BlockJob).where(BlockJob.ward_job_id == ward_job_id)):
        block_job.status = "pending"
        block_job.error = None
    job.status = "pending"
    session.commit()
    return {"transform": transform, "residual_m": residual}


def _fit_affine(control_points: list) -> tuple[list, float]:
    """Least-squares 2D affine (6 params: x' = a*x+b*y+c, y' = d*x+e*y+f)
    fit from (src_x, src_y) -> (dst_x, dst_y) pairs, plus the fit's own
    RMS residual in the destination frame's units (this project's working
    CRS, metres) -- the number `jobs/orchestrator.py` widens fusion sigma
    by.

    Raises `ValueError` if the source points are degenerate (all the same
    point, or collinear) -- `np.linalg.lstsq` doesn't itself raise for
    this, it just returns SOME solution for the underdetermined system,
    and for a fully-degenerate case (every point identical) that
    "solution" spuriously reports a "perfect" 0.0 residual: exactly the
    wrong answer, since a co-registration this project's fusion sigma is
    about to trust MORE needs the fit to have actually been determined
    by the data, not have degenerated to an arbitrary zero (found by
    review, reproduced directly with three identical control points).
    """
    import numpy as np

    pts = np.asarray(control_points, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("control points must be finite")
    src, dst = pts[:, :2], pts[:, 2:]
    A = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("control points are degenerate (duplicate or collinear) -- can't determine a unique affine fit")
    coeffs_x, *_ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(A, dst[:, 1], rcond=None)
    if np.linalg.matrix_rank(np.stack([coeffs_x[:2], coeffs_y[:2]])) < 2:
        raise ValueError("control points imply a collapsed affine transform")
    pred = np.column_stack([A @ coeffs_x, A @ coeffs_y])
    residual = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return [coeffs_x.tolist(), coeffs_y.tolist()], residual


# ------------------------------------------------------- job submission --


@app.post("/wards/{ward_job_id}/run")
def run(ward_job_id: int, session: Session = Depends(get_session), db_config: tuple = Depends(get_db_config)):
    """Dispatch (or resume) processing for every block not yet `done` --
    the same call whether this is the first run or a resume after a
    killed worker, per the Stage 8 Done-when."""
    _get_ward_job_or_404(session, ward_job_id)
    db_url, schema = db_config
    return run_ward(session, db_url, schema, ward_job_id)


@app.get("/wards/{ward_job_id}/status")
def status(ward_job_id: int, session: Session = Depends(get_session)):
    job = _get_ward_job_or_404(session, ward_job_id)
    state, blocks = ward_status(session, ward_job_id)
    return {"ward_job_id": job.id, "status": state, "blocks": blocks}


# --------------------------------------------------------- parcel query --


@app.get("/wards/{ward_job_id}/parcels")
def parcels(ward_job_id: int, minx: float, miny: float, maxx: float, maxy: float, session: Session = Depends(get_session)):
    """Spatially indexed bbox query against `Face.geom`'s own GiST index
    (`&&`, PostGIS's index-accelerated bbox-overlap operator) -- not a
    scan-and-filter-in-Python, which is what "spatially indexed" in the
    doc's own bullet rules out.
    """
    _get_ward_job_or_404(session, ward_job_id)
    bbox = from_shape(box(minx, miny, maxx, maxy), srid=SRID)
    rows = session.execute(
        select(Face)
        .join(IngestedBlock, IngestedBlock.block_id == Face.block_id)
        .where(IngestedBlock.ward_job_id == ward_job_id, Face.geom.op("&&")(bbox))
    ).scalars().all()
    return {
        "parcels": [
            {"face_id": f.id, "block_id": f.block_id, "parcel_id": f.recorded_parcel_id, "geometry": to_shape(f.geom).__geo_interface__} for f in rows
        ]
    }


# ------------------------------------------------------------- conflicts --


@app.get("/wards/{ward_job_id}/conflicts")
def conflicts(ward_job_id: int, session: Session = Depends(get_session)):
    _get_ward_job_or_404(session, ward_job_id)
    rows = session.execute(select(PersistedConflict).where(PersistedConflict.ward_job_id == ward_job_id)).scalars().all()
    return {
        "conflicts": [
            {
                "id": c.id, "block_id": c.block_id, "node_id": c.node_id, "sources": c.sources,
                "disagreement_m": c.disagreement_m, "kind": c.kind, "detail": c.detail,
                "geometry": to_shape(c.geom).__geo_interface__,
            }
            for c in rows
        ]
    }


# ------------------------------------------------------------- human edit --


class EditRequest(BaseModel):
    block_id: int
    node_id: int
    x: float
    y: float
    author: str | None = None


class ParcelAssociationsRequest(BaseModel):
    block_id: int
    assignments: dict[int, int]  # complete face_id -> recorded_parcel_id mapping
    author: str | None = None


@app.post("/wards/{ward_job_id}/parcel-associations")
def parcel_associations(ward_job_id: int, req: ParcelAssociationsRequest, session: Session = Depends(get_session)):
    _get_ward_job_or_404(session, ward_job_id)
    if session.get(IngestedBlock, (ward_job_id, req.block_id)) is None:
        raise HTTPException(404, "block does not belong to ward")
    try:
        with ChangesetContext(session, req.block_id, author=req.author, description="resolve parcel identity") as cs:
            cs.associate_faces(req.assignments)
    except ConcurrentModificationError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return parcel_area_report(session, req.block_id, load_block_graph(session, req.block_id))


def _apply_move(
    session: Session, ward_job_id: int, block_id: int, node_id: int, x: float, y: float, author, description,
    evidence_type, *, survey_point: SurveyPoint | None = None,
):
    """Shared by `/edit` and `/field-verification`: apply one node move
    through a real changeset, translating its failure modes into the
    right HTTP status instead of a bare 500. `PlanarGraph.move_node()`
    raises plain `KeyError` for an unknown node id (found running this
    endpoint's own "unknown node" test: an earlier version here only
    caught `ConcurrentModificationError`/`ValueError`, so a bad node id
    crashed with an unhandled 500 instead of a client error) -- caught
    here alongside the other two, all as the caller's mistake (422/409),
    never a server fault.

    Also verifies `block_id` actually belongs to `ward_job_id` before
    touching anything (found by review): neither endpoint previously
    checked this, so a request naming a real block id from a DIFFERENT
    ward -- confusable in the first place because `block_id` used to be
    only ward-LOCAL before this same review round's schema fix -- could
    edit that other ward's data through the wrong URL with no error at
    all. `block_id` is globally unique now, closing the underlying
    collision, but the ownership check is the actual authorization
    boundary and belongs here regardless.
    """
    if session.get(IngestedBlock, (ward_job_id, block_id)) is None:
        raise HTTPException(404, f"block {block_id} does not belong to ward {ward_job_id}")
    try:
        with ChangesetContext(session, block_id=block_id, author=author, description=description) as cs:
            if survey_point is not None:
                incident_faces = {fid for eid, edge in cs.graph.edges.items()
                                  if node_id in (edge.n0, edge.n1)
                                  for fid in cs.graph.faces_of_edge(eid)}
                if not any(cs.graph.face_parcel_ids.get(fid) == survey_point.parcel_id for fid in incident_faces):
                    raise ValueError("node is not incident to the specified recorded parcel")
            cs.move_node(node_id, x, y, evidence_type=evidence_type,
                         detail={"parcel_id": survey_point.parcel_id} if survey_point is not None else None)
            if survey_point is not None:
                session.add(survey_point)
    except ConcurrentModificationError as e:
        raise HTTPException(409, str(e)) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e


@app.post("/wards/{ward_job_id}/edit")
def edit(ward_job_id: int, req: EditRequest, session: Session = Depends(get_session)):
    """A human correction, applied through the same transactional
    changeset every other write goes through -- "there is no such thing
    as editing one parcel of a shared boundary" (invariant #2) applies to
    a human edit exactly as much as a model's.
    """
    _get_ward_job_or_404(session, ward_job_id)
    _apply_move(session, ward_job_id, req.block_id, req.node_id, req.x, req.y, req.author, "human edit", "manual_edit")
    return {"status": "applied"}


# ----------------------------------------------- field verification --


class FieldVerificationRequest(BaseModel):
    block_id: int
    node_id: int
    parcel_id: int
    x: float
    y: float
    author: str | None = None


@app.post("/wards/{ward_job_id}/field-verification")
def field_verification(ward_job_id: int, req: FieldVerificationRequest, session: Session = Depends(get_session)):
    """A surveyor's on-the-ground confirmation of a boundary corner: both
    applied as a changeset move (`evidence_type="field_verification"`,
    same provenance chain every other evidence type uses) AND recorded as
    a durable `SurveyPoint` (`source="field_verification"`) -- the same
    shape as an ingest-time GT point, so Stage 6/7's own calibration and
    prioritisation code can use a field-verified corner exactly like an
    original one, without a special case.
    """
    _get_ward_job_or_404(session, ward_job_id)
    parcel = session.get(RecordedParcel, req.parcel_id)
    if parcel is None or parcel.ward_job_id != ward_job_id or parcel.block_id != req.block_id:
        raise HTTPException(422, "parcel does not belong to this ward and block")
    measurement = SurveyPoint(
        ward_job_id=ward_job_id, parcel_id=req.parcel_id, geom=from_shape(Point(req.x, req.y), srid=SRID),
        source="field_verification", purpose="fusion",
    )
    _apply_move(
        session, ward_job_id, req.block_id, req.node_id, req.x, req.y, req.author, "field verification",
        "field_verification", survey_point=measurement,
    )
    return {"status": "applied"}


# -------------------------------------------------------------- export --


@app.get("/wards/{ward_job_id}/tiles/{z}/{x}/{y}.mvt")
def tile(ward_job_id: int, z: int, x: int, y: int, session: Session = Depends(get_session)):
    """Parcels as an MVT vector tile via `ST_AsMVT`, not raw GeoJSON --
    "this matters at 200k parcels and it is what the eventual frontend
    will want" (the doc's own words). Standard slippy-map z/x/y -> Web
    Mercator (EPSG:3857) tile envelope, `ST_AsMVTGeom` clips/transforms
    this ward's faces (stored in this project's own working CRS,
    EPSG:32643) into it.
    """
    _get_ward_job_or_404(session, ward_job_id)
    # standard slippy-map bounds: z is capped well above any real zoom level
    # (22 is already finer than a single parcel needs), x/y must address an
    # actual tile in that zoom's 2^z grid. Without this, an extreme y (e.g.
    # 10**18) overflows math.sinh() inside lat() below with an unhandled
    # OverflowError -> 500 instead of a clean 422 (found by review, reproduced
    # directly: z=1, y=10**18).
    if not (0 <= z <= 22):
        raise HTTPException(422, f"z must be in [0, 22], got {z}")
    if not (0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(422, f"x, y must be in [0, 2^z) for z={z}")
    n = 2**z
    minx_merc = x / n * 360.0 - 180.0
    maxx_merc = (x + 1) / n * 360.0 - 180.0
    import math

    def lat(yt):
        n_ = math.pi - 2.0 * math.pi * yt / n
        return math.degrees(math.atan(math.sinh(n_)))

    maxy_deg, miny_deg = lat(y), lat(y + 1)
    # 4326 (lon/lat) -> 3857 envelope corners via ST_Transform, computed in SQL
    # so this doesn't need a second CRS library dependency just for tile math
    result = session.execute(
        text(
            """
            WITH env AS (
                SELECT ST_Transform(ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326), 3857) AS geom
            ),
            mvtgeom AS (
                SELECT ST_AsMVTGeom(ST_Transform(f.geom, 3857), env.geom, 4096, 64, true) AS geom, f.id, f.block_id
                FROM faces f
                JOIN ingested_blocks ib ON ib.block_id = f.block_id
                JOIN env ON true
                WHERE ib.ward_job_id = :ward_job_id AND ST_Intersects(f.geom, ST_Transform(env.geom, :srid))
            )
            SELECT ST_AsMVT(mvtgeom, 'parcels') FROM mvtgeom WHERE geom IS NOT NULL
            """
        ),
        {"minx": minx_merc, "miny": miny_deg, "maxx": maxx_merc, "maxy": maxy_deg, "ward_job_id": ward_job_id, "srid": SRID},
    )
    mvt_bytes = result.scalar()
    return Response(content=bytes(mvt_bytes) if mvt_bytes else b"", media_type="application/vnd.mapbox-vector-tile")


# ----------------------------------------------------------- analytics --


@app.get("/wards/{ward_job_id}/analytics")
def analytics(ward_job_id: int, session: Session = Depends(get_session)):
    """Every metric stratified by settlement type, never pooled (doc,
    section 6: "a mean number across formal and informal blocks hides the
    only failure mode that matters"; "never report mean IoU as a
    headline"). Per stratum: boundary position error (P50/P90 against
    GT), topology validity rate, parcel count error (over/under
    segmentation) -- `core/evaluate.py`'s own functions, one call per
    stratum here, never one pooling call.

    Area errors are paired by durable recorded parcel ID, summing all
    components of each parcel. Missing geometry counts as zero area and is
    separately identified; unassigned faces are never guessed into a stratum.
    """
    _get_ward_job_or_404(session, ward_job_id)
    recorded = session.execute(select(RecordedParcel).where(RecordedParcel.ward_job_id == ward_job_id)).scalars().all()
    styles = sorted({r.style for r in recorded})
    blocks = session.execute(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == ward_job_id)).scalars().all()

    result = {}
    for style in styles:
        style_recorded = [r for r in recorded if r.style == style]
        style_blocks = {r.block_id for r in style_recorded}

        residuals, fit_residuals, validity, n_faces_total = [], [], [], 0
        area_rows = []
        for block_id in style_blocks:
            graph = load_block_graph(session, block_id)
            report = parcel_area_report(session, block_id, graph)
            area_rows.extend(r for r in report["parcels"] if r["style"] == style)
            if not graph.faces:
                continue
            style_face_ids = {fid for fid, pid in graph.face_parcel_ids.items()
                              if pid in {r.id for r in style_recorded}}
            # GT points scoped to THIS block's own parcels, not the whole style --
            # a GT point from a different block of the same style is physically
            # nowhere near this block's edges, and would inflate its boundary
            # error with a meaningless large "nearest edge" distance
            block_parcel_ids = {r.id for r in style_recorded if r.block_id == block_id}
            gt_rows = session.execute(select(SurveyPoint).where(SurveyPoint.ward_job_id == ward_job_id, SurveyPoint.parcel_id.in_(block_parcel_ids))).scalars().all()
            held_out = [(to_shape(g.geom).x, to_shape(g.geom).y) for g in gt_rows if g.purpose == "evaluation"]
            controls = [(to_shape(g.geom).x, to_shape(g.geom).y) for g in gt_rows if g.purpose == "fusion"]
            residuals.extend(boundary_position_residuals(held_out, graph))
            fit_residuals.extend(boundary_position_residuals(controls, graph))
            v = topology_validity_rate(graph)
            validity.append(v)
            n_faces_total += len(style_face_ids)

        error = summarize_errors(residuals)
        result[style] = {
            "boundary_position_error_p50": error["p50"],
            "boundary_position_error_p90": error["p90"],
            "evaluation_population": "held_out",
            "fusion_control_residual": summarize_errors(fit_residuals),
            "topology_validity_rate": (
                sum(v["rate"] * (v["n_faces"] or 0) for v in validity if v["rate"] is not None) / sum(v["n_faces"] for v in validity)
                if sum(v["n_faces"] for v in validity)
                else None
            ),
            "parcel_count": parcel_count_error(len({r["parcel_id"] for r in area_rows if r["face_ids"]}), len(style_recorded)),
            "face_count": n_faces_total,
            "topology_scope": "blocks_containing_stratum",
            "area_error_relative": summarize_errors([abs(r["error_m2"]) / r["recorded_area_m2"] for r in area_rows if r["recorded_area_m2"] > 0]),
            "area_constraints": {"within_tolerance": sum(r["within_tolerance"] for r in area_rows),
                                 "total": len(area_rows),
                                 "missing_parcel_ids": [r["parcel_id"] for r in area_rows if not r["face_ids"]]},
            "n_gt_points": error["n"],
        }
    return result


@app.get("/wards/{ward_job_id}/constraints")
def constraints(ward_job_id: int, session: Session = Depends(get_session)):
    """Live identity/area readiness. This endpoint never certifies boundaries."""
    _get_ward_job_or_404(session, ward_job_id)
    blocks = session.scalars(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == ward_job_id)).all()
    reports = [{"block_id": bid, **parcel_area_report(session, bid, load_block_graph(session, bid))} for bid in blocks]
    return {"recorded_area_constraints_satisfied": bool(reports) and all(r["constraints_satisfied"] for r in reports),
            "boundary_certification": "not_calibrated", "blocks": reports}
