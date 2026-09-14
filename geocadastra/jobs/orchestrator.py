"""Synthetic ward orchestration: one atomic, resumable transaction per block.

Workers regenerate raster evidence -- from the model when
GEOCADASTRA_MODEL_WEIGHTS is set, simulated otherwise, with the stored
provenance naming which -- while vector facts come from storage. Completion
means geometry processing completed, not certification.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import os

from celery import Celery
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point
from shapely.ops import unary_union
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from geocadastra.core.crs import CRSMismatchError, Geom
from geocadastra.core.fusion import DEFAULT_SIGMA_LEGACY_BY_STYLE, fuse_block
from geocadastra.core.transport import assign_parcels, parcels_to_graph
from geocadastra.core.capacity import refine_recorded_areas
from geocadastra.store.changeset import allocate_ids, apply_fusion, load_block_graph, lock_block, seed_block_graph
from geocadastra.store.constraints import parcel_area_report
from geocadastra.store.schema import (
    SRID,
    BlockJob,
    Face,
    IngestedBlock,
    LegacyRecord,
    PersistedConflict,
    RecordedParcel,
    SurveyPoint,
    WardJob,
    block_id_seq,
    legacy_record_id_seq,
    recorded_parcel_id_seq,
)
from geocadastra.synth.generator import WardParams, generate_ward, simulate_evidence_field

app = Celery("geocadastra", broker=os.environ.get("GEOCADASTRA_CELERY_BROKER_URL", "redis://localhost:6379/0"))
# Production default: real async dispatch through a real broker/worker.
# Tests flip `app.conf.task_always_eager = True` (standard Celery testing
# pattern) so `process_block.delay(...)` runs synchronously in-process --
# no real Redis needed to exercise task logic, matching how this project's
# test suite otherwise avoids depending on infrastructure the environment
# doesn't have (no Redis server is installed here; only its Python client
# is a real dependency, for the broker URL config itself).
#
# task_eager_propagates stays Celery's own default (False): in real async
# dispatch, `.delay()` enqueues and returns immediately -- a task's eventual
# failure can never raise back through the `run_ward()` loop that dispatched
# it, so one bad block must not stop that loop from dispatching every other
# block. Eager mode is meant to be a faithful stand-in for that, not a
# shortcut that changes this behaviour just because there's no real broker
# in the loop -- `process_block`'s own try/except already records a failed
# block's status durably in `BlockJob`, which is how a caller is meant to
# learn about it (a poll, not a bubbled-up exception). Calling
# `process_block(...)` directly (bypassing Celery, as some tests do to
# simulate one block's own crash precisely) is unaffected either way --
# that's a plain Python function call, propagates regardless of this setting.


# one Engine (and its connection pool) per (db_url, schema), reused across
# every call -- an Engine is meant to be created once and shared; building a
# fresh one per `process_block` call (a real risk here, since one is created
# per BLOCK, and a ward can have thousands) opens a brand new connection pool
# every time and never disposes the old one, leaking Postgres connections
# until the server's own max_connections is exhausted. Cached at module
# level (not per-Celery-worker-process state) since a single worker process
# runs many tasks against the same store over its lifetime.
_engine_cache: dict[tuple[str, str], object] = {}


def make_session(db_url: str, schema: str) -> Session:
    key = (db_url, schema)
    engine = _engine_cache.get(key)
    if engine is None:
        engine = create_engine(db_url, connect_args={"options": f"-csearch_path={schema},public"})
        _engine_cache[key] = engine
    return sessionmaker(bind=engine)()


def ingest_synthetic_ward(session: Session, ward, seed: int) -> WardJob:
    """Persist a synthetic ward's every VECTOR fact -- block polygons,
    recorded parcel areas/styles/seed points, legacy layer, GT points --
    durably, as one `WardJob`. This is what `process_block`/`run_ward`
    resume FROM: only the raster evidence field is reconstructed rather
    than stored (see module docstring).

    `generate_ward()` numbers its own blocks and parcels starting from 0
    EVERY call, with no idea a store might already hold a different
    ward's "block 0"/"parcel 0" -- so every block/parcel/legacy-record id
    is remapped onto a fresh, globally-unique one (`block_id_seq`/
    `recorded_parcel_id_seq`/`legacy_record_id_seq`) before writing,
    exactly the same pattern `store/changeset.py`'s own `seed_block_
    graph()` already uses for node/edge/face ids (found by review,
    reproduced: without this, a second ward's ingest crashed with a
    duplicate-primary-key error, and worse, two wards' same-numbered
    blocks silently pooled into one `load_block_graph()` result once
    both got seeded -- `Face.block_id` has no `ward_job_id` column at
    all, so a block id has to be globally unique for that table to mean
    anything).

    A parcel's "seed point" (the capacity-constrained solver's per-parcel
    anchor, Stage 3) is its own true centroid here -- a real ingest would
    take this from wherever the department's existing records place a
    parcel (a deed's marked point, a legacy GIS centroid); the synthetic
    generator doesn't carry a separate one, and its true centroid is
    always inside its own polygon, unlike an arbitrary point that could
    fall in a neighbour's -- correctness over unnecessary noise here.
    """
    if ward.crs != f"EPSG:{SRID}":
        raise CRSMismatchError(f"store requires EPSG:{SRID}, got {ward.crs}")
    ward_job = WardJob(
        source=f"synthetic:seed={seed}", params=dataclasses.asdict(ward.params), status="pending", crs=ward.crs
    )
    session.add(ward_job)
    session.flush()

    block_id_map = dict(zip((b.id for b in ward.blocks), allocate_ids(session, block_id_seq, len(ward.blocks))))
    parcel_id_map = dict(zip((p.id for p in ward.parcels), allocate_ids(session, recorded_parcel_id_seq, len(ward.parcels))))
    legacy_id_map = dict(
        zip((lp.id for lp in ward.legacy_parcels), allocate_ids(session, legacy_record_id_seq, len(ward.legacy_parcels)))
    )

    for block in ward.blocks:
        session.add(
            IngestedBlock(
                ward_job_id=ward_job.id, block_id=block_id_map[block.id], local_block_id=block.id,
                geom=from_shape(block.polygon, srid=SRID),
            )
        )
    for p in ward.parcels:
        centroid = p.polygon.centroid
        session.add(
            RecordedParcel(
                id=parcel_id_map[p.id], ward_job_id=ward_job.id, block_id=block_id_map[p.block_id], area=p.area,
                style=p.style, seed_point=from_shape(centroid, srid=SRID),
            )
        )
    for lp in ward.legacy_parcels:
        # a legacy polygon can span parcels from only one block (Stage 0
        # never merges across a block boundary), so the block of its first
        # source parcel is the legacy record's own block
        source_parcel = next(p for p in ward.parcels if p.id == lp.source_parcel_ids[0])
        session.add(
            LegacyRecord(
                id=legacy_id_map[lp.id], ward_job_id=ward_job.id, block_id=block_id_map[source_parcel.block_id],
                geom=from_shape(lp.polygon, srid=SRID), original_geom=from_shape(lp.polygon, srid=SRID),
            )
        )
    # Decide roles before fitting, independently of the residual or settlement.
    session.flush()  # recorded parcels must exist before their survey FK rows
    for gt in ward.gt_points:
        key = f"{seed}:{gt.x:.6f}:{gt.y:.6f}".encode()
        held_out = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 5 == 0
        session.add(
            SurveyPoint(
                ward_job_id=ward_job.id, parcel_id=parcel_id_map[gt.parcel_id], geom=from_shape(Point(gt.x, gt.y), srid=SRID),
                source="synthetic_gt", purpose="evaluation" if held_out else "fusion",
            )
        )
    session.commit()
    return ward_job


def _regenerate_ward(source: str, params: dict):
    """Reconstruct the exact same ward `ingest_synthetic_ward()` stored --
    `params` must be the SAME `WardJob.params` that ward was ingested
    with, not a default: two different `WardParams` sharing a seed
    produce two different wards, and regenerating with the wrong params
    silently gives a block whose evidence-field raster window doesn't
    even overlap the real block's geometry (found exactly this way, not
    assumed from reading the code -- see `WardJob.params`'s own comment).
    """
    if not source.startswith("synthetic:seed="):
        raise ValueError(f"don't know how to regenerate a ward from source {source!r}")
    seed = int(source.removeprefix("synthetic:seed="))
    # Jobs ingested before versioning used the independent-child splitter.
    # Never regenerate their evidence using a different ground truth.
    return generate_ward(params=WardParams(**{"generator_version": 1, **params}), seed=seed)


@functools.lru_cache(maxsize=2)
def _load_model(weights_path: str):
    """Load once per worker process, not once per block.

    `train.py` writes its state under "model" and stores no architecture in
    the checkpoint -- it always builds the default net -- so this rebuilds the
    same default. `load_state_dict` stays strict on purpose: an architecture
    that has moved on since the checkpoint was written must raise here, not
    quietly run a differently-shaped network over real parcels.
    """
    import torch
    from geocadastra.models.backbone import MultiTaskNet
    from geocadastra.models.train import N_LANDUSE_CLASSES
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
    model = MultiTaskNet(n_landuse_classes=N_LANDUSE_CLASSES)
    model.load_state_dict(state)
    model.eval()
    return model, hashlib.sha256(open(weights_path, "rb").read()).hexdigest()[:16]


def _block_evidence(ward, local_block_id: int):
    """Model prediction when weights are configured, simulation otherwise.

    Returns `(evidence_field, transform, provenance)`. The simulated field is
    the default so that a worker without configured weights keeps producing a
    reproducible, auditable result rather than silently no-op-ing -- but the
    provenance says which one ran, so a stored block can never be mistaken for
    a model result it did not come from.
    """
    weights_path = os.environ.get("GEOCADASTRA_MODEL_WEIGHTS")
    if not weights_path:
        field, transform = simulate_evidence_field(ward, local_block_id)
        return field, transform, {"evidence_source": "simulated"}
    from geocadastra.models.infer import block_evidence_from_model
    model, digest = _load_model(weights_path)
    field, transform = block_evidence_from_model(model, ward, local_block_id)
    return field, transform, {"evidence_source": "model", "weights_sha256": digest,
                              "weights_path": weights_path}


@app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_block(self, db_url: str, schema: str, ward_job_id: int, block_id: int, ward_source: str, ward_params: dict) -> str:
    """Publish one complete block atomically; serialize duplicate deliveries.

    Each fusion candidate uses a savepoint. A refused candidate cannot undo
    the seed or other safe candidates, while a crash rolls back the entire
    attempt. The transaction-scoped lock cannot be returned to the pool.
    """
    session = make_session(db_url, schema)
    try:
        lock_block(session, block_id)
        job = session.get(BlockJob, (ward_job_id, block_id))
        if job is not None and job.status == "done":
            return "already done"

        ward_job = session.get(WardJob, ward_job_id)
        if ward_job is None:
            raise ValueError(f"no ward {ward_job_id}")
        crs = ward_job.crs
        block_row = session.get(IngestedBlock, (ward_job_id, block_id))
        if block_row is None:
            raise ValueError(f"block {block_id} does not belong to ward {ward_job_id}")
        block_geom = Geom(to_shape(block_row.geom), crs)

        recorded = (
            session.execute(
                select(RecordedParcel)
                .where(RecordedParcel.ward_job_id == ward_job_id, RecordedParcel.block_id == block_id)
                .order_by(RecordedParcel.id)
            )
            .scalars()
            .all()
        )
        legacy = (
            session.execute(
                select(LegacyRecord).where(LegacyRecord.ward_job_id == ward_job_id, LegacyRecord.block_id == block_id)
            )
            .scalars()
            .all()
        )
        parcel_ids = [r.id for r in recorded]
        gt_rows = (
            session.execute(select(SurveyPoint).where(SurveyPoint.ward_job_id == ward_job_id, SurveyPoint.parcel_id.in_(parcel_ids), SurveyPoint.purpose == "fusion"))
            .scalars()
            .all()
            if parcel_ids
            else []
        )

        ward = _regenerate_ward(ward_source, ward_params)
        # local_block_id, not the global block_id: the regenerated `ward` object
        # still numbers its own blocks from 0 (see IngestedBlock.local_block_id's
        # own docstring)
        evidence_field, transform, evidence_provenance = _block_evidence(ward, block_row.local_block_id)

        parcel_areas = [r.area for r in recorded]
        seed_points = [(to_shape(r.seed_point).x, to_shape(r.seed_point).y) for r in recorded]
        # n_segments scaled to the raster's own pixel count, not assign_parcels()'s
        # fixed 2000 default -- on a small block that default asks SLIC for far
        # more superpixels than there are pixels, producing degenerate slivers
        # prone to an enclosed-island topology (found running this end to end,
        # not assumed): ~15px/superpixel is comfortably fine-grained for
        # boundary detail without over-fragmenting a small raster; matches the
        # order of magnitude Stage 3's own tests already use for small rasters.
        n_segments = max(len(recorded), min(2000, evidence_field.size // 15))
        result = assign_parcels(block_geom, parcel_areas, seed_points, evidence_field, transform, n_segments=n_segments)
        parcel_polygons = {recorded[i].id: poly for i, poly in result.parcel_polygons.items()}

        # Also recover stores containing a partial graph from an older worker:
        # retain its IDs/history and re-run fusion; never seed on top of it.
        if session.scalar(select(Face.id).where(Face.block_id == block_id).limit(1)) is None:
            fresh_graph = parcels_to_graph(parcel_polygons, block_geom)
            weights = {}
            for nid, node in fresh_graph.nodes.items():
                col, row = ~transform @ (node.x, node.y)
                row = min(max(int(round(row)), 0), evidence_field.shape[0] - 1)
                col = min(max(int(round(col)), 0), evidence_field.shape[1] - 1)
                weights[nid] = 1.0 + 20.0 * float(evidence_field[row, col])
            refinement = refine_recorded_areas(
                fresh_graph, block_geom, {r.id: r.area for r in recorded},
                {r.id: r.area_tolerance_m2 for r in recorded}, node_weights=weights,
            )
            fresh_graph = refinement.graph
            if not refinement.converged:
                session.add(PersistedConflict(
                    ward_job_id=ward_job_id, block_id=block_id, node_id=-1,
                    kind="area_refinement_unresolved", detail=refinement.detail,
                    sources=["recorded_area", "constrained_refinement"], disagreement_m=None,
                    geom=from_shape(block_geom.geom, srid=SRID),
                ))
            seed = seed_block_graph(session, block_id, fresh_graph, author="orchestrator", description="ingest: parcel assignment",
                evidence={"method": "capacity_constrained_transport", "ward_job_id": ward_job_id,
                          **evidence_provenance,
                          "area_refinement": refinement.detail,
                          "recorded_parcel_ids": parcel_ids, "source": ward_source, "n_segments": n_segments,
                          "assignment_conflicts": [dataclasses.asdict(c) for c in result.conflicts]})
            for conflict in fresh_graph.identity_conflicts:
                session.add(PersistedConflict(
                    ward_job_id=ward_job_id, block_id=block_id, node_id=-1,
                    kind="parcel_identity_unresolved", detail={**conflict, "face_id": seed.affected_entities["source_face_ids"][str(conflict["face_id"])], "changeset_id": seed.id},
                    sources=["assignment", "polygonization"], disagreement_m=None,
                    geom=from_shape(fresh_graph.face_polygon(conflict["face_id"]), srid=SRID),
                ))
        # seed_block_graph() remaps `fresh_graph`'s own local 0..n-1 node/edge/face
        # ids onto new globally-unique ids pulled from the store's sequences before
        # writing (Stage 2's own documented id-collision fix) -- so `fresh_graph`'s
        # ids no longer match what's actually in the store. fuse_block() must run
        # against the graph AS THE STORE NOW HAS IT (reloaded, with the real ids),
        # not the pre-seed graph, or apply_fusion() tries to move a node id that
        # was only ever real inside this function's own local, already-stale
        # object (found running this end to end: KeyError on the very first
        # fusion move, since the store's node 0 is a different corner entirely).
        graph = load_block_graph(session, block_id)

        legacy_boundary = unary_union([to_shape(r.geom).boundary for r in legacy]) if legacy else None
        # fuse_block()/gt_estimate() want plain (x, y) points -- fusion is a
        # pure nearest-neighbour spatial search, not parcel-scoped (a GT point
        # near a shared corner should inform that corner regardless of which
        # parcel "owns" it) -- GTPoint's own .parcel_id is a Stage 6 stratum
        # concept, not something fuse_block() ever asked for (found by
        # running this: gt_estimate() does np.asarray(gt_points, dtype=float),
        # which can't convert a GTPoint object).
        gt_points = [(to_shape(g.geom).x, to_shape(g.geom).y) for g in gt_rows]
        style = recorded[0].style if recorded else "formal"
        # co-registration residual (api/main.py's /co-register endpoint) widens
        # every style's legacy sigma by the same amount -- a poorly-aligned
        # upload makes the WHOLE legacy layer less trustworthy, not just one
        # style's share of it, so this inflates DEFAULT_SIGMA_LEGACY_BY_STYLE
        # uniformly rather than picking one style to blame.
        sigma_legacy_by_style = {
            s: base + (ward_job.coreg_residual_m or 0.0) for s, base in DEFAULT_SIGMA_LEGACY_BY_STYLE.items()
        }
        fuse_result = fuse_block(
            graph, list(graph.nodes), style=style, legacy_boundary=legacy_boundary, gt_points=gt_points,
            block_boundary=block_geom, sigma_legacy_by_style=sigma_legacy_by_style,
        )
        apply_result = apply_fusion(session, block_id, fuse_result, author="orchestrator", commit=False)

        for c in result.conflicts:
            session.add(PersistedConflict(
                ward_job_id=ward_job_id, block_id=block_id, node_id=-1,
                kind=c.kind, detail=c.detail, sources=["recorded_area", "assignment"],
                disagreement_m=None, geom=from_shape(block_geom.geom, srid=SRID),
            ))
        for refusal in apply_result.topology_refused:
            node = graph.nodes[refusal["node_id"]]
            session.add(PersistedConflict(
                ward_job_id=ward_job_id, block_id=block_id, node_id=node.id,
                kind="topology_refused", detail={"reason": refusal["reason"]},
                sources=list(refusal["sources"]), disagreement_m=None,
                geom=from_shape(Point(node.x, node.y), srid=SRID),
            ))
        for c in apply_result.conflicts:
            session.add(
                PersistedConflict(
                    ward_job_id=ward_job_id,
                    block_id=block_id,
                    node_id=c.node_id if c.node_id is not None else -1,
                    sources=list(c.sources),
                    disagreement_m=c.disagreement_m,
                    geom=from_shape(c.geometry.geom, srid=SRID),
                )
            )

        final_graph = load_block_graph(session, block_id)
        area_report = parcel_area_report(session, block_id, final_graph)
        for row in area_report["parcels"]:
            if not row["within_tolerance"]:
                session.add(PersistedConflict(
                    ward_job_id=ward_job_id, block_id=block_id, node_id=-1,
                    kind="final_recorded_area_mismatch", detail=row,
                    sources=["recorded_area", "final_geometry"], disagreement_m=None,
                    geom=from_shape(block_geom.geom, srid=SRID),
                ))
        coverage_error = area_report["block_coverage_error_m2"]
        if coverage_error is None or coverage_error > area_report["block_coverage_tolerance_m2"]:
            session.add(PersistedConflict(
                ward_job_id=ward_job_id, block_id=block_id, node_id=-1,
                kind="block_coverage_mismatch",
                detail={"error_m2": coverage_error, "tolerance_m2": area_report["block_coverage_tolerance_m2"]},
                sources=["block_boundary", "final_geometry"], disagreement_m=None,
                geom=from_shape(block_geom.geom, srid=SRID),
            ))

        if job is None:
            job = BlockJob(ward_job_id=ward_job_id, block_id=block_id)
            session.add(job)
        job.status = "done"
        job.error = None
        session.commit()
        return "done"
    except Exception as e:
        session.rollback()
        lock_block(session, block_id)
        # A different delivery may have completed after our rollback released
        # the lock. Never replace that successful status with our stale error.
        job = session.get(BlockJob, (ward_job_id, block_id), populate_existing=True)
        if job is not None and job.status == "done":
            raise
        if session.get(IngestedBlock, (ward_job_id, block_id)) is None:
            raise
        if job is None:
            job = BlockJob(ward_job_id=ward_job_id, block_id=block_id)
            session.add(job)
        job.status = "failed"
        job.error = str(e)
        session.commit()
        raise
    finally:
        session.close()


def ward_status(session: Session, ward_job_id: int) -> tuple[str, list]:
    """Derive progress from every expected block, including undispatched rows."""
    ward = session.get(WardJob, ward_job_id)
    if ward is None:
        raise ValueError(f"no ward {ward_job_id}")
    rows = session.execute(
        select(IngestedBlock.block_id, BlockJob.status, BlockJob.error)
        .outerjoin(BlockJob, (BlockJob.ward_job_id == IngestedBlock.ward_job_id) &
                   (BlockJob.block_id == IngestedBlock.block_id))
        .where(IngestedBlock.ward_job_id == ward_job_id).order_by(IngestedBlock.block_id)
    ).all()
    blocks = [{"block_id": bid, "status": state or "pending", "error": error} for bid, state, error in rows]
    states = {b["status"] for b in blocks}
    if states <= {"done"}:
        state = "done"
    elif "failed" in states:
        state = "failed"
    elif ward.status == "pending" and states <= {"pending"}:
        state = "pending"
    else:
        state = "running"
    return state, blocks


def run_ward(session: Session, db_url: str, schema: str, ward_job_id: int) -> dict:
    """Dispatch `process_block` for every block of `ward_job_id` that
    isn't already `done` -- the actual "resume" entry point: calling this
    again after a worker died mid-run only re-dispatches the blocks that
    never finished, per the Stage 8 Done-when. In eager mode (tests) this
    runs every dispatched block synchronously before returning; in
    production each `.delay()` enqueues and returns immediately, and a
    separate poll (see `api/main.py`) reports progress via `BlockJob`.

    Returns `{"total": N, "already_done": N, "dispatched": [block_id, ...]}`.
    """
    ward_job = session.get(WardJob, ward_job_id)
    if ward_job is None:
        raise ValueError(f"no ward_job {ward_job_id}")
    all_block_ids = (
        session.execute(select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == ward_job_id))
        .scalars()
        .all()
    )
    done_block_ids = {
        bj.block_id
        for bj in session.execute(
            select(BlockJob).where(BlockJob.ward_job_id == ward_job_id, BlockJob.status == "done")
        ).scalars()
    }
    to_dispatch = [bid for bid in all_block_ids if bid not in done_block_ids]

    ward_job.status = "running"
    session.commit()
    for block_id in to_dispatch:
        process_block.delay(db_url, schema, ward_job_id, block_id, ward_job.source, ward_job.params)

    ward_job.status, _ = ward_status(session, ward_job_id)
    session.commit()

    return {"total": len(all_block_ids), "already_done": len(done_block_ids), "dispatched": to_dispatch}
