"""Orchestration (Stage 8): Celery tasks, one per block, idempotent and
resumable -- a job that dies mid-ward resumes without redoing completed
blocks, per the Stage 8 Done-when.

Each block is processed independently and atomically: parcel assignment
(Stage 3) -> seed the block's planar graph (Stage 1/2) -> fuse against
legacy/GT evidence (Stage 5) -> persist. `BlockJob.status` is the
resumability primitive -- `process_block` checks it first and is a no-op
if already `done`; `run_ward` (or a resumed call to it) only dispatches
blocks that aren't. A mid-block crash can't leave a half-seeded graph:
every write happens inside Stage 2's own transactional
`seed_block_graph()`/`apply_fusion()`, and `process_block`'s own outer
try/except marks the block `failed` (not `done`) on any exception, so a
retry re-attempts the whole block rather than resuming partway through
it -- a real, if slightly coarser than block-internal, resumability unit.

A Celery task's arguments must be picklable and, in a real deployment,
sent across a message queue to a separate worker process -- so `process_
block` takes a plain `db_url`/`schema` (to build its own DB session, not
a live `Session` object) and a `ward_source` string, not a Python object.
The block's own true geometry, recorded areas, legacy records, and GT
points were all durably ingested (see `api/main.py`'s ingest endpoint)
and are read back from the DB, exactly the state a resume needs. Only the
raster EVIDENCE FIELD (Stage 4's model output) is not read from storage:
this project has no real drone imagery or raster store yet ("Real drone
data does not exist yet in this repo... everything must work on
synthetic first" -- the doc's own words), so `_regenerate_ward()`
deterministically re-derives the same synthetic ward `generate_ward()`
produced at ingest time and calls `simulate_evidence_field()` on it --
the exact same stand-in every other stage's acceptance tests already use
in place of a real trained Stage 4 model. A real deployment would read a
stored raster tile by block id instead; this is an explicit, acknowledged
simplification, not a silently different code path production would take
too.
"""
from __future__ import annotations

import dataclasses
import os

from celery import Celery
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point
from shapely.ops import unary_union
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from geocadastra.core.crs import Geom
from geocadastra.core.fusion import DEFAULT_SIGMA_LEGACY_BY_STYLE, fuse_block
from geocadastra.core.transport import assign_parcels, parcels_to_graph
from geocadastra.store.changeset import allocate_ids, apply_fusion, load_block_graph, seed_block_graph
from geocadastra.store.schema import (
    SRID,
    BlockJob,
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
                geom=from_shape(lp.polygon, srid=SRID),
            )
        )
    for gt in ward.gt_points:
        session.add(
            SurveyPoint(
                ward_job_id=ward_job.id, parcel_id=parcel_id_map[gt.parcel_id], geom=from_shape(Point(gt.x, gt.y), srid=SRID),
                source="synthetic_gt",
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
    return generate_ward(params=WardParams(**params), seed=seed)


@app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_block(self, db_url: str, schema: str, ward_job_id: int, block_id: int, ward_source: str, ward_params: dict) -> str:
    """Process one block end to end. Returns `"already done"`, `"done"`,
    or raises (after recording the block `failed`).

    Two genuinely concurrent calls for the SAME block (a real risk in a
    multi-worker deployment -- e.g. a task redelivered after its worker
    was thought dead but wasn't) both racing past the "already done"
    check would both run the full, non-idempotent seed/fuse pipeline,
    producing two duplicate copies of the block's graph (found by
    review, reproduced directly: `seed_block_graph()` has no check for
    "does this block already have faces" and no uniqueness constraint
    stops it). A session-scoped advisory lock keyed on `(ward_job_id,
    block_id)`, held for this call's whole duration and explicitly
    released in `finally`, serializes that: a second concurrent call
    blocks here until the first finishes, then sees `status == "done"`
    and returns cleanly instead of racing it. Session-scoped (`pg_
    advisory_lock`/`_unlock`), not the transaction-scoped `_xact_lock`
    `store/provenance.py` uses elsewhere in this project -- this
    function's own pipeline already commits multiple times internally
    (one changeset per fused node, by design, so one unsafe move can't
    block every other safe one in the same block), which would release
    an xact-scoped lock at the FIRST of those commits, not at the end.
    """
    session = make_session(db_url, schema)
    session.execute(text("SELECT pg_advisory_lock(:a, :b)"), {"a": ward_job_id, "b": block_id})
    try:
        job = session.get(BlockJob, (ward_job_id, block_id))
        if job is not None and job.status == "done":
            return "already done"

        ward_job = session.get(WardJob, ward_job_id)
        crs = ward_job.crs
        block_row = session.get(IngestedBlock, (ward_job_id, block_id))
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
            session.execute(select(SurveyPoint).where(SurveyPoint.ward_job_id == ward_job_id, SurveyPoint.parcel_id.in_(parcel_ids)))
            .scalars()
            .all()
            if parcel_ids
            else []
        )

        ward = _regenerate_ward(ward_source, ward_params)
        # local_block_id, not the global block_id: the regenerated `ward` object
        # still numbers its own blocks from 0 (see IngestedBlock.local_block_id's
        # own docstring)
        evidence_field, transform = simulate_evidence_field(ward, block_row.local_block_id)

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

        fresh_graph = parcels_to_graph(parcel_polygons, block_geom)
        seed_block_graph(session, block_id, fresh_graph, author="orchestrator", description="ingest: parcel assignment")
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

        legacy_boundary = unary_union([to_shape(r.geom) for r in legacy]) if legacy else None
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
        apply_result = apply_fusion(session, block_id, fuse_result, author="orchestrator")

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

        if job is None:
            job = BlockJob(ward_job_id=ward_job_id, block_id=block_id)
            session.add(job)
        job.status = "done"
        job.error = None
        session.commit()
        return "done"
    except Exception as e:
        session.rollback()
        job = session.get(BlockJob, (ward_job_id, block_id))
        if job is None:
            job = BlockJob(ward_job_id=ward_job_id, block_id=block_id)
            session.add(job)
        job.status = "failed"
        job.error = str(e)
        session.commit()
        raise
    finally:
        try:
            session.execute(text("SELECT pg_advisory_unlock(:a, :b)"), {"a": ward_job_id, "b": block_id})
            session.commit()  # the unlock call itself opens a new implicit transaction after the prior rollback/commit
        except Exception:
            pass  # the connection is already broken -- Postgres releases session-scoped advisory
            # locks automatically when the backend connection itself closes, so this isn't a
            # permanent leak, just a delayed release; never mask whatever exception is already
            # propagating (or a clean return) over a failure to unlock promptly
        session.close()


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

    # re-query which blocks are ACTUALLY done now, rather than trusting the
    # pre-dispatch `done_block_ids`/`to_dispatch` sets: in eager mode the loop
    # above already ran every dispatched task synchronously, so this reflects
    # the real post-run state; in real async dispatch, a freshly-dispatched
    # block that hasn't executed yet has NO `BlockJob` row at all, so it's
    # correctly absent from this query too -- an earlier version here computed
    # "remaining" as `WHERE status != 'done'`, which only matches EXISTING
    # rows and so silently missed exactly that case (found by review,
    # reproduced: it made a fresh ward's `/run` report `status: "done"`
    # immediately after dispatch, before a single block had actually run, in
    # any real -- non-eager -- deployment).
    done_now = {
        bj.block_id
        for bj in session.execute(
            select(BlockJob).where(BlockJob.ward_job_id == ward_job_id, BlockJob.status == "done")
        ).scalars()
    }
    ward_job.status = "done" if set(all_block_ids) <= done_now else "running"
    session.commit()

    return {"total": len(all_block_ids), "already_done": len(done_block_ids), "dispatched": to_dispatch}
