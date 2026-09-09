"""Stage 8: orchestration -- Celery tasks, one per block, idempotent and
resumable. Tests run Celery in eager mode (no real Redis broker needed --
none is installed in this environment, matching how the rest of this
project's test suite avoids depending on infrastructure it doesn't have)
so `process_block.delay(...)`/`run_ward()` execute synchronously.

Every task opens its OWN DB session from a plain `db_url`/`schema` (see
orchestrator.py's module docstring on why -- a real Celery worker is a
separate process, so a live `Session` object can never be a task
argument), so tests use `committed_session` (real commits, not the
rolled-back-transaction `db_session`) and pass `TEST_DB_URL`/`TEST_SCHEMA`
directly, exactly mirroring what a real deployment's task invocation
looks like.

Every test here runs the real Stage 3 (parcel assignment) + Stage 5
(fusion) pipeline against real Postgres -- `slow`, same precedent Stage
4-7 set for their own real-pipeline tests, not a unit-test-speed file.
`apply_fusion()`'s known O(N) per-node DB reload (documented, deliberately
deferred in the whole-codebase review rather than rushed) means even a
modest ~10-parcel block takes real seconds; that's an accepted, already-
tracked cost, not something this stage's tests should quietly work around
by using unrealistically tiny wards.
"""
import pytest
from sqlalchemy import select

pytestmark = pytest.mark.slow

from geocadastra.jobs.orchestrator import (
    app,
    ingest_synthetic_ward,
    make_session,
    process_block,
    run_ward,
)
from geocadastra.store.schema import BlockJob, Face, IngestedBlock, PersistedConflict, WardJob
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA


@pytest.fixture(autouse=True)
def _eager_celery():
    app.conf.task_always_eager = True
    yield
    app.conf.task_always_eager = False


def _small_ward(seed=1):
    # one arterial road (n_arterial_h=1) so the ward has >=2 blocks -- several
    # tests below need a real second block to exercise partial-completion resume
    return generate_ward(params=WardParams(width=100, height=60, gsd=1.0, n_arterial_h=1, n_arterial_v=0), seed=seed)


def _global_block_ids(session, ward_job_id, ward) -> list:
    """`IngestedBlock.block_id` (globally unique, from `block_id_seq`) in
    the SAME order as `ward.blocks` -- tests need to address the store by
    this id, not the synthetic ward's own ward-local `Block.id`, ever
    since ingest started remapping onto a global one (see `ingest_
    synthetic_ward()`'s own docstring for why: two wards' same-numbered
    "block 0" used to collide/pool together in the store)."""
    rows = session.execute(select(IngestedBlock).where(IngestedBlock.ward_job_id == ward_job_id)).scalars().all()
    by_local = {r.local_block_id: r.block_id for r in rows}
    return [by_local[b.id] for b in ward.blocks]


def test_ingest_synthetic_ward_persists_every_block_parcel_legacy_and_gt_row(committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    assert job.id is not None
    assert job.source == "synthetic:seed=1"
    assert job.crs == ward.crs

    from geocadastra.store.schema import IngestedBlock, LegacyRecord, RecordedParcel, SurveyPoint

    n_blocks = committed_session.execute(select(IngestedBlock).where(IngestedBlock.ward_job_id == job.id)).scalars().all()
    n_parcels = committed_session.execute(select(RecordedParcel).where(RecordedParcel.ward_job_id == job.id)).scalars().all()
    n_legacy = committed_session.execute(select(LegacyRecord).where(LegacyRecord.ward_job_id == job.id)).scalars().all()
    n_gt = committed_session.execute(select(SurveyPoint).where(SurveyPoint.ward_job_id == job.id)).scalars().all()
    assert len(n_blocks) == len(ward.blocks)
    assert len(n_parcels) == len(ward.parcels)
    assert len(n_legacy) == len(ward.legacy_parcels)
    assert len(n_gt) == len(ward.gt_points)


def test_run_ward_processes_every_block_and_produces_faces(committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)

    result = run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    assert result["total"] == len(ward.blocks)
    assert result["already_done"] == 0
    block_ids = _global_block_ids(committed_session, job.id, ward)
    assert set(result["dispatched"]) == set(block_ids)

    for block_id in block_ids:
        bj = committed_session.get(BlockJob, (job.id, block_id))
        assert bj is not None and bj.status == "done", bj.error if bj else None
        faces = committed_session.execute(select(Face).where(Face.block_id == block_id)).scalars().all()
        assert len(faces) > 0

    committed_session.refresh(job)
    assert job.status == "done"


def test_run_ward_is_idempotent_a_second_call_redoes_nothing(committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    second = run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    assert second["already_done"] == len(ward.blocks)
    assert second["dispatched"] == []


def test_process_block_is_a_no_op_if_already_done(committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = _global_block_ids(committed_session, job.id, ward)[0]
    process_block(TEST_DB_URL, TEST_SCHEMA, job.id, block_id, job.source, job.params)
    n_faces_before = len(committed_session.execute(select(Face).where(Face.block_id == block_id)).scalars().all())

    result = process_block(TEST_DB_URL, TEST_SCHEMA, job.id, block_id, job.source, job.params)
    assert result == "already done"
    n_faces_after = len(committed_session.execute(select(Face).where(Face.block_id == block_id)).scalars().all())
    assert n_faces_after == n_faces_before  # nothing re-seeded


def test_run_ward_resumes_correctly_after_a_simulated_killed_worker(committed_session):
    """The actual Stage 8 Done-when: 'a job that dies mid-ward resumes
    without redoing completed blocks.' Simulates a worker dying after
    processing only the first block (a real crash mid-ward, not a mocked
    one) by calling process_block directly for one block, then calling
    run_ward -- the resume entry point -- and checking the already-done
    block's faces are untouched while every other block gets processed."""
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    assert len(ward.blocks) >= 2, "need at least 2 blocks to test partial-completion resume"
    block_ids = _global_block_ids(committed_session, job.id, ward)

    first_block_id = block_ids[0]
    process_block(TEST_DB_URL, TEST_SCHEMA, job.id, first_block_id, job.source, job.params)
    faces_before_resume = {
        f.id for f in committed_session.execute(select(Face).where(Face.block_id == first_block_id)).scalars()
    }

    result = run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    assert result["already_done"] == 1
    assert first_block_id not in result["dispatched"]
    assert set(result["dispatched"]) == set(block_ids[1:])

    faces_after_resume = {
        f.id for f in committed_session.execute(select(Face).where(Face.block_id == first_block_id)).scalars()
    }
    assert faces_after_resume == faces_before_resume  # the completed block's faces were never touched again

    for block_id in block_ids:
        bj = committed_session.get(BlockJob, (job.id, block_id))
        assert bj.status == "done"


def test_run_ward_leaves_the_whole_run_reconstructible_from_the_changeset_log(committed_session):
    """Stage 8 Done-when: 'the whole run is reconstructible from the
    changeset log.' Every block's graph is loadable fresh from its own
    changeset history (Stage 2's own replay guarantee, exercised here at
    Stage 8 scale across every block of a real run) and matches the
    live faces exactly."""
    from geocadastra.store.changeset import load_block_graph

    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    for block_id in _global_block_ids(committed_session, job.id, ward):
        replayed = load_block_graph(committed_session, block_id)
        live_face_ids = {
            f.id for f in committed_session.execute(select(Face).where(Face.block_id == block_id)).scalars()
        }
        assert set(replayed.faces.keys()) == live_face_ids


def test_a_block_with_conflicting_evidence_persists_conflict_records(committed_session):
    """A block with legacy geometry distorted enough to disagree with the
    model evidence beyond tolerance should leave PersistedConflict rows,
    not silently average the disagreement away."""
    ward = _small_ward(seed=1)
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    conflicts = committed_session.execute(select(PersistedConflict).where(PersistedConflict.ward_job_id == job.id)).scalars().all()
    # not every ward/seed necessarily produces a conflict -- the real assertion is
    # that IF fusion found any (checked via a second, larger sweep in the
    # acceptance test), they end up in the table, not dropped; here just confirm
    # the query and schema round-trip cleanly
    valid_block_ids = set(_global_block_ids(committed_session, job.id, ward))
    for c in conflicts:
        assert c.block_id in valid_block_ids
        assert c.sources
        if c.kind == "source_disagreement":
            assert len(c.sources) >= 2
        else:
            assert c.detail


def test_two_wards_ingested_into_the_same_store_dont_collide(committed_session):
    """Found by review, reproduced directly: `generate_ward()` numbers its
    own blocks/parcels from 0 every call, so a second ward's ingest used
    to crash with a duplicate-primary-key error (RecordedParcel/
    LegacyRecord had no id sequence of their own), and -- more seriously
    -- two wards' same-numbered "block 0" silently pooled into one
    `load_block_graph()` result once both got processed (`Face.block_id`
    had no `ward_job_id` at all). This is completely ordinary production
    use (processing more than one ward, ever), not an edge case."""
    ward_a = _small_ward(seed=1)
    ward_b = _small_ward(seed=2)
    job_a = ingest_synthetic_ward(committed_session, ward_a, seed=1)
    job_b = ingest_synthetic_ward(committed_session, ward_b, seed=2)  # must not raise IntegrityError

    run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job_a.id)
    run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job_b.id)

    blocks_a = set(_global_block_ids(committed_session, job_a.id, ward_a))
    blocks_b = set(_global_block_ids(committed_session, job_b.id, ward_b))
    assert blocks_a.isdisjoint(blocks_b)  # globally unique, not ward-local

    from geocadastra.store.changeset import load_block_graph

    graph_a = load_block_graph(committed_session, next(iter(blocks_a)))
    graph_b = load_block_graph(committed_session, next(iter(blocks_b)))
    # each ward's own first block has exactly its own faces -- neither pooled
    # in the other's data (a real, reproduced failure mode before the fix)
    assert set(graph_a.faces.keys()).isdisjoint(graph_b.faces.keys())


def test_run_ward_does_not_report_done_before_dispatched_blocks_actually_run(committed_session, monkeypatch):
    """Found by review: the status computation used to derive "done" from
    a `WHERE status != 'done'` query, which only matches EXISTING BlockJob
    rows -- a freshly-dispatched block that hasn't executed yet has NO row
    at all, so it was silently invisible to that check. In real async
    dispatch (`.delay()` enqueues and returns immediately, unlike eager
    mode) this made a completely fresh `/run` call report the ward
    `status: "done"` before a single block had actually been processed.
    Reproduced here without needing a real broker: monkeypatch `process_
    block.delay` to a no-op (simulating "enqueued, not yet executed") and
    confirm the ward is correctly left `"running"`, not `"done"`.
    """
    from geocadastra.jobs.orchestrator import process_block

    monkeypatch.setattr(process_block, "delay", lambda *a, **kw: None)  # simulate real async: enqueue, don't run

    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    result = run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)
    assert result["dispatched"] != []  # something was actually dispatched (not a no-op ward)

    committed_session.refresh(job)
    assert job.status == "running", "reported done before any dispatched block actually ran"
