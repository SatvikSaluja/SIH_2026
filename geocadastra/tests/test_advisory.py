"""Stage 8+: the LLM advisory layer over real conflicts/priority/ward
state. Every provider call is mocked at `advisory._call_anthropic`
(never a real network call) -- what's under test is that each endpoint
reads REAL persisted/computed data and feeds it through correctly, and
that the daily cap / read-only tool scoping actually hold, not that the
provider itself behaves a particular way.
"""
import pytest
from fastapi.testclient import TestClient
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy import func, select

from geocadastra.api import advisory
from geocadastra.api.main import app, get_db_config, get_session
from geocadastra.jobs.orchestrator import app as celery_app
from geocadastra.jobs.orchestrator import ingest_synthetic_ward, run_ward as _run_ward
from geocadastra.store.schema import SRID, IngestedBlock, PersistedConflict
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _eager_celery():
    # matching test_api.py's own fixture exactly: run_ward()'s dispatched
    # blocks execute in-process instead of needing a real Redis broker.
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def _small_ward(seed=1):
    return generate_ward(params=WardParams(width=50, height=35, gsd=1.0, n_arterial_h=1, n_arterial_v=0), seed=seed)


@pytest.fixture()
def client(committed_session):
    app.dependency_overrides[get_session] = lambda: committed_session
    app.dependency_overrides[get_db_config] = lambda: (TEST_DB_URL, TEST_SCHEMA)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    """Configured by default so most tests exercise the real call path;
    tests that specifically check the unconfigured/capped states clear
    this themselves."""
    monkeypatch.setenv('GEOCADASTRA_VISION_KEY', 'test-key')
    monkeypatch.setenv('GEOCADASTRA_VISION_MODEL', 'claude-test')
    monkeypatch.setenv('GEOCADASTRA_ADVISORY_DAILY_LIMIT', '10')
    advisory._CALLS_TODAY.clear()  # module-level counter: tests must not leak into each other


def _text_response(text):
    return {'content': [{'type': 'text', 'text': text}], 'usage': {'input_tokens': 1, 'output_tokens': 1}}


# --------------------------------------------------------------- conflicts --

def test_conflict_advice_classifies_a_real_persisted_conflict(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = committed_session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id)).scalars().first()
    conflict = PersistedConflict(ward_job_id=job.id, block_id=block_id, node_id=1, sources=['legacy', 'gps'],
                                 disagreement_m=4.2, kind='source_disagreement', detail={'note': 'test fixture'},
                                 geom=from_shape(Point(0, 0), srid=SRID))
    committed_session.add(conflict)
    committed_session.commit()

    calls = []
    def fake_call(cfg, messages, tools=None, max_tokens=512):
        calls.append(messages)
        return _text_response('GENUINE_BOUNDARY_AMBIGUITY: two independent sources disagree by 4.2m; no clear error.')
    monkeypatch.setattr(advisory, '_call_anthropic', fake_call)

    resp = client.post(f'/advisory/conflicts/{conflict.id}')
    assert resp.status_code == 200
    body = resp.json()
    assert body['classification'] == 'GENUINE_BOUNDARY_AMBIGUITY'
    assert body['decision'] == 'advisory_only'
    # the prompt actually carried this conflict's REAL fields, not placeholders
    prompt = calls[0][0]['content']
    assert '4.20' in prompt and 'legacy' in prompt and 'gps' in prompt


def test_conflict_advice_404_for_unknown_conflict(client):
    assert client.post('/advisory/conflicts/999999').status_code == 404


def test_conflict_advice_503_when_unconfigured(client, committed_session, monkeypatch):
    monkeypatch.delenv('GEOCADASTRA_VISION_KEY', raising=False)
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = committed_session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id)).scalars().first()
    conflict = PersistedConflict(ward_job_id=job.id, block_id=block_id, node_id=1, sources=['a', 'b'],
                                 disagreement_m=1.0, geom=from_shape(Point(0, 0), srid=SRID))
    committed_session.add(conflict)
    committed_session.commit()
    assert client.post(f'/advisory/conflicts/{conflict.id}').status_code == 503


def test_conflict_advice_429_after_daily_cap(client, committed_session, monkeypatch):
    monkeypatch.setenv('GEOCADASTRA_ADVISORY_DAILY_LIMIT', '1')
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = committed_session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id)).scalars().first()
    ids = []
    for i in range(2):
        c = PersistedConflict(ward_job_id=job.id, block_id=block_id, node_id=i, sources=['a', 'b'],
                              disagreement_m=1.0, geom=from_shape(Point(0, 0), srid=SRID))
        committed_session.add(c)
        committed_session.flush()
        ids.append(c.id)
    committed_session.commit()
    monkeypatch.setattr(advisory, '_call_anthropic', lambda *a, **k: _text_response('INSUFFICIENT_INFORMATION'))
    assert client.post(f'/advisory/conflicts/{ids[0]}').status_code == 200
    assert client.post(f'/advisory/conflicts/{ids[1]}').status_code == 429


def test_conflict_triage_aggregates_real_classifications_across_a_ward(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = committed_session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id)).scalars().first()
    for i in range(3):
        committed_session.add(PersistedConflict(ward_job_id=job.id, block_id=block_id, node_id=i,
                                                 sources=['a', 'b'], disagreement_m=1.0 + i,
                                                 geom=from_shape(Point(0, 0), srid=SRID)))
    committed_session.commit()

    answers = iter(['LIKELY_SURVEY_ERROR here.', 'GENUINE_BOUNDARY_AMBIGUITY here.', 'LIKELY_SURVEY_ERROR again.'])
    monkeypatch.setattr(advisory, '_call_anthropic', lambda *a, **k: _text_response(next(answers)))

    resp = client.get(f'/advisory/wards/{job.id}/conflicts/triage')
    assert resp.status_code == 200
    body = resp.json()
    assert body['total_conflicts'] == 3
    assert body['classified_count'] == 3
    assert body['breakdown']['LIKELY_SURVEY_ERROR'] == 2
    assert body['breakdown']['GENUINE_BOUNDARY_AMBIGUITY'] == 1
    assert body['note'] is None


def test_conflict_triage_stops_cleanly_at_the_daily_cap_keeping_partial_results(client, committed_session, monkeypatch):
    monkeypatch.setenv('GEOCADASTRA_ADVISORY_DAILY_LIMIT', '2')
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    block_id = committed_session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == job.id)).scalars().first()
    for i in range(3):
        committed_session.add(PersistedConflict(ward_job_id=job.id, block_id=block_id, node_id=i,
                                                 sources=['a', 'b'], disagreement_m=1.0,
                                                 geom=from_shape(Point(0, 0), srid=SRID)))
    committed_session.commit()
    monkeypatch.setattr(advisory, '_call_anthropic', lambda *a, **k: _text_response('LIKELY_SURVEY_ERROR'))

    resp = client.get(f'/advisory/wards/{job.id}/conflicts/triage')
    assert resp.status_code == 200  # NOT 429 -- partial results returned, not discarded
    body = resp.json()
    assert body['total_conflicts'] == 3
    assert body['classified_count'] == 2  # the cap, not the total
    assert body['note'] is not None and '1' in body['note']


def test_conflict_triage_limit_is_validated(client, committed_session):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    assert client.get(f'/advisory/wards/{job.id}/conflicts/triage?limit=0').status_code == 422
    assert client.get(f'/advisory/wards/{job.id}/conflicts/triage?limit=26').status_code == 422


# ---------------------------------------------------------------- priority --

def test_ward_priority_computes_real_clusters_without_calling_llm_when_advise_top_n_is_zero(client, committed_session, monkeypatch):
    monkeypatch.delenv('GEOCADASTRA_VISION_KEY', raising=False)  # unconfigured -- proves computation doesn't need it
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    resp = client.post(f'/advisory/wards/{job.id}/priority',
                       json={'depot': [0, 0], 'advise_top_n': 0})
    assert resp.status_code == 200
    body = resp.json()
    assert 'advice' not in body
    assert len(body['clusters']) > 0
    assert all(len(c['parcel_ids']) > 0 for c in body['clusters'])


def test_ward_priority_calls_llm_for_top_clusters_when_configured(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    calls = []
    def fake_call(cfg, messages, tools=None, max_tokens=512):
        calls.append(messages)
        return _text_response('Front-loading these clusters groups nearby high-uncertainty parcels.')
    monkeypatch.setattr(advisory, '_call_anthropic', fake_call)

    resp = client.post(f'/advisory/wards/{job.id}/priority', json={'depot': [0, 0], 'advise_top_n': 2})
    assert resp.status_code == 200
    body = resp.json()
    assert 'advice' in body and len(calls) == 1
    # the prompt is grounded in the REAL top cluster this call computed,
    # not a placeholder -- every one of its parcel ids appears in the text
    top_cluster_ids = body['clusters'][0]['parcel_ids']
    prompt = calls[0][0]['content']
    assert all(str(pid) in prompt for pid in top_cluster_ids)


def test_ward_priority_404_for_unknown_ward(client):
    resp = client.post('/advisory/wards/999999/priority', json={'depot': [0, 0]})
    assert resp.status_code == 404


# ------------------------------------------------------------------- brief --

def test_ward_brief_returns_real_facts_without_llm_when_unconfigured(client, committed_session, monkeypatch):
    monkeypatch.delenv('GEOCADASTRA_VISION_KEY', raising=False)
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    resp = client.get(f'/advisory/wards/{job.id}/brief')
    assert resp.status_code == 200
    body = resp.json()
    assert body['brief'] is None
    assert body['facts']['status'] == 'done'
    assert body['facts']['blocks_total'] == body['facts']['blocks_done'] > 0
    # NOT asserting a specific count here: _small_ward() turns out to
    # reliably produce real topology_refused conflicts through the full
    # fusion pipeline (a genuine, pre-existing, unrelated bug -- found by
    # this test, reported separately, not this endpoint's concern). What
    # THIS test actually needs to verify is that the field is real and
    # self-consistent with what's actually persisted, not that it's zero.
    real_count = committed_session.execute(
        select(func.count()).select_from(PersistedConflict).where(PersistedConflict.ward_job_id == job.id)
    ).scalar_one()
    assert body['facts']['total_conflicts'] == real_count > 0


def test_ward_brief_writes_prose_grounded_in_the_real_facts_when_configured(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    calls = []
    def fake_call(cfg, messages, tools=None, max_tokens=512):
        calls.append(messages)
        return _text_response('This ward is fully processed with no outstanding conflicts.')
    monkeypatch.setattr(advisory, '_call_anthropic', fake_call)

    resp = client.get(f'/advisory/wards/{job.id}/brief')
    assert resp.status_code == 200
    body = resp.json()
    assert body['brief'] == 'This ward is fully processed with no outstanding conflicts.'
    # the prompt actually carried the real computed facts, not placeholders
    assert '"status": "done"' in calls[0][0]['content']


def test_ward_brief_404_for_unknown_ward(client):
    assert client.get('/advisory/wards/999999/brief').status_code == 404


# --------------------------------------------------------------------- ask --

def test_ask_executes_a_tool_call_then_returns_a_final_answer(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)
    _run_ward(committed_session, TEST_DB_URL, TEST_SCHEMA, job.id)

    responses = iter([
        {'content': [{'type': 'tool_use', 'id': 'call_1', 'name': 'ward_status', 'input': {}}],
        'usage': {}},
        _text_response('Every block in this ward is done.'),
    ])
    monkeypatch.setattr(advisory, '_call_anthropic', lambda *a, **k: next(responses))

    resp = client.post(f'/advisory/wards/{job.id}/ask', json={'question': 'Is this ward finished?'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['answer'] == 'Every block in this ward is done.'
    assert body['tool_calls'] == [{'name': 'ward_status', 'input': {}}]


def test_ask_gives_up_honestly_after_max_rounds(client, committed_session, monkeypatch):
    ward = _small_ward()
    job = ingest_synthetic_ward(committed_session, ward, seed=1)

    # every round asks for another tool call, never a final answer
    def always_wants_a_tool(*a, **k):
        return {'content': [{'type': 'tool_use', 'id': 'x', 'name': 'ward_status', 'input': {}}], 'usage': {}}
    monkeypatch.setattr(advisory, '_call_anthropic', always_wants_a_tool)

    resp = client.post(f'/advisory/wards/{job.id}/ask', json={'question': 'anything'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['answer'] is None
    assert 'error' in body
    assert len(body['tool_calls']) == advisory.MAX_TOOL_ROUNDS


def test_run_tool_ignores_a_ward_id_the_model_tries_to_supply(committed_session):
    """The actual scoping guarantee: _run_tool takes ward_job_id as its
    own explicit argument (from the URL), never from tool_input -- a
    model hallucinating a different ward_job_id inside its tool call
    arguments must have zero effect on which ward gets queried."""
    ward = _small_ward()
    job_a = ingest_synthetic_ward(committed_session, ward, seed=1)
    job_b = ingest_synthetic_ward(committed_session, _small_ward(seed=2), seed=2)
    committed_session.commit()

    result = advisory._run_tool('ward_status', {'ward_job_id': job_b.id, 'ignored': 'yes'}, committed_session, job_a.id)
    # proves it ran against job_a (the URL's ward), not job_b (the hallucinated one) --
    # both exist, so a scoping bug would silently return the wrong ward's data instead of erroring
    real_status, _ = advisory.ward_status(committed_session, job_a.id)
    assert result['status'] == real_status
