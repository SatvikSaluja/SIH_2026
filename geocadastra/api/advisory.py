"""LLM advisory layer over already-computed GeoCadastra data: single and
batch-aggregated fusion-conflict summaries, ward-wide priority-order
justification, a plain-language ward status brief, and a bounded
read-only Q&A tool loop over a ward's real state.

Deliberately thin: every function here reads data through the SAME
session-scoped queries/helpers the rest of this API already uses
(_get_ward_job_or_404, ward_status, load_block_graph) rather than a
second way of reaching the store, and writes nothing back -- no
endpoint here resolves a conflict, reorders a survey, or edits
geometry. Every response is explicitly advisory, matching
workspace.py's vision feature's own contract.

Shares the Anthropic provider credentials with workspace.py's vision
feature (GEOCADASTRA_VISION_KEY/GEOCADASTRA_VISION_MODEL -- one
provider account, not a second credential to configure) but tracks its
own daily budget (GEOCADASTRA_ADVISORY_DAILY_LIMIT, default 10) so a
busy day of image review doesn't starve conflict/priority/chat
advisory, or vice versa.

# ponytail: the daily counter is an in-process dict, resets on API
# restart -- move to a persisted table (PersistedConflict's own
# pattern) if this needs to survive restarts. It also counts one
# reservation per /ask call, not per underlying provider round-trip --
# a single question can cost up to MAX_TOOL_ROUNDS real API calls, so
# the real provider spend can run ahead of what the counter shows.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from geoalchemy2.shape import from_shape
from pydantic import BaseModel, Field
from shapely.geometry import box
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from geocadastra.api.deps import get_session, _get_ward_job_or_404
from geocadastra.core.priority import CostModel, build_parcel_adjacency, build_priority_order, face_centroids, face_uncertainty
from geocadastra.jobs.orchestrator import ward_status
from geocadastra.store.changeset import load_block_graph
from geocadastra.store.constraints import parcel_area_report
from geocadastra.store.schema import SRID, Face, IngestedBlock, PersistedConflict

router = APIRouter(prefix='/advisory', tags=['LLM advisory'])

_CALL_LOCK = threading.Lock()
_CALLS_TODAY: dict[str, int] = {}


def _llm_config():
    return {'configured': bool(os.environ.get('GEOCADASTRA_VISION_KEY') and os.environ.get('GEOCADASTRA_VISION_MODEL')),
           'model': os.environ.get('GEOCADASTRA_VISION_MODEL'),
           'daily_limit': int(os.environ.get('GEOCADASTRA_ADVISORY_DAILY_LIMIT', '10'))}


def _reserve_call_or_429(cfg):
    if not cfg['configured']:
        raise HTTPException(503, 'Configure GEOCADASTRA_VISION_KEY and GEOCADASTRA_VISION_MODEL on the server.')
    day = datetime.now(timezone.utc).date().isoformat()
    with _CALL_LOCK:
        used = _CALLS_TODAY.get(day, 0)
        if used >= cfg['daily_limit']:
            raise HTTPException(429, 'Daily advisory request cap reached')
        _CALLS_TODAY[day] = used + 1


def _call_anthropic(cfg, messages, tools=None, max_tokens=512):
    import httpx
    payload = {'model': cfg['model'], 'max_tokens': max_tokens, 'messages': messages}
    if tools:
        payload['tools'] = tools
    response = httpx.post('https://api.anthropic.com/v1/messages',
                          headers={'x-api-key': os.environ['GEOCADASTRA_VISION_KEY'], 'anthropic-version': '2023-06-01'},
                          json=payload, timeout=45)
    response.raise_for_status()
    return response.json()


def _text_of(content_blocks):
    return '\n'.join(b.get('text', '') for b in content_blocks if b.get('type') == 'text')


# --------------------------------------------------------------- conflicts --

_CONFLICT_CLASSES = ('LIKELY_SURVEY_ERROR', 'LIKELY_DIGITIZATION_NOISE', 'GENUINE_BOUNDARY_AMBIGUITY', 'INSUFFICIENT_INFORMATION')


def _conflict_prompt(conflict):
    """Shared by conflict_advice (one conflict) and conflict_triage (a
    batch) -- one prompt definition, not two that can silently drift
    apart and make triage's aggregate stop meaning the same thing as a
    single conflict_advice call would have classified it as.
    """
    return (
        f'A cadastral boundary-fusion process found {len(conflict.sources)} sources disagreeing by '
        f'{conflict.disagreement_m:.2f} metres at one location (recorded kind: {conflict.kind}). '
        f'Contributing sources: {", ".join(conflict.sources)}. Additional detail: {json.dumps(conflict.detail)}.\n\n'
        f'Classify this as EXACTLY one of: {", ".join(_CONFLICT_CLASSES)}. Then give a short (2-3 '
        'sentence) plain-language explanation a non-surveyor could read. Do not resolve the conflict '
        'or pick a winning source -- classify and explain only, for human review.'
    )


def _classify(cfg, conflict):
    result = _call_anthropic(cfg, [{'role': 'user', 'content': _conflict_prompt(conflict)}])
    answer = _text_of(result.get('content', []))
    classification = next((c for c in _CONFLICT_CLASSES if c in answer), 'UNCLASSIFIED')
    return classification, answer, result.get('usage', {})


@router.post('/conflicts/{conflict_id}')
def conflict_advice(conflict_id: int, session: Session = Depends(get_session)):
    """Plain-language advisory over ONE real, persisted conflict -- the
    same disagreement a human currently has to read raw source names and
    a disagreement magnitude to interpret. Structured (a constrained
    classification alongside the prose), so a sample of these can be
    aggregated later into an actual measured breakdown instead of only
    ever being read one at a time -- see conflict_triage() below, which
    is exactly that aggregation.
    """
    conflict = session.get(PersistedConflict, conflict_id)
    if conflict is None:
        raise HTTPException(404, f'no conflict {conflict_id}')
    cfg = _llm_config()
    _reserve_call_or_429(cfg)
    classification, answer, usage = _classify(cfg, conflict)
    return {'conflict_id': conflict_id, 'classification': classification, 'advice': answer,
           'usage': usage, 'decision': 'advisory_only'}


@router.get('/wards/{ward_job_id}/conflicts/triage')
def conflict_triage(ward_job_id: int, limit: int = 10, session: Session = Depends(get_session)):
    """The aggregate conflict_advice()'s structured classification was
    built for: what fraction of this ward's conflicts are real survey
    disputes vs. digitization noise vs. genuinely ambiguous -- a measured
    breakdown, not a guess, the same rigor this project has applied to
    the ML side's own "how much of the gap is real vs. fixable" question.

    Capped at `limit` conflicts (default 10, max 25) -- triaging an
    entire large ward in one blocking request would be slow and could
    exhaust the whole daily budget in a single call. If the cap is hit
    partway through, returns what was actually classified rather than
    discarding it behind a 429 -- a partial, honest answer beats an
    all-or-nothing one here.
    """
    _get_ward_job_or_404(session, ward_job_id)
    if not (1 <= limit <= 25):
        raise HTTPException(422, 'limit must be between 1 and 25')
    conflicts = session.execute(
        select(PersistedConflict).where(PersistedConflict.ward_job_id == ward_job_id)
        .order_by(PersistedConflict.id).limit(limit)).scalars().all()
    total = session.execute(select(func.count()).select_from(PersistedConflict)
                            .where(PersistedConflict.ward_job_id == ward_job_id)).scalar_one()
    cfg = _llm_config()
    breakdown = {c: 0 for c in _CONFLICT_CLASSES}
    breakdown['UNCLASSIFIED'] = 0
    classified = []
    for conflict in conflicts:
        try:
            _reserve_call_or_429(cfg)
        except HTTPException:
            break  # daily cap hit mid-batch -- stop cleanly, keep what's already classified
        classification, _answer, _usage = _classify(cfg, conflict)
        breakdown[classification] += 1
        classified.append({'conflict_id': conflict.id, 'classification': classification})
    return {'ward_job_id': ward_job_id, 'total_conflicts': total, 'classified_count': len(classified),
           'breakdown': breakdown, 'classified': classified,
           'note': (f'{total - len(classified)} conflict(s) not yet triaged (limit or daily cap)'
                    if len(classified) < total else None),
           'decision': 'advisory_only'}


# ---------------------------------------------------------------- priority --

class PriorityRequest(BaseModel):
    edge_uncertainty: dict[int, float] = Field(
        default_factory=dict,
        description='edge_id -> uncertainty in metres. This project has no single already-persisted '
                    'per-edge uncertainty source across calibration/model-confidence yet, so it is '
                    'supplied by the caller rather than guessed here -- guessing one would produce an '
                    'order that LOOKS real but rests on a fabricated number. Edges not present default '
                    'to unknown (treated as needing a visit), the same safe default face_uncertainty() '
                    'already uses internally.')
    depot: tuple[float, float]
    minutes_per_parcel: float = Field(15.0, gt=0)
    travel_speed_m_per_min: float = Field(60.0, gt=0)
    cluster_distance: float = Field(50.0, gt=0)
    tolerance: float = Field(0.3, gt=0)
    advise_top_n: int = Field(3, ge=0, le=10, description='0 skips the LLM call entirely -- pure computation, no cost.')


@router.post('/wards/{ward_job_id}/priority')
def ward_priority(ward_job_id: int, req: PriorityRequest, session: Session = Depends(get_session)):
    """Real priority computation (Stage 7's build_priority_order) across
    every block in this ward, merged exactly the way priority.py's own
    docstring says it was designed for ("a caller can merge several
    blocks' worth of these into one ward-wide priority order") -- not a
    new aggregation invented here.
    """
    _get_ward_job_or_404(session, ward_job_id)
    block_ids = [row[0] for row in session.execute(
        select(IngestedBlock.block_id).where(IngestedBlock.ward_job_id == ward_job_id)).all()]
    if not block_ids:
        raise HTTPException(404, f'ward {ward_job_id} has no ingested blocks yet')
    adjacency, centroids, uncertainty = {}, {}, {}
    for block_id in block_ids:
        graph = load_block_graph(session, block_id)
        adjacency.update(build_parcel_adjacency(graph))
        centroids.update(face_centroids(graph))
        uncertainty.update(face_uncertainty(graph, req.edge_uncertainty))
    if not adjacency:
        raise HTTPException(404, f'ward {ward_job_id} has no parcels yet')
    cost_model = CostModel(minutes_per_parcel=req.minutes_per_parcel, travel_speed_m_per_min=req.travel_speed_m_per_min)
    ranked = build_priority_order(adjacency, uncertainty, centroids, req.depot, cost_model,
                                  req.cluster_distance, req.tolerance)
    result = {'ward_job_id': ward_job_id, 'clusters': [{'parcel_ids': c} for c in ranked]}
    if req.advise_top_n and ranked:
        cfg = _llm_config()
        if cfg['configured']:
            _reserve_call_or_429(cfg)
            top = ranked[:req.advise_top_n]
            prompt = (
                f'A field-survey priority queue ranked {len(ranked)} parcel clusters by uncertainty-'
                f'weighted centrality and travel cost from a fixed depot. The top {len(top)} clusters '
                f'(by parcel id) are: {top}. In 2-3 sentences, explain in plain language why front-'
                'loading these clusters is a reasonable survey order for a human coordinator, given '
                'they group both high-uncertainty AND spatially-clustered parcels together. Do not '
                'invent facts about any specific parcel you have no data on.'
            )
            resp = _call_anthropic(cfg, [{'role': 'user', 'content': prompt}])
            result['advice'] = _text_of(resp.get('content', []))
            result['usage'] = resp.get('usage', {})
    return result


# ------------------------------------------------------------------- brief --

@router.get('/wards/{ward_job_id}/brief')
def ward_brief(ward_job_id: int, session: Session = Depends(get_session)):
    """One short paragraph, plain-language state of this ward -- for a
    stakeholder who wants the gist without reading status/conflicts/
    constraints as three separate calls. The LLM's only job is turning
    already-computed real numbers into prose; it derives nothing itself,
    and `facts` (always present) is exactly what the prose is built
    from, so any claim in `brief` can be checked against it directly.

    Works without a configured provider too -- `facts` alone is real,
    useful data; `brief` is just `None` with a note, the same graceful-
    without-LLM shape as ward_priority()'s advise_top_n=0 path.
    """
    _get_ward_job_or_404(session, ward_job_id)
    state, blocks = ward_status(session, ward_job_id)
    conflict_rows = session.execute(
        select(PersistedConflict.kind, func.count()).where(PersistedConflict.ward_job_id == ward_job_id)
        .group_by(PersistedConflict.kind)).all()
    conflicts_by_kind = {kind: count for kind, count in conflict_rows}
    done_block_ids = [b['block_id'] for b in blocks if b['status'] == 'done']
    reports = [parcel_area_report(session, bid, load_block_graph(session, bid)) for bid in done_block_ids]
    facts = {
        'status': state,
        'blocks_total': len(blocks),
        'blocks_done': sum(1 for b in blocks if b['status'] == 'done'),
        'blocks_failed': sum(1 for b in blocks if b['status'] == 'failed'),
        'conflicts_by_kind': conflicts_by_kind,
        'total_conflicts': sum(conflicts_by_kind.values()),
        'blocks_with_satisfied_area_constraints': sum(1 for r in reports if r['constraints_satisfied']),
        'blocks_with_area_reports': len(reports),
    }
    cfg = _llm_config()
    if not cfg['configured']:
        return {'ward_job_id': ward_job_id, 'facts': facts, 'brief': None,
               'note': 'LLM not configured -- facts only.', 'decision': 'advisory_only'}
    _reserve_call_or_429(cfg)
    prompt = (
        'Write a short (3-4 sentence) plain-language status brief for a cadastral survey ward, for a '
        f'non-technical stakeholder. Use ONLY these real, already-computed facts: {json.dumps(facts)}. '
        'Do not invent anything else about this ward. State overall progress, flag clearly if conflicts '
        'or failed blocks need attention, and do not claim the ward is legally certified or complete '
        'unless status is literally "done" with zero conflicts.'
    )
    resp = _call_anthropic(cfg, [{'role': 'user', 'content': prompt}])
    return {'ward_job_id': ward_job_id, 'facts': facts, 'brief': _text_of(resp.get('content', [])),
           'usage': resp.get('usage', {}), 'decision': 'advisory_only'}


# --------------------------------------------------------------------- ask --

_TOOLS = [
    {'name': 'ward_status', 'description': "This ward's block-by-block processing status.",
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'list_conflicts', 'description': 'Every persisted fusion conflict for this ward: sources, disagreement magnitude, kind.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'parcels_in_bbox', 'description': "Parcel face ids within a bounding box, in this ward's own CRS units.",
     'input_schema': {'type': 'object', 'properties': {
         'minx': {'type': 'number'}, 'miny': {'type': 'number'}, 'maxx': {'type': 'number'}, 'maxy': {'type': 'number'}},
         'required': ['minx', 'miny', 'maxx', 'maxy']}},
]
MAX_TOOL_ROUNDS = 3


def _run_tool(name, tool_input, session, ward_job_id):
    """Every tool is read-only and hard-scoped to `ward_job_id` from the
    URL, not from anything the model's tool-call input supplies -- the
    model chooses WHICH tool and (for parcels_in_bbox) WHICH bbox, never
    which ward. Nothing here can write to the store.
    """
    if name == 'ward_status':
        state, blocks = ward_status(session, ward_job_id)
        return {'status': state, 'blocks': blocks}
    if name == 'list_conflicts':
        rows = session.execute(select(PersistedConflict).where(PersistedConflict.ward_job_id == ward_job_id)).scalars().all()
        return {'conflicts': [{'id': c.id, 'block_id': c.block_id, 'sources': c.sources,
                              'disagreement_m': c.disagreement_m, 'kind': c.kind} for c in rows]}
    if name == 'parcels_in_bbox':
        bbox = from_shape(box(tool_input['minx'], tool_input['miny'], tool_input['maxx'], tool_input['maxy']), srid=SRID)
        rows = session.execute(select(Face.id, Face.block_id, Face.recorded_parcel_id)
                               .join(IngestedBlock, IngestedBlock.block_id == Face.block_id)
                               .where(IngestedBlock.ward_job_id == ward_job_id, Face.geom.op('&&')(bbox))).all()
        return {'parcels': [{'face_id': fid, 'block_id': bid, 'parcel_id': pid} for fid, bid, pid in rows]}
    raise ValueError(f'unknown tool {name}')


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@router.post('/wards/{ward_job_id}/ask')
def ask(ward_job_id: int, req: AskRequest, session: Session = Depends(get_session)):
    """Bounded read-only Q&A over one ward's real, already-persisted
    state -- nothing it calls can write to the store, and every tool call
    is scoped to THIS ward regardless of what the model asks for.
    """
    _get_ward_job_or_404(session, ward_job_id)
    cfg = _llm_config()
    _reserve_call_or_429(cfg)
    messages = [{'role': 'user', 'content': req.question}]
    tool_calls_made = []
    for _round in range(MAX_TOOL_ROUNDS):
        result = _call_anthropic(cfg, messages, tools=_TOOLS, max_tokens=1024)
        blocks = result.get('content', [])
        messages.append({'role': 'assistant', 'content': blocks})
        tool_uses = [b for b in blocks if b.get('type') == 'tool_use']
        if not tool_uses:
            return {'ward_job_id': ward_job_id, 'answer': _text_of(blocks), 'tool_calls': tool_calls_made,
                   'decision': 'advisory_only'}
        tool_results = []
        for use in tool_uses:
            try:
                output = _run_tool(use['name'], use.get('input', {}), session, ward_job_id)
            except Exception as exc:
                output = {'error': str(exc)}
            tool_calls_made.append({'name': use['name'], 'input': use.get('input', {})})
            tool_results.append({'type': 'tool_result', 'tool_use_id': use['id'], 'content': json.dumps(output)})
        messages.append({'role': 'user', 'content': tool_results})
    # Ran out of rounds without a final text answer -- say so honestly
    # rather than returning nothing or a truncated mid-tool-call state.
    return {'ward_job_id': ward_job_id, 'answer': None, 'tool_calls': tool_calls_made,
           'error': f'No final answer within {MAX_TOOL_ROUNDS} tool-call rounds', 'decision': 'advisory_only'}
