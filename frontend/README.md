# GeoCadastra frontend

An operator console over the real API in `geocadastra/api/main.py` — every
screen calls an endpoint that exists and is tested; nothing here is a mock
or a placeholder for a future backend.

## Run it

```bash
npm install
cp .env.example .env.local   # point VITE_API_BASE at your running API
npm run dev
```

The API needs CORS enabled for the dev origin (already wired in
`geocadastra/api/main.py` via `GEOCADASTRA_CORS_ORIGINS`, defaulting to
`http://localhost:5173`).

## What's wired, and what isn't

| Screen | Backend | Notes |
|---|---|---|
| Create ward | `POST /wards/ingest` | Synthetic only — the real GeoTIFF/parcel path in `geocadastra/jobs/ingest.py` has no HTTP route yet |
| Status / run / resume | `POST /wards/{id}/run`, `GET /wards/{id}/status` | Dispatches Celery tasks; needs a real Redis broker + worker to actually process (see below) |
| Map | `GET /wards/{id}/parcels` (bbox) | Uses the plain bbox query, not the MVT tile endpoint (`/tiles/{z}/{x}/{y}.mvt` exists but needs a tile-decoding client this app doesn't carry) |
| Conflicts | `GET /wards/{id}/conflicts` | `node_id: -1` is a real backend sentinel for a block-level issue (e.g. `area_refinement_unresolved`) — the UI correctly refuses to treat those as an editable node |
| Constraints | `GET /wards/{id}/constraints` | `boundary_certification` is always `"not_calibrated"` today — Stage 6's calibration module isn't wired into this endpoint yet |
| Analytics | `GET /wards/{id}/analytics` | Reported per settlement style, never pooled — matches the backend's own doc rule |
| Move a node | `POST /wards/{id}/edit`, `POST /wards/{id}/field-verification` | Requires a **real** node id. There is no endpoint that hands one out for an arbitrary map click — only `/conflicts` carries real node ids, so the edit form's reliable path is picking a conflict, not clicking the map |
| Co-registration | `POST /wards/{id}/coregister` | Never rejects a poor-but-valid fit, only a degenerate one (per the backend's own contract) |

## Known gap: running a job needs a Celery worker + Redis

`POST /wards/{id}/run` dispatches via Celery (`process_block.delay(...)`).
Locally that needs a Redis broker (`redis://localhost:6379/0` by default,
`GEOCADASTRA_CELERY_BROKER_URL` to override) and a running worker:

```bash
redis-server &
celery -A geocadastra.jobs.orchestrator worker --loglevel=info &
```

Without both, `/run` returns `500` (`kombu.exceptions.OperationalError:
... Connection refused`) — this is a real infra gap, not a frontend bug.

## Verification done so far

- `tsc --noEmit` and `vite build` both clean.
- Every endpoint the frontend calls was driven against a real running API
  (`uvicorn geocadastra.api.main:app`) backed by a real Postgres/PostGIS
  schema, with real processed ward data (26 faces, 442 conflicts including
  block-level and node-level ones, real constraint/analytics reports) — not
  just typechecked against guessed shapes.
- CORS confirmed via an actual preflight + `Origin` header round trip
  against the dev origin.
- Not yet done: no browser was available in this environment (no Chrome
  binary), so nothing here has been visually rendered or click-tested end
  to end. Typecheck, build, and live API contract are verified; on-screen
  rendering is not.
