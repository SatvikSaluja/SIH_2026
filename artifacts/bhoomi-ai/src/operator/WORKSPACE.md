# GeoCadastra research workspace

Start the API from repository root:

```bash
.venv/bin/python -m uvicorn geocadastra.api.main:app --host 127.0.0.1 --port 8000
```

Start the frontend from `frontend/`:

```bash
npm run dev
```

Open http://localhost:5173. The proxy project remains separate at port 5000. The Vite development server proxies `/workspace` and `/wards` to the API. `VITE_API_BASE` may override this; use an allowed CORS origin. For a built production frontend, configure the reverse proxy with the same paths. This is a local research workspace: bind to localhost; no public authentication has been added.

## Implemented workflows

1. Overview: current dataset counts, available checkpoints, completed local inference jobs and persisted human reviews. Interactive image-coordinate map (pan/zoom), not a geographic basemap. Raster CRS and resolution are displayed separately; screen pixel coordinates are never presented as longitude/latitude.
2. Datasets: manifest-discovered NPZ collections, tile search, split filters, pagination, RGB/reference/height layers, metadata and checksums. Collections still being written are labelled in progress. Raw downloads without a prepared manifest are not counted as labelled samples.
3. Inference: one or two selected MultiTaskNet checkpoints; queued single-worker execution; 128px patches with 32px overlap; existing weighted stitching; measured elapsed time, input/checkpoint SHA256, numerical arrays and JSON export. Only parcel-distance predictions are presented. The API computes reference agreement only if labels exist; it does not claim independent accuracy or calibrated confidence. Other network heads may be untrained. `model_a.pt` from the one-tile diagnostic uses the same architecture, but evaluation here uses stitched whole-tile predictions rather than duplicated overlapping patch counts.
4. Evidence: RGB Sobel and optional nDSM gradient responses in a 5px neighbourhood, scaled relative to each tile. Reference-band pixels are weak (<0.25), uncertain (0.25–0.65) or supported (>=0.65). These are reproducible heuristics, NOT verified visibility labels. Parcel intent is not used yet because the real trainer arrays do not preserve per-pixel parcel intent. Shadow/texture edges can score highly. This same per-pixel score can optionally feed the boundary loss term directly (see Training, `--visibility-aware`) -- when it does, no OTHER masking of training data happens: no pixel is dropped or excluded, only how strongly the boundary term counts against it changes.
5. Label QA: candidate crops ranked by weak/uncertain reference support; human accept, flag-for-correction or needs-survey decisions stored with notes. This is triage, not an automated label-alignment verdict. Decisions do not edit parcel geometries or labels.
6. Vision assistance: a user explicitly selects and submits a 192px crop. Raw RGB and a duplicate marked reference crop are sent to Anthropic Messages. Advice is saved separately from human decisions. It cannot update geometry, certify ownership or automatically relabel training examples.
7. Training: reads actual history files and checkpoint names, distinguishes same-image diagnostics, plots losses, and starts bounded fresh real-data runs using the existing CLI. Jobs are queued; history updates after completed epochs. Model objectives/evaluation definitions can differ between existing runs, so displayed losses must not be compared blindly. No fake progress percentages. The underlying trainer selects CUDA only if available. An optional "visibility-aware boundary loss" checkbox passes `--visibility-aware` through to the trainer (off by default, reproducing the exact prior loss); see Evidence above for what it scales.

The prior ward operations UI is retained in `OperatorConsole.tsx`: geometry, constraints, conflicts, edits and co-registration still call existing APIs and require the original Postgres/worker setup.

## Optional vision provider (server environment only)

```bash
export GEOCADASTRA_VISION_KEY='your-provider-key'
export GEOCADASTRA_VISION_MODEL='your-enabled-vision-model-id'
export GEOCADASTRA_VISION_DAILY_LIMIT=10
```

Restart the API after configuring environment variables. Never put secrets in `VITE_*` variables or committed files. The UI does not collect credentials. Only the fixed `https://api.anthropic.com/v1/messages` endpoint is used. Each call permits at most 512 output tokens and one small side-by-side image. Every attempted provider request consumes a persisted daily slot, including failures. This limits requests, not a guaranteed monetary budget; price depends on the configured model. No requests are automatic. Local tests mock provider responses; a live provider call requires actual configuration and has not been claimed as verified.

## Persistence and execution limits

`.local_workspace/workspace.sqlite` stores jobs, review notes and daily vision reservations. Inference PNG/NPZ files are in per-job subdirectories there. `GEOCADASTRA_WORKSPACE_STATE` may override the location. Real training output goes to `runs/workspace_<job-id>/`, including `console.log` and the trainer's checkpoints/history.

Run one API worker. One compute worker executes inference/training sequentially, with a small inference queue. Training has a six-hour subprocess timeout. On API restart, pending jobs from an earlier instance are shown as interrupted; they are not automatically rerun. Do not restart during an active training subprocess. This is a local job runner, not a production distributed broker.

## Visual references

Reviewed https://21st.dev for navigation, cards and gallery patterns, alongside the local proxy's map/sidebar composition. The UI is original React/CSS using the existing Leaflet dependency; no third-party template source was copied. Fonts: DM Sans and Manrope through Google Fonts, with local sans-serif fallback.

## Validation

Backend regression tests cover manifest discovery (including unrelated JSON), path restrictions, real model execution/provenance, evidence counts, persistent review records, restart status, provider-disabled behavior and a mocked provider with a durable request cap. Browser checks cover dataset search, evidence/review, provider-disabled state, inference submission, training views, legacy ward entry and mobile sizing. Full cadastral correctness and externally trained model accuracy are outside this UI verification.
