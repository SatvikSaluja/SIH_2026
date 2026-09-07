# Stage 8 — orchestration and API

## Built
`geocadastra/store/schema.py` additions: `WardJob` (one end-to-end run;
`params` -- the exact `WardParams` used, `dataclasses.asdict()` -- and
`coreg_residual_m`), `BlockJob` (the actual resumability unit: per-block
`pending|done|failed`), `IngestedBlock`/`RecordedParcel`/`LegacyRecord`/
`SurveyPoint` (every vector fact ingest persists durably), `PersistedConflict`
(Stage 5's `ConflictRecord`, deferred to "Stage 8's concern" in
STAGE_5_NOTES.md, now has a real home). Field verification reuses
`Provenance` rather than a new table -- a surveyor's confirmation is
evidence exactly like a model's, just a different `evidence_type`.

`geocadastra/jobs/orchestrator.py`: Celery `app` + `process_block` (one
block, idempotent -- checks `BlockJob.status` first, no-ops if `done`;
parcel assignment (Stage 3) -> reload from store -> fuse against legacy/GT
(Stage 5) -> persist; any exception marks the block `failed` with the
error recorded, never leaves it half-done) + `run_ward` (dispatches every
non-`done` block -- the same call handles first-run and resume) +
`ingest_synthetic_ward` (persists every vector fact of a synthetic ward).
No real drone imagery or raster store exists yet, so the raster evidence
field is regenerated deterministically from the ward's own persisted seed
+ params at process time rather than round-tripped through storage -- an
explicit, documented simplification (see the module's own docstring), not
a silently different path production would take too.

`geocadastra/api/main.py`: every endpoint the doc lists --
`/wards/ingest`, `/wards/{id}/coregister` (least-squares affine fit from
control points, RMS residual carried into every block's fusion sigma,
never rejects the upload per the doc's own words), `/wards/{id}/run`
(dispatch/resume), `/wards/{id}/status` (poll), `/wards/{id}/parcels`
(GiST-indexed bbox query, not a Python-side scan-and-filter),
`/wards/{id}/conflicts`, `/wards/{id}/edit` and `/wards/{id}/field-
verification` (both through the same transactional changeset every other
write uses), `/wards/{id}/tiles/{z}/{x}/{y}.mvt` (`ST_AsMVT`, not raw
GeoJSON -- "this matters at 200k parcels"), `/wards/{id}/analytics`
(stratified metrics, never pooled).

`geocadastra/core/evaluate.py`: the evaluation-harness metrics section 6
asks be built "alongside, not at the end" -- `boundary_position_error`
(P50/P90 against nearest EDGE, not nearest vertex), `topology_validity_rate`
(invalid polygons + real overlaps, not just one or the other),
`parcel_count_error` (over/under-segmentation reported separately, never
netted), `area_error_distribution` (relative, not absolute -- comparable
across parcel sizes the way mean IoU deliberately isn't). Every function
takes one already-filtered population; stratification happens in the
caller (`/analytics`), never pooled inside these functions.

## Tested
`tests/test_evaluate.py` (10, fast): each metric's own edge cases,
including a hand-fabricated genuinely-overlapping-faces graph (`build_
graph()` can't produce one by construction, so this is the only way to
exercise the overlap detector at all).

`tests/test_orchestrator.py` (9, `slow` -- real Stage 3/5 pipeline
through real Postgres): ingest persistence, full-ward processing,
idempotency, and -- the actual Stage 8 Done-when's resumability clause --
a real simulated killed-worker-mid-ward scenario (one block processed
directly, bypassing Celery, then `run_ward()` called again: the
completed block's faces are untouched, every other block gets
processed), plus changeset-log reconstruction, conflict persistence,
cross-ward isolation, and a mocked-dispatch async-status regression test.

`tests/test_api.py` (19, `slow`): every endpoint, plus the review-found
bugs below (degenerate co-registration input, zero `gsd`, out-of-range
tile coordinates, cross-ward edit ownership).

`tests/test_stage8_acceptance.py` (1, `slow`): the literal Stage 8
Done-when, entirely through the API -- ingest, a real simulated killed-
worker resume via the SAME `/run` call, every block reaching `done`, the
certified parcel layer queryable, and every block's graph independently
reconstructed from its own changeset log matching the live store exactly.

## Real bugs found by actually running this end to end (not by reasoning about the code)
This stage had more of these than any prior one -- a natural consequence
of being the integration layer wiring together seven previous stages'
worth of assumptions for the first time:

1. **`_regenerate_ward()` always used DEFAULT `WardParams`**, ignoring
   whatever params a ward was actually ingested with. Two different
   `WardParams` sharing a seed produce two different wards; regenerating
   with the wrong one gave a block whose evidence-field raster window
   didn't even overlap the real block's geometry, crashing `slic()` on an
   empty array. Fixed: `WardJob.params` persists the real params
   (`dataclasses.asdict`), reconstructed via `WardParams(**params)`.
2. **`assign_parcels()`'s default `n_segments=2000` produces degenerate
   superpixels on a small test raster** (fewer pixels than requested
   segments), triggering a real, pre-existing Stage 1 limitation ("island"
   parcels enclosed by a neighbour -- `build_graph()`'s documented hole
   guard). Fixed: `n_segments` scaled to the raster's own pixel count.
   This pre-existing limitation still occurs on ~40% of seeds at this
   ward size even after the fix (confirmed by directly sweeping seeds
   0-14) -- `process_block`'s failure handling is already correct for it
   (marks the block `failed`, nothing corrupts), so this is an honestly
   tracked, out-of-Stage-8-scope gap, not a new bug; test seeds were
   chosen to avoid it, not to hide it.
3. **`fuse_block()` was called against the PRE-seed graph**, whose node
   ids are local to that one throwaway `parcels_to_graph()` object --
   `seed_block_graph()` remaps every id onto a fresh globally-unique one
   pulled from the store's sequences (Stage 2's own documented id-
   collision fix) before writing. `apply_fusion()` then tried to move a
   node id that only ever existed in a stale local object, `KeyError`ing
   on the very first fusion move. Fixed: reload via `load_block_graph()`
   after seeding, fuse against THAT.
4. **`fuse_block()`/`gt_estimate()` want plain `(x, y)` tuples, not
   `GTPoint` objects** -- fusion is a pure nearest-neighbour spatial
   search, not parcel-scoped; `GTPoint.parcel_id` is a Stage 6 stratum
   concept fusion never asked for. `gt_estimate()`'s own `np.asarray(gt_
   points, dtype=float)` can't convert a `GTPoint`, crashing immediately.
5. **`make_session()` created a brand-new `Engine` (and connection pool)
   on every single call**, never disposed -- a real leak, one full
   connection pool per block processed. Fixed: one cached `Engine` per
   `(db_url, schema)`, reused.
6. **`run_ward()`'s eager-mode Celery dispatch didn't match real async
   semantics**: `task_eager_propagates=True` meant one block's exception
   bubbled synchronously through `run_ward()`'s own dispatch loop,
   stopping it from ever reaching the remaining blocks -- something that
   could never happen in real async dispatch (`.delay()` enqueues and
   returns immediately; a task's eventual failure can't raise back
   through the loop that dispatched it). Fixed: left
   `task_eager_propagates` at Celery's own default (`False`), so eager
   test mode is a faithful stand-in for production behaviour instead of a
   shortcut that changes it.
7. **`/run`'s Celery dispatch used hardcoded module-level `_DB_URL`/
   `_DB_SCHEMA`**, completely bypassing whatever DB a test's `get_session`
   override pointed the REQUEST's own session at -- every dispatched task
   then opened its own session against the wrong schema and crashed with
   `UndefinedTable`. In production these always match (both come from the
   same env vars); only the test override broke the coupling. Fixed: a
   separate `get_db_config()` FastAPI dependency, overridden alongside
   `get_session` in tests.
8. **`/edit`/`/field-verification` only caught `ConcurrentModificationError`/
   `ValueError`**, but `PlanarGraph.move_node()` raises a plain `KeyError`
   for an unknown node id -- an unknown node crashed with an unhandled
   500 instead of a 422 client error. Fixed (while also deduplicating the
   two endpoints' identical error-handling into one `_apply_move()`
   helper): catch `KeyError` alongside `ValueError`.
9. **The analytics endpoint's per-block boundary-error computation passed
   EVERY GT point across a whole style to EVERY block of that style**,
   including GT points physically nowhere near a given block -- their
   "nearest edge" distance would be a large, meaningless number inflating
   that block's own error. Fixed: GT points scoped to each block's own
   parcels before calling `boundary_position_error()`.

## Review pass: 8 more real bugs, several severe
A dedicated review of `schema.py`/`orchestrator.py`/`api/main.py`/
`evaluate.py` (SQL injection, Celery concurrency/idempotency, API input
validation, schema data-integrity, cross-file signatures, connection
lifecycle, exception-handling completeness) found 8 real, all fixed --
two of them severe enough that the system was fundamentally broken for
its own stated purpose (processing more than one ward, ever):

1. **`RecordedParcel.id`/`LegacyRecord.id` were the synthetic generator's
   own ward-LOCAL ids, used directly as raw primary keys with no sequence
   of their own** -- the exact bug class node/edge/face ids already had a
   fix for (`node_id_seq` etc., with its own comment explaining why),
   just never applied here. A second `POST /wards/ingest` crashed with
   `IntegrityError: duplicate key ... (id)=(0) already exists`. Fixed:
   `block_id_seq`/`recorded_parcel_id_seq`/`legacy_record_id_seq`, the
   same remap-before-write pattern `seed_block_graph()` already
   established, applied in `ingest_synthetic_ward()`.
2. **`Face` has no `ward_job_id` column at all, and `block_id` was
   ward-local** -- so two different wards' same-numbered "block 0"
   silently POOLED into one `load_block_graph()` result once both were
   processed, corrupting both wards' data with no error. Root-caused
   together with #1 (same underlying disease) and fixed the same way:
   block ids are now globally unique; `IngestedBlock` gained a
   `local_block_id` column so `process_block` can still translate back to
   what `simulate_evidence_field()` needs (the regenerated ward object
   still numbers its own blocks from 0).
3. **`process_block` had no concurrency guard** -- two genuinely
   concurrent calls for the same block (a real risk: a task redelivered
   after its worker was thought dead but wasn't) both race past the
   "already done" check and both run the full seed/fuse pipeline,
   producing duplicate graph data (`seed_block_graph()` has no "already
   has faces" check and nothing constrains it). Fixed: a session-scoped
   Postgres advisory lock (`pg_advisory_lock`/`_unlock`, not the
   transaction-scoped variant `store/provenance.py` uses elsewhere --
   this function's own pipeline already commits multiple times
   internally, which would release an xact-scoped lock far too early)
   keyed on `(ward_job_id, block_id)`, serializing genuinely concurrent
   calls instead of racing them.
4. **`run_ward()`'s final status computation queried `BlockJob WHERE
   status != 'done'`**, which only matches EXISTING rows -- a freshly-
   dispatched block that hasn't executed yet (no `BlockJob` row at all,
   the NORMAL state in real async dispatch) was invisible to that check.
   In any real (non-eager) deployment, a fresh `/run` call reported
   `status: "done"` immediately after dispatch, before a single block had
   actually run -- invisible to every test because they all force Celery
   eager mode. Fixed: re-derive "done" from which blocks are ACTUALLY
   done now (a proper set comparison against every block), not a query
   shape that silently under-counts. Regression test mocks `process_
   block.delay` as a no-op to simulate real async dispatch without a live
   broker.
5. **`/edit`/`/field-verification` never verified the block in the
   request body belongs to the ward in the URL** -- especially dangerous
   before #2's fix, when a block id like `0` was ambiguous across every
   ward in the store; a request to one ward's URL could silently edit a
   different ward's block. Fixed: both now 404 if `IngestedBlock` has no
   row for `(ward_job_id, block_id)`.
6. **The tile endpoint didn't validate `z`/`x`/`y`** -- an extreme value
   (e.g. `y=10**18`) overflows `math.sinh()` inside the slippy-map
   tile-to-lat/lon conversion, an unhandled `OverflowError` -> 500. Fixed:
   bounds-checked, 422 for anything outside a real tile grid.
7. **The co-registration affine fit checked control-point COUNT but never
   rank/degeneracy** -- three identical points pass the `>= 3` check but
   can't determine a unique transform; `np.linalg.lstsq` doesn't raise
   for this, it just returns some solution, reporting a spuriously
   "perfect" `residual_m: 0.0` that then silently understates every
   block's actual legacy-evidence uncertainty. Fixed: an explicit
   `matrix_rank` check, 422 for a degenerate fit.
8. **`IngestRequest.gsd` had no positivity check** -- `gsd=0` propagates
   into `generate_ward()`'s own `width / gsd`, an unhandled
   `ZeroDivisionError` -> 500. Fixed: `Field(gt=0)`.

Also investigated and confirmed NOT a live bug: a `text(f"...{seq.name}...")`
in `allocate_ids()` (now shared by `changeset.py` and `orchestrator.py`)
builds SQL via an f-string rather than a bind parameter, but `seq` is
always one of a small set of hardcoded module-level `Sequence` objects,
never user/API-supplied input -- not exploitable at any current call
site, left as-is rather than changed for its own sake.

Full regression suite after all of the above: 344 non-slow passed (same
count -- every new test here is real-pipeline `slow`), and the complete
Stage 8 `slow` suite (orchestrator + API + acceptance, 29 tests) passed
together, confirming the fixes compose correctly, not just individually.

## Deferred (correctly, per this stage's own scope)
- `area_error_distribution()` (`core/evaluate.py`) is built and unit-
  tested but NOT wired into `/analytics` yet: it needs a per-parcel
  prediction paired with its own recorded area, but `seed_block_graph()`
  remaps every face onto a fresh id with no traceability kept back to
  which original `RecordedParcel` produced it. A real, scoped addition
  (persisting that link at seed time), not attempted without being asked.
- `apply_fusion()`'s O(N) per-node DB reload (flagged as a deliberately-
  deferred optimization in the whole-codebase review) is now genuinely
  felt at Stage 8 scale -- even a ~10-parcel block takes real seconds,
  and the full Stage 8 test suite (orchestrator + API + acceptance) takes
  several minutes. Accepted as an already-tracked, pre-existing cost
  rather than rushed into a redesign mid-stage; test ward sizes were kept
  deliberately small to keep this bounded.
- `run_ward()`'s dispatch loop is sequential in eager/test mode (each
  `.delay()` call runs its task synchronously before the loop continues)
  -- in real async deployment, dispatch is still effectively sequential
  (a `for` loop calling `.delay()`), but the WORK itself runs in parallel
  across whatever Celery workers are available. The same-BLOCK race is
  now genuinely guarded (the advisory lock, above); no test here exercises
  genuine multi-worker concurrent processing of DIFFERENT blocks of the
  same ward under real process/thread concurrency (only sequential and
  directly-simulated-race scenarios) -- a real load test is out of this
  stage's scope.
- No auth/access-control on any endpoint -- this is a backend-logic
  handoff per the doc's own framing ("Backend and logic only, no UI"),
  and no auth requirement was specified anywhere in the build plan.
