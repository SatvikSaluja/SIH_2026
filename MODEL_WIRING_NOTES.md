# Capacity tolerance fix + model wiring

Two changes, in order. Full regression after each.

## 1. Recorded-area tolerance is now spent before the solve

**The bug.** `refine_recorded_areas()` built an *equality* constraint against
exact recorded areas and only compared against the tolerance after snapping.
A recorded tolerance is a permitted interval, so any block whose recorded
total does not exactly equal its own area was refused even when the record
admitted a perfectly good allocation.

Reproduction (this is the case an external review flagged, confirmed by
running it):

    block 10m x 10m = 100 m2, records 34/34/34 m2, tolerance +/-1 m2 each
    start 20/20/60 -> converged: False
                      reason:    recorded_areas_not_met_after_snapping
                      max_error_m2: 2.0
                      optimizer_message: "Optimization terminated successfully"
    start 33/33/34 -> converged: True   (already_within_tolerance)

The solver forced parcels 1 and 2 to exactly 34, leaving 32 for parcel 3, and
then rejected its own answer. 33.33/33.33/33.33 satisfies every record.

**The fix.** `_feasible_targets()` projects the recorded areas onto the
nearest point inside the tolerance box whose total equals the block area,
and the equality solve targets *that*. The final verification still uses the
original recorded areas and tolerances, so nothing is loosened -- the
tolerance is spent where it is meant to be spent, not used to excuse a miss.

The projection is a bisection on one scalar: `clip(target + lam, lo, hi)`
has a total that is nondecreasing in `lam`. Because `lo/hi` are
`target -/+ tol`, this reduces to `sum(clip(lam, -tol, tol)) = deficit`.

Two properties worth noting, both covered by tests:

- When the recorded total already equals the block area, the projection is the
  identity, so every pre-existing case behaves exactly as before.
- Slack goes to the parcels whose records permit it, not split evenly. With
  tolerances 0.001/0.001/3.0 and a 2 m2 deficit, parcels 1 and 2 stay at
  34.000 and parcel 3 absorbs the whole 2 m2 -> 32.0.

A genuinely impossible total (50/50/50 +/-1 in a 100 m2 block) is still
refused with `infeasible_recorded_total`. The pre-existing feasibility check
is exactly right for a box: achievable totals span
`[sum(t - tol), sum(t + tol)]`.

**The comment that was wrong.** The old code dropped the last equality as
"redundant -- total domain area is fixed by its ring". That is only true when
the targets sum to the block area. They did not, so the dropped constraint
was not redundant, it was infeasible. It is now genuinely redundant.

## 2. The model now participates in block processing

**The gap.** Nothing in `jobs/` or `api/` imported `geocadastra/models/`.
Every green acceptance test proved the geometry engine worked; none proved
the network contributed anything. The worker used
`simulate_evidence_field()` unconditionally.

**The change.** `block_evidence_from_model()` in `models/infer.py` returns
`(evidence_field, transform)` -- the same contract `simulate_evidence_field()`
already had, so Stage 3 needed no change at all. `_block_evidence()` in the
orchestrator picks between them and returns the provenance alongside.

Configured by `GEOCADASTRA_MODEL_WEIGHTS`. Simulation stays the default so a
worker without weights keeps producing a reproducible, auditable result
instead of silently failing -- but the stored provenance always records
`evidence_source` and, for the model path, the weights' SHA-256 prefix. A
stored block can never be mistaken for a model result it did not come from.

Weights load once per worker process (`lru_cache`), not once per block.

**The window.** The evidence is cropped from the ward raster's own grid and
returned with that grid's transform, not resampled to a fixed gsd. Stage 3
consumes an `(array, transform)` pair, so a transform that disagrees with its
array puts every parcel in the wrong part of the block. There is a test that
asserts the returned window actually covers the block it claims to.

**Test that is not a tautology.** An untrained network still returns a
plausible-looking field, so shape assertions alone cannot distinguish a wired
model from a discarded one. `test_a_trained_boundary_response_changes_the_
evidence_field` runs two different weight sets and asserts the evidence
differs; `test_worker_processes_a_block_end_to_end_on_model_evidence` drives
a real block to `done` and reads the weights hash back out of provenance.

## What this does not do

The model is wired, not *trained*. The Stage 4 accuracy target at
`test_stage4_acceptance.py:64` is still `xfail` for compute reasons, so the
honest statement remains: the pipeline can consume real predictions, and
those predictions are not yet good. That is the GPU work, still blocked on a
provider decision.

## Testing note (cost me ~10 minutes, worth recording)

Do not run the slow suite concurrently with anything else that touches the
database. They share the test schema, and a teardown in one drops it under
the other: the symptom is
`psycopg.errors.AdminShutdown: terminating connection due to administrator
command`, plus wall times inflated ~50x by CPU contention. The same file that
appeared to take 556s and error passed in 10.8s run alone. The failure was
in the test harness, not the code.

## Counts

- fast suite: 404 passed (was 394; +3 capacity regressions, +7 model wiring)
- slow suite: unchanged, 1 known xfail (Stage 4 accuracy)
