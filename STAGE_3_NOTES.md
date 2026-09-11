Update: [CAPACITY_AND_SUBDIVISION_NOTES.md](CAPACITY_AND_SUBDIVISION_NOTES.md) documents generator version 2, constrained capacity refinement, and the corrected fusion corner behavior. Historical deferrals below are superseded where that note explicitly states so.

Update (September 9): see [FOUNDATION_GAPS_NOTES.md](FOUNDATION_GAPS_NOTES.md) for parcel identity, interior rings, area guards, replay, and resumable training. Historical limitations below are superseded only where that note explicitly says so.

# Stage 3 — capacity-constrained parcel assignment

The differentiator. Built and tested against synthetic data + a simulated
evidence field, per the build plan, since Stage 4's real model doesn't exist
yet.

## Built
`geocadastra/synth/generator.py`: `simulate_evidence_field(ward, block_id, ...)`
-- a fake "high value = boundary probably runs here" raster: a ridge near
VISIBLE true edges bordering that block's parcels, decaying with distance;
zero contribution from invisible edges (a real model can't see them either --
that's the whole premise this stage exists to handle). Stage 3 test support,
not a Stage 0 deliverable, since it stands in for Stage 4's future output.

`geocadastra/core/transport.py`: `assign_parcels()` -- oversegment the block
into SLIC superpixels on the evidence field, build a superpixel adjacency
graph (edge weight = centroid distance x (1 + evidence-strength penalty) at
the shared boundary), compute geodesic (Dijkstra) cost from each parcel's
seed, solve the resulting optimal-transport problem (supply = superpixel
areas, demand = recorded parcel areas, via POT's exact network simplex --
see below on why Sinkhorn essentially never triggers), hard-round the plan
(cost tie-break for the rare split superpixel), and extract label boundaries
into `Geom` polygons. Handles the documented broken cases: recorded areas
not summing to the block (rescale, record `area_sum_mismatch`), some parcels
missing a recorded area or seed (uncapacitated -- a "dummy reservoir" demand
column reserves exactly their share instead of letting the capacitated
parcels' rescale silently consume it, then nearest-centroid distributes the
reserved superpixels among them), and no usable record at all (evidence-only
watershed). `assign_parcels_evidence_only()` is the same pipeline with the
area constraint removed entirely -- the baseline the headline metric is
measured against. `parcels_to_graph()` completes the pipeline the module
docstring promises: extracted boundaries -> `planarize()` (block boundary
passed as authoritative `fixed` linework) -> `build_graph()`, a real Stage 1
`PlanarGraph`, not just polygons that happen to tile.

Tests: `test_transport.py` -- 27 tests including the two Stage 3 acceptance
criteria (output areas match recorded areas within one superpixel; the
labelling exactly tiles the block) and the headline test: averaged over 8
blocks per visible-edge fraction (0.9/0.5/0.15), the constrained solver's
boundary error stays flat (~0.21 of block area) regardless of evidence
sparsity while evidence-only degrades substantially (0.24 -> 0.37), so the
gap widens monotonically (0.03 -> 0.11 -> 0.16) -- exactly the thesis. 227
tests total across all four stages, all pass (`pytest geocadastra/`, ~90s).

## Full multi-angle review (post-build)
Same workflow as Stages 0-2. This module is the most algorithmically novel
so far (SLIC, geodesic Dijkstra, optimal transport, several fallback paths),
and the review found bugs of a different character than earlier stages --
less "wrong number," more "silently stops enforcing the constraint, or
crashes on an unremarkable input":

- **The Sinkhorn fallback silently discarded the area constraint.** Above
  `EXACT_SOLVE_MAX_SIZE` (originally 200,000), the entropic-regularized
  solver produces a *dense* transport plan, but the hard-rounding step's
  cost tie-break assumes a near-binary one (true only for the exact LP
  solver) -- once every candidate is nonzero, the tie-break degenerates into
  a plain cheapest-cost pick, becoming byte-for-byte identical to the
  no-constraint baseline it exists to beat, with zero conflicts recorded.
  Reproduced at 100 parcels x 2200 superpixels: 50.6% mean area error, 100%
  assignment overlap with the unconstrained baseline. Separately measured
  that exact solving stays fast (single-digit seconds) to ~100M -- 500x the
  old cutoff -- while Sinkhorn right at that cutoff produced a 59-176%
  higher-cost plan for no speed benefit. Fixed by raising the cutoff to
  5,000,000 (far above any realistic block) so ordinary use never reaches
  Sinkhorn, plus a general post-solve safety net (see below) as defense in
  depth.
- **An unclipped evidence value crashed the solver's memory, not its logic.**
  A plain finite value outside [0,1] becomes a *negative* graph edge weight;
  since the graph is undirected, one negative edge is already a negative
  cycle, and `scipy.sparse.csgraph.dijkstra` has no negative-cycle guard --
  it just warns and runs away in unbounded memory. Reproduced via the real
  public API (a 3x3-pixel out-of-range patch): unbounded memory growth,
  eventual crash. Fixed with entry-point validation that clips to [0,1] and
  raises a clear error on NaN/Inf.
- **A seed a few pixels outside its block's raster crashed, one-sided.**
  `_seed_to_superpixel`'s growing-window search floor-clamped one end of
  each bound but not the other; a negative pixel row/col (north/west of the
  raster only) let plain numpy slicing silently wrap a negative stop index,
  reaching a crash several lines later instead of the function's own clean
  fallback. Confirmed independently by two review passes, at just 2 pixels
  of drift -- not a pathological input, since this stage's entire premise is
  tolerating imprecise legacy centroids. Fixed, and consolidated three
  independently-written (and inconsistently-clamped) copies of the same
  world-to-pixel conversion into one shared helper.
- **A starved unhandled parcel could silently get zero area.** With two or
  more parcels competing for the same leftover ("uncapacitated") superpixels,
  nearest-centroid-by-raw-seed-position can leave one with nothing --
  reproduced with 3 parcels, one shortchanged parcel ending up completely
  absent from the output with an empty conflict list. Fixed: now recorded as
  `leftover_parcel_got_no_area`.
- **A general post-solve safety net, not six separate point-fixes.** The
  review's own altitude pass noted these silent-shortfall bugs share a
  shape: a helper handles the happy path but an edge case around it is
  unguarded, with nothing recorded. Added `_verify_recorded_areas()`: after
  every solve path, for every parcel with a recorded area, check what was
  actually assigned against what was recorded, and emit a
  `recorded_area_not_matched` conflict if they diverge by more than a few
  superpixels -- catches the Sinkhorn degeneration and shortfall bugs
  structurally, not case by case.
- **The pipeline never completed its own documented last step.** The module
  docstring and the build plan both say to feed extracted boundaries through
  `planarize.py` and add the result to the block graph -- the original
  implementation stopped at raw polygons. Added `parcels_to_graph()`,
  wired through Stage 1's `planarize()`/`build_graph()` exactly like Stage
  1's own `blocks.py` (block boundary as authoritative `fixed` linework);
  verified the shared parcel-parcel boundary comes back as one real edge
  referenced by both faces, not independent polygon copies.
- **`np.vectorize` was 17-24x slower than a lookup array** for the
  superpixel-to-parcel label remap -- fixed, verified bit-identical output.
- **Public API didn't use the project's own `Geom` convention.** Every other
  module passes CRS-tagged `Geom`s across boundaries; this one took a bare
  `Polygon` plus a separate `crs: str`. Fixed: `assign_parcels()` and friends
  now take a single `block: Geom`.
- Also fixed: a docstring/behavior mismatch (unassigned superpixels are
  represented by an absent dict key everywhere except one degenerate branch
  that used to store a stray `None`); `parcels_to_graph()` now wraps
  `build_graph()`'s generic hole-guard error with the likely cause (an
  "island" parcel from the leftover mechanism) named, instead of a bare
  "some face has a hole".

## Deferred (correctly, per Stage 3's own scope)
- Full multi-ring (hole) face support -- an "island" parcel pattern from the
  leftover/reservoir mechanism can still hit Stage 1's documented hole
  limitation. Same accepted gap as Stage 1; the error is at least now
  diagnosable.
- No test specifically drives the Sinkhorn code path (it requires an
  impractically large problem size to reach even after raising the
  threshold) -- covered instead by direct unit tests of the post-solve
  safety net that would catch its failure mode regardless of which code
  path produces it.
- The headline test caps blocks at 14 parcels (`_clean_blocks`'s
  `max_parcels`); an early attempt at a stricter per-instance (not
  aggregated) assertion found that a specific 22-parcel, 15%-visible block
  can let evidence-only "get lucky" on total boundary error even where the
  constrained solver recovers every individual parcel's area almost
  exactly -- a real, understood property of many-rival-neighbor blocks with
  very sparse evidence (swapping boundary-adjacent area between competing
  neighbors while preserving each one's own total), not a bug. The
  aggregate-over-many-blocks measurement the headline test actually uses is
  the correct way to state the thesis; per-instance guarantees on the
  hardest blocks are a separate, harder claim this stage doesn't make.
