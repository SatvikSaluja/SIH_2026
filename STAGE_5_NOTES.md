Update: [CAPACITY_AND_SUBDIVISION_NOTES.md](CAPACITY_AND_SUBDIVISION_NOTES.md) documents generator version 2, constrained capacity refinement, and the corrected fusion corner behavior. Historical deferrals below are superseded where that note explicitly states so.

# Stage 5 — fusion and conflict records

> Update (8 September 2026): the subsequent whole-codebase fixes supersede affected behavior described below. See [REVIEW_FIXES_NOTES.md](REVIEW_FIXES_NOTES.md) for the corrected transaction, evidence, calibration, and evaluation contracts.

## Built
`geocadastra/core/conflicts.py`: `ConflictRecord` (parcel/face IDs, sources,
disagreement magnitude, geometry -- exactly what the doc asks for) and
`max_pairwise_disagreement()`. Kept separate from the fusion math itself,
matching the repo layout's own split.

`geocadastra/core/fusion.py`: correspondence-free per-source local lookups
-- `legacy_estimate()` (nearest POINT on legacy linework, never the nearest
vertex, so an independently re-vertexed or merged legacy polygon still
works), `model_estimate()` (nearest low-SDF pixel in a small window --
literally a local, windowed "fuse in distance-field space and re-extract
the zero set", since the model's own SDF output already IS a distance
field), `gt_estimate()` (nearest GT point, but only within a capture
radius -- beyond it, *no* estimate at all, not a low-confidence one, per
the doc's "carries no information outside it"). `fuse_estimates()`
inverse-variance-weights whichever sources actually apply. `fuse_node()`/
`fuse_block()` tie it together per node and per block; disagreement beyond
`tolerance` becomes a `ConflictRecord` instead of an average, per doc.

An exterior (block-perimeter) node gets separate handling --
`_is_exterior_node()`, and, when a `block_boundary` is given,
`_project_onto_block_boundary()` / `_ring_neighbor_bounds()` project its
candidate exactly onto the block's own TRUE boundary (sliding along a
straight run, or snapping exactly to a known corner) instead of ever
moving freely. See "Making exterior fusion actually work" below --  this
went through several real design iterations, not just the first one that
compiled.

`geocadastra/store/changeset.py`: `ChangesetContext.move_node()` gained an
optional `evidence_type`/`detail` override (default still `"manual_edit"`,
zero behavior change for every existing caller) so a fusion-driven move is
honestly attributed in provenance. `apply_fusion()` applies a
`fuse_block()` result's moves one at a time, each in its own changeset.

Tests: `test_fusion.py` (24), `test_conflicts.py` (3),
`test_stage5_fusion_acceptance.py` (5: 3 fast + 2 `slow`, one of the slow
ones `xfail`), plus 5 regression tests in `test_changeset.py`. 278 tests
pass across all five stages (`pytest geocadastra/ -q -m "not slow"`,
~130s); the 2 real-data `slow` tests add ~110s more.

## Full multi-angle review (post-build)
This stage's review was almost entirely about topology safety under
multi-node edits -- the doc's own words turned out to be exactly right:
*"after fusion, every Stage 1 topology invariant still holds. That second
assertion is the one that catches the classic bug."* It did, repeatedly,
at every level of increasing rigor this review pushed to.

- **`ChangesetContext` checked a face against itself, never against every
  OTHER face.** The existing `is_valid()` guard catches one face
  self-intersecting; it cannot catch two *different* faces newly
  overlapping each other, which several nodes moved independently in one
  changeset can cause even though each individually stays simple.
  Reproduced deterministically (three hand-built boxes, one shared corner
  pushed into a third box sharing no node with it): 11.2 m² of overlap
  written with no error, before the fix. Also reproduced on real
  synthetic data completely by accident while testing something else (an
  ordinary single-node "+1, +1" nudge in three *existing* Stage 2 tests
  turned out to already trigger this on their chosen seed -- present since
  Stage 2, just never checked for). Fixed: `_persist()` now checks every
  touched face against every other face for overlap beyond a small
  absolute tolerance (`_OVERLAP_TOL`, ~1cm², far above GEOS float noise at
  this project's coordinate scale, far below any real crossed-edge
  overlap). The three affected Stage 2 tests were fixed to use a
  provably-safe displacement (a verify-and-backoff nudge toward the
  node's own incident-faces' centroid, re-checked after every halving --
  a plain "move it a smaller amount" direction heuristic alone was not
  reliably safe either, confirmed by testing it).
- **The overlap check itself could crash instead of refusing cleanly.**
  `poly.intersection(other.geom)` without an explicit precision grid can
  throw `shapely.errors.GEOSException` ("side location conflict") on
  real, individually-valid inputs -- this project's own established GEOS
  lesson, rediscovered here: adding `grid_size=GRID` reduces but does not
  eliminate this on real synthetic data (reproduced: still crashes on 3
  of 8 real blocks tested). Fixed by also catching `GEOSException` around
  the check and refusing the edit the same way an actual detected overlap
  does -- a numerically pathological comparison means the same thing
  ("don't trust this edit") regardless of whether it manifests as a
  computed number or a raised exception. A crash is strictly worse than
  the safe refusal this check exists to produce.
- **Bundling a whole block's fusion moves into one changeset meant one
  unsafe candidate blocked every other, safe one.** `legacy_estimate()`'s
  "nearest point on the block's legacy linework" has no awareness of
  *which* parcel a node belongs to -- in a dense block, the nearest legacy
  point can genuinely be on an unrelated neighbor's edge. Measured
  directly: 21 of 52 fused candidates on a real synthetic block were
  individually topologically unsafe, *independent of how far they moved*
  (a near-zero move was exactly as likely to be unsafe as an 8m one --
  ruled out "cap the displacement" as a fix). Applying all 52 in one
  changeset meant the 31 good ones were held hostage by the 21 bad ones.
  Fixed: `apply_fusion()` applies each move in its own changeset, catching
  a refusal and recording it (`.topology_refused`) instead of aborting
  the block.
- **A conflict record's `geometry` had a silent fallback to a fake CRS.**
  `fuse_node()` originally defaulted to `"unspecified"` when no `crs` was
  given, directly violating this project's own invariant ("never let a
  geometry cross a module boundary without a declared CRS"). Fixed to
  raise instead -- a caller error, not something to paper over.

## Making exterior fusion actually work (not just safely refuse)
The first, simplest fix for "fusion could silently move the block's own
exterior, changing its total area" was to just never move an exterior
node at all. Asked explicitly to not settle for that if a correct,
tested, more capable version was achievable -- so it was tried, properly,
through several real iterations:

1. **Infer corner-vs-T-junction from the node's own current position.**
   Needs a tolerance separating "ordinary legacy-scale noise" from "a
   real corner (perimeter changes direction)". On this project's actual
   data those two overlap in magnitude (an informal block can have a
   genuine corner as gentle as ~2.5m of deviation, well inside the ~9m of
   noise a 3-sigma legacy tolerance has to accept) -- no fixed threshold
   safely told them apart, and a too-generous one let a real corner
   drift, changing area with neither the overlap nor self-intersection
   guard tripping. **Rejected** -- confirmed to fail on real data, not
   just reasoned about.
2. **Project onto the block's own TRUE boundary instead** (from
   `build_blocks()` -- the road network, a fixed, more authoritative
   reference; same precedent as `transport.py`'s `parcels_to_graph()`
   passing the block boundary to `planarize()` as `fixed` linework).
   Requires correctly clamping a node's slide to stay between its
   immediate ring-neighbors, which requires correctly handling the
   ring's own arbitrary coordinate-list seam (`shapely.project()`'s
   `[0, length)` parametrization wraps there) -- a plain
   `sorted(dist_a, dist_b)` gives the WRONG arc whenever a neighbor is
   near that seam, and this is common, not rare: it reproduced in the
   very first hand-built test case tried. Fixed with the standard
   "shortest signed offset on a circle" trick
   (`_ring_neighbor_bounds`), verified with a dedicated test built
   specifically to sit on the seam. The projection itself
   (`_project_onto_block_boundary`) uses shapely's own `project()`/
   `interpolate()` (never hand-rolled point-to-segment math, per this
   project's own rule), clamped between the ring-neighbor bounds so a
   candidate can't slide past an adjacent node and fold the ring's own
   walk order.
3. **Verified against real synthetic data, not just clean hand-built
   cases -- and that surfaced a real, separate Stage 0 defect.** This
   project's recursive-split parcel generator can leave a near-zero-area
   but real-extent internal gap between two parcels that should be
   exactly adjacent (confirmed directly: `unary_union()` of one real
   block's 25 parcels has an actual hole, ~2e-8 m² but spanning 37m of
   extent -- a GEOS precision artifact in `_recursive_split`'s repeated
   `split()`+`set_precision()` calls, not a Stage 5 concern). A node on
   that gap's edge registers as "OUTER-adjacent" identically to a node on
   the block's real perimeter -- nothing in the graph alone can tell them
   apart -- and was measured up to ~16m from the block's TRUE boundary,
   so blindly projecting it produced a confidently wrong (not obviously
   invalid) answer. Fixed with `max_boundary_distance`: an exterior node
   farther than this from the given boundary is left alone rather than
   projected, since a genuine perimeter point has no reason to be that
   far from its own true boundary regardless of ordinary noise.
4. **A follow-up hypothesis (add a general "combined touched-face area
   must be conserved" check to `ChangesetContext._persist()`) was tried
   and correctly reverted.** Reasoned that any topologically valid
   single-node move should conserve its incident faces' combined area --
   this is FALSE in general (directly disproved: moving one shared
   vertex of two adjacent 2x2 squares from its original position to a
   point 8 units away changes that single face's area from 4.0 to 2.0,
   with nothing invalid about the edit). The check, once added, promptly
   refused several ordinary, perfectly legitimate edits in the existing
   Stage 2 test suite. Reverted -- area conservation is a real property
   Stage 5 specifically wants for its OWN exterior-node handling (via the
   ring projection, by construction), not a general `ChangesetContext`
   invariant; a wrong general fix is worse than no fix.
5. **A "41 m² overlap slipped through the safety net" scare turned out to
   be a test-methodology bug, not a Stage 5 regression.** `ward.blocks[i]
   .id` restarts at 0 for every `generate_ward()` call; a test iterating
   several seeds against the same `db_session` reused that id directly as
   the DB block_id, silently conflating two entirely unrelated wards'
   geometry under one block -- of course two different wards' faces can
   "overlap" once merged into the same block. Fixed the test (a unique
   per-iteration block_id); re-verified the real safety property (no
   overlap, no invalid face) genuinely holds, across 10+ real ward
   blocks, once the test itself was measuring the right thing. A separate,
   even simpler own manual verification script had *also* been silently
   swallowing a `GEOSException` into "no overlap detected" instead of
   surfacing it (`except Exception: ov = -1`) -- worth remembering: an
   exception-swallowing measurement script can look like a passing
   result and mean nothing.

**Net result**: `fuse_block(..., block_boundary=...)` is a real,
substantially more capable, and verified-safe mechanism -- 80-93% of
candidate exterior-node improvements apply successfully on real synthetic
blocks (`test_block_boundary_fusion_never_produces_overlapping_or_invalid
_geometry_on_real_wards`, `slow`), with zero overlaps and zero invalid
faces confirmed directly, not assumed. The one property NOT reliably met
on real data is exact total-area conservation (Stage 1's own
domain-scaled tolerance) -- root-caused to the Stage 0 recursive-split
issue in point 3 above (projecting onto `block.polygon` when it isn't
quite the same shape the rest of the untouched graph already assumes),
confirmed independent of Stage 5's own logic (baseline, zero-fusion graphs
already match the domain area to float precision; the drift is
specifically introduced by projecting onto a boundary that's a slightly
different shape). Tracked honestly, not hidden or weakened:
`test_block_boundary_fusion_conserves_total_area_on_real_wards`,
`xfail(strict=False)`, reports real numbers on failure. Fixing it needs
Stage 0's recursive-split precision fixed (a separate undertaking,
flagged here, not pursued in this stage without being asked).

## Deferred (correctly, per this stage's own scope)
- Conflict records are returned in-memory, not persisted to the store --
  same precedent Stage 3 set for its own conflicts; a real store/API home
  for them is Stage 8's concern once there's an actual endpoint.
- `model_estimate()` is wired and tested against a synthetic SDF/log_var
  raster, but not yet exercised end-to-end against a real trained Stage 4
  model's output in this stage's own tests -- consistent with Stage 4's
  own honestly-tracked accuracy gap; revisit together once GPU-scale
  training lands.
- The Stage 0 recursive-split precision defect (internal near-zero-area
  holes, boundary-shape drift from the input polygon) found during this
  review is real and root-caused but not fixed here -- it's Stage 0's
  code, out of this stage's own scope unless asked to pursue it.
