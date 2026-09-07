# Stage 5 — fusion and conflict records

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
inverse-variance-weights whichever sources actually apply -- the
closed-form combination of independent Gaussian estimates, so "a confident
source dominates" and "two agreeing sources beat either alone" both fall
out for free. `fuse_node()`/`fuse_block()` tie it together per node and
per block; disagreement beyond `tolerance` becomes a `ConflictRecord`
instead of an average, per doc.

`geocadastra/store/changeset.py`: `ChangesetContext.move_node()` gained an
optional `evidence_type`/`detail` override (default still `"manual_edit"`,
zero behavior change for every existing caller) so a fusion-driven move is
honestly attributed in provenance. `apply_fusion()` applies a
`fuse_block()` result's moves one at a time, each in its own changeset --
see Review below for why that's not just a style choice.

Tests: `test_fusion.py` (21), `test_conflicts.py` (4),
`test_stage5_fusion_acceptance.py` (3, the stage's own Done-when), plus 3
new regression tests in `test_changeset.py`. 272 tests pass across all
five stages (`pytest geocadastra/ -q -m "not slow"`, ~150s).

## Full multi-angle review (post-build)
This stage's review was almost entirely about topology safety under
multi-node edits -- the doc's own words turned out to be exactly right:
*"after fusion, every Stage 1 topology invariant still holds. That second
assertion is the one that catches the classic bug."* It did.

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
  the block -- "nothing is silently resolved" applied to a new failure
  mode (evidence disagreeing with existing *topology*, not just with
  another source).
- **Fusion could silently move the block's own exterior, changing its
  total area.** Nothing distinguished a node on the block's true perimeter
  (road-derived, fixed) from an ordinary interior parcel corner --
  `legacy_estimate()` could pull an exterior node toward a legacy-shifted
  position, changing the enclosed area the Stage 1 "total face area
  equals input domain area" invariant checks. An attempted fix that let a
  perimeter node slide along its own straight run (common in this
  project's own data -- see below) turned out to need a tolerance
  separating "ordinary legacy-scale noise" from "a real corner", and on
  this project's actual synthetic data those two overlap in magnitude (an
  informal block can have a genuine corner as gentle as ~2.5m of
  deviation, well inside the ~9m of noise a 3-sigma legacy tolerance has
  to accept) -- no single threshold safely told them apart, and a
  too-generous one let a real corner drift ~100m² of area with neither
  the overlap nor the self-intersection guard ever tripping. Per this
  project's own rule ("prefer deleting a feature to shipping it
  untested"), reverted to the simple, provably-safe version: an exterior
  node never moves, full stop.
- **A conflict record's `geometry` had a silent fallback to a fake CRS.**
  `fuse_node()` originally defaulted to `"unspecified"` when no `crs` was
  given, directly violating this project's own invariant ("never let a
  geometry cross a module boundary without a declared CRS"). Fixed to
  raise instead -- a caller error, not something to paper over.

## A structural finding about the interaction between Stage 0 and Stage 5
Every one of this project's three block-subdivision styles
(`_strip_split`, `_recursive_split`'s single institutional cut, and its
recursive informal case) produces layouts where **every interior
parcel-to-parcel edge's endpoints land back on the block's own true
perimeter** -- confirmed by reading the generator and empirically across
40 real synthetic seeds (zero genuinely interior nodes, every time). This
means fusion, correctly refusing to ever move an exterior node, currently
has *no* node it's allowed to move on a typical generated ward's own
blocks -- `test_exterior_nodes_are_never_moved_by_fusion` locks this in as
an explicit, tested property rather than a silent no-op. The real
"evidence corrects an imprecise position" mechanism is fully built,
tested, and demonstrated (`test_gt_points_improve_accuracy_near_them_
without_degrading_elsewhere`, `test_topology_invariants_still_hold_after_
fusion`) against a deterministic hand-built grid with genuine interior
structure, matching the precedent already set for the overlap regression
test. Whether Stage 0's generator should grow a layout style with real
interior structure (a genuine internal lane, say) is a legitimate future
item, not something this stage's own scope covers.

## Deferred (correctly, per this stage's own scope)
- Conflict records are returned in-memory, not persisted to the store --
  same precedent Stage 3 set for its own conflicts; a real store/API home
  for them is Stage 8's concern once there's an actual endpoint.
- `model_estimate()` is wired and tested against a synthetic SDF/log_var
  raster, but not yet exercised end-to-end against a real trained Stage 4
  model's output in this stage's own tests -- consistent with Stage 4's
  own honestly-tracked accuracy gap; revisit together once GPU-scale
  training lands.
