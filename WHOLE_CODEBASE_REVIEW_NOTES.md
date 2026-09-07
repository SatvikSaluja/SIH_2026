# Whole-codebase review (post-Stage-5)

Requested explicitly after Stage 5 closed out: a fresh, full-codebase pass
(not just the latest diff) across all 16 non-test source files (~3,500
lines, Stages 0-5), using 9 independent finder angles (line-by-line,
removed-behavior, cross-file tracing, language pitfalls, wrapper
correctness, reuse, simplification, efficiency, altitude) dispatched in
parallel, each candidate then verified by direct reproduction against the
actual code before being reported or fixed. The 10th angle (CLAUDE.md
conventions) found nothing to check against -- no CLAUDE.md exists
anywhere in this repo or at the user level.

## Fixed (10 of 14 reported findings)

- **`apply_fusion()` missed `ConcurrentModificationError`**
  (`store/changeset.py`). Its per-node loop caught only `ValueError`;
  `ConcurrentModificationError` is a `RuntimeError`, so an ordinary
  concurrent edit landing mid-batch crashed the whole function, losing
  every already-applied result -- directly contradicting the function's
  own documented purpose. Reproduced with a real second session
  committing inside the exact load-to-commit window of one move (a
  monkeypatch on `move_node` was needed to land the race deterministically
  -- true concurrency can't be timed reliably any other way in a single
  Python call). Fixed: catch `(ValueError, ConcurrentModificationError)`
  specifically -- not a blanket `RuntimeError`, since a genuine "store is
  inconsistent" error (a different `RuntimeError` from `load_block_graph`)
  must still propagate loudly, not be folded into "this move was refused".

- **`model_estimate()` omitted `shape=` on `_world_to_pixel()`**
  (`core/fusion.py`). Without it, a node outside the raster could produce
  a small *negative* window bound (ceiling-clamped to the raster size but
  never floor-clamped), and Python's negative-index slicing silently
  wrapped the search window to the array's own tail instead of being
  empty. Reproduced: a node 10px above a 40-row raster's top edge returned
  a confidently wrong `SourceEstimate` from an unrelated row instead of
  the documented `None`. `transport.py`'s own `_world_to_pixel` docstring
  already documents this exact bug class being found and fixed elsewhere;
  this newer call site had just never adopted it.

- **Three functions silently discarded a `Geom`'s own declared CRS**
  (`core/blocks.py:build_blocks`, `core/transport.py:parcels_to_graph`,
  `core/fusion.py:_boundary_ring`). Each took a `Geom` boundary parameter
  and re-wrapped its bare geometry under a different, caller-supplied CRS
  string with no check the two matched -- exactly the "silently coerce"
  pattern `Geom`/`CRSMismatchError` exist to rule out everywhere else.
  Reproduced for all three (a Geom tagged EPSG:4326 against an EPSG:32643
  caller was silently accepted). Fixed: each now raises `CRSMismatchError`
  on a mismatch instead of overwriting it.

- **Hardcoded 3.0m road buffer, ignoring `WardParams.arterial_width`**
  (`models/dataset.py`). Correct only for minor roads (6.0m width);
  arterial roads (default 12.0m) got a `road_true`/landuse-class-3 target
  roughly half their true width. Reproduced: 1368px vs. the correct
  2184px on the same ward. Never exercised by any existing test/training
  config (all disable arterial roads). Fixed properly, not just
  band-aided: added a `road_widths` field to `SyntheticWard` (parallel to
  `roads_centerline`, populated by `_make_roads()`) rather than relying on
  an undocumented list-ordering assumption between the two modules, and
  `dataset.py` now buffers each road by its own true width.

- **`Geom.intersection()`/`.union()`/`.difference()` had no `grid_size`**
  (`core/crs.py`). This project's own canonical "every geometry crossing a
  module boundary should be a `Geom`" wrapper had skipped the precision-
  grid fix this codebase applies everywhere else -- ironic, since its own
  docstring is the one making that promise. Currently dead code (no live
  caller), fixed anyway since it's the class of latent trap a future
  caller would reasonably assume was already safe. Verified with buffered-
  circle overlays (many non-round coordinates) landing exactly on the
  precision grid after the fix.

- **Ground-truth edge computation lacked `grid_size`**
  (`synth/generator.py`, `_boundary_edges`). The real shared-boundary
  computation every downstream training target is measured against used a
  raw overlay op, in the same file whose own header comment documents a
  real historical wrong-answer bug from exactly this omission. Fixed the
  `intersection()` call. The paired `difference()` call was investigated
  and *deliberately left unfixed*: adding `grid_size` there crashes with
  `GEOSException: Overlay input is mixed-dimension` on input this call can
  legitimately receive (`shared_union`, a `unary_union` of several
  boundary segments, can become a mixed-dimension GeometryCollection) --
  confirmed by reproduction, it broke 14 previously-green tests across
  Stage 0/1. Reverted that one specific line with a comment explaining
  why, rather than either leaving a silent inconsistency or forcing a
  fix that doesn't actually work. **Lesson: an established "always add
  grid_size" pattern isn't universally safe -- shapely's grid-size overlay
  mode has its own real limitations (no mixed-dimension input) the
  non-grid mode tolerates; verify each application, don't apply the
  pattern reflexively.**

- **Vertex-snap in boundary projection ignored its own clamp**
  (`core/fusion.py:_project_onto_block_boundary`). The snap-to-nearest-
  vertex step searched the ENTIRE ring, unconstrained by the `dist_lo`/
  `dist_hi` clamp computed two lines above it -- so a candidate correctly
  clamped to stay within a node's valid arc could still get snapped past
  that boundary to an out-of-range vertex that happened to sit close in
  plain 2D space. Reproduced with a hand-built "staple" ring (a thin near-
  self-touching slit puts two topologically-distant vertices only 0.2m
  apart in 2D): a node whose valid arc ended 0.1m short of a vertex, at
  risk of snapping to that vertex (or an even-more-wrong one across the
  slit) instead of staying at the clamped boundary. Fixed: vertex-snap
  candidates are now restricted to vertices actually reachable within the
  same clamped arc.

- **Bare `except Exception` in `_recursive_split()`**
  (`synth/generator.py`). Converted ANY exception -- a real future bug in
  this function's own inputs, not just an expected GEOS split failure --
  into the same silent "keep the piece whole" fallback. Narrowed to
  `except GEOSException`. Confirmed while investigating: shapely's
  `split()` doesn't actually raise at all for the ordinary "cut doesn't
  touch the polygon" case (it just returns the polygon unsplit, already
  handled by the `len(pieces) < 2` check a few lines down) -- so the
  broad catch was never even needed for that case, only for genuine GEOS
  robustness failures.

- **`gt_points` truthiness check broke for a numpy array**
  (`core/fusion.py:gt_estimate`/`fuse_node`). `if not gt_points:`/
  `if gt_points:` on a parameter documented only as "an iterable of
  (x, y)" -- `bool()` on a numpy array of 2+ points raises
  `ValueError: truth value... is ambiguous`. Reproduced directly. Fixed
  with explicit `len()`/`is None` checks.

- **Mutable dict used as a shared default argument**
  (`core/fusion.py:DEFAULT_SIGMA_LEGACY_BY_STYLE`). Bound directly as the
  default across three functions -- dormant only because nothing in-repo
  currently mutates it in place. Fixed by making the constant itself a
  `types.MappingProxyType` (read-only) -- any future in-place mutation now
  raises immediately instead of silently corrupting every later call's
  default for the rest of the process's lifetime.

## Found, documented, deliberately not fixed in this round

- **`apply_fusion()`'s O(N) full-block DB reloads.** Opens one
  `ChangesetContext` per moved node, and every `__enter__` re-fetches and
  reconstructs the *entire* block from scratch. For a fusion pass moving
  N nodes in an F-face block, this is O(N) round-trips and O(N·F) redundant
  reconstruction/overlap-check work (tens of thousands of avoidable units
  at a realistic ~300-node block). A real, worthwhile optimization, but a
  genuine redesign (batching, or restructuring `apply_fusion` to reuse one
  loaded graph across moves with incremental re-validation) -- not
  attempted without being asked, to avoid rushing a correctness-critical
  path.
- **`max_boundary_distance`/`_is_exterior_node` as a call-site workaround**
  for the Stage 0 recursive-split defect (already tracked in
  STAGE_5_NOTES.md) rather than a fix at the source (`build_graph()`
  detecting an emergent hole in the union of faces). Same reasoning:
  a real, deeper fix, but Stage 0/1 code out of this round's scope.
- **Two reuse/duplication findings** (a hand-rolled CRS-mismatch check in
  `planarize.py`/`graph.py` duplicating `Geom._require_same_crs`; ~20
  duplicated lines between `assign_parcels()`/`assign_parcels_evidence_
  only()` in `transport.py`) -- legitimate cleanup, lower priority than
  the correctness fixes above, and the second touches already-shipped
  Stage 3 code without a correctness bug driving the change.

## Regression tests added
7 new tests locking in the above: `test_changeset.py` (1, the
`ConcurrentModificationError` race), `test_fusion.py` (5: outside-raster
`model_estimate`, `gt_points` array, `DEFAULT_SIGMA_LEGACY_BY_STYLE`
immutability, vertex-snap clamp, CRS rejection x2), `test_crs.py` (1,
grid-size snapping), `test_blocks.py` (1, CRS rejection),
`test_transport.py` (1, CRS rejection), `test_train.py` (1, road width).
291 tests pass (`pytest geocadastra/ -q -m "not slow"`, ~150s); the 2
`slow` tests remain 1 pass / 1 xfail exactly as before (Stage 4's
accuracy gap, Stage 5's area-conservation gap) -- this review changed
neither, confirming no regression to either honestly-tracked property.
