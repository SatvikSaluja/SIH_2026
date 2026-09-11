Update: [CAPACITY_AND_SUBDIVISION_NOTES.md](CAPACITY_AND_SUBDIVISION_NOTES.md) documents generator version 2, constrained capacity refinement, and the corrected fusion corner behavior. Historical deferrals below are superseded where that note explicitly states so.

# Stage 0 — synthetic ward generator

## Built
`geocadastra/synth/generator.py`: `generate_ward(params: WardParams, seed: int) -> SyntheticWard`.

- Roads: arterials + a jittered minor grid, unioned with the ward boundary and
  `polygonize()`'d into blocks (zero-width centerlines for topology; road
  *width* is only used later, to rasterize road surface for the fake ortho).
- Each block gets a style (`formal` / `informal` / `institutional`, weighted
  random) that drives its subdivision:
  - formal → straight strips cut perpendicular to the block's longest
    oriented-bbox edge (frontage).
  - informal → recursive random guillotine cuts (irregular organic shapes).
  - institutional → usually left as one parcel, occasionally one cut.
  All three reduce to shapely `intersection`/`split`, so tiling is exact by
  construction rather than by fitting/snapping afterward.
- Buildings placed per parcel with a setback-eroded envelope; a fraction of
  informal-block buildings are deliberately pushed across a parcel line and
  clipped against the block instead of the parcel, to simulate encroachment.
- Boundary "edges" = shared-boundary linework between adjacent parcels (and
  between a parcel and the block exterior), each independently marked
  visible/invisible by `visible_edge_fraction`. Only visible edges get a
  rendered wall stroke in the ortho.
- Legacy GIS layer: per-parcel vertex jitter + a per-parcel systematic shift
  (both keyed by block style), then independent per-parcel "missing" drop and
  a "merge" pass that folds a parcel into one real spatial neighbor (drawn
  from the same shared-edge adjacency used for rendering, not id-adjacency).
- GT survey points: exact coordinates of a sparse, deduplicated subset of true
  parcel corners.
- Rendering: fake ortho (soil → veg → roads → buildings → visible walls,
  painted back to front) plus a DSM/DTM pair on the same grid/transform
  (DSM = smooth DTM + building height field + veg height field).
- All randomness flows through one `numpy.random.Generator(seed)` in a fixed
  call order — no `random` module, no unordered-set iteration in the path —
  so a `(params, seed)` pair is byte-identical across runs.

`geocadastra/tests/test_synth.py`: 92 tests (see "Full multi-angle review"
below for how it grew from 19) — 6 fixed seeds × the two required invariants
(zero gap/overlap tiling per block, exact area sum per block), a 60-example
`hypothesis` sweep of the same invariant over random seeds, determinism,
seed-sensitivity, style mix, the `visible_edge_fraction` knob at 0/1,
legacy-layer sanity/dedup/geometry-quality, GT-point sampling-bias, the
`_recursive_split` work budget, degenerate-block robustness (30 seeds of a
dense/jittery road config), `WardParams` validation, raster/DSM consistency,
and that crossing buildings occur. All pass
(`pytest geocadastra/tests/test_synth.py`, ~70s).

## Full multi-angle review (post-build)
Ran a 10-angle review (correctness × 5, reuse/simplification/efficiency/
altitude/conventions) over the Stage 0 diff, verified every candidate by
direct reproduction rather than taking the reviewing agents' word for it, and
fixed everything that reproduced. Two are worth flagging specifically because
they were caught only by testing *more*, which is exactly what was asked:

- **A phantom GEOS overlay bug**, found by a 60-example `hypothesis` seed
  sweep the 6 hand-picked seeds never exercised: at seed=660, block 18, two
  recursively-split parcels several generations deep — provably disjoint by
  construction (every ancestor split individually verified as an exact,
  non-overlapping partition) — had `.intersection()` report 209 m² of overlap
  (100% of the smaller parcel). Confirmed wrong by hand (ray-casting two
  sample points, both landing outside the "overlapping" polygon) and by
  bisecting `shapely.set_precision()` grid sizes: an explicit precision grid
  (even a very fine one) eliminates it entirely, because it forces GEOS onto
  its precision-aware, robustness-guaranteed overlay path instead of raw
  float overlay. Fixed by snapping every `split()`/`intersection()` output in
  `_strip_split`/`_recursive_split` to a `1e-9` m grid — checked over 200
  wards afterward: overlap and area-sum error both dropped to ~1e-10
  relative (float noise), no measurable precision cost. This is the same
  class of GEOS robustness issue the original build plan's invariant #5
  anticipates (snap-rounding before polygonizing); Stage 1's `planarize.py`
  is the real fix, this is the minimum needed to make Stage 0's own ground
  truth trustworthy in the meantime.
- **A numerically fragile test**, found at seed=21 by the same sweep: the
  zero-gap check used `shapely.ops.unary_union(polys).area`, which lost 88 m²
  (3.15%) on a 24-parcel informal block via a *separate* GEOS batch-union
  robustness issue — even though the parcels themselves were correct (areas
  summed exactly, pairwise overlap ~0, and an incremental left-fold union
  recovered the exact area). The test's own gap-detection method was the
  unreliable part, not the generator. Fixed by switching the gap check to an
  incremental union.

Other fixed correctness bugs: a legacy-jitter ring-closure bug that made
~every jittered legacy parcel self-intersect (independent per-vertex jitter
on a *closed* ring gave the duplicated first/last point two different
offsets); a legacy-layer merge-order bug that could duplicate a parcel's
geometry across two `legacy_parcels` entries (100% reproducible before the
fix); unbounded recursion in `_recursive_split` that hung >150s for an
extreme-but-plausible `min_area`/`max_depth` combination (fixed with a
shared work budget, degrading to coarser-but-still-exact tiling past the
cap); GT survey points sampled with probability biased toward corners shared
by many parcels (up to ~3.6× over target); a missing area filter in
`_polygonize_blocks` that could hand a degenerate face to `_strip_split` and
crash it; and three `WardParams` misconfigurations that previously crashed
deep inside generation with unclear errors (a style key present in
`style_weights` but missing from the four legacy-distortion dicts, a
reversed `strip_count_range`, an all-zero `style_weights`) — all now raise a
clear `ValueError` at construction via `__post_init__`.

Applied (not just reported) a few cheap, zero-risk cleanups flagged by
multiple review angles independently: a manual pairwise-`.union()` loop in
`_boundary_edges` that reinvented `unary_union` (already used three other
places in this file), duplicated "group parcels by block" logic in two
functions (factored into `_group_by_block`), and a buffer rebuilt per-face
instead of once. Declined the larger restructuring suggestions (merging the
four style-keyed legacy dicts into one, merging `edges_geom`/`edges_visible`
into one dict, dropping the derivable `Parcel.area` field) as bigger diffs
for marginal benefit at this stage — noted, not applied.

One reviewing agent's raw output was flagged by the harness as matching an
instruction-shaped pattern and had its control tags neutralized before
reaching this session — the actual content was benign (just "no CLAUDE.md
files found"), but noting it here for the record since it's a security-
relevant harness behavior, not a code finding.

## Deferred (correctly, per Stage 0's own scope)
- No CRS transform machinery — `CRS = "EPSG:32643"` is a plain string label,
  not a `pyproj`-backed reprojection. Real CRS handling is Stage 1's `crs.py`.
- `scikit-image` was installed pre-emptively for SLIC (Stage 3) but Stage 0
  never ended up needing it — dropped from `requirements.txt` until it's
  actually imported. Same for `pyproj`.
- No graph/edge persistence — `edges_geom`/`edges_visible` are plain dicts
  keyed by parcel-id pairs, good enough for Stage 0's own tests but not the
  node/edge/face model; that's Stage 1's `graph.py`.

## Assumption that needed correcting mid-build
The per-building ortho coloring originally called `rng.normal(0, 15, size=(H,W,3))`
*inside* the per-building loop — one full-frame noise array per building. On a
154-block/1151-parcel/715-building demo ward that was 27.8s. Rewrote it as one
`rasterize()` call tagging every building by index, plus one batched
noise/color array indexed by building id: 1.0s, same output shape/semantics
(later shapes still win on overlap, same as the loop). Left a comment at the
call site; no further batching attempted since 50k–200k-parcel-scale
rendering is explicitly a later-stage (tiled inference) concern, not Stage 0's.

## Knobs available now (per the "parameterise everything" requirement)
`visible_edge_fraction`, `legacy_jitter_std` / `legacy_shift_range` /
`legacy_p_missing` / `legacy_p_merge` (per style), `strip_count_range`,
`informal_min_area` / `informal_max_depth`, ward `width`/`height`/`gsd`, road
spacing/width, building setback/coverage/crossing probability, vegetation
count/size, GT point fraction.
