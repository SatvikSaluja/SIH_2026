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

`geocadastra/tests/test_synth.py`: 19 tests, 5 seeds × the two required
invariants (zero gap/overlap tiling per block, exact area sum per block) plus
determinism, seed-sensitivity, style mix, the `visible_edge_fraction` knob at
0/1, legacy-layer sanity, raster/DSM consistency, and that crossing buildings
actually occur. All pass (`pytest geocadastra/tests/test_synth.py`, ~5s).

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
