GeoCadastra — subdivision and capacity refinement

**Subdivision version 2**

The old generator snapped each recursively split child independently and
rotated already-quantized formal strips back into world coordinates. Repeated
operations left hairline internal gaps: their area was near zero while their
boundaries extended 10–28 metres into blocks on reproduced seeds 21, 300 and
660. Merely changing the precision grid sometimes moved the failure elsewhere.

The new implementation collects subdivision cuts, nodes one shared boundary
network, and polygonizes it once. Formal subdivision rotates cutters, not
quantized output parcels. Recursive splitting determines where finite cuts
belong; its intermediate child polygons are not the final parcel geometry.
This preserves shared segments and junctions and eliminates the reproduced
internal crack boundaries. The split-work budget remains explicit.

WardParams.generator_version defaults to 2 and is persisted in ward params.
Version 1 retains the original implementation. Stored jobs whose params lack
a version regenerate with version 1, rather than silently changing their
raster evidence. A regression checks legacy vector/raster bytes against a
pre-change SHA-256 digest. No database migration is needed for this JSON field.

**Fusion corner correction**

Widening the area-conservation acceptance check from one seed to 40 exposed
a separate defect: fusion classified a corner from the proposed position,
which could project along an edge and remove a triangle of block land without
invalidating any polygon. It now recognizes an established corner from the
original node position within twice the graph precision grid of an
explicit authoritative block vertex. Such a corner retains that vertex;
non-corner boundary junctions retain their existing sliding behavior.

The Stage 5 area-conservation test is no longer an expected failure. The
40-ward safety sweep also checks area conservation at the original tolerance.
An older test's assumption that generated blocks contain no interior nodes
was replaced by the intended assertion that exterior nodes remain fixed when
no block boundary is supplied. Interior nodes may legitimately use evidence.

**Recorded-area refinement**

After raster assignment and parcel identity resolution, and before graph
seeding, the worker calls core.capacity.refine_recorded_areas(). The solver
adjusts shared graph node positions jointly toward recorded parcel areas.
It minimizes displacement, with larger penalties where the simulated boundary
evidence is stronger, retaining the original topology and record associations.
Corners remain fixed; other exterior nodes move along their original block
segment. Interior nodes have two coordinate degrees of freedom.

Acceptance is based on the actual final geometry, not the optimizer's success
flag. The GEOS-snapped graph must have distinct nodes, valid non-overlapping
faces, block coverage within 0.01 m², and every recorded area within its
explicit stored tolerance. A failed candidate leaves the input graph intact.
The worker records area_refinement_unresolved with a diagnostic and retains
its existing final area/coverage conflict reporting. Accepted refinement is
included in initial-edge provenance and the initial graph replay event.
Subsequent fusion and human changesets still enforce the area/coverage guards.

This is a bounded local constrained solver, not a universal feasibility
algorithm. It requires complete parcel identity and an initially tiled graph.
It declines inconsistent totals, insufficient degrees of freedom, more than
256 optimization variables, failed numerical solves, or results that fail
after snapping. The iteration limit is 120. Some feasible area allocations
require a different topology or sparse/global optimization; some recorded
areas cannot coexist on the declared precision grid. These remain visible
conflicts rather than rescaled records or false certification.

**Validation**

New tests cover the reproduced crack seeds plus additional wards, rotated
strips, subdivision budgets, legacy replay bytes, two- and three-record
capacity corrections, infeasible totals, real worker persistence, and corner
pinning. The targeted fusion/capacity suite passed 39 tests, including the
40-ward area/topology sweep. The completed full suite ran every slow test:
**427 passed, 1 xfailed, zero unexpected failures**, in **1767.00 seconds
(29 minutes 26 seconds)**. Only the existing CPU model-accuracy acceptance
test remains expected to fail. The Stage 5 area-conservation target was not
weakened and now passes as a normal test.

Command: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
--tb=short -ra`. Output: `review_2026_09_08/CAPACITY_VALIDATION.log`.
After that full run, the new evidence lookup's deprecated affine `*` operator
was replaced with `@`; all five capacity regressions, including real worker
persistence, passed again (7.66 seconds). Existing dependency warnings remain.
`git diff --check` passed. No public application schema was modified.

The model/calibration/prioritisation integration, real-data ingest,
authentication, GPU-scale accuracy and worker/broker deployment testing remain
separate work. Area compliance is not boundary certification.
