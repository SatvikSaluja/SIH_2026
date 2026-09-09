GeoCadastra — foundational gaps, September 9, 2026

This change builds on the 18 fixes in REVIEW_FIXES_NOTES.md. It closes the
missing parcel-identity and interior-ring paths, adds recorded-area and block
coverage enforcement to edits, implements independent changeset replay for
new histories, and makes training device-aware and resumable. It does not
claim the entire certification product is complete.

**Parcel identity and review**

`parcels_to_graph()` carries source recorded-parcel IDs through polygonization.
Association requires one unambiguous area-covering source, allowing only a
precision-grid boundary strip. Ambiguous or uncovered faces retain no parcel
ID and produce typed conflicts. Multiple faces may belong to one record;
area accounting sums their components. Initial IDs are remapped globally at
seeding and preserved in face rows, changeset events, and parcel API output.

Old faces remain unassigned after migration. Resolve them explicitly with
`POST /wards/{ward_id}/parcel-associations`, providing `block_id`, a complete
`assignments` mapping from face IDs to recorded parcel IDs, and `author`.
Association changes use the same transactional changeset and provenance
mechanism as geometry. Concurrent identity changes invalidate stale geometry
changesets. Field verification now requires the node to be incident to the
specified parcel, in addition to ward/block checks.

**Area and block coverage**

Each RecordedParcel has an explicit `area_tolerance_m2` (engineering default
0.01 m², not a jurisdictional acceptance rule). Changesets reject edits that
push a compliant parcel outside tolerance or worsen an existing discrepancy.
They also reject increasing the area of the symmetric difference between the
parcel union and the authoritative block beyond the existing error or the
0.01 m² block tolerance. Thus equal-area translation outside the block is
not an allowed workaround. Existing bad initial allocations can be improved
incrementally; they are never silently declared compliant.

`GET /wards/{ward_id}/constraints` reports current per-record areas, tolerance,
missing and unassigned geometry, and block coverage. Analytics now includes
per-record relative area errors and separates parcel count from face-component
count. Topology statistics explicitly identify their whole-block scope when
blocks contain multiple settlement styles. Workers retain final area and
coverage mismatches as typed conflict records after fusion.

This is constraint enforcement and transparent reporting, not an exact-area
solver redesign. Initial superpixel allocation still approximates capacities;
records with infeasible totals remain conflicts. Coupled geometry adjustment
that actually solves all feasible capacities is still required. Existing
conflict rows are historical observations; the constraints endpoint is the
live readiness check after subsequent edits.

**Interior rings**

A face has an exterior edge walk plus zero or more interior edge walks.
Shared island boundaries remain the same edge objects for both faces.
FaceBoundary stores a ring index; load, seed, geometry derivation, edits,
provenance, adjacency and survey uncertainty include interior rings. Enclosed
parcels are supported without filling or discarding their holes. The separate
Stage 0 subdivision precision defect is not claimed fixed by this change.

**Independent replay**

New seed changesets record the complete graph topology and initial positions;
edit events record their committed positions and parcel associations.
`store.replay.replay_block_graph()` reconstructs from these changeset events
without reading live faces, boundaries, nodes or edges. A regression deletes
those live rows before reconstruction. Stage 8 acceptance now compares actual
replayed geometry and associations, rather than just reloading face IDs.
Historical logs without a seed event are rejected explicitly; migration does
not fabricate a historical baseline. This reconstructs the graph, not all
ward imagery, source records, conflict history or external survey documents.

**Training**

TrainConfig.device declares the target device. The model, inputs and all
training targets move to it, and training explicitly enters train mode.
Optional epoch checkpoints atomically replace the previous checkpoint and
contain model, optimizer, scheduler, RNG state, completed epochs, losses and
dataset/schedule configuration. Resume rejects changed training configuration
but allows a device change. Supply the same model architecture when resuming
a custom model. Checkpoints are written at epoch boundaries; an interrupted
epoch is rerun. CPU resume is tested against uninterrupted training with exact
parameter equality. CUDA execution has not been tested on this machine.

Example:

```python
config = TrainConfig(epochs=100, device="cuda")
model, losses = train(config, checkpoint_path="checkpoints/ward.pt")
# After interruption, with the same configuration:
model, losses = train(config, resume_from="checkpoints/ward.pt",
                      checkpoint_path="checkpoints/ward.pt")
```

**Migration and validation**

Existing schemas must apply `migrations/0001_review_integrity.sql` followed by
`migrations/0002_parcel_identity_and_rings.sql`, using the application schema
first in search_path and `psql -v ON_ERROR_STOP=1`. Migration 0002 preserves
old exterior walks as ring 0, adds explicit tolerances and leaves unknown
parcel identity null. Repeat application is covered by an isolated-schema
regression. Fresh schemas use updated SQLAlchemy metadata. Public application
tables have not been migrated by this work.

Focused validation passed 78 tests before the final coverage guard, plus two
training recovery tests. The completed full suite included every slow test:
**408 passed, 2 xfailed, zero unexpected failures**, in **1669.97 seconds
(27 minutes 49 seconds)** on Python 3.12.3 with real local PostgreSQL/PostGIS.
The two expected failures retain the existing CPU accuracy and synthetic
subdivision area-preservation targets. The 1282 warnings are dependency
deprecations. Command: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
-m pytest -q --tb=short -ra`. Full output is saved in
`review_2026_09_08/FOUNDATION_VALIDATION.log`. `git diff --check` also passed.

**Remaining major work**

The model/calibration/prioritisation modules still need orchestration and API
integration with versioned evidence and certification invalidation after edits.
The constraints endpoint deliberately reports `boundary_certification` as
`not_calibrated`. No conformal whole-boundary guarantee is inferred from an
area check or from merely producing parcels. Real-data ingest, authentication,
GPU-scale accuracy, the generator precision defect, the exact-area allocation
redesign, real broker/process failure testing, and throughput work remain.
