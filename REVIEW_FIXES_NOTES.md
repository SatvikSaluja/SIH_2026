GeoCadastra — September review fixes

The 18 numbered findings in `review_2026_09_08/REVIEW.md` are addressed in this change. That review remains an historical record of the pre-fix behavior; its print-only probes are not regression tests for the corrected implementation.

**Transactions and recovery (#1–3, #10)**

All block writers use the same schema-qualified transaction advisory lock. Changeset commit acquires the lock before checking the loaded version snapshot and holds it through persistence. A worker owns one transaction from its idempotency check through graph publication, conflict writes, and completion status. Each fusion move uses a savepoint, so a refused candidate preserves other staged work; a crash aborts the entire attempt. The worker no longer uses a session-scoped lock, so no lock can escape into the connection pool. A legacy partially populated block retains its existing graph IDs on retry instead of receiving a duplicate seed.

Field verification inserts its survey measurement inside the geometry changeset transaction. Storage failures roll back both the measurement and the geometry. The existing globally ordered provenance chain still serializes appends across blocks while a transaction holds its chain lock; throughput scaling remains separate work.

**Evidence and geometry (#5–7, #9, #12–14)**

Co-registration retains the original legacy geometry, applies the affine transform from that original on every registration, stores the current transform/residual, and appends a control-point/transform history row referencing a changeset. It locks the ward's blocks in stable order and marks existing block jobs pending so a subsequent run can fuse against the new evidence. Repeating an alignment does not compound its transform.

Workers pass the union of individual legacy boundary lines to fusion. Conflicts now carry a `kind` and `detail`: area assignment conflicts and topology refusals survive alongside source disagreements. A displacement is nullable when it is not an appropriate measure of that conflict.

Graph seeding and synthetic ingest reject a CRS other than the store's declared EPSG:32643. Node edits use GEOS's 1 mm precision grid before topology validation and reject node collisions. New graph edges receive initial provenance including the assignment method, ward/source, parcel record references, and constraint conflicts when seeded by the worker.

Field verification validates parcel ownership against both ward and block. A composite database foreign key also prevents orphan or cross-ward survey references.

**Certification, uncertainty, and evaluation (#4, #8, #11, #15–18)**

Conformal calibration uses the augmented infinity order statistic when sample size cannot support the requested coverage. Missing edge bands are unbounded and need field work; they no longer count as zero-hour certification. Zero uncertainty next to an unbounded neighbour does not produce a NaN priority score.

The SDF head applies its log-variance parameterization once. The loss and direct inference consume that effective value. Tiled inference combines the component variances and mean disagreement through moment matching, then exports effective log variance; it also transfers input/output tensors to/from the model's device. Existing model checkpoints keep their parameter shapes, but cached inference/calibration outputs should be regenerated under this corrected variance contract.

Survey points have explicit fusion/calibration/evaluation purposes. New synthetic ingest assigns a deterministic, residual-independent 20% evaluation holdout; workers exclude it from fusion. Analytics computes P50 and P90 from the individual held-out residuals within each stratum and reports fusion-control residuals separately. A stratum with no held-out observations returns null accuracy values. Status polling derives progress from all expected blocks, including blocks without a job row, so asynchronous completion is reflected immediately.

Topology evaluation counts invalid faces without overlaying them and reports unchecked pairs explicitly. Valid faces with indeterminate overlaps are not counted as verified valid. No prediction is silently repaired to improve the metric.

**Database upgrade**

Fresh schemas use the updated SQLAlchemy metadata. Existing schemas use `migrations/0001_review_integrity.sql`, with the application's schema first in `search_path`, followed by `public`, and `psql -v ON_ERROR_STOP=1`. The migration is transactional and repeatable. It preserves legacy source geometry and keeps all historical survey observations marked as fusion inputs; previously used observations cannot honestly become held-out evaluation data. Foreign-key validation aborts rather than deleting any pre-existing invalid survey reference.

The migration is tested on an isolated schema containing real ingested rows, downgraded to the old columns and then upgraded twice. Application data in the public schema is not modified by the tests.

**Validation**

Regression cases in `geocadastra/tests/test_review_regressions.py` cover simultaneous commit attempts, simultaneous block deliveries, crash rollback and retry, savepoint refusal, affine transform persistence, conflict retention, small-sample bands, unknown edges, CRS rejection, provenance, field-verification atomicity and ownership, async status, precision, held-out metrics, variance consistency, invalid geometry, and schema migration. Existing tests that asserted the old incorrect semantics have been corrected rather than weakening their tolerances.

Full-suite command: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q --tb=short -ra`.

Completed on September 8, 2026: **395 passed, 2 xfailed, zero unexpected failures**, in **1584.04 seconds (26 minutes 24 seconds)**, on Python 3.12.3 with real local PostgreSQL/PostGIS. No slow tests were excluded. The two existing expected-failure acceptance tests (CPU model accuracy and the synthetic subdivision/boundary-area limitation) retained their original targets and executed. The 1268 warnings concern dependency deprecations. The complete output is preserved in `review_2026_09_08/FIX_VALIDATION.log`. An earlier interrupted run is not counted as validation.

The separate integration gaps in the original review remain: this task fixes the 18 numbered defects, not the unimplemented model/calibration/prioritisation API integration, exact-area redesign, interior-ring graph support, authentication, or GPU-scale training.

Subsequent foundational work is documented in [FOUNDATION_GAPS_NOTES.md](FOUNDATION_GAPS_NOTES.md); its tests and remaining limitations supersede the relevant historical statements above.
