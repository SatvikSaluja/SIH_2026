GeoCadastra codebase review — 8 September 2026

I inspected every implementation module, traced the cross-module data flow, compared the stage notes and relevant test assertions, ran the fast suite, and reproduced failure cases against real Postgres/PostGIS in isolated temporary schemas. No application code or existing tests were modified. This directory contains only review artifacts. This is a review, not a claim that every possible defect has been found.

**What the code actually does**

The running application is a synthetic parcel reconstruction backend. `synth/generator.py` makes a ward, subdivides blocks into parcels, renders RGB/DSM/DTM rasters, distorts a legacy layer, and samples survey points. Road centerlines define blocks; road widths affect rendering, rather than excluding road land from parcel capacity.

The geometry core carries CRS labels, nodes linework with GEOS, polygonizes faces, and constructs ordered edge walks. The store assigns global IDs and persists node/edge versions, face-edge relationships, derived polygon caches, changesets, and an evidence hash chain.

The API ingests synthetic vector facts. A Celery task regenerates the ward and simulated evidence raster, assigns superpixels to parcel capacities using optimal transport, converts them into a graph, seeds the store, fuses legacy/survey evidence into node moves, saves some conflicts, and marks the block done. Parcel queries and vector tiles read the cached face polygons.

The model is a separate RGB/nDSM network with cross attention, a ResNet/FPN backbone/decoder, and four task heads. Its distance target is unsigned and measured in metres, despite the SDF name. Training and tiled inference exist, but the worker never invokes them. Conformal calibration and survey prioritisation also exist as standalone modules; the worker does not compute or persist their outputs. Consequently, a completed job currently delivers geometry, not a calibrated, certified parcel layer or a surveyor-hours-to-90% result.

The headline promises need qualification: capacities are rounded to superpixels, mismatched totals are rescaled with a conflict, and later geometry moves do not enforce recorded areas. Face identity is also lost through polygonization and store remapping; there is no durable face-to-recorded-parcel relationship. These are material integration/design gaps beyond merely waiting for GPU training.

**Findings, ordered by practical impact**

P1 means a correctness failure affecting geometry, evidence, or certification that should be resolved before relying on the output. P2 means an incorrect API result, metric, or invariant that also needs correction. “Reproduced” below means an executed diagnostic; “traced” means the conclusion follows from the implementation and its callers, without a full dedicated end-to-end reproduction.

1. **P1 — Concurrent changesets can both succeed and corrupt the face cache.** [changeset.py:280](/home/satvik/SIH_2026/geocadastra/store/changeset.py:280)

   The version check and persistence are separate operations without a block-level serialization mechanism. Two callers editing different nodes can both pass the check before either commits. The later caller overwrites `Face.geom` using its stale graph even though both node versions survive. A barrier after the real checks reproduced two successful commits, no concurrency error, and a 1 m² symmetric difference between the stored polygon and the polygon reconstructed from current nodes. The provenance lock is acquired too late to prevent this. Existing concurrency tests commit the first edit before the second performs its check, so they miss this interleaving. Serialize block commits and validate versions while holding the lock, or use an atomic block revision comparison.

2. **P1 — The worker's advisory lock is not pinned to a physical database connection.** [orchestrator.py:228](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:228), [orchestrator.py:349](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:349)

   `make_session()` binds a Session to an Engine. Every internal commit can return its connection to the pool, while the session-scoped Postgres lock remains on that physical connection. A subsequent query or unlock may use another connection. Reproduced with a two-connection pool: the backend PID changed after commit and `pg_advisory_unlock` returned false. Another task can borrow a connection already holding the lock, or the original task can wait on its own pooled lock; the finalizer ignores the false result. Pin an explicit Connection for the complete task and release the lock on that same connection. This reproduction exercises the exact Session/Engine lifecycle; it is not a full multi-process Celery load test.

3. **P1 — A crash during fusion leaves a block that cannot safely resume.** [orchestrator.py:283](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:283), [changeset.py:410](/home/satvik/SIH_2026/geocadastra/store/changeset.py:410)

   Seeding flushes the initial graph; the first fusion changeset commits that graph together with its move. An exception afterwards cannot roll it back. The retry unconditionally seeds another graph into the same block. Reproduced by raising after the first real fusion commit: one face remained under a failed block, and the retry raised `KeyError: move_node(): no node 9 in this graph`. In this case, duplicate overlapping faces caused a refusal to roll back the new seed, invalidating the remaining fusion node IDs. Other interleavings can retain duplicate graph data. Persist resumable phase/attempt state or make block publication atomic, using savepoints for individually refused moves. The existing crash tests stop between completed blocks, not inside a block.

4. **P1 — Small-stratum conformal bands do not provide their advertised coverage.** [conformal.py:71](/home/satvik/SIH_2026/geocadastra/core/conformal.py:71)

   Capping `ceil((n+1)*(1-alpha))` at `n` returns a finite maximum when the required order statistic is the extra infinity value. With one calibration point and alpha=0.1, the maximum covers an independent continuous test score only 50% of the time. Executing 10,000 independent uniform calibration/test pairs gave **50.22%**, not 90%. Return an unbounded band or explicitly decline certification when there are too few examples. The current tests and Stage 6 notes explicitly endorse the incorrect clamp, so they need correction too. The finite-sample coverage requirement is described in [Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6); the 50% counterexample is reproduced locally.

5. **P1 — Co-registration computes a transform but never applies or persists it.** [main.py:135](/home/satvik/SIH_2026/geocadastra/api/main.py:135)

   The endpoint stores only `coreg_residual_m`; neither the affine coefficients nor transformed geometry are retained. Reproduced a clean (+10,+20) translation: residual was approximately 1.4e-14 m, but the legacy geometry remained byte-for-byte unchanged. The worker therefore trusts an uncorrected layer with a near-zero alignment residual. Persist the transform and apply it to the relevant evidence, retaining its source and revision.

6. **P1 — Production fusion receives polygon interiors instead of legacy boundaries.** [orchestrator.py:296](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:296)

   `unary_union([to_shape(r.geom) ...])` unions filled polygons. `legacy_estimate()` then computes nearest points to that filled region. Any node inside it becomes its own supposed legacy boundary observation, often eliminating both correction and disagreement. Reproduced with a 10×10 square and node (5,5): the worker-shaped input returns (5,5), whereas the boundary input returns (10,5). Union the individual polygon boundaries; taking the boundary after unioning polygons would still erase internal shared lines. Stage 6 acceptance already uses the correct individual-boundary form, so it tests a different predictor from the worker.

7. **P1 — Assignment conflicts and topology refusals disappear at the worker boundary.** [orchestrator.py:280](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:280), [orchestrator.py:320](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:320)

   Only `apply_result.conflicts` is saved. Neither `AssignmentResult.conflicts` nor `apply_result.topology_refused` is persisted. A real 400 m² block given an 800 m² recorded parcel produced both `area_sum_mismatch` and `recorded_area_not_matched`, yet stored zero conflicts and completed. The topology-refusal omission is confirmed by tracing the return value. Store typed conflict records with the relevant area details or refusal reason; do not force area errors into a displacement-only schema.

8. **P1 — Unmeasured edges count as perfectly certain.** [priority.py:68](/home/satvik/SIH_2026/geocadastra/core/priority.py:68), [priority.py:263](/home/satvik/SIH_2026/geocadastra/core/priority.py:263)

   Missing edge values are skipped or default to zero. A face with no measurements gets uncertainty 0; `simulate_survey()` reports it certified before any work. Reproduced an empty uncertainty map producing a 100% certified fraction at zero hours. A partially measured face also ignores its unknown edges. Represent unavailable certification explicitly, reject incomplete inputs, or treat unknown uncertainty as unbounded. Existing tests currently assert this unsafe behavior.

9. **P1 — The store silently changes a graph's CRS label.** [changeset.py:137](/home/satvik/SIH_2026/geocadastra/store/changeset.py:137)

   `seed_block_graph()` writes every geometry with fixed SRID 32643 without checking `graph.crs`; loading always declares EPSG:32643. Reproduced seeding EPSG:4326 coordinates and reading them back as EPSG:32643 without transformation or rejection. Validate CRS before any write. The same fixed-SRID assumption needs an explicit guard on ward ingest.

10. **P1 — Field verification is split across two committed transactions.** [main.py:334](/home/satvik/SIH_2026/geocadastra/api/main.py:334)

    `_apply_move()` commits the geometry before the endpoint inserts its survey evidence. A survey insertion/commit failure leaves a geometry change even though the request fails and the measurement is absent. Reproduced by failing the second commit and rolling back the request: the node remained moved. Coordinate the changeset and survey attachment under one transaction, preserving savepoint behavior where needed.

11. **P2 — Ward status never finishes after real asynchronous execution.** [orchestrator.py:332](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:332), [main.py:211](/home/satvik/SIH_2026/geocadastra/api/main.py:211)

    Only `run_ward()` recomputes `WardJob.status`, immediately after dispatch. Workers update `BlockJob`, and `/status` simply returns the stale ward value. Reproduced dispatch returning before execution, then processing the sole block: the API returned ward `running` with its block `done`. Derive status from the complete expected block set at poll time or update it transactionally when tasks finish. Define failed/mixed states as well.

12. **P2 — Field verification accepts nonexistent or unrelated parcel IDs.** [main.py:347](/home/satvik/SIH_2026/geocadastra/api/main.py:347), [schema.py:227](/home/satvik/SIH_2026/geocadastra/store/schema.py:227)

    The block is checked against the ward, but `parcel_id` is neither validated nor backed by a foreign key. The endpoint accepted and persisted parcel ID 987654321, which did not exist. Another ward's parcel ID can similarly be attached, and analytics queries points by parcel ID without a ward condition. Validate ward/block/parcel ownership and, once face identity is persisted, the node's incidence to that parcel.

13. **P2 — Geometry edits bypass the fixed precision grid.** [graph.py:78](/home/satvik/SIH_2026/geocadastra/core/graph.py:78), [changeset.py:365](/home/satvik/SIH_2026/geocadastra/store/changeset.py:365)

    Node moves store the raw floats. Rounding a lookup key does not snap the node geometry; checking overlap with `grid_size` does not snap persisted coordinates either. Reproduced persisting x=10.000123 on the project's 0.001 m grid. Snap through the approved GEOS precision operation before topology validation and persistence, including collision checks. Raw survey observations can remain separately available as evidence.

14. **P2 — Initial graph edges have no provenance records.** [changeset.py:137](/home/satvik/SIH_2026/geocadastra/store/changeset.py:137)

    Seeding records a changeset but never calls `append_provenance()`. A seeded square had four edges and zero provenance rows. Edges that never receive a later fusion/manual move can remain permanently unexplained. The seed changeset description is not the promised per-edge evidence record. Append initial provenance containing assignment method, source references, and the relevant constraint/conflict context.

15. **P2 — Analytics labels an average of block medians as a stratum P50.** [main.py:467](/home/satvik/SIH_2026/geocadastra/api/main.py:467)

    `sum(p50s)/len(p50s)` is not the median over a stratum's survey errors, even if weighted by sample count. For blocks with errors [0,0,0] and [100], the endpoint formula yields 50 while the actual stratum P50 is 0. This is a direct arithmetic/call-site finding. Aggregate raw residuals or a valid mergeable quantile representation within each stratum, then compute the advertised percentile. The docstring also promises P90, but the response omits it.

16. **P2 — API boundary analytics reuses the survey points supplied to fusion.** [orchestrator.py:304](/home/satvik/SIH_2026/geocadastra/jobs/orchestrator.py:304), [main.py:457](/home/satvik/SIH_2026/geocadastra/api/main.py:457)

    The worker fits positions using the block's survey points; analytics then measures against the same table without a holdout split or a distinction between fit residual and validation error. Tracing confirms the identical population. These numbers can describe agreement with supplied control points, but cannot establish unseen-boundary accuracy. Persist training/fusion/calibration/evaluation roles and report held-out error separately. Stage 6's standalone held-out test does not make this API metric held out.

17. **P2 — Training and inference interpret log variance differently.** [heads.py:98](/home/satvik/SIH_2026/geocadastra/models/heads.py:98), [infer.py:49](/home/satvik/SIH_2026/geocadastra/models/infer.py:49), [fusion.py:173](/home/satvik/SIH_2026/geocadastra/core/fusion.py:173)

    The loss uses effective log variance `10*tanh(raw/10)`, but inference exports raw values and fusion takes `sqrt(exp(raw))`. Thus reliability weights are calculated from a different variance than the one optimized in training. For raw=20, the fitted sigma is about 124 while the interpreted sigma is about 22,026. Large raw values also restore the overflow that the loss transformation avoids. Define one variance parameterization used consistently at training and inference, with an explicit contract for tiled aggregation. The numerical mismatch was also executed in the math probe; no GPU training was needed to identify it.

18. **P2 — The topology evaluator can crash on the invalid geometry it is meant to count.** [evaluate.py:47](/home/satvik/SIH_2026/geocadastra/core/evaluate.py:47)

    The function identifies invalid polygons, then still performs intersection overlays on them. Reproduced a bow-tie face and a nearby square raising `GEOSException` instead of returning an invalid-face count. Classify invalid faces without feeding them into unsafe overlay, and define how numerically indeterminate overlaps are reported. Do not repair evaluated predictions silently or simply add a grid and assume that makes invalid input safe.

**Existing limitations and integration gaps, separate from new findings**

- The Stage 0 recursive split defect and Stage 1 lack of interior-ring support remain documented. I did not fix or remeasure their seed-frequency claims.
- The API does not expose calibrated bands, a survey priority plan, or the surveyor-hours curve. Neither the schema nor orchestration connects Stages 6/7 to completed jobs. The Stage 8 “certified layer” acceptance assertion only checks that parcels exist.
- Exact recorded-area preservation is not an end-to-end invariant: transport uses whole superpixels and a tolerance, graph construction may change faces, and fusion/manual edits validate topology without checking parcel capacity. The documented missing face-to-recorded-parcel link also prevents reliable per-parcel area reporting and legal record attribution.
- The current Stage 4 target is an unsigned distance field. If signed distance is truly required, that target contract needs a design change rather than just more training.
- Training and inference helpers create CPU tensors and provide no device transfer path. GPU rental alone will not enable these helpers to operate on a GPU-resident model.
- No authentication, fusion reload cost, synthetic-only ingest, and unresolved model acceptance accuracy are already tracked limitations, not newly discovered regressions.
- The available virtual environment is Python **3.12.3**, although the project brief specifies 3.11. Results here do not certify compatibility with the pinned intended interpreter.

**Validation and reproduction**

- `.venv/bin/python -m pytest -m 'not slow' -q`: **344 passed, 34 deselected**, 193 seconds.
- `.venv/bin/python -m pytest geocadastra/tests/test_api.py geocadastra/tests/test_orchestrator.py geocadastra/tests/test_stage8_acceptance.py -q`: **29 passed**, 1,193.81 seconds (19m 53s). Together with the fast suite, **373 existing tests passed**; the other five slow tests were not rerun.
- The diagnostic scripts [probe_db.py](probe_db.py) and [probe_math.py](probe_math.py) print the observed failure cases. They are explicit-run review probes, not regression tests that assert corrected behavior. Run with `.venv/bin/python -m pytest review_2026_09_08/probe_db.py review_2026_09_08/probe_math.py -s -q`.
- The database probe uses the project's default local test credentials, creates a UUID-named schema, and drops only that schema in `finally`. It uses real geometry operations and real database commits, with narrow injected failures/barriers to reach crash and race windows deterministically. It does not alter the application tables in `public`.
- The full Stage 4–7 slow acceptance group was not rerun; no GPU training or real Redis/multi-process Celery load test was performed. Asynchronous completion was exercised by delaying dispatch and then invoking the real task.
- An additional triangular-block probe recovered the exact block area and boundary; that candidate was not reported as a defect. Candidate findings were not accepted solely because they sounded plausible.

The first repair work should establish a consistent block transaction/locking/resume protocol, then preserve all evidence and conflicts, then repair certification and metric semantics. Add regression tests that exercise these failure windows before changing the implementation. This review makes no application-code changes.
