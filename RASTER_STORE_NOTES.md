# Durable ward imagery

The prerequisite for ingesting anything real.

## The problem

`process_block` reconstructed its rasters by re-running the synthetic
generator from a stored seed:

    ward = _regenerate_ward(ward_source, ward_params)
    evidence_field, transform = simulate_evidence_field(ward, local_block_id)

That works only for a synthetic ward, and only when the stored params match
exactly -- `_regenerate_ward`'s own docstring records that mismatched params
silently produce a raster window that does not even overlap the real block.
Real imagery has no seed to regenerate from at all, so the pixels have to be
stored before any real ingest is possible.

## What was built

`ward_rasters` holds one row per ward per kind (ortho / dsm / dtm). Pixels go
to disk as GeoTIFF; the row carries width, height, affine transform and CRS,
so a reader never has to trust a filename. `path` is always derived from
`ward_job_id` and a checked `kind` -- nothing a caller supplies reaches the
filesystem, so there is no traversal to sanitise.

`read_block_window()` reads one block's window rather than materialising a
ward-sized array and slicing it.

`ingest_synthetic_ward()` now stores the rasters it generated. The model path
in `_block_evidence()` prefers them and falls back to regeneration for jobs
ingested earlier, recording `pixels: stored | regenerated` in provenance
alongside `evidence_source`.

## The bug the equivalence test caught

Reading a block's window off disk and cropping it from an in-memory ward are
the same computation. I wrote it twice, and the two disagreed:

    disk:   (3, 60, 68)
    memory: (3, 61, 69)

rasterio's `from_bounds(...).round_offsets().round_lengths()` floors the
offset and then ceils the *length*, which drops the partially covered pixel
at the far edge. Flooring the near edge and ceiling the far edge keeps it.

The covering version is the correct one: a window that undercovers its block
feeds the model a strip of missing evidence along two sides, which would have
shown up as systematically worse boundaries near two edges of every block --
a bug that is very hard to attribute after the fact.

The fix was not to match one rounding to the other but to delete one of the
implementations. `core/window.py::pixel_window()` is now the only place this
is computed, and both callers use it. Two implementations of one computation
was the actual defect; the pixel difference was only its symptom.

This is exactly why the test compares the two paths against each other rather
than asserting each one's shape separately. Two independent shape assertions
would both have passed.

## Deliberate limits

- Simulation still needs the regenerated ward. The simulated field is derived
  from the ward's ground-truth edges, not from pixels, so storing rasters does
  not free that path from `_regenerate_ward`. Only synthetic wards use it.
- Files are written before the caller's transaction commits, so a rolled-back
  ingest leaves orphaned GeoTIFFs. Garbage, not corruption: the rows naming
  them roll back, and a later ingest of the same ward overwrites them.
- There is still no real ingest endpoint. This stage makes one possible; it
  does not provide one.

## Test hygiene fix

A filesystem does not roll back with the test transaction. Without an autouse
fixture pointing `GEOCADASTRA_RASTER_ROOT` at a throwaway directory, every
ingesting test would write GeoTIFFs into `rasters/` in the working directory.
The conftest fixture now gives the same "leaves nothing behind" guarantee the
schema fixture already made for the database.

## Counts

fast: 413 passed (was 405; +8 raster store).
