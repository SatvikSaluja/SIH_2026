"""Durable ward imagery: write once at ingest, read a window per block.

Processing used to reconstruct its rasters by re-running the synthetic
generator from a seed. Real imagery has no seed, so the pixels have to be
stored -- and once they are, a block reads only its own window instead of
materialising a ward-sized array to slice.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from sqlalchemy import select

from geocadastra.core.window import pixel_window
from geocadastra.store.schema import SRID, WardRaster

KINDS = ("ortho", "dsm", "dtm")
DTYPES = {"ortho": "uint8", "dsm": "float32", "dtm": "float32"}


def raster_root() -> Path:
    return Path(os.environ.get("GEOCADASTRA_RASTER_ROOT", "rasters"))


def _path(ward_job_id: int, kind: str) -> Path:
    # Server-derived from an integer id and a checked kind: nothing a caller
    # supplies reaches the filesystem, so there is no traversal to sanitise.
    if kind not in KINDS:
        raise ValueError(f"unknown raster kind {kind!r}")
    return raster_root() / f"ward_{int(ward_job_id)}" / f"{kind}.tif"


def store_ward_rasters(session, ward_job_id: int, *, ortho, dsm, dtm, transform, crs: str) -> None:
    """`ortho` is (H, W, 3) uint8; `dsm`/`dtm` are (H, W) float32.

    All three must share one grid -- they are read back as a single window
    per block, and a DSM on a different grid than its ortho would silently
    subtract heights from the wrong pixels.

    The files are written before the caller's transaction commits, so a
    rolled-back ingest leaves orphaned GeoTIFFs. That is garbage, not
    corruption: the rows that name them roll back with the transaction, and
    a later ingest of the same ward_job_id overwrites them.
    """
    if crs != f"EPSG:{SRID}":
        from geocadastra.core.crs import CRSMismatchError
        raise CRSMismatchError(f"store requires EPSG:{SRID}, got {crs}")
    height, width = dtm.shape
    if ortho.shape != (height, width, 3) or dsm.shape != (height, width):
        raise ValueError(f"ortho/dsm/dtm must share one grid, got "
                         f"{ortho.shape}, {dsm.shape}, {dtm.shape}")
    bands = {"ortho": np.asarray(ortho).transpose(2, 0, 1), "dsm": np.asarray(dsm)[None], "dtm": np.asarray(dtm)[None]}
    for kind, array in bands.items():
        path = _path(ward_job_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=array.shape[0], dtype=DTYPES[kind], crs=crs,
                           transform=transform, compress="deflate") as dst:
            dst.write(array.astype(DTYPES[kind]))
        session.merge(WardRaster(ward_job_id=ward_job_id, kind=kind, path=str(path),
                                 width=width, height=height, crs=crs,
                                 transform=list(transform)[:6]))


def has_ward_rasters(session, ward_job_id: int) -> bool:
    stored = set(session.scalars(select(WardRaster.kind).where(WardRaster.ward_job_id == ward_job_id)))
    return stored.issuperset(KINDS)


def read_block_window(session, ward_job_id: int, bounds):
    """Return `(rgb, ndsm, transform)` for `bounds`, read from disk.

    `rgb` is (3, h, w) float32 in [0, 1] and `ndsm` is (h, w) metres -- the
    exact shapes `run_tiled_inference` takes. The returned transform is the
    window's own, because a transform that disagrees with its array puts
    every parcel in the wrong part of the block.
    """
    rows = {r.kind: r for r in session.scalars(select(WardRaster).where(WardRaster.ward_job_id == ward_job_id))}
    missing = [k for k in KINDS if k not in rows]
    if missing:
        raise LookupError(f"ward {ward_job_id} has no stored {', '.join(missing)} raster")

    arrays = {}
    window = transform = None
    for kind in KINDS:
        row = rows[kind]
        with rasterio.open(row.path) as src:
            if window is None:
                # One window, computed once from the ortho's grid and reused:
                # recomputing per band would let a rounding difference give
                # the three bands different shapes.
                row0, row1, col0, col1 = pixel_window(src.transform, src.height, src.width, bounds)
                window = Window(col0, row0, col1 - col0, row1 - row0)
                transform = src.window_transform(window)
            arrays[kind] = src.read(window=window)

    rgb = arrays["ortho"].astype(np.float32) / 255.0
    ndsm = (arrays["dsm"][0] - arrays["dtm"][0]).astype(np.float32)
    return rgb, ndsm, transform
