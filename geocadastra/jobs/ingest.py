"""Ingest a real ward: imagery from GeoTIFFs, records from the department.

`ingest_synthetic_ward` persists a ward the generator invented, and the
worker can regenerate its pixels from a seed. Nothing here can be
regenerated, so everything a later stage needs has to be stored now: the
imagery, the block polygons, the recorded areas and their seed points.

Every geometric fact is validated at this boundary rather than trusted,
because it is the one place data the project did not produce gets in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from geoalchemy2.shape import from_shape
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from geocadastra.core.crs import CRSMismatchError
from geocadastra.store.changeset import allocate_ids
from geocadastra.store.rasters import store_ward_rasters
from geocadastra.store.schema import (SRID, IngestedBlock, RecordedParcel, WardJob,
                                      block_id_seq, recorded_parcel_id_seq)

EXPECTED_CRS = f"EPSG:{SRID}"


def read_raster_triple(ortho_path, dsm_path, dtm_path):
    """`(ortho, dsm, dtm, transform, crs)` from three GeoTIFFs on one grid.

    The three are read as one grid on purpose. A DSM on a different grid,
    extent, or resolution than its ortho subtracts heights from the wrong
    pixels, and the result looks like plausible imagery rather than an
    error -- so the mismatch is refused here, where it is still legible.
    """
    with rasterio.open(ortho_path) as ortho_src:
        if ortho_src.count < 3:
            raise ValueError(f"ortho must have at least 3 bands, got {ortho_src.count}")
        crs = ortho_src.crs
        transform, height, width = ortho_src.transform, ortho_src.height, ortho_src.width
        ortho = ortho_src.read([1, 2, 3]).transpose(1, 2, 0)

    surfaces = {}
    for name, path in (("dsm", dsm_path), ("dtm", dtm_path)):
        with rasterio.open(path) as src:
            if (src.height, src.width) != (height, width):
                raise ValueError(f"{name} is {src.height}x{src.width}, ortho is {height}x{width}")
            if src.crs != crs:
                raise CRSMismatchError(f"{name} is {src.crs}, ortho is {crs}")
            if not np.allclose(tuple(src.transform)[:6], tuple(transform)[:6], rtol=0, atol=1e-6):
                raise ValueError(f"{name} georeferencing differs from the ortho's")
            surfaces[name] = src.read(1)

    if crs is None:
        raise CRSMismatchError("ortho has no CRS; a raster without georeferencing cannot be ingested")
    if crs.to_string() != EXPECTED_CRS:
        raise CRSMismatchError(f"ingest requires {EXPECTED_CRS}, got {crs.to_string()} -- reproject before ingest")
    return (ortho.astype(np.uint8), surfaces["dsm"].astype(np.float32),
            surfaces["dtm"].astype(np.float32), transform, EXPECTED_CRS)


def ingest_ward(session, *, source: str, blocks, parcels, ortho_path, dsm_path, dtm_path) -> WardJob:
    """Persist one real ward and return its job.

    `blocks` is an iterable of block polygons (shapely, in EXPECTED_CRS).
    `parcels` is an iterable of dicts with `block` (index into `blocks`),
    `area`, `style`, `seed_point` (an (x, y) tuple) and an optional
    `area_tolerance_m2`.

    Ids are allocated from the shared sequences, never taken from the
    caller's file: two uploads' "parcel 0" would otherwise collide on a
    primary key, and worse, two wards' same-numbered blocks would pool into
    one `load_block_graph()` result (`Face.block_id` has no ward column).
    """
    blocks = list(blocks)
    parcels = list(parcels)
    if not blocks:
        raise ValueError("a ward must contain at least one block")
    if not parcels:
        raise ValueError("a ward must contain at least one recorded parcel")

    ortho, dsm, dtm, transform, crs = read_raster_triple(ortho_path, dsm_path, dtm_path)
    _validate_geometry(blocks, parcels, ortho.shape[:2], transform)

    job = WardJob(source=source, params={}, status="pending", crs=crs)
    session.add(job)
    session.flush()

    block_ids = allocate_ids(session, block_id_seq, len(blocks))
    parcel_ids = allocate_ids(session, recorded_parcel_id_seq, len(parcels))
    for local_id, (block_id, polygon) in enumerate(zip(block_ids, blocks)):
        session.add(IngestedBlock(ward_job_id=job.id, block_id=block_id, local_block_id=local_id,
                                  geom=from_shape(polygon, srid=SRID)))
    for parcel_id, record in zip(parcel_ids, parcels):
        session.add(RecordedParcel(
            id=parcel_id, ward_job_id=job.id, block_id=block_ids[record["block"]],
            area=float(record["area"]), style=record["style"],
            area_tolerance_m2=float(record.get("area_tolerance_m2", 0.01)),
            seed_point=from_shape(Point(*record["seed_point"]), srid=SRID)))

    store_ward_rasters(session, job.id, ortho=ortho, dsm=dsm, dtm=dtm, transform=transform, crs=crs)
    session.flush()
    return job


def _validate_geometry(blocks, parcels, raster_shape, transform):
    """Refuse what later stages cannot recover from.

    Each check is here because the failure it prevents is silent downstream:
    an invalid block polygon produces an unusable face set, overlapping
    blocks double-count land, a seed point outside its block anchors the
    capacity solver in a neighbour, recorded areas exceeding the block make
    every refinement infeasible, and imagery that does not cover a block
    yields an evidence field of edge pixels.
    """
    for index, polygon in enumerate(blocks):
        if not polygon.is_valid:
            raise ValueError(f"block {index} is not a valid polygon")
        if polygon.area <= 0:
            raise ValueError(f"block {index} has no area")
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks[i + 1:], start=i + 1):
            if a.intersection(b).area > 1e-6:
                raise ValueError(f"blocks {i} and {j} overlap")

    height, width = raster_shape
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (width, height)
    covered = unary_union(blocks)
    if not (minx <= covered.bounds[0] and miny <= covered.bounds[1]
            and maxx >= covered.bounds[2] and maxy >= covered.bounds[3]):
        raise ValueError("imagery does not cover every block")

    recorded_by_block = {}
    for index, record in enumerate(parcels):
        for key in ("block", "area", "style", "seed_point"):
            if key not in record:
                raise ValueError(f"parcel {index} is missing {key!r}")
        if not 0 <= record["block"] < len(blocks):
            raise ValueError(f"parcel {index} references block {record['block']}, which does not exist")
        if not float(record["area"]) > 0:
            raise ValueError(f"parcel {index} has a non-positive recorded area")
        block = blocks[record["block"]]
        if not block.covers(Point(*record["seed_point"])):
            raise ValueError(f"parcel {index}'s seed point lies outside its own block")
        recorded_by_block.setdefault(record["block"], 0.0)
        recorded_by_block[record["block"]] += float(record["area"])

    for block_index, recorded in recorded_by_block.items():
        available = blocks[block_index].area
        tolerance = sum(float(p.get("area_tolerance_m2", 0.01))
                        for p in parcels if p["block"] == block_index)
        if abs(recorded - available) > tolerance + 0.01:
            raise ValueError(
                f"block {block_index}: recorded areas total {recorded:.2f} m2 but the block is "
                f"{available:.2f} m2, outside the {tolerance:.2f} m2 the records themselves allow")


def parcels_from_geojson(path, blocks):
    """Recorded parcels from a GeoJSON FeatureCollection.

    Each feature supplies `area` and `style` in its properties; the seed
    point is the feature's representative point, which -- unlike a centroid
    -- is guaranteed to lie inside its own polygon, so it cannot anchor the
    capacity solver in a neighbouring parcel.
    """
    import json

    blocks = list(blocks)
    features = json.loads(Path(path).read_text())["features"]
    records = []
    for index, feature in enumerate(features):
        geometry = shape(feature["geometry"])
        point = geometry.representative_point()
        containing = [i for i, block in enumerate(blocks) if block.covers(point)]
        if not containing:
            raise ValueError(f"parcel feature {index} does not fall inside any block")
        properties = feature.get("properties", {})
        if "area" not in properties:
            raise ValueError(f"parcel feature {index} has no recorded 'area' property")
        records.append({"block": containing[0], "area": float(properties["area"]),
                        "style": properties.get("style", "formal"),
                        "area_tolerance_m2": float(properties.get("area_tolerance_m2", 0.01)),
                        "seed_point": (point.x, point.y)})
    return records
