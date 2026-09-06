"""Synthetic training data (Stage 4): SyntheticWard -> model input/target tensors.

Stage 0's ward already carries everything needed as ground truth: true
parcel boundaries (for the SDF target), roads, buildings, and parcel style
(as a land-use proxy) -- plus a rendered ortho + DSM/DTM as the model's
input. Train on synthetic first; this is the whole reason Stage 0 exists.
"""
from __future__ import annotations

import numpy as np
import torch
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from geocadastra.synth.generator import SyntheticWard

STYLE_TO_CLASS = {"formal": 0, "informal": 1, "institutional": 2}  # class 3 = under a road, see below
N_LANDUSE_CLASSES = 4

# A line rasterized at zero width is silently dropped wherever it falls
# exactly on the raster's own extent (GDAL has no pixel *centre* out there
# to test) -- found by review: a parcel's exterior boundary running along
# a ward's edge vanished entirely, giving `sdf_true` errors of ~20m in a
# 64m ward at every ward's border. A small buffer moves every line's
# rasterized footprint just inside the valid pixel-centre grid; 0.51 gsd
# is deliberately just past the exact half-pixel boundary, which is itself
# an exact-tie case GDAL resolves as "outside" (confirmed empirically: a
# 0.50 gsd buffer still drops the line, 0.51 does not).
_RASTERIZE_LINE_BUFFER_GSD_FACTOR = 0.51


def ward_to_tensors(ward: SyntheticWard) -> dict[str, torch.Tensor]:
    """The whole ward as one training example. A real training loop crops
    tiles from this; tiling for *inference* on a much larger raster is
    infer.py's job, a separate concern from how one training example is
    built here.
    """
    h, w = ward.dtm.shape
    gsd = abs(ward.transform.a)
    line_buf = gsd * _RASTERIZE_LINE_BUFFER_GSD_FACTOR

    # SDF target: distance to the nearest TRUE parcel boundary (every edge,
    # not just visible ones -- this is the supervised target; the model
    # only gets to SEE the visible ones in `rgb`, which is exactly why its
    # predicted variance should come out higher where a boundary is
    # invisible). Not signed in the classical inside/outside-of-one-region
    # sense -- parcels tile the whole ward, so there's no single global
    # inside/outside to sign against -- zero exactly at a boundary, growing
    # with distance into either neighbor, which is what "the zero level set
    # recovers the boundary" actually needs.
    rgb = torch.from_numpy(ward.ortho.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
    ndsm = torch.from_numpy((ward.dsm - ward.dtm).astype(np.float32)).unsqueeze(0)  # (1,H,W)

    boundary_lines = list(ward.edges_geom.values())
    boundary_mask = (
        rasterize(
            [(ln.buffer(line_buf), 1) for ln in boundary_lines], (h, w), transform=ward.transform, fill=0, dtype="uint8"
        ).astype(bool)
        if boundary_lines
        else np.zeros((h, w), dtype=bool)
    )
    sdf_true = torch.from_numpy((distance_transform_edt(~boundary_mask) * gsd).astype(np.float32)).unsqueeze(0)

    visible_lines = [ward.edges_geom[k] for k, v in ward.edges_visible.items() if v]
    visible_mask = (
        rasterize(
            [(ln.buffer(line_buf), 1) for ln in visible_lines], (h, w), transform=ward.transform, fill=0, dtype="uint8"
        ).astype(bool)
        if visible_lines
        else np.zeros((h, w), dtype=bool)
    )

    road_polys = [ln.buffer(3.0) for ln in ward.roads_centerline]
    road_true = (
        rasterize([(p, 1) for p in road_polys], (h, w), transform=ward.transform, fill=0, dtype="uint8").astype(np.float32)
        if road_polys
        else np.zeros((h, w), dtype=np.float32)
    )

    building_true = (
        rasterize([(b.polygon, 1) for b in ward.buildings], (h, w), transform=ward.transform, fill=0, dtype="uint8").astype(
            np.float32
        )
        if ward.buildings
        else np.zeros((h, w), dtype=np.float32)
    )

    # Class 3 ("under a road") needs its own rasterize pass drawn *after*
    # (i.e. overriding) the parcel classes: parcels tile the whole ward
    # including the land under a road (Stage 0 renders road surface on top
    # of parcels for the ortho, it doesn't carve road right-of-way out of
    # parcel geometry) -- so without this override, class 3 was reachable
    # in principle but never actually present in a single training pixel,
    # leaving that logit permanently untrained (found by review).
    landuse_shapes = [(p.polygon, STYLE_TO_CLASS[p.style]) for p in ward.parcels]
    landuse_true = rasterize(landuse_shapes, (h, w), transform=ward.transform, fill=3, dtype="int64")
    if road_polys:
        road_class_mask = rasterize(
            [(p, 1) for p in road_polys], (h, w), transform=ward.transform, fill=0, dtype="uint8"
        ).astype(bool)
        landuse_true[road_class_mask] = 3

    return {
        "rgb": rgb,
        "ndsm": ndsm,
        "sdf_true": sdf_true,
        "visible_mask": torch.from_numpy(visible_mask),
        "road_true": torch.from_numpy(road_true).unsqueeze(0),
        "building_true": torch.from_numpy(building_true).unsqueeze(0),
        "landuse_true": torch.from_numpy(landuse_true),
    }
