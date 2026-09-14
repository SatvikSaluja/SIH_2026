"""Tiled inference (Stage 4): run the model over a raster larger than one
tile, blending predictions in SDF space, and derive the evidence field
Stage 3 consumes.

Uncertainty comes from the predicted log-variance head in the same single
forward pass -- no MC dropout here. That's a real accuracy/cost tradeoff
(10x inference cost for MC dropout's repeated passes), so it's deferred
until Stage 6's calibration work shows the single-pass variance isn't
enough, not added pre-emptively.
"""
from __future__ import annotations

import numpy as np
import torch
from affine import Affine

from geocadastra.core.window import pixel_window
from geocadastra.models.backbone import MultiTaskNet

EVIDENCE_DECAY = 1.5  # metres; matches synth.generator.simulate_evidence_field's default, so
# Stage 3 sees the same functional shape from a real model as from the synthetic stand-in


def sdf_to_evidence(sdf: np.ndarray, decay: float = EVIDENCE_DECAY) -> np.ndarray:
    """High near sdf=0 (a predicted boundary), decaying with distance --
    the "high value = boundary probably runs here" convention Stage 3's
    assign_parcels() expects."""
    return np.exp(-np.abs(sdf) / decay).astype(np.float32)


def _tile_window(tile_size: int) -> np.ndarray:
    """2D weight, peak at the tile centre tapering toward its edges: a
    pixel predicted by a tile with more surrounding context on all sides
    (i.e. nearer that tile's centre) counts for more when two overlapping
    tiles disagree, instead of an unweighted average that treats a
    least-context edge prediction the same as a full-context centre one."""
    w1 = np.hanning(tile_size) if tile_size > 1 else np.ones(1)
    w1 = np.clip(w1, 1e-3, None)  # never exactly zero, or a pixel covered by only one tile's edge gets weight 0
    return np.outer(w1, w1).astype(np.float64)


def _tile_starts(total: int, tile_size: int, step: int) -> list[int]:
    if total <= tile_size:
        return [0]
    starts = list(range(0, total - tile_size + 1, step))
    if starts[-1] != total - tile_size:
        starts.append(total - tile_size)
    return starts


def run_tiled_inference(
    model: MultiTaskNet,
    rgb: np.ndarray,
    ndsm: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    n_landuse_classes: int = 4,
) -> dict[str, np.ndarray]:
    """`rgb`: (3, H, W) in [0,1]. `ndsm`: (H, W) or (1, H, W), metres.
    Returns per-pixel `sdf`, `log_var`, `road`, `building` (all (H, W),
    road/building already sigmoid-ed to [0,1]), `landuse`
    ((n_landuse_classes, H, W), softmax-ed), and `evidence` (H, W)."""
    model.eval()
    device = next(model.parameters()).device
    if ndsm.ndim == 2:
        ndsm = ndsm[None]
    _, h, w = rgb.shape
    tile_size = min(tile_size, h, w)
    step = max(tile_size - overlap, 1)

    accum = {k: np.zeros((h, w), dtype=np.float64) for k in ("sdf", "log_var", "road", "building")}
    accum_landuse = np.zeros((n_landuse_classes, h, w), dtype=np.float64)
    second_moment = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    window = _tile_window(tile_size)

    ys = _tile_starts(h, tile_size, step)
    xs = _tile_starts(w, tile_size, step)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y1, x1 = y + tile_size, x + tile_size
                rgb_tile = torch.from_numpy(rgb[:, y:y1, x:x1]).unsqueeze(0).float().to(device)
                ndsm_tile = torch.from_numpy(ndsm[:, y:y1, x:x1]).unsqueeze(0).float().to(device)
                out = model(rgb_tile, ndsm_tile)
                win = window[: y1 - y, : x1 - x]

                accum["sdf"][y:y1, x:x1] += out["sdf"][0, 0].cpu().numpy() * win
                # Moment-match the overlapping tile predictions: include
                # both their effective variances and disagreement in means.
                mu = out["sdf"][0, 0].cpu().numpy().astype(np.float64)
                variance = torch.exp(out["log_var"][0, 0]).cpu().numpy()
                second_moment[y:y1, x:x1] += (variance + mu * mu) * win
                accum["road"][y:y1, x:x1] += torch.sigmoid(out["road"][0, 0]).cpu().numpy() * win
                accum["building"][y:y1, x:x1] += torch.sigmoid(out["building"][0, 0]).cpu().numpy() * win
                accum_landuse[:, y:y1, x:x1] += torch.softmax(out["landuse"][0], dim=0).cpu().numpy() * win[None]
                weight_sum[y:y1, x:x1] += win

    weight_sum = np.clip(weight_sum, 1e-9, None)
    result = {k: (v / weight_sum).astype(np.float32) for k, v in accum.items()}
    variance = np.maximum(second_moment / weight_sum - (accum["sdf"] / weight_sum)**2, np.finfo(np.float32).tiny)
    result["log_var"] = np.log(variance).astype(np.float32)
    result["landuse"] = (accum_landuse / weight_sum[None]).astype(np.float32)
    result["evidence"] = sdf_to_evidence(result["sdf"])
    return result


def evidence_from_rasters(model, rgb, ndsm, transform, *, tile_size: int = 64, overlap: int = 16):
    """`(evidence_field, transform)` from pixels already on their final grid.

    `rgb` is (3, h, w) in [0, 1], `ndsm` is (h, w) in metres, and `transform`
    is that array's own georeferencing -- returned unchanged, because Stage 3
    consumes an (array, transform) pair and a transform that disagrees with
    its array puts every parcel in the wrong part of the block.
    """
    evidence = run_tiled_inference(model, rgb, ndsm, tile_size=tile_size, overlap=overlap)["evidence"]
    return evidence, transform


def crop_ward_block(ward, block_id: int):
    """`(rgb, ndsm, transform)` for one block of an in-memory synthetic ward.

    The stored-raster path reads the same window off disk instead; this one
    exists for wards that are regenerated rather than stored.
    """
    bounds = next(b for b in ward.blocks if b.id == block_id).polygon.bounds
    height, width = ward.dtm.shape
    row0, row1, col0, col1 = pixel_window(ward.transform, height, width, bounds)

    rgb = ward.ortho[row0:row1, col0:col1].astype(np.float32).transpose(2, 0, 1) / 255.0
    ndsm = (ward.dsm - ward.dtm)[row0:row1, col0:col1].astype(np.float32)
    return rgb, ndsm, ward.transform * Affine.translation(col0, row0)


def block_evidence_from_model(model, ward, block_id: int, *, tile_size: int = 64, overlap: int = 16):
    """Model evidence for one block of an in-memory ward, with
    `simulate_evidence_field`'s `(evidence_field, transform)` contract."""
    rgb, ndsm, transform = crop_ward_block(ward, block_id)
    return evidence_from_rasters(model, rgb, ndsm, transform, tile_size=tile_size, overlap=overlap)
