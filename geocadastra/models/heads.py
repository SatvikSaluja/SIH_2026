"""Task heads (Stage 4): sdf, roads, buildings, landuse.

Each head is a thin conv stack off the shared decoder features `backbone.py`
produces -- these are heads on one network, not four separate models.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SDFHead(nn.Module):
    """Signed distance to the nearest parcel boundary, plus a per-pixel
    log-variance. Regressing distance rather than predicting a boundary
    mask gives sub-pixel boundary location (the zero level set) and can't
    average a thin boundary away, since there's no area term in the loss.

    The log-variance is what the heteroscedastic loss (`sdf_nll_loss`) is
    for: it lets the network say "I don't know" where the image gives no
    real cue (an invisible boundary) instead of being forced to commit to a
    number it can't actually predict -- which is the whole point of
    training this head with an uncertainty term instead of plain MSE.
    """

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),  # [sdf, log_variance]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


class RoadHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BuildingHead(nn.Module):
    """A binary building mask, not full per-instance ids: instances are
    separated post-hoc (connected components on this mask, or a watershed
    using the SDF head's own energy) rather than predicted directly here --
    keeps this head's loss simple (plain per-pixel BCE) and reuses the SDF
    head's boundary signal instead of duplicating it."""

    def __init__(self, in_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LandUseHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sdf_nll_loss(sdf_pred: torch.Tensor, log_var: torch.Tensor, sdf_true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Heteroscedastic (Kendall & Gal) negative log-likelihood: the model
    pays a squared-error penalty scaled down by its own predicted variance,
    plus a log-variance penalty that stops it claiming infinite uncertainty
    everywhere. This is *why* variance comes out higher on invisible
    boundaries: claiming high uncertainty there is the cheapest way to
    reduce loss on a target the image gives it no real way to predict.
    """
    # a hard clamp() gives EXACTLY zero gradient past its bounds (confirmed by
    # review with a direct autograd check) -- a pixel whose log_var races to
    # the clamp early in training (e.g. an easy match on an invisible-
    # boundary-adjacent pixel by chance) then has its calibration frozen for
    # the rest of training, regardless of how wrong later predictions are.
    # A soft (tanh) bound keeps exp() just as stable but never has literally
    # zero gradient, so a miscalibrated pixel can still be corrected.
    log_var = 10.0 * torch.tanh(log_var / 10.0)
    precision = torch.exp(-log_var)
    per_pixel = 0.5 * precision * (sdf_pred - sdf_true) ** 2 + 0.5 * log_var
    if mask is not None:
        per_pixel = per_pixel * mask
        return per_pixel.sum() / mask.sum().clamp(min=1.0)
    return per_pixel.mean()
