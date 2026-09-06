"""Multi-task network (Stage 4): one shared encoder, four heads.

Two-stream stems (RGB, nDSM) fused via cross-attention at a coarse scale --
tolerant to a few pixels of residual misregistration between the two
sensors, unlike a rigid per-pixel concat/add -- feed one shared timm
backbone trunk and one shared FPN-lite decoder. `heads.py`'s four heads
branch off that single decoder output; this is one network, not four
services.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from geocadastra.models.heads import BuildingHead, LandUseHead, RoadHead, SDFHead


class _ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Stem(nn.Module):
    """in_channels -> width, at 1/4 input resolution."""

    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.net = nn.Sequential(_ConvBNAct(in_channels, width, stride=2), _ConvBNAct(width, width, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Fuse two same-shape feature maps via bidirectional multi-head
    attention over flattened spatial tokens. Each RGB location can attend
    to any nDSM location (and vice versa) instead of only the one at the
    identical pixel coordinate -- a few pixels of residual misregistration
    between the two sensors degrades this gracefully instead of destroying
    the fusion the way a rigid per-pixel concat/add would.

    O((H*W)^2) in the fused feature map's token count -- fine at the coarse
    (1/4 input) resolution this runs at for the tile sizes infer.py uses;
    would need windowed/local attention if ever run at a much larger single
    tile.
    """

    def __init__(self, width: int, n_heads: int = 4):
        super().__init__()
        self.rgb_to_ndsm = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ndsm_to_rgb = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.norm_rgb = nn.LayerNorm(width)
        self.norm_ndsm = nn.LayerNorm(width)
        self.out_proj = nn.Conv2d(width * 2, width, kernel_size=1)

    def forward(self, rgb_feat: torch.Tensor, ndsm_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        ndsm_tokens = ndsm_feat.flatten(2).transpose(1, 2)

        rgb_attended, _ = self.rgb_to_ndsm(rgb_tokens, ndsm_tokens, ndsm_tokens)
        ndsm_attended, _ = self.ndsm_to_rgb(ndsm_tokens, rgb_tokens, rgb_tokens)

        rgb_out = self.norm_rgb(rgb_tokens + rgb_attended).transpose(1, 2).reshape(b, c, h, w)
        ndsm_out = self.norm_ndsm(ndsm_tokens + ndsm_attended).transpose(1, 2).reshape(b, c, h, w)

        return self.out_proj(torch.cat([rgb_out, ndsm_out], dim=1))


class FPNDecoder(nn.Module):
    """Top-down feature pyramid, finest-scale output. `in_channels_list`
    must be ordered finest-to-coarsest (matches timm's `features_only`
    output order)."""

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list])
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        x = laterals[-1]
        for lat in reversed(laterals[:-1]):
            x = F.interpolate(x, size=lat.shape[-2:], mode="nearest") + lat
        return self.smooth(x)


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        n_landuse_classes: int = 4,
        stem_width: int = 32,
        decoder_width: int = 32,
        backbone_name: str = "resnet18",
    ):
        super().__init__()
        self.rgb_stem = Stem(3, stem_width)
        self.ndsm_stem = Stem(1, stem_width)
        self.fusion = CrossAttentionFusion(stem_width)
        # out_indices=(0,1,2), not timm's default (all 5 stages): the deeper
        # stages are meant for 224px+ ImageNet inputs and, stacked on this
        # module's own /4 stem, collapse a modest tile (e.g. 64px) to a 1x1
        # feature map -- besides breaking BatchNorm outright at batch size 1
        # (found in training), a per-pixel regression task (the SDF head)
        # wants the finest stage close to native resolution, not buried
        # under abstraction depth built for image-level classification.
        self.trunk = timm.create_model(
            backbone_name, pretrained=False, features_only=True, in_chans=stem_width, out_indices=(0, 1, 2)
        )
        trunk_channels = self.trunk.feature_info.channels()
        self.decoder = FPNDecoder(trunk_channels, decoder_width)

        self.sdf_head = SDFHead(decoder_width)
        self.road_head = RoadHead(decoder_width)
        self.building_head = BuildingHead(decoder_width)
        self.landuse_head = LandUseHead(decoder_width, n_landuse_classes)

    def forward(self, rgb: torch.Tensor, ndsm: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb_feat = self.rgb_stem(rgb)
        ndsm_feat = self.ndsm_stem(ndsm)
        fused = self.fusion(rgb_feat, ndsm_feat)
        trunk_feats = self.trunk(fused)
        decoded = self.decoder(trunk_feats)
        decoded = F.interpolate(decoded, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        sdf, log_var = self.sdf_head(decoded)
        return {
            "sdf": sdf,
            "log_var": log_var,
            "road": self.road_head(decoded),
            "building": self.building_head(decoded),
            "landuse": self.landuse_head(decoded),
        }
