"""Diagnostic, not a training run: can either model formulation learn to
reproduce boundaries on ONE manually-verified real tile, training and
evaluating on the exact same patches?

This is deliberately not a generalization claim -- train==eval on purpose.
If neither model can even overfit one well-labelled image, that means the
problem is upstream of data volume: a label, loss, or architecture issue
that more tiles and more GPU hours cannot fix. If one CAN, that tells us
the SDF-regression formulation (or the direct-classification one) is
learnable at all, and the next real question becomes generalization to a
different neighbourhood -- a separate, later check.

Two formulations, same input (RGB + nDSM), same tile, same patches:
  A. The production approach: MultiTaskNet's SDF head, regressing a
     continuous distance-to-boundary field, warm-started the same way as
     every other real run tonight.
  B. A direct boundary-pixel classifier: is this pixel within 0.3m of a
     boundary, yes/no, via a small CNN that never downsamples -- no
     stride, no pooling, full input resolution throughout. This tests
     whether MultiTaskNet's downsample-then-upsample path is itself
     losing the fine (sub-metre) detail a 0.3m-wide boundary line needs,
     independent of the loss-shape question already addressed elsewhere.

Both losses are weighted against the same ~90/10 class imbalance (this
tile's own boundary_weight matches masked_loss's decay=1.5 convention for
A; B uses BCE pos_weight computed from this tile's own boundary fraction)
so the comparison is between formulations, not between "one is weighted
and one isn't".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

from geocadastra.models.backbone import MultiTaskNet


class FullResBoundaryNet(nn.Module):
    """No stride, no pooling, anywhere -- every layer operates at the
    input's own resolution, deliberately so a thin boundary line is never
    downsampled away and has to be re-guessed by upsampling."""

    def __init__(self, width=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, rgb, ndsm):
        return self.net(torch.cat([rgb, ndsm], dim=1))


def load_tile_patches(npz_path, size=128, stride=64, min_valid=0.90):
    """Overlapping windows (stride < size) -- deliberately, since this is
    ONE tile: more patches to actually train on, at the cost of the usual
    non-overlap assumption, which does not matter for an overfit check."""
    z = np.load(npz_path)
    rgb, ndsm, distance, valid = z['rgb'], z['ndsm'], z['distance'], z['valid']
    h, w = valid.shape
    patches = []
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            v = valid[r:r + size, c:c + size]
            if v.mean() >= min_valid:
                patches.append((r, c))
    if not patches:
        raise ValueError(f'No patches meet valid>={min_valid} at size={size} in this tile')
    tensors, offsets = [], []
    for r, c in patches:
        tensors.append((
            torch.from_numpy(rgb[:, r:r + size, c:c + size].astype('float32') / 255.0),
            torch.from_numpy(ndsm[None, r:r + size, c:c + size].astype('float32')),
            torch.from_numpy(distance[None, r:r + size, c:c + size].astype('float32')),
            torch.from_numpy(valid[None, r:r + size, c:c + size].astype('float32')),
        ))
        offsets.append((r, c))
    return tensors, offsets, (rgb, ndsm, distance, valid)


def train_A_sdf(patches, warm_start, epochs, boundary_weight=15.0, decay=1.5, lr=1e-3):
    model = MultiTaskNet()
    if warm_start:
        source = torch.load(warm_start, map_location='cpu', weights_only=True)
        model.load_state_dict(source.get('model', source), strict=True)
    for head in (model.road_head, model.building_head, model.landuse_head):
        for param in head.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for rgb, ndsm, target, valid in patches:
            rgb, ndsm, target, valid = rgb.unsqueeze(0), ndsm.unsqueeze(0), target.unsqueeze(0), valid.unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            out = model(rgb, ndsm)
            precision = torch.exp(-out['log_var'])
            per_pixel = 0.5 * precision * (out['sdf'] - target) ** 2 + 0.5 * out['log_var']
            weight = valid * (1.0 + boundary_weight * torch.exp(-target.clamp(min=0) / decay))
            loss = (per_pixel * weight).sum() / weight.sum().clamp(min=1.0)
            loss.backward(); optimizer.step(); total += loss.item()
        scheduler.step()
        history.append(total / len(patches))
    return model, history


def train_B_classifier(patches, epochs, lr=1e-3):
    all_target = torch.cat([(t <= 0.3).float() for _, _, t, _ in patches])
    all_valid = torch.cat([v for _, _, _, v in patches])
    boundary_fraction = (all_target * all_valid).sum() / all_valid.sum().clamp(min=1.0)
    pos_weight = ((1 - boundary_fraction) / boundary_fraction.clamp(min=1e-4)).clamp(max=50.0)
    print(f'  Model B: boundary fraction {boundary_fraction:.4f}, pos_weight {pos_weight:.2f}', flush=True)

    model = FullResBoundaryNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for rgb, ndsm, target, valid in patches:
            rgb, ndsm, valid = rgb.unsqueeze(0), ndsm.unsqueeze(0), valid.unsqueeze(0)
            binary_target = (target.unsqueeze(0) <= 0.3).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, ndsm)
            per_pixel = F.binary_cross_entropy_with_logits(logits, binary_target, pos_weight=pos_weight, reduction='none')
            loss = (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)
            loss.backward(); optimizer.step(); total += loss.item()
        scheduler.step()
        history.append(total / len(patches))
    return model, history


@torch.no_grad()
def evaluate_A(model, patches):
    model.eval()
    tp = fp = fn = 0
    for rgb, ndsm, target, valid in patches:
        out = model(rgb.unsqueeze(0), ndsm.unsqueeze(0))
        pred = (out['sdf'].abs() <= 0.3) & valid.unsqueeze(0).bool()
        truth = (target.unsqueeze(0) <= 0.3) & valid.unsqueeze(0).bool()
        tp += (pred & truth).sum().item(); fp += (pred & ~truth).sum().item(); fn += (~pred & truth).sum().item()
    model.train()
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))


@torch.no_grad()
def evaluate_B(model, patches):
    model.eval()
    tp = fp = fn = 0
    for rgb, ndsm, target, valid in patches:
        logits = model(rgb.unsqueeze(0), ndsm.unsqueeze(0))
        pred = (torch.sigmoid(logits) > 0.5) & valid.unsqueeze(0).bool()
        truth = (target.unsqueeze(0) <= 0.3) & valid.unsqueeze(0).bool()
        tp += (pred & truth).sum().item(); fp += (pred & ~truth).sum().item(); fn += (~pred & truth).sum().item()
    model.train()
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))


def render_overlay(rgb, true_mask, pred_a_mask, pred_b_mask, out_path):
    img = rgb.transpose(1, 2, 0).copy()
    img[binary_dilation(true_mask, iterations=1)] = [0, 255, 0]       # true boundary: green
    img[binary_dilation(pred_a_mask, iterations=1)] = [255, 0, 0]     # model A: red
    img[binary_dilation(pred_b_mask, iterations=1)] = [0, 128, 255]   # model B: blue
    # Where predictions and truth coincide, later assignment wins -- report
    # overlap numerically instead of relying on colour blending.
    Image.fromarray(img).save(out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', required=True)
    p.add_argument('--warm-start', default='runs/synthetic_pretrain/last.pt.best')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--out', default='runs/overfit_one_tile')
    args = p.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    patches, offsets, (rgb_full, _, distance_full, valid_full) = load_tile_patches(args.npz)
    print(f'{len(patches)} overlapping patches from this one tile, training==evaluating on all of them', flush=True)

    print('Training model A (MultiTaskNet SDF regression, warm-started)...', flush=True)
    model_a, history_a = train_A_sdf(patches, args.warm_start, args.epochs)
    metrics_a = evaluate_A(model_a, patches)
    print(f'  A: train loss {history_a[0]:.4f} -> {history_a[-1]:.4f}; {metrics_a}', flush=True)
    torch.save(model_a.state_dict(), outdir / 'model_a.pt')  # before anything else can lose 40 minutes of training

    print('Training model B (full-resolution direct classifier, cold)...', flush=True)
    model_b, history_b = train_B_classifier(patches, args.epochs)
    metrics_b = evaluate_B(model_b, patches)
    print(f'  B: train loss {history_b[0]:.4f} -> {history_b[-1]:.4f}; {metrics_b}', flush=True)
    torch.save(model_b.state_dict(), outdir / 'model_b.pt')

    # MultiTaskNet's cross-attention fusion is O((H*W)^2) in token count --
    # correct on a 128x128 training patch, but a full ~1200x800 tile needs a
    # ~60,000x60,000 attention matrix per head (confirmed: this crashed
    # trying to allocate 57.6GB the first time this script ran it that way).
    # Stitch the SAME patches the model was actually trained/evaluated on
    # instead of a single whole-tile forward pass it was never sized for.
    size = 128
    h, w = valid_full.shape
    pred_a_full = np.zeros((h, w), dtype=bool)
    pred_b_full = np.zeros((h, w), dtype=bool)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        for (rgb, ndsm, _target, _valid), (r, c) in zip(patches, offsets):
            out_a = model_a(rgb.unsqueeze(0), ndsm.unsqueeze(0))
            pred_a_full[r:r + size, c:c + size] |= (out_a['sdf'][0, 0].abs() <= 0.3).numpy()
            logits_b = model_b(rgb.unsqueeze(0), ndsm.unsqueeze(0))
            pred_b_full[r:r + size, c:c + size] |= (torch.sigmoid(logits_b)[0, 0] > 0.5).numpy()
    pred_a_mask = pred_a_full & valid_full.astype(bool)
    pred_b_mask = pred_b_full & valid_full.astype(bool)

    true_mask = (distance_full <= 0.3) & valid_full.astype(bool)
    render_overlay(rgb_full, true_mask, pred_a_mask, pred_b_mask, outdir / 'overlay.png')

    import json
    result = {'n_patches': len(patches), 'epochs': args.epochs,
             'model_a_sdf_regression': {'train_loss_start': history_a[0], 'train_loss_end': history_a[-1], **metrics_a},
             'model_b_direct_classifier': {'train_loss_start': history_b[0], 'train_loss_end': history_b[-1], **metrics_b}}
    (outdir / 'result.json').write_text(json.dumps(result, indent=2))
    print('\n' + json.dumps(result, indent=2))
    print(f'\nOverlay: {outdir / "overlay.png"} (green=true, red=model A, blue=model B)')


if __name__ == '__main__':
    main()
