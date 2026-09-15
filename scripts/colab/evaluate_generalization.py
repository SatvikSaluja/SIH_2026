"""Evaluate the two overfit_one_tile.py checkpoints on a DIFFERENT tile,
without any further training and without touching either model's fixed
decision threshold (0.3m for A, 0.5 probability for B -- the same ones
used to score the training tile). This is the generalization check the
one-image experiment's own decision rule calls for.

Not the project's reserved final test tile (BX24_500_015024) -- this is
a separate, geographically distinct validation probe, kept apart from
that so the actual held-out test evaluation is still available later,
unspent.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from geocadastra.models.backbone import MultiTaskNet
from overfit_one_tile import FullResBoundaryNet, evaluate_A, evaluate_B, load_tile_patches, render_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', required=True)
    p.add_argument('--checkpoints', default='runs/overfit_one_tile')
    p.add_argument('--out', default='runs/overfit_one_tile/generalization_check')
    args = p.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ckdir = Path(args.checkpoints)

    patches, offsets, (rgb_full, _, distance_full, valid_full) = load_tile_patches(args.npz)
    print(f'{len(patches)} patches from a DIFFERENT tile than either model was trained on', flush=True)

    model_a = MultiTaskNet()
    model_a.load_state_dict(torch.load(ckdir / 'model_a.pt', map_location='cpu', weights_only=True))
    model_b = FullResBoundaryNet()
    model_b.load_state_dict(torch.load(ckdir / 'model_b.pt', map_location='cpu', weights_only=True))

    metrics_a = evaluate_A(model_a, patches)  # same >= .3m threshold as training-tile scoring
    metrics_b = evaluate_B(model_b, patches)  # same >0.5 probability threshold
    print(f'Model A on new neighbourhood: {metrics_a}', flush=True)
    print(f'Model B on new neighbourhood: {metrics_b}', flush=True)

    size = 128
    h, w = valid_full.shape
    pred_a_full = torch.zeros((h, w), dtype=torch.bool).numpy()
    pred_b_full = torch.zeros((h, w), dtype=torch.bool).numpy()
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        for (rgb, ndsm, _t, _v), (r, c) in zip(patches, offsets):
            out_a = model_a(rgb.unsqueeze(0), ndsm.unsqueeze(0))
            pred_a_full[r:r + size, c:c + size] |= (out_a['sdf'][0, 0].abs() <= 0.3).numpy()
            logits_b = model_b(rgb.unsqueeze(0), ndsm.unsqueeze(0))
            pred_b_full[r:r + size, c:c + size] |= (torch.sigmoid(logits_b)[0, 0] > 0.5).numpy()
    import numpy as np
    true_mask = (distance_full <= 0.3) & valid_full.astype(bool)
    render_overlay(rgb_full, true_mask, pred_a_full & valid_full.astype(bool),
                   pred_b_full & valid_full.astype(bool), outdir / 'overlay.png')

    result = {'npz': args.npz, 'n_patches': len(patches),
             'note': 'evaluated with the SAME fixed thresholds as the training-tile scoring; no tuning here',
             'model_a_sdf_regression': metrics_a, 'model_b_direct_classifier': metrics_b}
    (outdir / 'result.json').write_text(json.dumps(result, indent=2))
    print('\n' + json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
