"""Train the actual GeoCadastra MultiTaskNet with resumable synthetic supervision.
The exported checkpoint is worker-format compatible; accuracy is measured separately.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from rasterio.features import rasterize
from geocadastra.models.train import train,TrainConfig
from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.dataset import ward_to_tensors
from geocadastra.synth.generator import WardParams,generate_ward

def evaluate(model,params,device,seeds):
    model.eval();rows=[]
    for seed in seeds:
        ward=generate_ward(params=params,seed=seed);t=ward_to_tensors(ward)
        with torch.no_grad():out=model(t['rgb'][None].to(device),t['ndsm'][None].to(device))
        sdf=out['sdf'][0,0].cpu().numpy();lv=out['log_var'][0,0].cpu().numpy()
        all_lines=list(ward.edges_geom.values());visible=[ward.edges_geom[k] for k,v in ward.edges_visible.items() if v]
        def mask(lines):
            return rasterize([(line.buffer(1.5),1) for line in lines],out_shape=(params.height,params.width),transform=ward.transform,fill=0,dtype='uint8').astype(bool) if lines else np.zeros((params.height,params.width),bool)
        vis=mask(visible);all_mask=mask(all_lines);inv=all_mask&~vis;pred=np.abs(sdf)<0.5*params.gsd
        to_reference=distance_transform_edt(~all_mask) if all_mask.any() else None
        # Explicitly report absent predictions; scipy EDT alone would give a misleading finite result.
        dist=distance_transform_edt(~pred)*params.gsd if pred.any() else None
        rows.append({'seed':seed,'predicted_boundary_pixels':int(pred.sum()),
          'predicted_boundary_fraction':float(pred.mean()),
          'precision_within_1px_of_reference':float((to_reference[pred]<=1).mean()) if pred.any() and to_reference is not None else None,
          'median_prediction_to_reference_error_px':float(np.median(to_reference[pred])) if pred.any() and to_reference is not None else None,
          'median_visible_error_px':float(np.median(dist[vis])/params.gsd) if vis.sum()>5 and dist is not None else None,
          'mean_visible_log_variance':float(lv[vis].mean()) if vis.sum()>5 else None,
          'mean_invisible_log_variance':float(lv[inv].mean()) if inv.sum()>5 else None})
    errors=[r['median_visible_error_px'] for r in rows if r['median_visible_error_px'] is not None]
    vv=[r['mean_visible_log_variance'] for r in rows if r['mean_visible_log_variance'] is not None]
    vi=[r['mean_invisible_log_variance'] for r in rows if r['mean_invisible_log_variance'] is not None]
    no_prediction=any(r['predicted_boundary_pixels']==0 for r in rows)
    median=float(np.median(errors)) if errors else None
    separated=bool(vv and vi and np.mean(vi)>np.mean(vv))
    return {'scope':'Held-out synthetic wards only; not field accuracy or calibrated certification',
       'median_boundary_error_px':median,'any_ward_without_predicted_boundary':no_prediction,
       'invisible_log_variance_greater_than_visible':separated,
       'degenerate_full_image_prediction':any(r['predicted_boundary_fraction']==1 for r in rows),
       'targets_met':bool(median is not None and median<=1 and separated and not no_prediction and all(r['predicted_boundary_fraction']<1 for r in rows)),
       'metric_warning':'Visible-boundary distance is one-sided and can be zero when every pixel is predicted boundary; inspect precision and reverse distance too. Targets are synthetic criteria with an added full-image-degeneracy guard, not real-world certification.', 'wards':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/stage4_gpu');p.add_argument('--wards',type=int,default=64)
    p.add_argument('--val-wards',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=400);p.add_argument('--epochs-this-run',type=int,default=5)
    p.add_argument('--resume',action='store_true');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--cpu',action='store_true');p.add_argument('--eval-wards',type=int,default=10)
    a=p.parse_args()
    if min(a.wards,a.epochs,a.epochs_this_run,a.eval_wards,a.batch_size)<1 or a.val_wards<1 or a.wards+a.val_wards>100000:raise ValueError('Invalid counts')
    if not a.cpu and not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime in Colab, or explicitly use --cpu for a local smoke test')
    device='cpu' if a.cpu else 'cuda';torch.set_num_threads(2);torch.manual_seed(0)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);checkpoint=root/'last.pt'
    params=WardParams(width=64,height=64,gsd=1,n_arterial_h=0,n_arterial_v=0,minor_spacing=30,wall_render_width=1.5)
    config=TrainConfig(n_wards=a.wards,epochs=a.epochs,device=device,batch_size=a.batch_size,n_val_wards=a.val_wards,ward_params=params,seed_offset=0,
        loss_weights={'sdf':2.,'road':.5,'building':.5,'landuse':.5})
    if a.evaluate:
        checkpoint=checkpoint.with_name(checkpoint.name+".best")
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        model=MultiTaskNet().to(device);model.load_state_dict(saved['model'])
        saved_params=WardParams(**saved['config']['ward_params'])
        # These seeds never overlap the permitted training seed range.
        metrics=evaluate(model,saved_params,device,range(100000,100000+a.eval_wards))
        metrics['checkpoint']=str(checkpoint)
        metrics['selected_epoch']=saved['epoch']
        (root/'heldout_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2));return
    if checkpoint.exists() and not a.resume:raise ValueError('Existing checkpoint: use --resume or another output folder')
    print('Device:',device,'GPU:',torch.cuda.get_device_name(0) if device=='cuda' else 'none',flush=True)
    start=time.monotonic()
    _,history=train(config,checkpoint_path=checkpoint,resume_from=checkpoint if a.resume else None,max_epochs_per_run=a.epochs_this_run)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    result={'best_val_loss':saved['best_val_loss'],'last_val_loss':saved['val_loss_history'][-1],
        'epochs_completed':len(history),'last_train_loss':history[-1],'run_seconds':time.monotonic()-start,
        'checkpoint':str(checkpoint),'data':'synthetic','configuration':asdict(config)}
    (root/'progress.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
