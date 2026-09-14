"""Focused checks for portable checkpoints and masked boundary training."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('train_nz',Path(__file__).parents[1]/'train_nz.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def test_validation_metrics_load_with_weights_only(tmp_path):
    model=module.BoundaryNet()
    x=torch.zeros(1,3,16,16); y=torch.zeros(1,1,16,16); v=torch.ones_like(y)
    metrics=module.evaluate(model,[(x,y,v)],torch.device('cpu'))
    path=tmp_path/'last.pt'
    module.atomic_save({'history':[metrics],'model':model.state_dict()},path)
    restored=torch.load(path,weights_only=True)
    assert restored['history']==[metrics]

def test_invalid_pixels_do_not_contribute_to_loss_or_gradient():
    logits=torch.zeros(1,1,2,2,requires_grad=True)
    target=torch.ones_like(logits); valid=torch.tensor([[[[1.,0.],[0.,0.]]]])
    loss=module.loss_fn(logits,target,valid);loss.backward()
    assert logits.grad[0,0,0,0]!=0
    assert torch.count_nonzero(logits.grad)==1
