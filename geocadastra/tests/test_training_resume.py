"""Training recovery must reproduce an uninterrupted optimizer schedule."""
import pytest
import torch
from geocadastra.models.backbone import MultiTaskNet
from geocadastra.models.train import TrainConfig, train
from geocadastra.synth.generator import WardParams


def model():
    torch.manual_seed(42)
    return MultiTaskNet(stem_width=16,decoder_width=16)


def config():
    return TrainConfig(n_wards=1,epochs=3,ward_params=WardParams(width=48,height=48,gsd=1,n_arterial_h=0,n_arterial_v=0,minor_spacing=25))


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    cfg=config()
    uninterrupted,history=train(cfg,model())
    path=tmp_path/"checkpoint.pt"
    _,partial=train(cfg,model(),checkpoint_path=path,max_epochs_per_run=1)
    assert len(partial)==1
    resumed,actual=train(cfg,model(),checkpoint_path=path,resume_from=path)
    assert actual==pytest.approx(history,abs=1e-7)
    for name,value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)
    assert len(list(tmp_path.iterdir()))==1


def test_resume_rejects_changed_dataset_and_training_enters_train_mode(tmp_path):
    path=tmp_path/"checkpoint.pt"
    net=model().eval()
    train(config(),net,checkpoint_path=path,max_epochs_per_run=1)
    assert net.training
    cfg=config()
    cfg.seed_offset=10
    with pytest.raises(ValueError,match="configuration"):
        train(cfg,model(),resume_from=path)


def test_resume_preserves_best_when_validation_worsens(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("geocadastra.models.train")
    losses = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr(module, "evaluate", lambda *args: next(losses))
    cfg = config()
    cfg.n_val_wards = 1
    path = tmp_path / "checkpoint.pt"
    train(cfg, model(), checkpoint_path=path, max_epochs_per_run=1)
    saved = torch.load(path, weights_only=True)
    assert saved["best_val_loss"] == 1.0
    best_path = path.with_name(path.name + ".best")
    best_bytes = best_path.read_bytes()
    # Also recover checkpoints written by the previous implementation.
    saved["best_val_loss"] = float("inf")
    torch.save(saved, path)
    train(cfg, model(), checkpoint_path=path, resume_from=path)
    assert best_path.read_bytes() == best_bytes
    latest = torch.load(path, weights_only=True)
    assert latest["best_val_loss"] == min(latest["val_loss_history"]) == 1.0
