import torch

from geocadastra.models.backbone import CrossAttentionFusion, FPNDecoder, MultiTaskNet, Stem
from geocadastra.models.heads import BuildingHead, LandUseHead, RoadHead, SDFHead, sdf_nll_loss


def test_stem_downsamples_to_quarter_resolution():
    stem = Stem(3, 16)
    x = torch.randn(2, 3, 64, 64)
    out = stem(x)
    assert out.shape == (2, 16, 16, 16)


def test_cross_attention_fusion_preserves_shape_and_is_not_a_noop():
    fusion = CrossAttentionFusion(16, n_heads=4)
    rgb = torch.randn(2, 16, 8, 8)
    ndsm = torch.randn(2, 16, 8, 8)
    out = fusion(rgb, ndsm)
    assert out.shape == (2, 16, 8, 8)
    assert not torch.allclose(out, rgb)  # actually mixes information in, not a pass-through


def test_cross_attention_actually_attends_across_positions_not_just_the_aligned_one():
    """The structural reason attention tolerates a few pixels of
    misregistration, where a rigid per-pixel concat/add structurally
    cannot: a token CAN mix in information from other spatial positions,
    not only the one at its own coordinate. This is a capability the
    architecture provides -- whether a *trained* network uses it for
    exactly this purpose is a training outcome, not something a randomly
    initialized network would already exhibit, so this checks the
    mechanism directly rather than an emergent numerical property."""
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(16, n_heads=4)
    rgb = torch.randn(1, 16, 6, 6)
    ndsm = torch.randn(1, 16, 6, 6)
    rgb_tokens = rgb.flatten(2).transpose(1, 2)
    ndsm_tokens = ndsm.flatten(2).transpose(1, 2)

    _, attn_weights = fusion.rgb_to_ndsm(rgb_tokens, ndsm_tokens, ndsm_tokens, need_weights=True, average_attn_weights=True)
    # attn_weights: (batch, n_query_tokens, n_key_tokens) -- if it were just
    # "look at the same position," this would be a near-identity matrix
    n_tokens = attn_weights.shape[-1]
    diagonal_mass = attn_weights.diagonal(dim1=-2, dim2=-1).mean()
    off_diagonal_mass = (attn_weights.sum() - attn_weights.diagonal(dim1=-2, dim2=-1).sum()) / (n_tokens * n_tokens - n_tokens)
    assert off_diagonal_mass > 0  # genuinely spreads attention beyond the matching position
    assert diagonal_mass < 1.0 - 1e-4  # not a disguised identity/pass-through


def test_fpn_decoder_merges_multiscale_features_to_finest_resolution():
    decoder = FPNDecoder([16, 32, 64], out_channels=8)
    feats = [torch.randn(1, 16, 32, 32), torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8)]
    out = decoder(feats)
    assert out.shape == (1, 8, 32, 32)


def test_heads_produce_expected_channel_counts():
    x = torch.randn(2, 32, 16, 16)
    sdf, log_var = SDFHead(32)(x)
    assert sdf.shape == (2, 1, 16, 16)
    assert log_var.shape == (2, 1, 16, 16)
    assert RoadHead(32)(x).shape == (2, 1, 16, 16)
    assert BuildingHead(32)(x).shape == (2, 1, 16, 16)
    assert LandUseHead(32, n_classes=4)(x).shape == (2, 4, 16, 16)


def test_sdf_nll_loss_prefers_high_variance_where_prediction_is_wrong():
    """This is the mechanism the Done-when property (higher variance on
    invisible boundaries) depends on: for a fixed, large error, does the
    loss actually prefer a higher log_var over a lower one?"""
    sdf_pred = torch.zeros(1, 1, 4, 4)
    sdf_true = torch.full((1, 1, 4, 4), 5.0)  # a large, unavoidable error
    low_var_loss = sdf_nll_loss(sdf_pred, torch.zeros(1, 1, 4, 4), sdf_true)
    high_var_loss = sdf_nll_loss(sdf_pred, torch.full((1, 1, 4, 4), 3.0), sdf_true)
    assert high_var_loss < low_var_loss


def test_sdf_nll_loss_prefers_low_variance_where_prediction_is_right():
    sdf_pred = torch.zeros(1, 1, 4, 4)
    sdf_true = torch.zeros(1, 1, 4, 4)  # perfect prediction
    low_var_loss = sdf_nll_loss(sdf_pred, torch.full((1, 1, 4, 4), -5.0), sdf_true)
    high_var_loss = sdf_nll_loss(sdf_pred, torch.zeros(1, 1, 4, 4), sdf_true)
    assert low_var_loss < high_var_loss


def test_multitasknet_forward_and_backward_produce_all_four_heads():
    net = MultiTaskNet(n_landuse_classes=4, stem_width=16, decoder_width=16, backbone_name="resnet18")
    rgb = torch.randn(2, 3, 96, 96)
    ndsm = torch.randn(2, 1, 96, 96)
    out = net(rgb, ndsm)
    for key, channels in [("sdf", 1), ("log_var", 1), ("road", 1), ("building", 1), ("landuse", 4)]:
        assert out[key].shape == (2, channels, 96, 96)

    loss = sum(v.mean() for v in out.values())
    loss.backward()
    # gradients reach both stems -- confirms the fusion doesn't accidentally
    # cut one stream off from the loss
    assert net.rgb_stem.net[0].conv.weight.grad is not None
    assert net.rgb_stem.net[0].conv.weight.grad.abs().sum() > 0
    assert net.ndsm_stem.net[0].conv.weight.grad is not None
    assert net.ndsm_stem.net[0].conv.weight.grad.abs().sum() > 0
