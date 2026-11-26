import torch

from morloquio.stage_b.losses import masked_focal_bce, masked_soft_dice, soft_cldice, soft_skeleton, total_loss

LOSS = {"focal_gamma": 2.0, "dice_weight": 1.0, "cldice_weight": 0.5, "cldice_iters": 3}


def _line(width=1, gap=False, size=64):
    t = torch.zeros(1, 1, size, size)
    c = size // 2
    t[..., c - width // 2 : c + width // 2 + 1, 8:-8] = 1.0
    if gap:
        t[..., :, 28:36] = 0.0
    return t


def test_zero_weight_pixels_do_not_affect_loss():
    torch.manual_seed(0)
    target = _line()
    weight = torch.ones_like(target)
    weight[..., 32:] = 0.0
    logits = torch.randn_like(target)
    other = logits.clone()
    other[..., 48:] = torch.randn_like(other[..., 48:]) * 5
    a, _ = total_loss(logits, target, weight, LOSS)
    b, _ = total_loss(other, target, weight, LOSS)
    assert torch.allclose(a, b, atol=1e-6)


def test_focal_and_dice_reward_correct_prediction():
    target = _line(width=3)
    weight = torch.ones_like(target)
    good = target * 12 - 6
    bad = -good
    assert masked_focal_bce(good, target, weight, 2.0) < masked_focal_bce(bad, target, weight, 2.0)
    assert masked_soft_dice(torch.sigmoid(good), target, weight) < 0.05


def test_soft_skeleton_of_thick_line_is_thin():
    skel = soft_skeleton(_line(width=5), iters=5)
    rows = (skel[0, 0, :, 32] > 0.5).nonzero().flatten().tolist()
    assert len(rows) <= 2 and 31 <= rows[0] <= 33


def test_cldice_penalizes_broken_line_more_than_thick_line():
    target = _line(width=1)
    weight = torch.ones_like(target)
    broken = soft_cldice(_line(width=1, gap=True), target, weight, iters=3)
    thick = soft_cldice(_line(width=3), target, weight, iters=3)
    assert broken > thick + 0.03
