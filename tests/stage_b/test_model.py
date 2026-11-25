import torch

from morloquio.stage_b.losses import total_loss
from morloquio.stage_b.model import UNet

LOSS = {"focal_gamma": 2.0, "dice_weight": 1.0, "cldice_weight": 0.5, "cldice_iters": 3}


def test_unet_shape_any_multiple_of_16():
    net = UNet(14, base=8, depth=4, attention=True)
    assert net(torch.zeros(2, 14, 64, 64)).shape == (2, 1, 64, 64)
    assert net(torch.zeros(1, 14, 96, 80)).shape == (1, 1, 96, 80)
    assert UNet(14, base=8, depth=3, attention=False)(torch.zeros(1, 14, 40, 48)).shape == (1, 1, 40, 48)


def test_overfits_single_patch():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    target[..., 30:33, :] = 1.0
    x[:, 0] += 3 * target[:, 0]  # il segnale c'è in un canale
    weight = torch.ones_like(target)
    net = UNet(4, base=8, depth=3)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss, _ = total_loss(net(x), target, weight, LOSS)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first
