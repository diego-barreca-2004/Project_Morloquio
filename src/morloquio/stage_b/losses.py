"""Loss pesate per pixel: focal BCE, soft Dice e soft clDice (Shit et al., arXiv 2003.07311).

Il peso viene da :func:`morloquio.stage_b.labels.rasterize_labels`: 1 sui positivi OSM, 0 dove la posizione è
incerta o la tile è di test, basso sui pixel non etichettati. clDice confronta gli scheletri: penalizza le linee
interrotte, che per la completezza di una rete stradale contano più dei pixel di bordo.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

_EPS = 1.0


def masked_focal_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt) ** gamma * bce * weight).sum() / weight.sum().clamp_min(1e-6)


def masked_soft_dice(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    inter = (prob * target * weight).sum()
    denom = ((prob + target) * weight).sum()
    return 1 - (2 * inter + _EPS) / (denom + _EPS)


def _erode(img: torch.Tensor) -> torch.Tensor:
    a = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    b = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(a, b)


def _dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, 1, 1)


def _open(img: torch.Tensor) -> torch.Tensor:
    return _dilate(_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int) -> torch.Tensor:
    """Scheletro morbido e differenziabile (Shit et al., algoritmo 1)."""
    skel = F.relu(img - _open(img))
    for _ in range(iters):
        img = _erode(img)
        delta = F.relu(img - _open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, iters: int) -> torch.Tensor:
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = ((sp * target * weight).sum() + _EPS) / ((sp * weight).sum() + _EPS)
    tsens = ((st * prob * weight).sum() + _EPS) / ((st * weight).sum() + _EPS)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def total_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, params: dict
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = logits.float()
    prob = torch.sigmoid(logits)
    parts = {"focal": masked_focal_bce(logits, target, weight, float(params["focal_gamma"]))}
    if params.get("dice_weight", 0):
        parts["dice"] = float(params["dice_weight"]) * masked_soft_dice(prob, target, weight)
    if params.get("cldice_weight", 0):
        parts["cldice"] = float(params["cldice_weight"]) * soft_cldice(
            prob, target, weight, int(params["cldice_iters"])
        )
    loss = sum(parts.values())
    return loss, {k: float(v.detach()) for k, v in parts.items()}
