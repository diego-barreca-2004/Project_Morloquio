"""U-Net con blocchi residui e attention gate sulle skip connection (Attention Residual U-Net).

Architettura usata per le strade forestali da LiDAR in ISPRS Archives XLIX-B1-2026 (11 canali, 1 m); qui il numero di
canali è libero. GroupNorm invece di BatchNorm: stabile anche con batch piccoli. L'uscita è un logit per pixel.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm1 = _norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = _norm(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(y + self.skip(x))


class AttentionGate(nn.Module):
    """Pesa la skip ``x`` con una mappa 0–1 calcolata da ``x`` e dal segnale del decoder ``g`` (stessa risoluzione)."""

    def __init__(self, ch: int):
        super().__init__()
        inter = max(ch // 2, 1)
        self.wx = nn.Conv2d(ch, inter, 1)
        self.wg = nn.Conv2d(ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.psi(F.relu(self.wx(x) + self.wg(g))))


class UNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, depth: int = 4, attention: bool = True):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([ResBlock(in_ch, chs[0])] + [ResBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.gates = nn.ModuleList(
            [AttentionGate(chs[i]) if attention else nn.Identity() for i in reversed(range(depth))]
        )
        self.dec = nn.ModuleList([ResBlock(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, gate, dec in zip(self.up, self.gates, self.dec, strict=True):
            g = up(x)
            skip = skips.pop()
            skip = gate(skip, g) if isinstance(gate, AttentionGate) else skip
            x = dec(torch.cat([skip, g], dim=1))
        return self.head(x)
