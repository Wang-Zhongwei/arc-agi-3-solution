"""Tiny hybrid CNN-ViT for 64x64 ARC-AGI-3 frames (~0.85M params).

Input: 16-channel colour one-hot + 4 palette-invariant 'same colour as neighbour' channels (R, D, L, U).
Body: 2 conv blocks at 64x64 -> stride-2 convs to 16x16 -> 3 transformer blocks on 256 tokens
(128-dim, 4 heads, conv positional encoding so it stays translation-equivariant) -> U-Net upsampler with skips.
Heads: affinity logits (right/down at distance 1, 2, 4 = 6 ch), semantic bg/sprite/HUD (3 ch), clickable (1 ch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

OFFS = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 4), (4, 0)]


def features(fr):
    """fr: (B,64,64) long colour indices -> (B,20,64,64) float."""
    oh = F.one_hot(fr.clamp(0, 15), 16).permute(0, 3, 1, 2).float()
    same = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        sh = torch.roll(fr, shifts=(-dy, -dx), dims=(1, 2))
        s = (sh == fr).float()
        if dx == 1: s[:, :, -1] = 0
        if dx == -1: s[:, :, 0] = 0
        if dy == 1: s[:, -1, :] = 0
        if dy == -1: s[:, 0, :] = 0
        same.append(s)
    return torch.cat([oh, torch.stack(same, 1)], 1)


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())


class Block(nn.Module):
    def __init__(self, d=128, h=4):
        super().__init__()
        self.pe = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.n1 = nn.LayerNorm(d); self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        x = x + self.pe(x)
        t = x.flatten(2).transpose(1, 2)
        q = self.n1(t)
        t = t + self.att(q, q, q, need_weights=False)[0]
        t = t + self.mlp(self.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class TinyNet(nn.Module):
    def __init__(self, transformer=True):
        super().__init__()
        self.s1 = nn.Sequential(cbr(20, 32), cbr(32, 32))
        self.d1 = cbr(32, 64, 2)
        self.d2 = cbr(64, 128, 2)
        self.tf = nn.Sequential(*[Block() for _ in range(3)]) if transformer else nn.Sequential(cbr(128, 128), cbr(128, 128), cbr(128, 128))
        self.u1 = cbr(128 + 64, 64)
        self.u2 = cbr(64 + 32, 32)
        self.aff = nn.Conv2d(32, len(OFFS), 1); self.sem = nn.Conv2d(32, 3, 1); self.clk = nn.Conv2d(32, 1, 1)

    def forward(self, fr):
        x0 = self.s1(features(fr))
        x1 = self.d1(x0)
        x2 = self.tf(self.d2(x1))
        y = self.u1(torch.cat([F.interpolate(x2, scale_factor=2, mode="nearest"), x1], 1))
        y = self.u2(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), x0], 1))
        return self.aff(y), self.sem(y), self.clk(y)
