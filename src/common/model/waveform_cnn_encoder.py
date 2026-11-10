"""
ABR Waveform CNN Encoder
========================
Robust 1D CNN encoder for raw ABR waveforms.

- Uses GroupNorm (stable for small / variable batch sizes)
- Uses dilated convolutions to capture morphology at different temporal scales
- pool_k=1 (default): global average pooling
- pool_k=K (e.g. 8): preserves K spatial bins → retains Wave-V latency info
- Maps [T, 1, L] → [T, embed_dim]
"""

import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, dilation=1, groups=8, dropout=0.0):
        super().__init__()
        padding = ((k - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # Choose groups safely (must divide out_ch)
        g = groups
        while out_ch % g != 0 and g > 1:
            g //= 2
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ABRWaveformCNNEncoder(nn.Module):
    """
    Maps [T, 1, L] → [T, embed_dim]

    Args:
        waveform_length: number of raw waveform points (default 256)
        embed_dim:       output feature size
        dropout:         applied after each conv
        gn_groups:       target GroupNorm groups (auto-reduced if needed)
        pool_k:          spatial bins after adaptive avg pool.
                         1 = global pool.
                         8 = keeps 8 bins, preserving Wave-V latency gradient.
    """
    def __init__(self, waveform_length=256, embed_dim=48, dropout=0.15,
                 gn_groups=8, pool_k=1):
        super().__init__()
        self.pool_k = pool_k

        # Stem: wide kernel to capture early waveform morphology
        self.block1 = nn.Sequential(
            ConvGNAct(1,  32, k=7, dilation=1, groups=gn_groups, dropout=dropout),
            ConvGNAct(32, 32, k=5, dilation=1, groups=gn_groups, dropout=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 256 → 128
        )

        # Mid: dilated convolutions for multi-scale peak / latency features
        self.block2 = nn.Sequential(
            ConvGNAct(32, 64, k=5, dilation=2, groups=gn_groups, dropout=dropout),
            ConvGNAct(64, 64, k=3, dilation=2, groups=gn_groups, dropout=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 128 → 64
        )

        # Late: broader context with larger dilation
        self.block3 = nn.Sequential(
            ConvGNAct(64, 96, k=3, dilation=4, groups=gn_groups, dropout=dropout),
            ConvGNAct(96, 96, k=3, dilation=4, groups=gn_groups, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(pool_k)
        self.fc = nn.Linear(96 * pool_k, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def normalize_waveform(self, x):
        """Mean-center only — preserves amplitude scale (clinically important)."""
        return x - x.mean(dim=-1, keepdim=True)

    def forward(self, x):
        # x: [T, 1, L]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.pool_k == 1:
            x = self.pool(x).squeeze(-1)  # [T, 96]
        else:
            x = self.pool(x)              # [T, 96, pool_k]
            x = x.flatten(1)             # [T, 96 * pool_k]
        x = self.fc(x)                   # [T, embed_dim]
        x = self.out_norm(x)
        return x

