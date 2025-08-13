# models/frontend/frontend3dcvt.py
import torch
import torch.nn as nn
from typing import Optional


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class Conv3DStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 5, 5),
                              stride=(1, 2, 2), padding=(1, 2, 2), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x  # shape: (B, out_channels, T, H', W')


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        # reduce spatial resolution and project channels -> embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(1, 3, 3),
                              stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(embed_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.bn(x)
        x = self.act(x)
        return x  # shape: (B, embed_dim, T, H'', W'')


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # shape (max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0).to(x.device)
        return x


class Frontend3DCVT(nn.Module):
    """
    Lightweight 3D Conv -> patch embed -> Transformer frontend.

    Input: (B, C, T, H, W)
    Output: (B, T_out, embed_dim)
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        stem_out: int = 64,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.stem = Conv3DStem(in_channels, stem_out)
        self.embed = PatchEmbedding(stem_out, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns: (B, T_out, embed_dim)
        """
        # stem -> (B, stem_out, T, H', W')
        x = self.stem(x)

        # embed -> (B, embed_dim, T, H'', W'')
        x = self.embed(x)

        b, d, t, h, w = x.shape
        # collapse spatial dims by average pooling -> (B, d, t)
        x = x.view(b, d, t, -1).mean(-1)   # (B, d, t)
        x = x.permute(0, 2, 1).contiguous()  # (B, T, D)

        # positional encoding + transformer (batch_first=True)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, D)
        return x


# quick sanity check
if __name__ == "__main__":
    # dummy: (B, C, T, H, W)
    model = Frontend3DCVT(in_channels=1, embed_dim=256)
    dummy = torch.randn(2, 1, 75, 96, 96)
    out = model(dummy)
    print("Frontend out shape:", out.shape)  # expect (2, T_out, 256)
