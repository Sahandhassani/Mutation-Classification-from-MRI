import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)           # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)
        return x


class SwinBlock2D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SwinTransformer2D(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=1,
        embed_dim=96,
        depth=6,
        num_heads=4,
        num_classes=4
    ):
        super().__init__()

        self.patch_embed = PatchEmbed2D(in_chans, embed_dim)
        self.blocks = nn.Sequential(*[
            SwinBlock2D(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)     # (B, N, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
