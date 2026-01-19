import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, D, H', W')
        x = x.flatten(2)              # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        return x

class VisionLSTMBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Linear(2 * hidden_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, D)

        # --- Recurrent global modeling ---
        residual = x
        x = self.norm1(x)

        x_lstm, _ = self.lstm(x)              # (B, N, 2H)
        x = self.proj(x_lstm)                 # (B, N, D)
        x = x + residual

        # --- Feed-forward ---
        x = x + self.mlp(self.norm2(x))
        return x

class VisionLSTM(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=256,
        hidden_dim=256,
        depth=6,
        num_labels=4
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_chans, embed_dim
        )

        self.blocks = nn.Sequential(
            *[VisionLSTMBlock(embed_dim, hidden_dim) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        # x: (B, C, H, W)

        x = self.patch_embed(x)        # (B, N, D)
        x = self.blocks(x)             # (B, N, D)
        x = self.norm(x)

        # Global pooling (token-agnostic)
        x = x.mean(dim=1)              # (B, D)
        return self.head(x)
