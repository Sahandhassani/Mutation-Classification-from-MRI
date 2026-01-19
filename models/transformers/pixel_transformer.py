import torch
import torch.nn as nn


class PixelTransformer2D(nn.Module):
    """
    Transformer operating on individual pixels (1x1 patches).
    Faithful to: 'An Image Is Worth More Than 16×16 Patches' (ICLR 2025)
    """

    def __init__(
        self,
        img_size=224,
        in_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=4,
        dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.num_tokens = img_size * img_size

        # ---- Pixel projection ----
        self.pixel_embed = nn.Linear(in_channels, embed_dim)

        # ---- Position embedding (learned, NO locality prior) ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # ---- Head ----
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, "Image size mismatch"

        # (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Pixel embedding
        x = self.pixel_embed(x)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Global average pooling over pixels
        x = x.mean(dim=1)

        x = self.norm(x)
        out = self.head(x)

        return out
