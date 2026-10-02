"""
LeJEPA VisionTransformer for probe evaluation.
This is the backbone-only VisionTransformer (no ViTEncoder/projector).
"""
from typing import Optional

import torch
from torch import nn

from blocks import Block


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ViT backbone for LeJEPA — no classification head.

    Returns CLS token embedding and patch-average embedding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_register_tokens: int = 0,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_register_tokens = num_register_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None

        token_count = 1 + num_register_tokens + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                gate_type=gate_type,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'gate_proj' in name:
                    nn.init.normal_(m.weight, std=1e-6)
                    nn.init.constant_(m.bias, 2.0)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass returning CLS embedding.

        Args:
            x: Images of shape (B, C, H, W).

        Returns:
            cls_embed: CLS token embedding of shape (B, embed_dim).
        """
        x = self.patch_embed(x)
        bsz = x.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        if self.register_tokens is not None:
            reg_tokens = self.register_tokens.expand(bsz, -1, -1)
            x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token = x[:, 0]
        return cls_token
