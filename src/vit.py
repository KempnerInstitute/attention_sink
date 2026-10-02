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

    Returns the CLS token embedding.
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
        global_pathway: Optional[dict] = None,
        ls_init: float = 1e-4,
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
                global_pathway=global_pathway,
                ls_init=ls_init,
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
                if '.gp.' in name:
                    continue  # GlobalPathway initialises its own layers
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


class ViTEncoder(nn.Module):
    """ViT backbone + MLP projector for LeJEPA pretraining.

    Takes multi-view input (N, V, C, H, W) and returns:
      - emb: backbone CLS embeddings (N*V, embed_dim)
      - proj: projections reshaped to (V, N, proj_dim)
    """

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        patch_size: int = 16,
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
        global_pathway: Optional[dict] = None,
        ls_init: float = 1e-4,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_register_tokens=num_register_tokens,
            gate_type=gate_type,
            global_pathway=global_pathway,
            ls_init=ls_init,
        )
        self.embed_dim = embed_dim

        # MLP projector: embed_dim -> 2048 -> 2048 -> proj_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Multi-view images of shape (N, V, C, H, W).

        Returns:
            emb: Backbone CLS embeddings of shape (N*V, embed_dim).
            proj: Projections of shape (V, N, proj_dim).
        """
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))        # (N*V, embed_dim)
        proj = self.proj(emb).reshape(N, V, -1)      # (N, V, proj_dim)
        proj = proj.transpose(0, 1)                   # (V, N, proj_dim)
        return emb, proj
