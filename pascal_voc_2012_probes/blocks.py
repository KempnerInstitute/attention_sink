import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate_type = gate_type
        self.head_dim = dim // num_heads
        if gate_type == "elementwise":
            self.gate_proj = nn.Linear(dim, dim, bias=True)
        elif gate_type == "headwise":
            # One scalar gate per head (projects to num_heads values)
            self.gate_proj = nn.Linear(dim, num_heads, bias=True)

    def forward(self, x):
        bsz, n, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's efficient attention (FlashAttention/memory-efficient backend)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # 3. Concat
        # The .transpose and .reshape here IS the "Concat" operation in the diagram
        out = out.transpose(1, 2).reshape(bsz, n, dim)

        # 4. G1 Application (The "Most Effective" part)
        # Applied AFTER Concat, BEFORE Wo
        if self.gate_type == "elementwise":
            # Generate gate from Input x (standard residual gating practice)
            gate = torch.sigmoid(self.gate_proj(x)) 
            out = out * gate
        elif self.gate_type == "headwise":
            # One scalar gate per head, broadcast across head dim
            # gate_proj(x): (B, N, num_heads) -> sigmoid -> (B, N, H, 1) -> repeat -> (B, N, H, D) -> reshape (B, N, dim)
            gate = torch.sigmoid(self.gate_proj(x))  # (B, N, H)
            gate = gate.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  # (B, N, H, D)
            gate = gate.reshape(bsz, n, dim)  # (B, N, dim)
            out = out * gate

        # 5. Wo (Dense Layer)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            gate_type=gate_type,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        self.ls1 = LayerScale(dim, 1e-4)
        self.ls2 = LayerScale(dim, 1e-4)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
