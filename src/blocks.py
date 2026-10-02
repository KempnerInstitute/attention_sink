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
        # Set by the trainer for --cm_pre_gate: keep W_O concat(A V) + b_O computed without the gate.
        self.record_pre_gate = False
        self.pre_gate_out: Optional[torch.Tensor] = None
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
        self.pre_gate_out = self.proj(out) if self.record_pre_gate else None

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


class GlobalPathway(nn.Module):
    """Explicit global pathway (paper Eq. 4), computed from the block's pre-normalised tokens x (B, N, D):

        pi_j    = softmax_j(w_P^T x_j)              pooling weights over tokens
        g       = sum_j pi_j W_VG x_j               global state, R^{state_dim}, not tied to any position
        o_i     = gamma_i W_OG g                    update for token i, rank one across tokens

    Recipient gate gamma_i:
      'learned'  sigmoid(w_gamma^T x_i + b_gamma)                          (Eq. 4)
      'hier'     sigmoid(a_i) * softmax([r_i0, r_ibc])_bc, [a_i, r_i0, r_ibc] = Linear(D, 3)(x_i):
                 a parent gate times a two-way choice between no update and the broadcast.
    """
    GATE_LOGITS = {"learned": 1, "hier": 3}

    def __init__(self, dim: int, state_dim: int = 64, gate: str = "learned"):
        super().__init__()
        if gate not in self.GATE_LOGITS:
            raise ValueError(f"unknown global-pathway gate {gate!r} (expected one of {list(self.GATE_LOGITS)})")
        self.gate_mode = gate
        self.w_P = nn.Linear(dim, 1, bias=False)
        self.W_VG = nn.Linear(dim, state_dim, bias=False)
        self.W_OG = nn.Linear(state_dim, dim, bias=False)
        self.gate = nn.Linear(dim, self.GATE_LOGITS[gate], bias=True)
        for m in (self.w_P, self.W_VG, self.W_OG, self.gate):
            nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)  # learned: gamma = 0.5 at init; hier: 0.25

    @staticmethod
    def _fp32(t: torch.Tensor) -> torch.Tensor:
        return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t

    def pool_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> pi (B, N); softmax in at least fp32 under autocast."""
        return torch.softmax(self._fp32(self.w_P(x).squeeze(-1)), dim=1)

    def global_state(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> g (B, state_dim)."""
        pi = self.pool_weights(x).to(x.dtype)
        return torch.einsum("bn,bne->be", pi, self.W_VG(x))

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> gamma (B, N, 1)."""
        logits = self.gate(x)
        if self.gate_mode == "learned":
            return torch.sigmoid(logits)
        q = torch.sigmoid(self._fp32(logits[..., :1]))
        p_bc = torch.softmax(self._fp32(logits[..., 1:]), dim=-1)[..., 1:2]
        return (q * p_bc).to(x.dtype)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`g` overrides the global state (e.g. zero or another image's state for interventions)."""
        if g is None:
            g = self.global_state(x)
        return self.gates(x) * self.W_OG(g).unsqueeze(1)


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
        global_pathway: Optional[dict] = None,
        ls_init: float = 1e-4,
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

        self.ls1 = LayerScale(dim, float(ls_init))
        self.ls2 = LayerScale(dim, float(ls_init))

        # Optional global pathway on the same pre-normalised tokens as attention, added to the attention update.
        # global_pathway = {"arch": "paper", "state_dim": d_g, "gate": "learned" | "hier", "ls_init": "none" | float};
        # ls_init "none" = no LayerScale on the pathway branch (ls3 is the identity).
        if global_pathway:
            if global_pathway.get("arch", "paper") != "paper":
                raise ValueError(f"unknown global pathway arch {global_pathway['arch']!r}")
            self.gp = GlobalPathway(dim, state_dim=global_pathway.get("state_dim", 64), gate=global_pathway.get("gate", "learned"))
            gp_ls = global_pathway.get("ls_init", 1e-4)
            self.ls3 = nn.Identity() if str(gp_ls) == "none" else LayerScale(dim, float(gp_ls))
        else:
            self.gp = None
            self.ls3 = None

        # Common-mode penalty (paper Eq. 5), switched on by the trainer. With `record_common_mode` the forward stores
        # common_mode_frac = E(mean_i o_i) / (E(o) + eps), E(Z) = mean squared entry, for the attention update
        # o = ls1(attn(x)) (includes b_O and LayerScale); with `common_mode_pre_gate` o = ls1(W_O concat(A V) + b_O),
        # i.e. the same update without the attention gate.
        self.record_common_mode = False
        self.common_mode_pre_gate = False
        self.common_mode_frac: Optional[torch.Tensor] = None

    def forward(self, x):
        h = self.norm1(x)
        update = self.ls1(self.attn(h))
        if self.record_common_mode:
            o = (self.ls1(self.attn.pre_gate_out) if self.common_mode_pre_gate else update).float()
            self.common_mode_frac = o.mean(dim=1).pow(2).mean() / (o.pow(2).mean() + 1e-12)
        if self.gp is not None:
            update = update + self.ls3(self.gp(h))
        x = x + self.drop_path(update)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
