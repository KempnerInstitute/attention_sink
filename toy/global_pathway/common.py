"""Two-layer broadcast toy with an explicit global pathway (Section 5, Figure 7B).

Base model / task / recipe are the ones of the broadcast toy in
``toy/notebooks/broadcast_causal_intervention.ipynb`` (App. A.1): d = 64, head dim 64,
L = 32 tokens, one head per layer, pre-norm LayerNorm, no biases in Wq/Wk/Wv/Wo, a learned
positional embedding added before the LayerNorm of layer 1 only,
MLP = LN -> Linear(d, 4d) -> GELU -> Linear(4d, d), a residual after each attention layer and
each MLP, AdamW(lr 5e-4, wd 1e-4), 30k steps, batch 1024, fresh i.i.d. N(0, I) tokens every
step, alpha = 1.

On top of that, switchable per layer (both layers):
  * ``pathway`` - the global pathway of Eq. 4:
        pi_j    = softmax_j(z_j . w_P)                 (over tokens)
        g       = sum_j pi_j W_VG z_j                  (global state, R^dg)
        gamma_i = sigmoid(z_i . w_gamma + b_gamma)     (learned recipient gate; = 1 with gp_gate='fixed')
        o_gp_i  = gamma_i W_OG g                       (rank one over tokens)
    computed from the same pre-normalised tokens z as attention; block update
    x <- x + o_attn + o_gp, then x <- x + MLP(x).
  * ``penalty`` - the scale-invariant common-mode penalty of Eq. 5 on the attention update
    o_attn = W_O A V of BOTH layers (see ``penalty_frac``).

Randomness contract (matched batches across arms):
  * ``torch.manual_seed(seed)``                -> parameter init (base parameters are created
                                                  before the pathway, so they are identical
                                                  across arms of one seed)
  * ``data_gen`` seeded ``1000 + seed``        -> training data stream (identical across arms)
  * ``task_gen`` seeded ``2000 + seed``        -> u, R (identical across arms)
  * ``eval_gen`` seeded ``5000 + seed``        -> evaluation + donor batches (evaluate.py)
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

DMODEL = 64
HEAD_DIM = 64
SEQ_LEN = 32
ALPHA = 1.0
EPS_FRAC = 1e-12          # stabiliser of the Eq.-5 ratio

# Released arms: name -> (pathway, gp_gate, penalty). The names are the run-directory tags.
# "pl12_frac" = penalty on layers 1 and 2, scale-invariant ratio form (Eq. 5).
ARMS = {
    "base":                (False, "learned", False),   # attention only
    "gp":                  (True,  "learned", False),   # + global pathway (learned recipient gate)
    "gp_pen_pl12_frac":    (True,  "learned", True),    # + global pathway + common-mode penalty
    "gp_g1":               (True,  "fixed",   False),   # optional: uniform recipients, gamma_i = 1
    "gp_pen_g1_pl12_frac": (True,  "fixed",   True),    # optional: gamma_i = 1 + penalty
}
PAPER_ARMS = ["base", "gp", "gp_pen_pl12_frac"]
PEN_LAYERS = (1, 2)


def fp32_exact() -> None:
    """fp32 with TF32 off, so runs are numerically reproducible on a given GPU model."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class GlobalPathway(nn.Module):
    """Eq. 4: softmax pooling to one global state g, returned through a recipient gate."""

    def __init__(self, dim: int, state_dim: int = 64, gate: str = "learned"):
        super().__init__()
        if gate not in ("learned", "fixed"):
            raise ValueError(f"unknown recipient gate {gate!r}")
        self.gate_mode = gate
        self.w_P = nn.Linear(dim, 1, bias=False)
        self.W_VG = nn.Linear(dim, state_dim, bias=False)
        self.W_OG = nn.Linear(state_dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1, bias=True) if gate == "learned" else None
        mods = (self.w_P, self.W_VG, self.W_OG) + ((self.gate,) if self.gate is not None else ())
        for m in mods:
            nn.init.trunc_normal_(m.weight, std=0.02)
        if self.gate is not None:
            nn.init.zeros_(self.gate.bias)      # gamma = 0.5 at init

    def forward(self, z, g_override=None):
        pi = torch.softmax(self.w_P(z).squeeze(-1), dim=1)              # (B, n)
        g = torch.einsum("bn,bne->be", pi, self.W_VG(z))                  # (B, dg)
        if isinstance(g_override, str):
            if g_override != "zero":
                raise ValueError(g_override)
            g = torch.zeros_like(g)
        elif g_override is not None:
            g = g_override
        if self.gate is None:
            gamma = torch.ones(z.shape[0], z.shape[1], 1, dtype=z.dtype, device=z.device)
        else:
            gamma = torch.sigmoid(self.gate(z))                           # (B, n, 1)
        o_gp = gamma * self.W_OG(g).unsqueeze(1)                          # rank one over tokens
        return pi, g, gamma, o_gp


class Layer(nn.Module):
    """One pre-norm attention layer (+ optional global pathway, attached by TwoLayerToy)."""

    def __init__(self, d=DMODEL, h=HEAD_DIM, length=SEQ_LEN, add_pos=True):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, h, bias=False)
        self.Wk = nn.Linear(d, h, bias=False)
        self.Wv = nn.Linear(d, h, bias=False)
        self.Wo = nn.Linear(h, d, bias=False)
        self.pos = nn.Parameter(torch.randn(length, d) * 0.02) if add_pos else None
        self.gp: Optional[GlobalPathway] = None

    def forward(self, x, A_override=None, V_override=None, g_override=None,
                zero_attn=False, zero_gp=False):
        residual = x
        z = self.norm(x + self.pos.unsqueeze(0) if self.pos is not None else x)
        Q, K, V = self.Wq(z), self.Wk(z), self.Wv(z)
        scores = Q @ K.transpose(1, 2) / math.sqrt(Q.shape[-1])
        A = torch.softmax(scores, dim=-1)
        A_eff = A if A_override is None else A_override
        V_eff = V if V_override is None else V_override
        o_attn = self.Wo(A_eff @ V_eff)                       # W_O A V: the penalised tensor
        if zero_attn:
            o_attn = torch.zeros_like(o_attn)
        pi = g = gamma = None
        o_gp = torch.zeros_like(o_attn)
        if self.gp is not None:
            pi, g, gamma, o_gp = self.gp(z, g_override=g_override)
            if zero_gp:
                o_gp = torch.zeros_like(o_gp)
        out = residual + o_attn + o_gp
        cache = dict(Q=Q, K=K, V=V, A=A, z=z, residual=residual, pi=pi, g=g, gamma=gamma,
                     o_attn=o_attn, o_gp=o_gp, update=o_attn + o_gp)
        return out, A, cache


def _mlp(d: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class TwoLayerToy(nn.Module):
    def __init__(self, d=DMODEL, h=HEAD_DIM, L=SEQ_LEN, pathway=False, gp_gate="learned", dg=64):
        super().__init__()
        self.use_pathway = bool(pathway)
        # base parameters first (identical across arms for a given seed) ...
        self.l1 = Layer(d, h, L, add_pos=True)
        self.mlp1 = _mlp(d)
        self.l2 = Layer(d, h, L, add_pos=False)
        self.mlp2 = _mlp(d)
        # ... optional modules afterwards
        if self.use_pathway:
            for lay in (self.l1, self.l2):
                lay.gp = GlobalPathway(d, state_dim=dg, gate=gp_gate)

    def forward(self, x, ret=True, ov: Optional[Dict[int, dict]] = None):
        """``ov = {layer: overrides}`` applies an intervention to one layer (see Layer.forward)."""
        ov = ov or {}
        o1, A1, c1 = self.l1(x, **ov.get(1, {}))
        h1 = o1 + self.mlp1(o1)
        o2, A2, c2 = self.l2(h1, **ov.get(2, {}))
        out = o2 + self.mlp2(o2)
        if not ret:
            return out
        return out, (A1, A2), {1: c1, 2: c2, "l1_out": o1, "l2_out": o2}

    def layer(self, l: int) -> Layer:
        return self.l1 if l == 1 else self.l2


def build_model(cfg: dict) -> TwoLayerToy:
    return TwoLayerToy(d=cfg["d"], h=cfg["d"], L=cfg["L"], pathway=cfg["pathway"],
                       gp_gate=cfg["gp_gate"], dg=cfg["dg"])


# --------------------------------------------------------------------------------------
# task, loss, penalty
# --------------------------------------------------------------------------------------
def make_task(x, u, R, alpha=ALPHA):
    """Eq. 3: j* = argmax_j <x_j, u>, y_i = x_i + alpha x_{j*} R (every token in the loss)."""
    b = torch.arange(x.shape[0], device=x.device)
    j_star = (x @ u).argmax(dim=-1)
    payload = x[b, j_star] @ R                                   # (B, d)
    target = x + alpha * payload.unsqueeze(1)
    return dict(target=target, j_star=j_star, payload=payload)


def task_mse(pred, target):
    return ((pred - target) ** 2).sum() / pred.numel()


def cm_numerator(o):
    """E(obar) with obar_b = mean over the n tokens of example b; = ||P_com O||_F^2 / (B n d)."""
    return o.mean(dim=1).pow(2).mean()


def penalty_frac(o):
    """Eq. 5 ratio for one layer: E(obar) / (E(O) + eps), gradient through numerator and
    denominator (scale invariant). NOTE: train.py adds lambda_B * (pen_1 + pen_2), i.e. the SUM
    over the penalised layers, not the mean of Eq. 5 (lambda_B = 1 here = 2 under Eq. 5)."""
    return cm_numerator(o) / (o.pow(2).mean() + EPS_FRAC)


def common_mode_fraction(o):
    with torch.no_grad():
        return (cm_numerator(o) / (o.pow(2).mean() + EPS_FRAC)).item()


# --------------------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------------------
def make_generators(seed: int, device, d=DMODEL, task_gen="device"):
    """data_gen = 1000 + seed (matched batches); task tensors u, R from 2000 + seed.

    task_gen='device' (default) draws u and R with a generator on ``device`` in fp32 and takes
    the QR there. This is how the paper runs were produced; on CUDA the fp32 QR goes through
    cuSOLVER, so R can differ by ~1e-6 between GPU models (effect on task MSE ~1e-8).
    task_gen='cpu64' draws u, R on the CPU in float64 (bit-identical on every machine), but it
    is a DIFFERENT random task than the paper's for the same seed.
    """
    data_gen = torch.Generator(device=device)
    data_gen.manual_seed(1000 + int(seed))
    if task_gen == "device":
        tg = torch.Generator(device=device)
        tg.manual_seed(2000 + int(seed))
        u = torch.randn(d, device=device, generator=tg)
        u = u / u.norm().clamp_min(1e-12)
        R, _ = torch.linalg.qr(torch.randn(d, d, device=device, generator=tg))
        return data_gen, u, R.contiguous()
    if task_gen == "cpu64":
        tg = torch.Generator()
        tg.manual_seed(2000 + int(seed))
        u = torch.randn(d, dtype=torch.float64, generator=tg)
        u = u / u.norm().clamp_min(1e-12)
        R, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=tg))
        dt = torch.get_default_dtype()
        return data_gen, u.to(device=device, dtype=dt), R.contiguous().to(device=device, dtype=dt)
    raise ValueError(task_gen)


def task_hash(u, R) -> str:
    """Short hash of the task tensors (compare across the arms of one seed)."""
    hh = hashlib.sha1()
    for t in (u, R):
        hh.update(t.detach().cpu().contiguous().numpy().tobytes())
    return hh.hexdigest()[:12]


def sample_batch(gen, batch, device, L=SEQ_LEN, d=DMODEL, dtype=torch.float32):
    return torch.randn(batch, L, d, device=device, generator=gen, dtype=dtype)


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def recipient_mask(j_star, n):
    """Recipients = every token except the source j*."""
    return torch.arange(n, device=j_star.device).unsqueeze(0) != j_star.unsqueeze(1)


def _mean(t):
    return float(t.mean().item()) if t.numel() else float("nan")


def _gram_eigs(U):
    """Eigenvalues of U U^T per example (float64). A converged broadcast head is almost exactly
    rank one, which can make a batched eigvalsh fail to converge; fall back to svdvals and then
    to a per-example loop (NaN rows for examples that never work)."""
    U = U.double()
    B, n = U.shape[0], U.shape[1]
    G = U @ U.transpose(1, 2)
    G = 0.5 * (G + G.transpose(1, 2))
    scale = G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    jit = (1e-10 * scale).view(B, 1, 1) * torch.eye(n, dtype=G.dtype, device=G.device)
    for fn in (lambda: torch.linalg.eigvalsh(G + jit), lambda: torch.linalg.svdvals(U) ** 2):
        try:
            ev = fn()
            if torch.isfinite(ev).all():
                return ev.clamp_min(0.0)
        except Exception:
            pass
    rows = []
    for b in range(B):
        ev = None
        for fn in (lambda: torch.linalg.eigvalsh(G[b] + jit[b]),
                   lambda: torch.linalg.svdvals(U[b]) ** 2):
            try:
                cand = fn()
                if torch.isfinite(cand).all():
                    ev = cand.clamp_min(0.0)
                    break
            except Exception:
                pass
        rows.append(ev if ev is not None
                    else torch.full((n,), float("nan"), dtype=G.dtype, device=G.device))
    return torch.stack(rows)


def stable_rank(U, max_examples=256):
    """sum(eig)/max(eig) of U U^T per example, mean over at most `max_examples` examples
    (1.0 = exactly rank one). Diagnostic only: returns NaN instead of raising."""
    try:
        ev = _gram_eigs(U[:max_examples])
        tot, mx = ev.sum(-1), ev.max(-1).values
        ok = torch.isfinite(tot) & torch.isfinite(mx) & (mx > 1e-20)
        if not bool(ok.any()):
            return float("nan")
        return float((tot[ok] / mx[ok]).mean().item())
    except Exception as e:
        print(f"[toy] warn: stable rank unavailable ({type(e).__name__}: {e})", flush=True)
        return float("nan")


@torch.no_grad()
def layer_diagnostics(model, cache, l, j_star):
    """Scalar diagnostics of layer l, keys prefixed ``l{l}_``."""
    c = cache[l]
    lay = model.layer(l)
    A, V = c["A"], c["V"]
    B, n, _ = A.shape
    b = torch.arange(B, device=A.device)
    colmass = A.mean(dim=1)                                  # (B, n) mean inflow per key
    a_js = A[b, :, j_star]                                   # (B, n) attention to j*
    nonsink = recipient_mask(j_star, n)

    def _vnr(T):                                             # ||T_{j*}|| / mean_{j != j*} ||T_j||
        nrm = T.norm(dim=-1)
        oth = (nrm * nonsink).sum(1) / nonsink.sum(1).clamp_min(1)
        return float((nrm[b, j_star] / oth.clamp_min(1e-12)).mean().item())

    o_attn, o_gp = c["o_attn"], c["o_gp"]
    rms_attn = float(o_attn.pow(2).mean().sqrt().item())
    rms_gp = float(o_gp.pow(2).mean().sqrt().item())
    p = f"l{l}_"
    routed = (a_js >= 0.5).float()
    d = {
        p + "max_inflow": _mean(colmass.max(dim=1).values),       # Eq. 2 sink strength
        p + "sink_match": _mean((colmass.argmax(dim=-1) == j_star).float()),
        p + "attn_to_jstar": _mean(a_js),
        p + "routed_frac": _mean(routed),
        p + "eps_sink_frac": _mean((routed.mean(dim=1) >= 0.9).float()),
        p + "vnr_out": _vnr(lay.Wo(V)),
        p + "vnr_head": _vnr(V),
        p + "stable_rank": stable_rank(o_attn),
        p + "cm_frac_attn": common_mode_fraction(o_attn),
        p + "cm_frac_gp": common_mode_fraction(o_gp) if model.use_pathway else float("nan"),
        p + "rms_attn": rms_attn,
        p + "rms_gp": rms_gp,
        p + "gp_usage": rms_gp / max(rms_attn, 1e-12) if model.use_pathway else float("nan"),
        p + "pen_frac": float(penalty_frac(o_attn).item()),
    }
    if model.use_pathway:
        pi, gamma = c["pi"], c["gamma"].squeeze(-1)
        d[p + "pi_max"] = _mean(pi.max(dim=1).values)
        d[p + "pi_jstar"] = _mean(pi[b, j_star])
        d[p + "gamma_mean"] = _mean(gamma)
    else:
        for k in ("pi_max", "pi_jstar", "gamma_mean"):
            d[p + k] = float("nan")
    return d
