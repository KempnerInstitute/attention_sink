#!/usr/bin/env python
"""CPU / float64 correctness tests for the controlled broadcast task (run before training).

    python tests.py            # prints PASS/FAIL per check, exit code 1 on any failure
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import evaluate as E
from common import DMODEL, SEQ_LEN, GlobalPathway, TwoLayerToy

torch.set_default_dtype(torch.float64)
DEV = torch.device("cpu")
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# ======================================================================================
# 1. the baseline is the broadcast notebook's model (toy/notebooks/broadcast_causal_intervention.ipynb)
# ======================================================================================
class NBOneHead(nn.Module):
    def __init__(self, d=DMODEL, h=DMODEL, length=SEQ_LEN, add_pos=True):
        super().__init__()
        self.add_pos = add_pos
        self.norm = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, h, bias=False)
        self.Wk = nn.Linear(d, h, bias=False)
        self.Wv = nn.Linear(d, h, bias=False)
        self.Wo = nn.Linear(h, d, bias=False)
        self.pos = nn.Parameter(torch.randn(length, d) * 0.02)

    def forward(self, x):
        residual = x
        z = x + self.pos.unsqueeze(0) if self.add_pos else x
        z = self.norm(z)
        Q, K, V = self.Wq(z), self.Wk(z), self.Wv(z)
        A = torch.softmax(Q @ K.transpose(1, 2) / math.sqrt(Q.shape[-1]), dim=-1)
        return residual + self.Wo(A @ V)


class NBTwoLayer(nn.Module):
    def __init__(self, d=DMODEL, h=DMODEL, length=SEQ_LEN):
        super().__init__()
        self.l1 = NBOneHead(d, h, length, add_pos=True)
        self.mlp1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.l2 = NBOneHead(d, h, length, add_pos=False)
        self.mlp2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        l1 = self.l1(x)
        h1 = l1 + self.mlp1(l1)
        l2 = self.l2(h1)
        return l2 + self.mlp2(l2)


def nb_make_task(x, u, R, alpha=1.0):
    j_star = (x @ u).argmax(dim=-1)
    payload = x[torch.arange(x.shape[0]), j_star] @ R
    return x + alpha * payload.unsqueeze(1), j_star


def test_notebook_equivalence():
    torch.manual_seed(0)
    mine = TwoLayerToy(pathway=False)
    nb = NBTwoLayer()
    for a, b in ((mine.l1, nb.l1), (mine.l2, nb.l2)):
        b.norm.load_state_dict(a.norm.state_dict())
        for w in ("Wq", "Wk", "Wv", "Wo"):
            getattr(b, w).load_state_dict(getattr(a, w).state_dict())
        if a.pos is not None:
            b.pos.data.copy_(a.pos.data)
    nb.mlp1.load_state_dict(mine.mlp1.state_dict())
    nb.mlp2.load_state_dict(mine.mlp2.state_dict())
    x = torch.randn(7, SEQ_LEN, DMODEL)
    with torch.no_grad():
        err = (mine(x, ret=False) - nb(x)).abs().max().item()
    check("attention-only model == notebook TwoLayer", err < 1e-12, f"max|diff| = {err:.3e}")
    _, u, R = C.make_generators(0, DEV)
    t = C.make_task(x, u, R)
    tgt, js = nb_make_task(x, u, R)
    d = (t["target"] - tgt).abs().max().item()
    check("make_task == notebook make_task", torch.equal(t["j_star"], js) and d < 1e-14,
          f"max|diff| = {d:.3e}")


# ======================================================================================
# 2. global pathway (Eq. 4)
# ======================================================================================
def test_pathway():
    torch.manual_seed(1)
    gp = GlobalPathway(DMODEL, 64, "learned")
    z = torch.randn(5, SEQ_LEN, DMODEL)
    _, g, gamma, o = gp(z)
    s = torch.linalg.svdvals(o)
    ratio = (s[:, 1] / s[:, 0].clamp_min(1e-300)).max().item()
    check("pathway update is rank one per example", ratio < 1e-12, f"max s2/s1 = {ratio:.3e}")
    perm = torch.randperm(SEQ_LEN)
    _, g_p, _, o_p = gp(z[:, perm])
    dg, do = (g_p - g).abs().max().item(), (o_p - o[:, perm]).abs().max().item()
    check("g permutation-invariant, update permutation-equivariant", dg < 1e-14 and do < 1e-14,
          f"max|dg| = {dg:.3e}, max|o(Pz) - P o(z)| = {do:.3e}")
    _, g0, _, o0 = gp(z, g_override="zero")
    check("g = 0 gives a zero update", o0.abs().max().item() == 0.0 and g0.abs().max().item() == 0.0,
          f"max|o| = {o0.abs().max().item():.3e}")
    check("learned gamma in (0, 1), 0.5 at init", bool((gamma > 0).all() and (gamma < 1).all()),
          f"gamma in [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
    gpf = GlobalPathway(DMODEL, 64, "fixed")
    _, _, gam_f, o_f = gpf(z)
    spread = (o_f - o_f[:, :1]).abs().max().item()
    cm = (C.cm_numerator(o_f) / o_f.pow(2).mean()).item()
    check("fixed gate: identical update for every token (exactly common mode)",
          spread < 1e-15 and gpf.gate is None and float(gam_f.min()) == 1.0 and abs(cm - 1) < 1e-12,
          f"max token spread = {spread:.3e}, common-mode fraction = {cm:.12f}")


def test_model_and_arms():
    torch.manual_seed(2)
    m = TwoLayerToy(pathway=True)
    x = torch.randn(4, SEQ_LEN, DMODEL)
    with torch.no_grad():
        _, _, c0 = m(x, ov={1: dict(g_override="zero"), 2: dict(g_override="zero")})
        _, _, cz = m(x, ov={1: dict(zero_gp=True), 2: dict(zero_gp=True)})
        out, _, c = m(x)
    same = max((c0[l]["o_gp"] - cz[l]["o_gp"]).abs().max().item() for l in (1, 2))
    check("zero_g == remove_pathway for this parameterisation", same == 0.0, f"max|diff| = {same:.3e}")
    rec = max((c[l]["residual"] + c[l]["update"] - c[f"l{l}_out"]).abs().max().item() for l in (1, 2))
    check("layer output = residual + o_attn + o_gp", rec < 1e-14, f"max|diff| = {rec:.3e}")
    # matched init: the base parameters of every arm are identical for the same seed
    sds = {}
    for arm, (pw, gg, _) in C.ARMS.items():
        torch.manual_seed(5)
        sds[arm] = TwoLayerToy(pathway=pw, gp_gate=gg).state_dict()
    base = sds["base"]
    ok = all(torch.equal(base[k], sd[k]) for sd in sds.values() for k in base)
    extra = sorted(set(sds["gp"]) - set(base))
    check("base parameters identical across arms (same seed)", ok,
          f"{len(base)} base tensors equal in all {len(sds)} arms; pathway adds {len(extra)} tensors")
    tags = sorted(C.ARMS)
    check("arm names = run-directory tags of the paper runs",
          set(C.PAPER_ARMS) <= set(tags), ", ".join(tags))


# ======================================================================================
# 3. penalty (Eq. 5 ratio)
# ======================================================================================
def test_penalty():
    torch.manual_seed(3)
    B, n, d = 6, SEQ_LEN, DMODEL
    o = torch.randn(B, n, d)
    P = torch.full((n, n), 1.0 / n)
    fro = ((P @ o) ** 2).sum()
    lhs, rhs = C.cm_numerator(o).item(), (fro / (n * d * B)).item()
    check("E(obar) == ||P_com O||_F^2 / (B n d)", abs(lhs - rhs) / rhs < 1e-13, f"{lhs:.12e} vs {rhs:.12e}")
    f1, f2 = C.penalty_frac(o).item(), C.penalty_frac(137.0 * o).item()
    check("penalty is scale invariant", abs(f1 - f2) / f1 < 1e-12, f"{f1:.12f} vs {f2:.12f}")
    ratio = (fro / (o ** 2).sum()).item()
    check("penalty == ||P_com O||_F^2 / ||O||_F^2", abs(f1 - ratio) < 1e-12, f"{f1:.12f} vs {ratio:.12f}")
    c = torch.randn(1, 1, d).expand(B, n, d).clone()
    check("penalty = 1 for a purely common update, ~0 for a centred one",
          abs(C.penalty_frac(c).item() - 1) < 1e-12 and C.penalty_frac(o - o.mean(1, keepdim=True)).item() < 1e-28,
          f"{C.penalty_frac(c).item():.12f}, {C.penalty_frac(o - o.mean(1, keepdim=True)).item():.2e}")
    # gradient flows through numerator AND denominator: the gradient is orthogonal to O
    # (d/dt pen(tO) = 0 at t = 1, up to the 1e-12 stabiliser), which a stop-gradient
    # denominator would violate (<grad, O> = 2 pen)
    oo = o.clone().requires_grad_(True)
    C.penalty_frac(oo).backward()
    radial = (oo.grad * o).sum().item()
    cos = radial / (oo.grad.norm() * o.norm()).item()
    og = o.clone().requires_grad_(True)
    (C.cm_numerator(og) / (og.pow(2).mean().detach() + C.EPS_FRAC)).backward()
    radial_sg = (og.grad * o).sum().item()
    check("penalty gradient is orthogonal to O (no stop-gradient)", abs(cos) < 1e-10,
          f"cos(grad, O) = {cos:.1e} (stop-gradient denominator would give <grad, O> = {radial_sg:.3e})")


# ======================================================================================
# 4. generators (matched batches)
# ======================================================================================
def test_generators():
    for mode in ("device", "cpu64"):
        outs = []
        for _ in range(2):
            torch.manual_seed(999)                        # global state must not matter
            dg, u, R = C.make_generators(3, DEV, task_gen=mode)
            outs.append((u, R, [C.sample_batch(dg, 8, DEV, dtype=torch.float64) for _ in range(3)]))
        (u1, R1, x1), (u2, R2, x2) = outs
        same = (all(torch.equal(a, b) for a, b in zip(x1, x2)) and torch.equal(u1, u2)
                and torch.equal(R1, R2))
        ort = (R1 @ R1.T - torch.eye(DMODEL)).abs().max().item()
        check(f"task_gen={mode}: same seed -> identical batches, u, R; R orthogonal",
              same and ort < 1e-12, f"max|R R^T - I| = {ort:.3e}, hash {C.task_hash(u1, R1)}")
    dg4, u4, _ = C.make_generators(4, DEV)
    x4 = C.sample_batch(dg4, 8, DEV, dtype=torch.float64)
    check("different seed -> different batches and task",
          (x4 - x1[0]).abs().max().item() > 1e-3 and (u4 - u1).abs().max().item() > 1e-3,
          f"max|dx| = {(x4 - x1[0]).abs().max().item():.3f}")


# ======================================================================================
# 5. evaluate.py: closed-form local predictions are exact
# ======================================================================================
def test_interventions():
    worst, n = 0.0, 0
    for arm in ("base", "gp", "gp_g1"):
        pw, gg, _ = C.ARMS[arm]
        torch.manual_seed(6)
        m = TwoLayerToy(pathway=pw, gp_gate=gg)
        x, xd = torch.randn(8, SEQ_LEN, DMODEL), torch.randn(8, SEQ_LEN, DMODEL)
        _, u, R = C.make_generators(0, DEV)
        js, jd = C.make_task(x, u, R)["j_star"], C.make_task(xd, u, R)["j_star"]
        with torch.no_grad():
            _, _, cache = m(x)
            _, _, dcache = m(xd)
            for l in (1, 2):
                for name in E.INTERVENTIONS[1:]:
                    ov, pred = E.build_intervention(name, m, l, cache, dcache, js, jd)
                    if ov is None:
                        continue
                    _, _, cn = m(x, ov={l: ov})
                    worst = max(worst, (cn[l]["update"] - cache[l]["update"] - pred).abs().max().item())
                    n += 1
    check("closed-form local change of every intervention is exact", worst < 1e-12 and n == 30,
          f"{n} (arm, layer, intervention) cases, max|err| = {worst:.3e}")


if __name__ == "__main__":
    test_notebook_equivalence()
    test_pathway()
    test_model_and_arms()
    test_penalty()
    test_generators()
    test_interventions()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    raise SystemExit(1 if n_fail else 0)
