# Toy-model notebooks

Self-contained Jupyter notebooks for the controlled NOP and broadcast tasks. Each notebook defines its own model,
trains it and plots the results; stored outputs are kept, so every number and figure can be read without re-running.

| notebook | paper | what it does |
|---|---|---|
| `noop_causal_intervention.ipynb` | App. A.3: Table 2, Figure 8 (and the A.3 routing / value-norm numbers) | single-head NOP model with a dedicated token 0; zero / donor / non-sink value at the sink, redirect sink attention |
| `broadcast_causal_intervention.ipynb` | App. A.4: Table 3, Figures 9 and 10 (and the A.4 sink-match / attention numbers) | two-layer broadcast model; value and routing interventions per layer, donor-payload transfer |
| `broadcast_global_pathway_penalty_causal_intervention.ipynb` | App. E.1: Figure 19 and the E.1 numbers | attention baseline / + global pathway / + global pathway + penalty, one seed, uniform recipients (`gamma_i = 1`) |
| `noop_dynamics.ipynb` | Figure 2B (cell 16), Figure 11A (cell 13), Figure 11B (cell 9) | NOP sink formation over training: damping sweep, NOP-frequency sweep, `W_Q W_K` spectrum of a sink vs a no-sink model |

Not in these notebooks: the sources of **Figure 2A** (sink / non-sink value-norm histogram) and **Figure 3** (broadcast
attention patterns, value norms, query-key PCA, update rank). The paper versions of Figure 2B and Figure 11 are
restyled re-plots of the notebook curves; the re-plotting script is not included either. The five-seed controlled task
of Section 5 / Figure 7B is in `../global_pathway/`.

**Dependencies.** `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` (`noop_dynamics`), Jupyter; all in
`environment.yml`. A GPU is strongly recommended.

**Rough runtimes on one recent GPU** (estimates, not measured; the stored outputs come from a Colab T4 for the three
causal notebooks): `noop_causal_intervention` a few minutes; `broadcast_causal_intervention` ~5 min;
`broadcast_global_pathway_penalty_causal_intervention` ~10-20 min (three 30k-step runs; set
`BROADCAST_RUN_MODE=smoke` for a 3-step end-to-end check); `noop_dynamics` 1-3 h (27 training runs of 10k-30k steps,
batch 2048, with a 64-batch sink-score evaluation every 10 steps).

**Outputs written to the working directory.** Checkpoints `toy_noop_dedicated_sink_checkpoint.pt`,
`toy_broadcast_causal_checkpoint.pt`, `broadcast_global_pathway_checkpoints/*.pt`, and PDFs / CSVs next to them. The
three causal notebooks **load an existing checkpoint instead of training**, so delete it to retrain.

## Known issues (code cells are kept as originally run; nothing below is fixed)

`noop_dynamics.ipynb`
1. **Not runnable top to bottom.** Cell 8 evaluates `no_ops_values`, which is first assigned in cell 9 (the stored cell-8
   output has execution count 70, i.e. it came from an out-of-order session). A fresh kernel raises `NameError` there.
2. **No random seeds** anywhere (no `torch.manual_seed`); re-runs give different curves.
3. Writes `data/*.pt` and `figs/*.pdf` without creating `data/` and `figs/` (`FileNotFoundError` unless they exist).
4. `device = "cuda"` is hard-coded in four cells.
5. **Formulation differs from App. A.1 / A.5.** The model adds the attention output to the *normalised* input,
   `O = LN(x + pos + bos) + W_O(A V)`, and the loss is on the block output, `MSE(O, m*gamma*x + (1-m)*x)` with
   `m = 1[<x, v> > lambda]`. Eqs. 6-7 instead define `O = X + dO` with the loss on the attention branch `dO` against
   `(1-m) X + m gamma X`, and A.5 says the sweeps use this branch-supervised form. In the notebook the `m = 0` tokens
   are the NOP tokens (target ~ residual) and `m = 1` tokens need an active update; under Eq. 7 at `gamma = 0` it is the
   other way round, which also flips the meaning of the NOP probability `p` in Eq. 22.
6. **Spectral signature (Figure 2B).** Cell 16 takes the singular values of `Wq.weight @ Wk.weight.T`. With the paper's
   convention `q = W_Q^T x` (so `nn.Linear.weight = W_Q^T`) this is `W_Q^T W_K`, not `Theta = W_Q W_K^T` of Eq. 24
   (`Wq.weight.T @ Wk.weight`); the two have different singular values in general. The axis says "Eigenvalue" but the
   values are singular values.
7. **Figure 11A colour bar.** Cell 13 normalises the colour bar to the thresholds `lambda` (-1.28 ... 1.28) but labels it
   "Probability of no-op"; colours follow the rank of `lambda`, so yellow = largest `lambda` = smallest `P(m = 1)`. The
   paper's re-plot shows 0.1 ... 0.9 with yellow = 0.9, i.e. `P(m = 0)` (the notebook's own NOP tokens), whereas Eq. 22
   (`lambda = Phi^-1(1 - p)`) defines `p = P(m = 1)`.
8. **Sweep sizes.** The frequency sweep trains 20 values `p = linspace(0.1, 0.9, 20)` for 20k steps each (App. A.5 lists
   `p in {0.1, 0.3, 0.5, 0.7, 0.9}`; A.1's default is 30k steps). The damping sweep trains the six `gamma` of Eq. 23,
   Figure 11B plots four (0, 0.01, 0.1, 1). In the stored outputs `gamma = 0.1` and `1.0` never form a sink, and the
   high-`lambda` end of the frequency sweep stays at the uniform level 1/32 through 20k steps.

`noop_causal_intervention.ipynb`
9. "Donor sink value" permutes token 0's value within the batch (`roll(1)`). Token 0's input is zeroed, so its value is
   identical in every example and this intervention is an exact no-op by construction (Table 2: change 0.000000).

`broadcast_global_pathway_penalty_causal_intervention.ipynb` (App. E.1)
10. **Not Eq. 5.** The penalty is a per-example ratio with a stop-gradient denominator,
    `mean_b ||P_com O_b||^2 / (sg(||O_b||^2) + 1e-12)`, averaged over the two layers, `lambda_B = 1`. Eq. 5 uses the
    batch-level ratio `E(O_bar) / E(O)` with gradients through both. The stored penalised model satisfies it by
    shrinking attention: attention-update RMS 1e-6 (L1) and 1.6e-5 (L2), L1 attention exactly uniform (max inflow
    0.03125), the behaviour App. E.2 ("Penalty normalization") attributes to stop-gradient denominators.
11. One seed; recipient gate omitted (`gamma_i = 1`, as the E.1 text says); pathway weights use the default
    `nn.Linear` init. It is therefore a different experiment from `../global_pathway/` (Eq.-5 ratio summed over
    layers, learned `gamma_i`, five seeds), whose optional `gp_g1` / `gp_pen_g1_pl12_frac` arms are the closest
    counterpart.

All notebooks: exact numbers need the same GPU type (seeded notebooks draw from the CUDA RNG when a GPU is present).
