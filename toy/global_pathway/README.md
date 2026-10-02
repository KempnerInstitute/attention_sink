# Controlled broadcast task: attention vs global pathway vs global pathway + penalty

Code for the controlled-task experiment of Section 5 (paragraphs "Experiments" and "Controlled-Task Results") and for
panel B of Figure 7 (`figures/fig7_mitigation.py`). Five seeds of three matched arms on the two-layer broadcast task of
Section 3.2 / Appendix A.1.

| file | what it does |
|---|---|
| `common.py` | model, task, penalty, generators, diagnostics, arm definitions |
| `train.py` | trains one (arm, seed) -> `runs/broadcast/<arm>/seed<S>/{ckpt.pt, train_log.npz, train_summary.json}` |
| `evaluate.py` | diagnostics + causal interventions of one run -> `eval.json` in the run directory |
| `summarize.py` | prints the Section-5 numbers and the Figure-7B values from the `eval.json` files |
| `tests.py` | CPU / float64 correctness tests (20 checks, ~10 s) |
| `run.sh` | all 15 runs: SLURM array (`sbatch run.sh`) or sequential (`bash run.sh`), then `summarize.py` |

## Setup

**Task** (Eq. 3). Tokens `x_i ~ N(0, I_64)`, `L = 32`, fresh every step. A fixed unit vector `u` selects the source
`j* = argmax_j <x_j, u>`; every token must output `y_i = x_i + alpha x_{j*} R` with a fixed orthogonal `R`, `alpha = 1`.

**Model.** The two-layer, one-head-per-layer pre-norm model of `toy/notebooks/broadcast_causal_intervention.ipynb`
(`tests.py` checks the forward pass is identical): `d = d_h = 64`, no biases in `W_Q, W_K, W_V, W_O`, learned
positional embedding before the first LayerNorm only, `MLP = LN -> Linear(d, 4d) -> GELU -> Linear(4d, d)`, residual
after every attention layer and MLP. AdamW, lr 5e-4, weight decay 1e-4, batch 1024, 30 000 steps, fp32 with TF32 off.

**Global pathway** (Eq. 4, both layers, from the same normalised tokens `z` as attention):
`pi_j = softmax_j(z_j . w_P)`, `g = sum_j pi_j W_VG z_j` (`d_g = 64`), `gamma_i = sigmoid(z_i . w_gamma + b_gamma)`,
update `x_i <- x_i + o_i^attn + gamma_i W_OG g`. Pathway weights: truncated normal, std 0.02; gate bias 0
(`gamma = 0.5` at init). The base parameters are created first, so all arms of one seed share them.

**Penalty** (Eq. 5 ratio). For each layer `l`, `pen_l = E(mean_i o_i^attn) / (E(O^attn) + 1e-12)` with
`E(Z) = mean squared entry` over the whole batch and `O^attn = W_O A V`; gradients flow through numerator and
denominator (scale invariant). The loss is `MSE + lambda_B * (pen_1 + pen_2)` with `lambda_B = 1`, i.e. the penalty
terms of the two layers are **summed** (Eq. 5 writes the mean over layers; see "Notes").

| arm (`--arm`, = run directory) | pathway | recipient gate | penalty | in the paper |
|---|---|---|---|---|
| `base` | no | - | no | "Attention" |
| `gp` | yes | learned `gamma_i` | no | "+ Global" |
| `gp_pen_pl12_frac` | yes | learned `gamma_i` | layers 1+2 | "+ Global + Pen." |
| `gp_g1` | yes | `gamma_i = 1` | no | optional, not used in the paper |
| `gp_pen_g1_pl12_frac` | yes | `gamma_i = 1` | layers 1+2 | optional, not used in the paper |

**Matched batches.** `torch.manual_seed(seed)` sets the parameter init; separate generators seeded `1000 + seed`
(training data) and `2000 + seed` (`u`, `R`) make every arm of a seed see the same task and the same batches;
evaluation uses `5000 + seed`. Always compare arms per seed: the seed moves the task MSE far more than the arm does.

## Running

```bash
python tests.py                                             # CPU, before anything else
python train.py --arm gp_pen_pl12_frac --seed 0             # ~4 min on one GPU (30k steps)
python evaluate.py --run runs/broadcast/gp_pen_pl12_frac/seed0
bash run.sh              # or: mkdir -p logs && sbatch run.sh (edit <PARTITION>/<ACCOUNT>; one array task per run)
python summarize.py                                         # Section-5 numbers from runs/
python ../../figures/fig7_mitigation.py --toy_runs runs     # Figure 7 (panel A needs vit_causal/ outputs)
```

`train.py` and `evaluate.py` skip runs whose `train_summary.json` / `eval.json` already exist (`evaluate.py --force`
overwrites). Training runs on the CPU too (`--device cpu`), but far too slowly for the full protocol.

## Outputs

* `train_summary.json`: `config`, `task_hash` (hash of `u`, `R`; must agree across the arms of a seed), timing, and the
  training diagnostics at the last logged step (`final`) and averaged over the last ten (`final_mean_last10`).
  `train_log.npz` holds the same keys at step 1 and every 100 steps (measured on the training batch before the update).
* `eval.json`: 64 batches x 256 sequences; every leaf is `{mean, sem, n}` over batches, `NaN` when not applicable.
  `performance.task_mse`; `diagnostics.l{1,2}_*` (`max_inflow` = Eq. 2 sink strength `max_s n^-1 sum_i A_is`,
  `sink_match`, `attn_to_jstar`, `eps_sink_frac`, value-norm ratios, `stable_rank` of `W_O A V`, common-mode fraction,
  branch RMS, pooling weight on `j*`, mean `gamma`); `interventions.L{1,2}.<intervention>.<metric>` for
  `baseline, zero_sink_value, donor_sink_value, remove_attention, zero_g, remove_pathway, donor_g`, each applied to one
  layer with its attention matrix fixed and propagated through the rest of the network. `donor_payload_transfer_cosine`
  is the cosine between the mean recipient output change and `alpha (payload_donor - payload_own)`.
  `local_identity_ok` checks the measured local change against its closed form (`checks.local_identity_failures` must
  be 0).

## Numbers of the paper runs

`python summarize.py` on the paper's 15 runs (means over seeds 0-4):

| arm | task MSE | inflow L1 | inflow L2 | L2: dMSE remove attention | L2: dMSE remove global | L2: cosine via sink | L2: cosine via g |
|---|---|---|---|---|---|---|---|
| `base` | 0.1397 | 0.888 | 0.906 | 2.906 | - | 0.962 | - |
| `gp` | 0.1402 | 0.103 | 0.908 | 2.875 | 0.515 | 0.957 | 0.203 |
| `gp_pen_pl12_frac` | 0.1441 | 0.062 | 0.057 | 0.004 | 1.644 | -0.035 | 0.892 |

## Notes

* **Reproducing the paper runs.** The paper runs were trained on CUDA GPUs with the default `--task_gen device`
  (`u`, `R` and the data stream drawn with generators on the GPU; the fp32 QR of `R` differs by ~1e-6 between GPU
  models, which changed task MSE by ~1e-8). On another GPU model expect agreement to the reported precision, not
  bit-identical runs. `--task_gen cpu64` builds `u`, `R` on the CPU in float64 (machine independent) but is a different
  random task for the same seed, so per-seed numbers change; do not mix the two in one set of runs. CPU runs use a
  different random stream than CUDA runs.
* **Penalty weight.** `train.py` adds `lambda_B * pen_l` for each of the two layers, so `lambda_B = 1` here corresponds
  to `lambda_B = 2` in the mean-over-layers form of Eq. 5.
* **Differences from the Appendix E.1 notebook** (`toy/notebooks/broadcast_global_pathway_penalty_causal_intervention.ipynb`):
  that notebook trains one seed with `gamma_i = 1`, default PyTorch init of the pathway, and a per-example penalty
  ratio with a stop-gradient denominator averaged over the two layers. It is a separate experiment; its numbers are
  not produced by this directory.
