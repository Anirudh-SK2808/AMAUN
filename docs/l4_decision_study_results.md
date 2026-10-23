# L4 decision study: results

Results of the run described in [l4_decision_study.md](l4_decision_study.md).

| | |
|---|---|
| Results directory | `results/local_smoke/l4_decision_study/` |
| Adapter directory | `adapters/local_smoke/l4_decision_study/` (`seed-sum`, `seed-qa`, `seed-sentiment`) |
| Run mode | `local_smoke` (**not reportable**, pipeline validation tier) |
| Backbone | Qwen/Qwen2.5-1.5B-Instruct @ `989aa79`, bf16, no quantisation, 3.09 GB |
| Hardware | GCP `g2` VM, 1 × NVIDIA L4 (24 GB), 16 vCPU, CUDA 13.0 |
| Software | Python 3.11.17, torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0, datasets 5.0.1 |
| Commit | `6211a15` (clean tree), config hash `2992ac81…` |
| Started | 2026-10-22 20:33 UTC |
| Seed | 1234 (single seed) |
| Total wall time | 27 596 s = **7.67 h** (budget 12 h, fits) |
| Failures / skipped stages | none / none |
| Backbone fingerprint unchanged | yes (checked before and after every adapter's training as well) |
| Split leakage audit | clean (190 split pairs checked, 0 overlaps) |

Sources: `decision_study/summary.json`, `standalone_test_scores.json`, `utility_supervision.json`,
`evaluation_<task>.json`, `preflight.json`, `data.json`, `memory_audit.json`, `metrics.jsonl`,
`fusion_logs/*.json`, `utility_network/reuse_gate.json`, and `adapters/.../<id>/metadata.json`.
TEST scores are on 150 examples per task. Perplexity = exp(mean token loss). GB = 10⁹ bytes.

The "ablations" in this study are the integration choices: for each new task, every option
(CREATE, REUSE with each adapter, FUSE with each adapter) was measured on TEST, and each decision
policy was scored by the option it picked.

---

## 0. Data

| Task | Dataset | train / val / support / test | Pool | Inputs over the input limit¹ |
|---|---|---|---|---|
| sum | abisee/cnn_dailymail 3.0.0 | 6000 / 750 / 750 / 300 | 9 000 | **65.1 %** |
| qa | rajpurkar/squad | 6000 / 750 / 750 / 300 | 9 000 | 0.0 % |
| sentiment | stanfordnlp/imdb | 6000 / 750 / 750 / 300 | 9 000 | 8.2 % |
| longform | Pavithree/eli5 | 6000 / 750 / 750 / 300 | 9 000 | 0.0 % |
| domain | dreamerdeo/finqa | 6000 / **64 / 64** / 300 | 6 203² | **45.8 %** |

¹ Whitespace-token proxy (`data.json`, computed before tokenisation) against each task's `max_input_tokens`: 512 for sum, 768 for domain (`configs/experiments/l4_decision_study.yaml`). Over-length prompts are cut from the middle (`prompt_truncation: middle`), keeping the question and answer cue.
² FinQA's train split was exhausted (6 251 rows read, 48 produced no example), which left only
64 rows each for validation and support.

---

## 1. Initial adapters (phase 1)

All three: LoRA r = 16, α = 32 (scaling 2.0), dropout 0.05, on q/k/v/o/gate/up/down projections
(196 layers); 18 464 768 trainable parameters; lr 2e-4 cosine, 3 % warmup, batch 4 × grad-accum 4,
gradient checkpointing; 6 000 TRAIN rows, 1 epoch = 375 optimizer steps, no early stop. No two
adapters share training rows. Each stores as 73.9 MB.

| Adapter | Dataset revision | Train seed | Max target tokens | Tokens seen | Train loss step 1 | Mean train loss, last 50 steps | Final train loss | Val loss @125 / @250 / @375 | Train wall (s) | Peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| `seed-sum` | `96df5e6` | 70419137 | 128 | 3 324 037 | 1.9110 | 1.6168 | 1.5375 | 1.4733 / 1.4543 / 1.4509 | 2 257 | 5.97 GB |
| `seed-qa` | `7b6d24c` | 1667571188 | 48 | 1 243 139 | 2.4619 | 0.1658 | 0.0139 | 0.2828 / 0.2617 / 0.2286 | 1 018 | 5.50 GB |
| `seed-sentiment` | `e628166` | 632721731 | 4 | 1 850 880 | 2.2331 | 0.0467 | 0.0016 | 0.0596 / 0.0473 / 0.0455 | 1 649 | 5.46 GB |

The in-training validation losses are on 16 VALIDATION examples
(`intermediate_eval_examples`). All three still improved at step 375. `seed-sum` was nearly flat
from step 250 on.

### Zero-shot backbone vs trained adapter (TEST)

| Task | Metric | Zero-shot | Own adapter | Δ | Token loss (ppl): zero-shot → adapter |
|---|---|---|---|---|---|
| sum | ROUGE-L | 0.2523 | 0.2885 | +0.0362 | 2.0040 (7.42) → 1.5021 (4.49) |
| sum | ROUGE-1 / ROUGE-2 | 0.3839 / 0.1614 | 0.4065 / 0.1875 | +0.0226 / +0.0261 | |
| qa | token F1 | 0.4708 | 0.8988 | +0.4280 | 1.3187 (3.74) → 0.2254 (1.25) |
| qa | exact match | 0.1933 | 0.7600 | +0.5667 | |
| sentiment | exact match | 0.8800 | 0.9333 | +0.0533 | 2.5535 (12.85) → 0.0902 (1.09) |
| longform | ROUGE-L | 0.1281 | (no seed adapter) | | 2.7977 (16.41) |
| domain (FinQA) | token F1 / EM / ROUGE-L | 0.0069 / 0 / 0.0062 | (no seed adapter) | | 2.7707 (15.97) |

TEST evaluation wall time (s): adapters sum 950, qa 73, sentiment 36. Zero-shot: sum 1 397,
qa 204, sentiment 37, longform 1 478, domain 1 261.

---

## 2. Utility supervision (phase 2)

Six ordered pair integrations: fuse the new adapter into the old one (layer-wise softmax α,
init 0.5, plus the LoRA weights themselves; 100 steps; KD from the old adapter, λ = 1, T = 2,
16 retention examples), then measure on VALIDATION. Losses use up to 750 examples and generation
scores use 16.

| New → old | New-task val loss (standalone → fused) | Old-task val loss (standalone → fused) | New-task gen (standalone → fused) | Old-task gen (standalone → fused) | Forgetting proxy | Beneficial |
|---|---|---|---|---|---|---|
| qa → sentiment | 0.2068 → 0.3510 | 0.0460 → 0.0663 | 0.9479 → 0.9021 | 0.9375 → 0.9375 | 0.0020 | no |
| qa → sum | 0.2068 → 0.2995 | 1.6630 → 1.7070 | 0.9479 → 0.8574 | 0.2514 → 0.2285 | 0.0037 | no |
| sentiment → qa | 0.0460 → 0.0645 | 0.2068 → 0.2128 | 0.9375 → 1.0000 | 0.9479 → 0.9479 | 0.0006 | no |
| sentiment → sum | 0.0460 → 0.0991 | 1.6630 → 1.6924 | 0.9375 → 0.9375 | 0.2514 → 0.2501 | 0.0025 | no |
| sum → qa | 1.6630 → 1.7657 | 0.2068 → 0.2123 | 0.2514 → 0.2422 | 0.9479 → 0.9479 | 0.0005 | no |
| sum → sentiment | 1.6630 → 1.7873 | 0.0460 → 0.0610 | 0.2514 → 0.2197 | 0.9375 → 0.9375 | 0.0015 | no |

**0 of 6 integrations were beneficial.** Fusion raised validation loss on both the new and the old
task in every pair. Generation scores mostly held on the old task. The new task lost up to 0.09
(qa → sum), and gained once (sentiment → qa, 0.9375 → 1.0 on 16 examples).

### Fusion diagnostics (supervision pairs)

| New → old | Mean α (old / new) | α range over layers (old) | L_new, mean of first → last 10 steps | L_KD, mean of first → last 10 steps | SVD re-factorisation error, mean / max | Fusion wall (s) | Pair wall (s) |
|---|---|---|---|---|---|---|---|
| qa → sentiment | 0.526 / 0.474 | 0.427 – 0.631 | 0.1894 → 0.2640 | 0.1302 → 0.0111 | 0.404 / 0.563 | 90 | 350 |
| qa → sum | 0.569 / 0.431 | 0.433 – 0.682 | 0.1906 → 0.2689 | 0.1464 → 0.0158 | 0.463 / 0.576 | 102 | 529 |
| sentiment → qa | 0.609 / 0.391 | 0.503 – 0.684 | 0.0191 → 0.0341 | 0.3816 → 0.0102 | 0.397 / 0.549 | 115 | 301 |
| sentiment → sum | 0.595 / 0.405 | 0.411 – 0.717 | 0.0218 → 0.0200 | 0.1016 → 0.0082 | 0.394 / 0.566 | 130 | 440 |
| sum → qa | 0.532 / 0.468 | 0.390 – 0.609 | 1.6643 → 1.5132 | 0.2170 → 0.0132 | 0.459 / 0.575 | 144 | 498 |
| sum → sentiment | 0.467 / 0.533 | 0.306 – 0.572 | 1.6504 → 1.5017 | 0.2326 → 0.0075 | 0.362 / 0.498 | 148 | 534 |

Two things in these logs bear directly on "fusion is always harmful":

- **The KD term won.** L_KD fell 9–37× in every fusion, while L_new rose in 3 of the 6 pairs.
  At λ = 1 the optimiser preserved the old adapter's outputs at the expense of the new task.
- **The stored fused adapter is not the one that was optimised.** `fusion.storage =
  refactorize_svd` with `refactorize_rank = null` truncates each fused layer back to rank 16.
  The relative reconstruction error is 36–46 % on average per layer and up to 58 %. Every fused
  score in this document was measured on the truncated adapter.

### What was fitted on these 6 examples

| Component | Result |
|---|---|
| Utility network g_ω | MLP [64, 64], 6 heads, 54 features (8 always missing), 200 epochs, training loss 1 817.8 → 18.48, in-sample only |
| Per-head final MSE (in-sample) | primary 0.313, forgetting 0.215, confidence 0.014, memory 3.846, communication 4.088, long-term loss 0.102 |
| τ_u = τ_reuse | −1.1188, set by the one-class "conservative" policy (just above the max of 6 harmful observations), **not learned** |
| Fixed-rule threshold | `semantic.d_task_sq` < −0.0450 (same one-class policy) |
| REUSE gate | ridge regression (λ = 1) on 9 features, 6 training rows, 16 probe examples, tolerance −0.01 |

In-sample predictions of the utility network per pair (predicted U / predicted primary-score
target vs the target it was fitted to):

| New → old | Predicted U | Predicted primary | Target primary |
|---|---|---|---|
| qa → sentiment | −1.3867 | 0.1469 | 0.9655 |
| qa → sum | −1.1818 | 0.9036 | 0.9705 |
| sentiment → qa | −1.1382 | 0.3890 | 0.9936 |
| sentiment → sum | −1.1188 | 0.5670 | 0.9901 |
| sum → qa | −1.6879 | 0.2890 | 0.8381 |
| sum → sentiment | −1.7070 | 0.2395 | 0.8363 |

Even in-sample, the primary-score head misses its own targets by up to 0.82. The network did not
fit its 6 training points.

The run records two limitations against itself: 6 < `min_supervision_examples` = 20, and no
variation in the outcome label.

REUSE-gate in-sample fit (VALIDATION gap = existing adapter's score − dedicated adapter's score):

| New task | Candidate | Measured gap | Predicted gap | Abs. error |
|---|---|---|---|---|
| qa | seed-sentiment | −0.0399 | −0.0499 | 0.0100 |
| qa | seed-sum | −0.1186 | −0.1069 | 0.0117 |
| sentiment | seed-qa | −0.0057 | −0.0045 | 0.0013 |
| sentiment | seed-sum | −0.1126 | −0.1079 | 0.0048 |
| sum | seed-qa | −0.0606 | −0.0633 | 0.0027 |
| sum | seed-sentiment | −0.0433 | −0.0483 | 0.0050 |

REUSE-gate ridge weights (on standardised features; bias −0.0635):

| Feature | Weight | Mean | Std |
|---|---|---|---|
| d_task_sq | −0.0018 | 0.0516 | 0.0080 |
| cand_loss | −0.0009 | 1.3634 | 0.8262 |
| base_loss | +0.0014 | 1.8761 | 0.6619 |
| loss_gain | +0.0017 | 0.5127 | 1.0050 |
| rel_loss_gain | +0.0146 | 0.1711 | 0.5139 |
| cand_confidence | −0.0021 | 0.7659 | 0.1279 |
| cand_entropy | +0.0074 | 1.1062 | 0.6229 |
| cand_ece | −0.0209 | 0.1556 | 0.1025 |
| cand_token_accuracy | +0.0119 | 0.6801 | 0.1885 |

---

## 3. Evaluation tasks (phase 3): every option measured on TEST

A temporary adapter was trained on each evaluation task (6 000 rows, 375 steps) so CREATE and FUSE
could be measured even where a policy would not have trained.

| Task | Temp adapter final train loss | Train wall (s) | Peak memory |
|---|---|---|---|
| longform (ELI5) | 2.8136 | 1 538 | 8.11 GB |
| domain (FinQA) | 1.3357 | 3 011 | 6.95 GB |

Columns: **new** = new-task primary score; **NT** = negative transfer vs a separate adapter;
**fgt** = forgetting on old tasks; **avg** = average over all four tasks; **regret** = oracle avg −
this option's avg; **repo** = repository size after integration.

### 3a. longform (ELI5), primary metric ROUGE-L, expected relation: seed-sum

| Option | new | loss (ppl) | NT | qa | sentiment | sum | fgt | avg | regret | adapters | repo (MB) | integ. (s) | eval (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CREATE | 0.1270 | 2.5626 (12.97) | 0 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5619 | 0.0012 | 4 | 295.4 | 0 | 1 055 |
| FUSE:seed-qa | 0.1057 | 2.6081 (13.57) | 0.0213 | 0.8960 | 0.9333 | 0.2885 | 0.0009 | 0.5559 | 0.0072 | 3 | 221.6 | 112 | 781 |
| FUSE:seed-sentiment | 0.1106 | 2.6222 (13.77) | 0.0164 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5578 | 0.0053 | 3 | 221.6 | 115 | 1 258 |
| FUSE:seed-sum | 0.1065 | 2.6275 (13.84) | 0.0205 | 0.8988 | 0.9333 | 0.2808 | 0.0026 | 0.5549 | 0.0082 | 3 | 221.6 | 127 | 2 232 |
| REUSE:seed-qa | 0.0642 | 2.8368 (17.06) | 0.0628 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5462 | 0.0169 | 3 | 221.6 | 0 | 212 |
| REUSE:seed-sentiment | 0.0550 | 2.7705 (15.97) | 0.0720 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5439 | 0.0192 | 3 | 221.6 | 0 | 345 |
| **REUSE:seed-sum** (oracle) | **0.1318** | 2.7527 (15.68) | 0 | 0.8988 | 0.9333 | 0.2885 | 0 | **0.5631** | 0 | 3 | 221.6 | 0 | 934 |

Zero-shot backbone on longform: ROUGE-L 0.1281, loss 2.7977 (ppl 16.41).
Old-task token loss after fusion: qa 0.2324 (standalone 0.2254), sentiment 0.0801 (0.0902),
sum 1.5353 (1.5021). The fused `sum` adapter dropped ROUGE-1/2/L to 0.3924 / 0.1811 / 0.2808.

Note that REUSE:seed-sum has the best ROUGE-L but a *worse* perplexity (15.68) than CREATE (12.97)
or any fusion. Generation score and likelihood disagree on this task.

### 3b. domain (FinQA), primary metric token F1, expected relation: seed-qa

| Option | new (F1) | EM | loss (ppl) | NT | qa | sentiment | sum | fgt | avg | regret | adapters | repo (MB) | integ. (s) | eval (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CREATE** (oracle) | **0.1844** | 0.1133 | 1.3081 (3.70) | 0 | 0.8988 | 0.9333 | 0.2885 | 0 | **0.5763** | 0 | 4 | 295.4 | 0 | 86 |
| FUSE:seed-qa | 0.1067 | 0.0400 | 1.5657 (4.79) | 0.0778 | 0.8869 | 0.9333 | 0.2885 | 0.0040 | 0.5538 | 0.0224 | 3 | 221.6 | 181 | 162 |
| FUSE:seed-sentiment | 0.1322 | 0.0467 | 1.5512 (4.72) | 0.0522 | 0.8988 | 0.9133 | 0.2885 | 0.0067 | 0.5582 | 0.0181 | 3 | 221.6 | 185 | 124 |
| FUSE:seed-sum | 0.1300 | 0.0267 | 1.5504 (4.71) | 0.0544 | 0.8988 | 0.9333 | 0.2683 | 0.0067 | 0.5576 | 0.0187 | 3 | 221.6 | 197 | 1 191 |
| REUSE:seed-qa | 0.0769 | 0.0267 | 1.9777 (7.23) | 0.1075 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5494 | 0.0269 | 3 | 221.6 | 0 | 94 |
| REUSE:seed-sentiment | 0.0600 | 0.0333 | 2.1727 (8.78) | 0.1244 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5452 | 0.0311 | 3 | 221.6 | 0 | 65 |
| REUSE:seed-sum | 0.0513 | 0.0000 | 2.5426 (12.71) | 0.1331 | 0.8988 | 0.9333 | 0.2885 | 0 | 0.5430 | 0.0333 | 3 | 221.6 | 0 | 299 |

Zero-shot backbone on domain: token F1 0.0069, EM 0, loss 2.7707 (ppl 15.97).
Old-task effects of fusion: fusing into `seed-qa` dropped SQuAD EM 0.76 → 0.72 (F1 0.8988 →
0.8869, loss 0.2254 → 0.2561); into `seed-sentiment` dropped IMDb accuracy 0.9333 → 0.9133;
into `seed-sum` dropped ROUGE-1/2/L to 0.3855 / 0.1711 / 0.2683.

### Fusion diagnostics (evaluation fusions)

| Fusion | Mean α (old / new) | α range over layers (old) | L_new, first → last 10 steps | L_KD, first → last 10 steps | SVD error mean / max | Fusion wall (s) |
|---|---|---|---|---|---|---|
| longform → seed-qa | 0.550 / 0.450 | 0.432 – 0.631 | 2.5002 → 2.6622 | 0.4130 → 0.0115 | 0.475 / 0.580 | 111 |
| longform → seed-sentiment | 0.495 / 0.505 | 0.366 – 0.636 | 2.4962 → 2.6460 | 0.2648 → 0.0067 | 0.401 / 0.542 | 114 |
| longform → seed-sum | 0.506 / 0.494 | 0.248 – 0.628 | 2.4766 → 2.6459 | 0.0909 → 0.0121 | 0.455 / 0.558 | 126 |
| domain → seed-qa | 0.539 / 0.461 | 0.380 – 0.657 | 1.3578 → 1.4469 | 0.4034 → 0.0211 | 0.449 / 0.554 | 179 |
| domain → seed-sentiment | 0.517 / 0.483 | 0.396 – 0.659 | 1.3382 → 1.4210 | 0.1982 → 0.0122 | 0.390 / 0.539 | 183 |
| domain → seed-sum | 0.522 / 0.478 | 0.363 – 0.704 | 1.3243 → 1.4324 | 0.0935 → 0.0168 | 0.439 / 0.557 | 195 |

In all 6 evaluation fusions, L_new **rose** during fusion while L_KD fell, the same pattern as in
phase 2. Across all 12 fusions, L_new rose in 9.

### Option-level observations

- **FUSE never beat CREATE on the new task.** It cost 0.016–0.021 ROUGE-L on longform and
  0.052–0.078 F1 on domain, and caused small forgetting (≤ 0.0067) on the fused-into task.
- **REUSE was only competitive where the relation was real:** `seed-sum` on longform beat
  CREATE (0.1318 vs 0.1270) with one fewer adapter. On domain, reusing the "related" `seed-qa`
  lost 0.1075 F1.
- **The expected relation did not help fusion.** On domain, fusing into the "related" `seed-qa`
  was the worst fusion (0.1067 F1). On longform, the three fusions were within 0.005 ROUGE-L of
  each other.

---

## 4. Policy comparison (the main result)

| Policy | longform | domain | Regret longform | Regret domain | Mean avg perf | Mean forgetting | Mean regret vs oracle | Flags |
|---|---|---|---|---|---|---|---|---|
| `amaun` | CREATE | CREATE | 0.0012 | 0 | 0.5691 | 0 | 0.0006 | longform: unnecessary_create |
| `amaun_train_first` | CREATE | CREATE | 0.0012 | 0 | 0.5691 | 0 | 0.0006 | longform: unnecessary_create |
| `always_separate` | CREATE | CREATE | 0.0012 | 0 | 0.5691 | 0 | 0.0006 | longform: unnecessary_create |
| `fixed_rule` | CREATE | CREATE | 0.0012 | 0 | 0.5691 | 0 | 0.0006 | longform: unnecessary_create |
| `always_merge` | FUSE:seed-qa | FUSE:seed-qa | 0.0072 | 0.0224 | 0.5549 | 0.0025 | 0.0148 | domain: harmful_integration |

Oracle: REUSE:seed-sum on longform (avg 0.5631), CREATE on domain (avg 0.5763).

Decision reasons as logged:

| Policy | longform | domain |
|---|---|---|
| `amaun`, `amaun_train_first` | max U −1.5390 < τ_u −1.1188 → CREATE | max U −1.2522 < τ_u −1.1188 → CREATE |
| `fixed_rule` | max −d_task_sq −0.0514 < −0.0450 → CREATE | max −d_task_sq −0.1112 < −0.0450 → CREATE |
| `always_merge` | nearest by d_task_sq = seed-qa (0.0514) | nearest by d_task_sq = seed-qa (0.1117) |
| `always_separate` | CREATE | CREATE |

The longform `unnecessary_create` flag lists FUSE:seed-qa, FUSE:seed-sentiment, FUSE:seed-sum and
REUSE:seed-sum as matching CREATE "within tolerance" (0.01 on avg) with one fewer adapter.

**Interpretation.** Four of five policies made identical decisions, so this run does not separate
AMAUN from the CREATE-always or fixed-rule baselines. The reason is upstream: all 6 supervision
integrations were harmful, so τ_u came from the one-class policy, and every predicted utility fell
below it. AMAUN was effectively a CREATE-always policy here. It did beat `always_merge` (regret
0.0006 vs 0.0148, no forgetting). The run itself labels these differences "descriptive, not
statistically tested" (2 tasks × 1 seed).

### Utility network: decision-time prediction vs TEST-measured outcome

The utility network scores *fusing* with each candidate. "Actual" is the corresponding FUSE option,
scored on TEST with the supervision target definition (loss → score proxy).

| Task | Candidate | Predicted U | Actual U | Predicted primary target | Actual | Predicted forgetting | Actual | Predicted memory | Actual | Predicted long-term loss | Actual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| longform | seed-sum | −1.5390 | −1.1782 | 0.6950 | 0.7689 | 0.7540 | 0.0029 | 108.2 | 73.9 | 2.6831 | 2.6275 |
| longform | seed-qa | −1.7076 | −1.1753 | 0.2587 | 0.7704 | 0.3813 | 0.0007 | 66.5 | 73.9 | 1.5687 | 2.6081 |
| longform | seed-sentiment | −1.8455 | −1.1759 | 0.0651 | 0.7693 | 0.4378 | 0.0000 | 62.7 | 73.9 | 1.5546 | 2.6222 |
| domain | seed-sum | −1.2522 | −1.0597 | 1.1111 | 0.8564 | 0.6784 | 0.0037 | 194.1 | 73.9 | 3.5665 | 1.5504 |
| domain | seed-sentiment | −1.3656 | −1.0573 | 0.5510 | 0.8563 | 0.4292 | 0.0001 | 127.1 | 73.9 | 2.2329 | 1.5512 |
| domain | seed-qa | −1.4901 | −1.0618 | 0.4129 | 0.8551 | 0.4021 | 0.0030 | 87.0 | 73.9 | 1.6562 | 1.5657 |

The predictions span a wide range (primary 0.07–1.11, forgetting 0.38–0.75), but the measured
outcomes of the three candidates are nearly identical within each task (U spread ≤ 0.005,
forgetting ≤ 0.004). The network ranked `seed-sum` first on both tasks. By measured score,
FUSE:seed-sum was the middle of the three fusions on both tasks (longform 0.1065 ROUGE-L, domain
0.1300 F1). Forgetting was overpredicted by two orders of magnitude or more. With 6 in-sample training points, none of
this shows generalisation.

### REUSE gate (before training): predicted vs TEST

| Task | Candidate | Loss gain over backbone (probe) | Eligible | Predicted gap | Actual gap on TEST (proxy) | Measured score, REUSE − CREATE | Gate said OK | Actually OK |
|---|---|---|---|---|---|---|---|---|
| longform | seed-qa | −0.0345 | no | −0.0428 | −0.0209 | −0.0628 | no | no |
| longform | seed-sentiment | −0.0030 | no | −0.0398 | −0.0159 | −0.0720 | no | no |
| longform | seed-sum | +0.0429 | yes | −0.0561 | −0.0146 | **+0.0048** | no | no |
| domain | seed-qa | +0.9814 | yes | −0.0929 | −0.0568 | −0.1075 | no | no |
| domain | seed-sentiment | +0.5173 | yes | −0.1322 | −0.0727 | −0.1244 | no | no |
| domain | seed-sum | +0.1236 | yes | −0.1177 | −0.1019 | −0.1331 | no | no |

"Eligible" = `require_gain_over_backbone`: the candidate must have lower probe loss than the bare
backbone. On longform only `seed-sum` passed. The gate stayed closed on both tasks
(tolerance −0.01; gate time 15.8 s longform, 34.5 s domain), so it saved no training time. It
agreed with the TEST label in 6 of 6 cases and overestimated the reuse penalty every time. Its
"actual gap" is the proxy-based gap: by measured ROUGE-L, REUSE:seed-sum on longform scored
**+0.0048 over CREATE**, which a stricter score-based label would have counted as acceptable.

---

## 5. Resources

### Stage durations

| Stage | Seconds | Hours |
|---|---|---|
| Preflight | 116 | 0.03 |
| Train initial adapters | 4 925 | 1.37 |
| Zero-shot evaluation | 4 377 | 1.22 |
| Utility supervision | 2 652 | 0.74 |
| Train temporary adapters | 4 550 | 1.26 |
| Evaluation tasks | 14 431 | 4.01 |
| Memory audit | 27 | 0.01 |
| **Total** | **27 596** | **7.67** |

### Training budget ledger

| Optimizer steps | Examples seen | Tokens seen | Trainable params | Training wall (s) |
|---|---|---|---|---|
| 1 875 | 30 000 | 12 329 542 | 18 464 768 | 9 473 |

Fusion adds 392 α parameters per fusion (196 layers × 2 components) on top of the LoRA weights.

### Preflight: peak GPU memory on the token-longest TRAIN batch (batch 4)

| Task | Seq len (train / gen) | Train step | Fusion + KD step | Generation |
|---|---|---|---|---|
| sum | 641 / 512 | 5.74 GB | 16.26 GB | 3.26 GB |
| qa | 522 / 512 | 5.27 GB | 14.22 GB | 3.26 GB |
| sentiment | 514 / 512 | 5.24 GB | 13.70 GB | 3.26 GB |
| longform | 621 / 115 | 7.91 GB | 18.54 GB | 3.20 GB |
| domain | 884 / 768 | 6.73 GB | **21.85 GB** | 3.30 GB |

Peak allocated memory. Reserved memory peaked at 22.61 GB. The domain fusion step uses 91 % of the
L4's 24 GB, which leaves little headroom for longer inputs or a larger backbone.

### Memory audit

Verdict **stable** over 3 cycles of load → evaluate → fuse+KD (2 steps) → evaluate → release → gc.
GPU allocated 3.18 GB, reserved 22.68 GB, live tensors 1 924, CPU RSS 3.08 GB, all with zero growth
between cycles.

| Cycle | Standalone qa (F1 / loss) | Fused qa (F1 / loss) |
|---|---|---|
| 0 | 1.000 / 0.6075 | 1.000 / 0.7427 |
| 1 | 1.000 / 0.6075 | 0.875 / 0.7484 |
| 2 | 1.000 / 0.6075 | 0.875 / 0.7484 |

(4 VALIDATION examples.) Cycle 0's fused adapter differs from cycles 1 and 2, so the fusion step is
not bit-for-bit reproducible across identical cycles. The standalone path is.

### Repository

| State | Adapters | Bytes |
|---|---|---|
| After phase 1 | 3 | 221 577 216 (221.6 MB) |
| After CREATE | 4 | 295 436 288 (295.4 MB) |
| After REUSE / FUSE | 3 | 221 577 216 (221.6 MB) |

73.9 MB per adapter, including fused ones, because fusion re-factorises to rank 16. Backbone:
3.09 GB.

---

## 6. What this run does and does not show

**Shows**
- The full decision-study pipeline runs end to end on an L4 in 7.7 h with no failures, no
  split leakage, an unchanged backbone, and no memory growth.
- On these tasks, fusion as configured (100 steps, KD λ = 1, SVD back to rank 16) is harmful in
  every measured case: 6/6 supervision pairs and 6/6 evaluation fusions lost new-task score
  against a dedicated adapter.
- Reusing an adapter only works when the task relation is genuine (sum → ELI5), not merely
  nominal (SQuAD → FinQA).

**Does not show**
- That fusion as a method is harmful. Two configuration effects confound the result: KD dominated
  the objective (L_new rose in 9 of 12 fusions), and the scored adapter is an SVD truncation with
  ~36–48 % mean per-layer reconstruction error. Neither was isolated in this run.
- That AMAUN's learned decision beats a baseline. With no beneficial supervision examples, it
  reduced to CREATE-always and tied three baselines.
- Anything reportable: `local_smoke` tier, 1 seed, 2 evaluation tasks, 150 TEST examples each,
  no significance test. 65 % of CNN/DM (limit 512) and 46 % of FinQA (limit 768) inputs exceeded the input limit by a whitespace proxy, and
  FinQA had only 64 validation rows.

**What the next run would need** to make the comparison informative: supervision pairs that include
beneficial integrations (same-task or closely related pairs, ≥ 20 examples); a fusion ablation
(`storage: keep_components` or `dense_delta` vs `refactorize_svd`, and a lower KD λ); a genuinely
different evaluation task (e.g. `dialog`); more than one seed; and the `actual` tier.
