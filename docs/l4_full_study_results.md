# L4 full decision study — results

Results of the run specified in `docs/l4_next_run_prompt.md` (Part C). Every number below is read
from the run's own outputs in `results/local_smoke/l4_full_study/` (git-ignored; copied from the VM
and checked file by file against a SHA-256 manifest, `results/local_smoke/l4_full_study_vm_logs/`).

**Tier `local_smoke`: not a reportable result.** One seed, one backbone size, four policy-eligible
evaluation tasks. Every comparison below is descriptive; the confidence intervals resample TEST
examples only and say nothing about seed-to-seed variation.

## 1. Run header

| | |
|---|---|
| Experiment | `l4_full_study`, tier `local_smoke`, seed 1234 |
| Backbone | Qwen/Qwen2.5-1.5B-Instruct @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, bf16, frozen (fingerprint unchanged at the end) |
| Hardware | 1 × NVIDIA L4 (GCP `asia-south1-c`), CUDA 13.0, torch 2.14.0+cu130, Python 3.11.17 |
| Final commit | `4c84ec5` on `master`, clean tree |
| Config hash | `f863ed09…` (started under `a2161b04…`; changed by the D-029 surgery, §7) |
| Started / finished | 2026-10-23 15:41 UTC / 2026-10-24 14:44 UTC (23.1 h wall clock) |
| Compute time | **14.23 h** of a 32 h budget (preflight projection 29.06 h) |
| Failures / skipped units | none / none |
| Determinism | strict (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`); no op raised |
| Evaluation path | adapter merged into the base weights, greedy, 4 prompts per batch (D-026) |

**Incidents and resumes.** Two resumes, both recorded in the checkpoint:

1. Phase 4 crashed on its first fit of the utility network (a fold's normaliser was given a split
   name the leakage guard rejects; D-028). The run sat idle from 23:12 to 07:26 UTC; nothing
   finished was lost. Resumed on `f636fdc`, which also introduced ELI5's gain-retention criteria
   (D-027) before any ELI5 VALIDATION or TEST row was read.
2. After the first fit overfitted and a normaliser defect was found, the utility model was revised
   during the run (D-029). The run was stopped at 08:07 UTC, its checkpoint changed by
   `scripts/checkpoint_surgery_d029.py` (allow-listed config keys only; Phase 4 redone; no
   measurement discarded), and resumed on `4c84ec5`. The stop discarded ~45 min of an unfinished
   FinQA adapter, which was retrained from the start.

Two earlier attempts are archived on the VM (`results/local_smoke/l4_full_study.preflight_78h`,
`.resumed_attempt`) and are not part of this run: a launch stopped by its own preflight (projected
78.4 h before D-026) and a resumed launch replaced by a clean restart.

## 2. Projected vs actual time

| Phase | Projected (h) | Actual (h) |
|---|---|---|
| P0 preflight | — | 0.14 |
| P1 seeds | 2.03 | 1.59 |
| P2 fusion selection | 6.76 | 2.61 |
| P3 supervision | 8.56 | 3.27 |
| P4 utility | — | 0.00 |
| P5 evaluation tasks | 8.61 | 6.13 |
| P6 stream | 2.92 | 0.31 |
| P7 audit | 0.18 | 0.15 |
| **Total** | **29.06** | **14.23** |

The projection was conservative by about 2×. Phase 6 was cheap because nearly every stream
evaluation repeated a measurement already cached in phases 1 and 5 (R2/R3). Phases 2 and 3 were
overestimated because their loss passes and generations cost less than the timing probe's cold
first call suggested.

## 3. Data integrity

**Run-1 splits reproduced exactly** for the three reused adapters (D-022). TRAIN token counts
equal run 1's `tokens_seen`: sum 3 324 037, qa 1 243 139, sentiment 1 850 880.

| Task | Data config | Snapshot keys | Split sizes | Row digests | TRAIN tokens = run 1 |
|---|---|---|---|---|---|
| sum | ✓ | ✓ | ✓ | ✓ | ✓ |
| qa | ✓ | ✓ | ✓ | ✓ | ✓ |
| sentiment | ✓ | ✓ | ✓ | ✓ | ✓ |

**Leakage audit: clean.** Example-id audit over 946 split pairs, content audit over 880. The only
cross-task TEST overlaps are the intended ones (qa ↔ qa-B and sentiment ↔ sentiment-B share their
300 TEST rows).

| Task | TRAIN | VAL | SUPPORT | TEST | Pool | Dropped: repeat of a TEST row (D-025) | Excluded: in the original's pool |
|---|---|---|---|---|---|---|---|
| sum | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| qa | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| sentiment | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| qa-B | 6000 | 750 | 750 | 300 | 8997 | 0 | 3 |
| sentiment-B | 6000 | 750 | 750 | 300 | 8974 | 0 | 26 |
| instr | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| dialog | 6000 | 750 | 750 | 300 | 8988 | 12 | — |
| xsum | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| longform | 6000 | 750 | 750 | 300 | 9000 | 0 | — |
| domain (FinQA) | 5656 | 250 | 250 | 300 | 6156 | 0 | — |
| ner | 6000 | 750 | 750 | 300 | 8971 | 29 | — |

**Duplicate-pool provenance.** qa-B and sentiment-B read their originals' shuffled streams after
the first 9 000 rows and drop any row whose content is in the original's pool or TEST; no
adapter's TRAIN rows overlap another's. FinQA's TRAIN is 5 656 rather than 5 700 because 47 of its
6 203 usable rows repeat another row exactly (D-021).

## 4. Seed adapters

| Adapter | Source | Steps | First / final train loss | Metric | Zero-shot TEST | Adapter TEST | Zero-shot VAL | Adapter VAL |
|---|---|---|---|---|---|---|---|---|
| seed-sum | run 1 (loaded) | 375 | 1.911 / 1.538 | ROUGE-L | 0.2490 | 0.2905 | 0.2066 | 0.2563 |
| seed-qa | run 1 (loaded) | 375 | 2.462 / 0.014 | token F1 | 0.4160 | 0.8549 | 0.4535 | 0.8760 |
| seed-sentiment | run 1 (loaded) | 375 | 2.233 / 0.002 | exact match | 0.8967 | 0.9533 | 0.9067 | 0.9600 |
| seed-qa-B | trained | 375 | 1.767 / 0.090 | token F1 | 0.4159 | 0.8718 | 0.4279 | 0.8973 |
| seed-instr | trained | 375 | 2.416 / 1.202 | ROUGE-L | 0.2296 | 0.2899 | 0.2734 | 0.3301 |
| seed-dialog | trained | 375 | 3.413 / 1.648 | ROUGE-L | 0.0915 | 0.1761 | 0.1060 | 0.1877 |

Every seed adapter beats the frozen backbone on its own task. Full loss and validation
trajectories are in `decision_study/initial_adapters.json`.

## 5. Fusion selection (Phase 2, VALIDATION)

Six ordered pairs of sum / qa / sentiment; 100 fusion steps (F-e: none). "Rel." is fused ÷
standalone on the new task; "old drop" is the relative loss on the old task. A pair is within both
limits when Rel. ≥ 0.98 and old drop ≤ 1 %.

| Config | Storage | KD λ | α | Mean Rel. | Mean old drop | Pairs within both |
|---|---|---|---|---|---|---|
| F-0 (run 1's setting; reference, not selectable) | SVD rank 16 | 1.0 | softmax | 0.992 | 1.2 % | 2/6 |
| F-a | lossless | 0.1 | softmax | 0.971 | 1.2 % | 2/6 |
| F-b | lossless | 0 | softmax | 0.963 | 7.7 % | 1/6 |
| **F-c ← chosen** | lossless | 0.1 | free | 0.957 | **1.1 %** | 1/6 |
| F-e | lossless | — | uniform average, no training | 0.991 | 1.2 % | **3/6** |

**Chosen: F-c. `fusion_viable` = false.** No config kept the mean old drop at or below 1 %, so the
written rule fell back to the lowest mean forgetting: F-c, by 0.1 points over F-e. F-c has one
pair within both limits; `fusion_viable` needs two. On every other measure the training-free
uniform average (F-e) matched or beat the trained configs, at a fraction of the memory.

<details><summary>All 30 config × pair measurements</summary>

L_new and L_KD are the first and last 10-step window means; α is the mean coefficient of the old /
new component; "SVD err" is the mean per-layer relative error of a rank-16 SVD (the actual one for
F-0, what it would have been for the lossless configs).

| Config | Pair (new→old) | New fused | New alone | Rel. | Old fused | Old alone | Old drop | Within both | L_new | L_KD | α (old/new) | Stored MB | Peak GiB | SVD err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F-0 | sum→qa | 0.2505 | 0.2563 | 0.977 | 0.9049 | 0.8760 | 0.0% |  | 1.66→1.51 | 0.225→0.020 | 0.53/0.47 | 74 | 15.8 | 0.458 |
| F-0 | sum→sentiment | 0.2478 | 0.2563 | 0.967 | 0.9600 | 0.9600 | 0.0% |  | 1.65→1.50 | 0.231→0.008 | 0.47/0.53 | 74 | 15.8 | 0.361 |
| F-0 | qa→sum | 0.8723 | 0.8760 | 0.996 | 0.2497 | 0.2563 | 2.6% |  | 0.19→0.27 | 0.145→0.016 | 0.57/0.43 | 74 | 13.3 | 0.463 |
| F-0 | qa→sentiment | 0.8972 | 0.8760 | 1.024 | 0.9533 | 0.9600 | 0.7% | ✓ | 0.19→0.27 | 0.132→0.009 | 0.52/0.48 | 74 | 13.3 | 0.404 |
| F-0 | sentiment→sum | 0.9533 | 0.9600 | 0.993 | 0.2459 | 0.2563 | 4.0% |  | 0.02→0.02 | 0.102→0.008 | 0.60/0.40 | 74 | 13.3 | 0.394 |
| F-0 | sentiment→qa | 0.9533 | 0.9600 | 0.993 | 0.9011 | 0.8760 | 0.0% | ✓ | 0.02→0.03 | 0.373→0.010 | 0.61/0.39 | 74 | 13.3 | 0.397 |
| F-a | sum→qa | 0.2326 | 0.2563 | 0.907 | 0.8836 | 0.8760 | 0.0% |  | 1.64→1.50 | 0.284→0.037 | 0.36/0.64 | 148 | 15.8 | 0.309 |
| F-a | sum→sentiment | 0.2427 | 0.2563 | 0.947 | 0.9533 | 0.9600 | 0.7% |  | 1.63→1.49 | 0.236→0.025 | 0.36/0.64 | 148 | 15.8 | 0.261 |
| F-a | qa→sum | 0.8612 | 0.8760 | 0.983 | 0.2410 | 0.2563 | 5.9% |  | 0.19→0.24 | 0.191→0.069 | 0.49/0.51 | 148 | 13.3 | 0.441 |
| F-a | qa→sentiment | 0.8864 | 0.8760 | 1.012 | 0.9600 | 0.9600 | 0.0% | ✓ | 0.19→0.26 | 0.175→0.034 | 0.49/0.51 | 148 | 13.3 | 0.380 |
| F-a | sentiment→sum | 0.9400 | 0.9600 | 0.979 | 0.2554 | 0.2563 | 0.4% |  | 0.03→0.02 | 0.104→0.027 | 0.54/0.46 | 148 | 13.3 | 0.407 |
| F-a | sentiment→qa | 0.9600 | 0.9600 | 1.000 | 0.8816 | 0.8760 | 0.0% | ✓ | 0.02→0.02 | 0.281→0.020 | 0.55/0.45 | 148 | 13.3 | 0.402 |
| F-b | sum→qa | 0.2383 | 0.2563 | 0.930 | 0.7051 | 0.8760 | 19.5% |  | 1.64→1.50 | — | 0.35/0.65 | 148 | 15.8 | 0.296 |
| F-b | sum→sentiment | 0.2399 | 0.2563 | 0.936 | 0.9600 | 0.9600 | 0.0% |  | 1.63→1.49 | — | 0.34/0.66 | 148 | 15.8 | 0.252 |
| F-b | qa→sum | 0.8448 | 0.8760 | 0.964 | 0.2046 | 0.2563 | 20.2% |  | 0.19→0.22 | — | 0.46/0.54 | 148 | 13.4 | 0.422 |
| F-b | qa→sentiment | 0.8645 | 0.8760 | 0.987 | 0.9533 | 0.9600 | 0.7% | ✓ | 0.20→0.26 | — | 0.48/0.52 | 148 | 13.4 | 0.374 |
| F-b | sentiment→sum | 0.9400 | 0.9600 | 0.979 | 0.2620 | 0.2563 | 0.0% |  | 0.03→0.02 | — | 0.52/0.48 | 148 | 13.3 | 0.404 |
| F-b | sentiment→qa | 0.9400 | 0.9600 | 0.979 | 0.8246 | 0.8760 | 5.9% |  | 0.02→0.03 | — | 0.52/0.48 | 148 | 13.3 | 0.407 |
| F-c | sum→qa | 0.2331 | 0.2563 | 0.910 | 0.8823 | 0.8760 | 0.0% |  | 1.58→1.48 | 0.296→0.037 | 0.78/1.13 | 148 | 15.8 | 0.370 |
| F-c | sum→sentiment | 0.2333 | 0.2563 | 0.910 | 0.9600 | 0.9600 | 0.0% |  | 1.56→1.47 | 0.229→0.019 | 0.81/1.16 | 148 | 15.8 | 0.322 |
| F-c | qa→sum | 0.8463 | 0.8760 | 0.966 | 0.2483 | 0.2563 | 3.1% |  | 0.21→0.25 | 0.213→0.067 | 0.91/0.88 | 148 | 13.3 | 0.461 |
| F-c | qa→sentiment | 0.8480 | 0.8760 | 0.968 | 0.9467 | 0.9600 | 1.4% |  | 0.23→0.20 | 0.134→0.031 | 0.90/0.90 | 148 | 13.3 | 0.391 |
| F-c | sentiment→sum | 0.9533 | 0.9600 | 0.993 | 0.2517 | 0.2563 | 1.8% |  | 0.09→0.02 | 0.147→0.022 | 0.98/0.89 | 148 | 13.3 | 0.426 |
| F-c | sentiment→qa | 0.9533 | 0.9600 | 0.993 | 0.8742 | 0.8760 | 0.2% | ✓ | 0.04→0.04 | 0.713→0.036 | 0.97/0.83 | 148 | 13.3 | 0.420 |
| F-e | sum→qa | 0.2566 | 0.2563 | 1.001 | 0.8860 | 0.8760 | 0.0% | ✓ | — | — | 0.50/0.50 | 148 | 3.2 | 0.469 |
| F-e | sum→sentiment | 0.2544 | 0.2563 | 0.993 | 0.9533 | 0.9600 | 0.7% | ✓ | — | — | 0.50/0.50 | 148 | 3.2 | 0.414 |
| F-e | qa→sum | 0.8875 | 0.8760 | 1.013 | 0.2595 | 0.2563 | 0.0% | ✓ | — | — | 0.50/0.50 | 148 | 3.2 | 0.469 |
| F-e | qa→sentiment | 0.8361 | 0.8760 | 0.954 | 0.9533 | 0.9600 | 0.7% |  | — | — | 0.50/0.50 | 148 | 3.2 | 0.416 |
| F-e | sentiment→sum | 0.9533 | 0.9600 | 0.993 | 0.2517 | 0.2563 | 1.8% |  | — | — | 0.50/0.50 | 148 | 3.2 | 0.414 |
| F-e | sentiment→qa | 0.9533 | 0.9600 | 0.993 | 0.8408 | 0.8760 | 4.0% |  | — | — | 0.50/0.50 | 148 | 3.2 | 0.416 |

Run 1's stored checks for the same pairs (16 VALIDATION examples, SVD storage, λ = 1, installed
at 2× its value — D-017):

| Pair (new→old) | New fused / alone | Old fused / alone |
|---|---|---|
| qa→sentiment | 0.902 / 0.948 | 0.938 / 0.938 |
| qa→sum | 0.857 / 0.948 | 0.229 / 0.251 |
| sentiment→qa | 1.000 / 0.938 | 0.948 / 0.948 |
| sentiment→sum | 0.938 / 0.938 | 0.250 / 0.251 |
| sum→qa | 0.242 / 0.251 | 0.948 / 0.948 |
| sum→sentiment | 0.220 / 0.251 | 0.938 / 0.938 |

</details>

What the grid shows:

- **KD is what limits forgetting.** Without it (F-b, λ = 0) two pairs lost about 20 % on the old
  task (sum→qa 19.5 %, qa→sum 20.2 %). With λ = 0.1 (F-a, F-c) the KD loss still fell 3–20×,
  while the new-task loss fell a little when sum was the new task and rose for qa; at λ = 1 (F-0)
  KD fell 9–37×.
- **Free α stayed close to 1** (component means 0.78–1.16; above 1 only for the new component when
  sum was the new task) and did not raise the new-task score over softmax α.
- **Lossless storage removed a large approximation** (the would-be rank-16 SVD error is 0.25–0.47
  per layer) but did not on its own bring fused adapters within 2 %.
- **Memory.** Trained fusion peaked at 13.3–15.8 GiB here; the uniform average at 3.2 GiB, a
  quarter to a fifth of that.

## 6. Supervision (Phase 3)

30 ordered pairs of the six seeds with F-c; labels from the measured task metric on 100–150
VALIDATION examples (D-019).

| Pair type | Pairs | Beneficial (measured) | Beneficial (loss proxy) | Labels agree |
|---|---|---|---|---|
| same task (qa ↔ qa-B) | 2 | 2 | 0 | 0/2 |
| related | 2 | 0 | 0 | 2/2 |
| unrelated | 26 | 8 | 0 | 18/26 |
| **all** | **30** | **10 (33 %)** | **0** | **20/30** |

The beneficial pairs: qa→qa-B, qa-B→qa, sentiment→qa, sentiment→qa-B, sentiment→dialog,
qa-B→sum, instr→sum, instr→qa, instr→sentiment, instr→qa-B. The loss proxy called every pair
harmful, so it disagrees with the measured metric on all ten beneficial pairs, including both
same-task pairs.

The labels sit close to their own noise: the relative new-task score runs from 0.902 to 1.093 with
a median of 0.977, against the 0.98 line. Reusing an existing adapter for another seed task was
within the 2 % margin in only 2 of 30 pairs (the same-task pairs); the median reuse gap was −23.5 %.

## 7. Utility network (Phase 4)

The gate passed (30 pairs, 33 % beneficial), so the network was fitted; `probe_rule` was not
needed as a fallback. Out-of-fold numbers are leave-one-new-task-out over the six seed tasks.

| Model | Setting | Features | OOF decision accuracy | OOF MSE: primary | forgetting | confidence | memory | communication | long-term loss | τ_u |
|---|---|---|---|---|---|---|---|---|---|---|
| `amaun_v0` (first fit, D-019) | [64, 64], 200 epochs | 54 | 0.333 | 4.761 | 0.750 | 0.106 | 0.001 | 0.000 | 0.139 | −1.207 |
| `nn_small` (D-029) | [8], epochs ≤ 20 → 20 | 32 | 0.000 | 2.527 | 1.146 | 0.907 | 0.044 | 0.036 | 0.449 | — |
| **`ridge` (D-029) ← selected** | λ ∈ {0.1, 1, 10, 100} → 100 | 32 | 0.333 | 3.686 | 1.052 | 0.256 | 0.000 | 0.000 | 0.269 | 0.903 |

MSEs are in z units: predicting the mean scores 1.0. Decision accuracy is the share of the six
held-out tasks for which the highest predicted utility picked the best measured candidate (chance
is 1 in 5).

**What happened.** The first fit memorised its 30 pairs (training MSE 0.03 against 4.8 out of
fold). 22 of its 54 features were constant across the pairs, and the normaliser turned any later
change in them into values near 10⁸, which would have broken the Phase 6 stream. D-029 replaced it
mid-run with two smaller models on the 32 features that vary, each tuned by an inner
leave-one-task-out loop and compared by nested CV under a rule fixed in advance. Ridge won on
decision accuracy. Both chosen settings sit at the edge of their ranges (λ = 100 in four of six
folds and in the final fit; 20 epochs in four of six folds and in the final fit), and both models
remain worse than the mean on the primary head.
The first fit was refitted identically and kept as policy `amaun_v0`.

Out of fold, every model got right only the two same-task duplicates (qa → seed-qa-B and qa-B →
seed-qa). The reuse gate (ridge on SUPPORT probe losses) was trained; the fixed-rule threshold on
`semantic.d_task_sq` is −0.048.

## 8. Evaluation tasks (Phase 5, TEST)

For each task: the dedicated adapter's eligibility on VALIDATION (it must beat zero-shot by ≥ 5 %
relative), then every measured option on TEST. An option is **acceptable** when it reaches the
equivalence bar (0.98 × CREATE; for ELI5, 98 % of CREATE's gain over zero-shot, D-027) and no old
task drops by more than 1 %. **Inconclusive** means the 95 % paired-bootstrap CI of (option − bar)
contains 0. The **oracle** is the cheapest acceptable option. Training minutes include the
dedicated adapter for CREATE and FUSE (FUSE adds the fusion). Stored size after the option is
443 MB for BACKBONE and REUSE (the six seeds), and 517 MB for CREATE and FUSE (a lossless fused
adapter has twice the rank of the one it replaces).

### FinQA (`domain`): unlike every seed. Eligible: 0.2356 vs 0.0122 zero-shot (token F1)

| Option | TEST token F1 | CI of (option − bar) | Old-task change | Adapters | Train min | Acceptable | Inconclusive |
|---|---|---|---|---|---|---|---|
| **CREATE (oracle)** | **0.2030** | — | — | 7 | 48 | ✓ | — |
| FUSE seed-sum | 0.1722 | [−0.056, +0.004] | sum −2.6 % | 6 | 51 | ✗ | yes |
| FUSE seed-qa | 0.1493 | [−0.081, −0.019] | qa −1.7 % | 6 | 51 | ✗ | no |
| FUSE seed-qa-B | 0.1417 | [−0.090, −0.026] | qa-B −7.2 % | 6 | 51 | ✗ | no |
| REUSE seed-dialog | 0.0841 | [−0.149, −0.079] | — | 6 | 0 | ✗ | no |
| REUSE seed-instr | 0.0775 | [−0.158, −0.083] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa | 0.0723 | [−0.162, −0.088] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa-B | 0.0712 | [−0.164, −0.092] | — | 6 | 0 | ✗ | no |
| REUSE seed-sum | 0.0576 | [−0.174, −0.105] | — | 6 | 0 | ✗ | no |
| REUSE seed-sentiment | 0.0522 | [−0.184, −0.111] | — | 6 | 0 | ✗ | no |
| BACKBONE | 0.0111 | [−0.224, −0.152] | — | 6 | 0 | ✗ | no |

Bar 0.1989. Under the cap of 6 the oracle is REUSE seed-dialog (the best feasible option within
the forgetting limit). As expected, FinQA needs its own adapter.

### sentiment-B: same task as `seed-sentiment`. Eligible: 0.9867 vs 0.9333 (exact match; +5.7 %, just over the bar)

| Option | TEST accuracy | CI of (option − bar) | Old-task change | Adapters | Train min | Acceptable | Inconclusive |
|---|---|---|---|---|---|---|---|
| **REUSE seed-sentiment (oracle)** | **0.9533** | [+0.009, +0.042] | — | 6 | 0 | ✓ | no |
| CREATE | 0.9467 | — | — | 7 | 28 | ✓ | — |
| REUSE seed-qa | 0.9433 | [−0.004, +0.035] | — | 6 | 0 | ✓ | yes |
| REUSE seed-qa-B | 0.9400 | [−0.011, +0.032] | — | 6 | 0 | ✓ | yes |
| REUSE seed-instr | 0.9167 | [−0.038, +0.015] | — | 6 | 0 | ✗ | yes |
| REUSE seed-dialog | 0.9100 | [−0.048, +0.009] | — | 6 | 0 | ✗ | yes |
| FUSE seed-sentiment | 0.9100 | [−0.047, +0.012] | sentiment −4.5 % | 6 | 30 | ✗ | yes |
| FUSE seed-qa | 0.9100 | [−0.045, +0.012] | qa 0.0 % | 6 | 30 | ✗ | yes |
| BACKBONE | 0.8967 | [−0.058, −0.005] | — | 6 | 0 | ✗ | no |
| REUSE seed-sum | 0.8100 | [−0.161, −0.078] | — | 6 | 0 | ✗ | no |

Bar 0.9277. Reusing the original sentiment adapter is free and best. The backbone is already
strong here, so several unrelated adapters are also acceptable on point estimates.

### xsum: related to `seed-sum`. Eligible: 0.3007 vs 0.2029 (ROUGE-L)

| Option | TEST ROUGE-L | CI of (option − bar) | Old-task change | Adapters | Train min | Acceptable | Inconclusive |
|---|---|---|---|---|---|---|---|
| **CREATE (oracle)** | **0.2624** | — | — | 7 | 34 | ✓ | — |
| FUSE seed-dialog | 0.2562 | [−0.018, +0.016] | dialog −9.2 % | 6 | 36 | ✗ | yes |
| FUSE seed-sentiment | 0.2500 | [−0.022, +0.008] | sentiment 0.0 % | 6 | 36 | ✗ | yes |
| FUSE seed-sum | 0.2491 | [−0.024, +0.007] | **sum −28.0 %** | 6 | 36 | ✗ | yes |
| REUSE seed-dialog | 0.2218 | [−0.056, −0.013] | — | 6 | 0 | ✗ | no |
| REUSE seed-instr | 0.1961 | [−0.080, −0.042] | — | 6 | 0 | ✗ | no |
| BACKBONE | 0.1928 | [−0.083, −0.046] | — | 6 | 0 | ✗ | no |
| REUSE seed-sentiment | 0.1629 | [−0.119, −0.069] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa | 0.1560 | [−0.122, −0.081] | — | 6 | 0 | ✗ | no |
| REUSE seed-sum | 0.1372 | [−0.139, −0.101] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa-B | 0.1362 | [−0.140, −0.102] | — | 6 | 0 | ✗ | no |

Bar 0.2571. Every FUSE option comes within a few hundredths of CREATE (all CIs contain 0), but
fusing into the related adapter (`seed-sum`) costs that adapter 28 % of its own score. Reusing
`seed-sum` as-is is among the worst options. Under the cap the oracle is FUSE seed-sentiment.

### longform (ELI5): related to `seed-sum`. **Non-discriminating**: dedicated −0.1321 vs zero-shot −0.1123 (rescaled BERTScore-F1)

| Option | TEST BERTScore-F1 (rescaled) | CI of (option − bar) | Old-task change | Acceptable | Inconclusive |
|---|---|---|---|---|---|
| **REUSE seed-dialog (oracle)** | **−0.0165** | [+0.103, +0.171] | — | ✓ | no |
| BACKBONE | −0.0972 | [+0.026, +0.086] | — | ✓ | no |
| REUSE seed-instr | −0.1244 | [−0.010, +0.068] | — | ✓ | yes |
| REUSE seed-sum | −0.1264 | [−0.013, +0.064] | — | ✓ | yes |
| REUSE seed-qa | −0.1312 | [−0.015, +0.061] | — | ✓ | yes |
| FUSE seed-instr | −0.1443 | [−0.027, +0.044] | instr −4.2 % | ✗ | yes |
| CREATE | −0.1550 | — | — | ✗ | — |
| REUSE seed-qa-B | −0.1641 | [−0.050, +0.030] | — | ✗ | yes |
| FUSE seed-sum | −0.1779 | [−0.064, +0.015] | sum −4.4 % | ✗ | yes |
| REUSE seed-sentiment | −0.2208 | [−0.116, −0.019] | — | ✗ | no |

Bar −0.1539 (gain retention, D-027). The dedicated ELI5 adapter scores *below* the frozen backbone,
as ROUGE-L did in run 1, so ELI5 cannot separate the options and is left out of the policy tables.
The conversational `seed-dialog` adapter is by far the best ELI5 option.

### NER (CoNLL-2003): unlike every seed. Eligible: 0.9300 vs 0.1931 (entity F1)

| Option | TEST entity F1 | CI of (option − bar) | Old-task change | Adapters | Train min | Acceptable | Inconclusive |
|---|---|---|---|---|---|---|---|
| **CREATE (oracle)** | **0.9131** | — | — | 7 | 11 | ✓ | — |
| FUSE seed-sum | 0.8405 | [−0.083, −0.028] | sum −1.4 % | 6 | 13 | ✗ | no |
| FUSE seed-instr | 0.8039 | [−0.126, −0.059] | instr −5.9 % | 6 | 12 | ✗ | no |
| FUSE seed-qa | 0.7804 | [−0.149, −0.073] | qa −1.5 % | 6 | 12 | ✗ | no |
| REUSE seed-dialog | 0.2603 | [−0.685, −0.591] | — | 6 | 0 | ✗ | no |
| REUSE seed-sentiment | 0.2246 | [−0.718, −0.622] | — | 6 | 0 | ✗ | no |
| REUSE seed-instr | 0.1894 | [−0.749, −0.665] | — | 6 | 0 | ✗ | no |
| BACKBONE | 0.1633 | [−0.774, −0.693] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa | 0.1474 | [−0.791, −0.703] | — | 6 | 0 | ✗ | no |
| REUSE seed-sum | 0.1086 | [−0.826, −0.747] | — | 6 | 0 | ✗ | no |
| REUSE seed-qa-B | 0.0833 | [−0.848, −0.774] | — | 6 | 0 | ✗ | no |

Bar 0.8948. NER needs its own adapter. Fusion recovers most of it (0.78–0.84), but not within 2 %.

**Every FUSE option a policy chose was measured** in all five tasks; the measured set was the top
two candidates by SUPPORT probe-loss gain plus every policy's choice. Per-example predictions and
scores for every evaluation are in `decision_study/predictions/` (292 files).

## 9. Policy tables (4 eligible tasks)

**Correct** means the executed option is acceptable and costs no more than the oracle. **Shortfall**
is how far below the equivalence bar the executed option scored. Cost excess is measured against
the oracle (adapters; training minutes). The last column is the pooled per-example difference from
`always_separate` (paired bootstrap, 1 000 resamples). Pooling mixes metrics and is descriptive.

### No adapter cap

| Policy | Executed (FinQA / sentiment-B / xsum / NER) | Correct | Acceptable | Mean shortfall | Excess adapters | Excess train (min) | Max forgetting | vs `always_separate` [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| **`amaun`** (ridge, D-029) | CREATE / REUSE qa-B / CREATE / CREATE | **4/4** | 4/4 | 0.000 | +0.00 | +0.0 | 0.0 % | −0.002 [−0.009, +0.005] |
| `amaun_train_first` | CREATE / BACKBONE / CREATE / CREATE | 3/4 | 3/4 | 0.008 | +0.00 | +0.0 | 0.0 % | −0.014 [−0.023, −0.007] |
| `amaun_v0` (first fit) | FUSE sum / REUSE qa-B / FUSE sum / FUSE sum | 1/4 | 1/4 | 0.022 | −0.75 | +1.8 | 28.0 % | −0.033 [−0.048, −0.020] |
| `probe_rule` | REUSE qa-B / REUSE sentiment / REUSE dialog / CREATE | 2/4 | 2/4 | 0.041 | −0.50 | −20.4 | 0.0 % | −0.042 [−0.054, −0.029] |
| `fixed_rule` | CREATE / FUSE sentiment / FUSE sum / CREATE | 2/4 | 2/4 | 0.006 | −0.25 | +8.1 | 28.0 % | −0.012 [−0.022, −0.003] |
| `always_separate` | CREATE ×4 | 3/4 | 4/4 | 0.000 | +0.25 | +7.0 | 0.0 % | — |
| `always_merge` | FUSE qa / FUSE sentiment / FUSE sum / FUSE instr | 0/4 | 0/4 | 0.042 | −0.75 | +9.1 | 28.0 % | −0.059 [−0.076, −0.043] |
| `reuse_only` | REUSE qa-B / REUSE sentiment / REUSE dialog / REUSE sentiment | 1/4 | 1/4 | 0.208 | −0.75 | −23.2 | 0.0 % | −0.238 [−0.264, −0.212] |
| `fuse_only` | FUSE qa / FUSE sentiment / FUSE dialog / FUSE qa | 0/4 | 0/4 | 0.046 | −0.75 | +9.1 | 9.2 % | −0.065 [−0.083, −0.048] |

### Cap of 6 adapters (CREATE infeasible; a CREATE decision executes as BACKBONE, D-018)

| Policy | Executed (FinQA / sentiment-B / xsum / NER) | Correct | Acceptable | Mean shortfall | Excess train (min) | Max forgetting | vs `always_separate` [95 % CI] |
|---|---|---|---|---|---|---|---|
| `amaun` | BACKBONE / REUSE qa-B / BACKBONE / BACKBONE | 1/4 | 1/4 | 0.246 | −9.0 | 0.0 % | +0.012 [+0.007, +0.019] |
| `amaun_train_first` | BACKBONE ×4 | 0/4 | 0/4 | 0.254 | −9.0 | 0.0 % | +0.000 [+0.000, +0.000] |
| `amaun_v0` | FUSE sum / REUSE qa-B / FUSE sum / FUSE sum | 1/4 | 1/4 | 0.022 | +16.1 | 28.0 % | +0.260 [+0.235, +0.288] |
| `probe_rule` | REUSE qa-B / REUSE sentiment / REUSE dialog / BACKBONE | 1/4 | 1/4 | 0.224 | −9.0 | 0.0 % | +0.038 [+0.028, +0.050] |
| `fixed_rule` | BACKBONE / FUSE sentiment / FUSE sum / BACKBONE | 0/4 | 0/4 | 0.236 | +7.5 | 28.0 % | +0.012 [+0.001, +0.023] |
| `always_separate` | BACKBONE ×4 | 0/4 | 0/4 | 0.254 | −9.0 | 0.0 % | — |
| `always_merge` | FUSE qa / FUSE sentiment / FUSE sum / FUSE instr | 0/4 | 0/4 | 0.042 | +23.4 | 28.0 % | +0.234 [+0.208, +0.261] |
| `reuse_only` | REUSE qa-B / REUSE sentiment / REUSE dialog / REUSE sentiment | 1/4 | 1/4 | 0.208 | −9.0 | 0.0 % | +0.055 [+0.037, +0.076] |
| `fuse_only` | FUSE qa / FUSE sentiment / FUSE dialog / FUSE qa | 0/4 | 0/4 | 0.046 | +23.3 | 9.2 % | +0.229 [+0.202, +0.255] |

No policy is cap-aware: each decides as if CREATE were available. At the cap, every CREATE turns
into the backbone, which is weak on three of the four tasks. Only fusion recovers quality there
(shortfall about 0.02–0.05 instead of about 0.25), and it pays with up to 28 % forgetting.

**Why ridge-`amaun` scored 4/4.** Its τ_u (0.903) sat above every utility it produced, so it never
fused. Its decisions were: the reuse gate firing on sentiment-B (REUSE seed-qa-B, acceptable at no
cost), and CREATE everywhere else. On these four tasks "CREATE unless reuse is clearly good" was the
right rule. The learned utility did not get to rank any FUSE candidate.

## 10. Sequential stream (Phase 6)

Eleven tasks arrive in a fixed order into an empty repository; each decision changes it. CREATE
reuses the adapter already trained for that task (R2). Scores are TEST primary metrics: ROUGE-L,
token F1, accuracy, rescaled BERTScore and entity F1 are mixed, so means are descriptive.

| Task | `amaun` no cap | `amaun` cap 6 | `always_separate` no cap | `always_separate` cap 6 |
|---|---|---|---|---|
| sum | CREATE 0.2905 | CREATE 0.2905 | CREATE 0.2905 | CREATE 0.2905 |
| qa | **BACKBONE 0.4160** | **BACKBONE 0.4160** | CREATE 0.8549 | CREATE 0.8549 |
| sentiment | **BACKBONE 0.8967** | **BACKBONE 0.8967** | CREATE 0.9533 | CREATE 0.9533 |
| instr | CREATE 0.2899 | CREATE 0.2899 | CREATE 0.2899 | CREATE 0.2899 |
| dialog | CREATE 0.1761 | CREATE 0.1761 | CREATE 0.1761 | CREATE 0.1761 |
| qa-B | CREATE 0.8718 | CREATE 0.8718 | CREATE 0.8718 | CREATE 0.8718 |
| sentiment-B | REUSE qa-B 0.9400 | REUSE qa-B 0.9400 | CREATE 0.9467 | BACKBONE 0.8967 |
| xsum | CREATE 0.2624 | CREATE 0.2624 | CREATE 0.2624 | BACKBONE 0.1928 |
| longform | CREATE −0.1550 | CREATE −0.1550 | CREATE −0.1550 | BACKBONE −0.0972 |
| domain | CREATE 0.2030 | BACKBONE (cap) 0.0111 | CREATE 0.2030 | BACKBONE 0.0111 |
| ner | CREATE 0.9131 | BACKBONE (cap) 0.1633 | CREATE 0.9131 | BACKBONE 0.1633 |
| **Decisions** | 8 CREATE, 2 BACKBONE, 1 REUSE | 6 CREATE, 4 BACKBONE, 1 REUSE | 11 CREATE | 6 CREATE, 5 BACKBONE |
| **Final repository** | 8 adapters, 591 MB | 6, 443 MB | 11, 812 MB | 6, 443 MB |
| **Mean final score** | 0.4640 | 0.3784 | 0.5097 | 0.4185 |

No run fused, so every task's arrival score equals its final score: zero forgetting in all four
runs. `amaun` used three fewer adapters than `always_separate` but lost 0.046 in mean score (0.040
under the cap). Almost all of the loss is qa and sentiment at steps 1–2. With only `seed-sum` in
the repository, the backbone's SUPPORT probe matched that adapter's, and the specified decision
order (item 15) then chooses BACKBONE without comparing it to CREATE. qa's own adapter would have
scored 0.855 instead of 0.416. Per-step trajectories, repository sizes and decision reasons are in
`decision_study/stream_*.json`.

## 11. Repeatability and memory (Phase 7)

Three identical cycles (load adapter → evaluate → fuse with F-c → evaluate fused → release):

| Cycle | Standalone | Fused | Fused adapter hash | Fusion (s) | Peak GiB |
|---|---|---|---|---|---|
| 0 | 0.2497 | 0.2263 | `0d9052812c7f` | 144 | 15.8 |
| 1 | 0.2497 | 0.2263 | `0d9052812c7f` | 144 | 15.8 |
| 2 | 0.2497 | 0.2263 | `0d9052812c7f` | 144 | 15.8 |

**Fully repeatable**: identical fused weights and scores across cycles (run 1's three cycles gave
three different scores). No growth in allocated or reserved GPU memory, live tensors or CPU RSS:
verdict `stable`. Preflight's highest peak was FinQA's fusion step at 20.4 GiB, within the 22.5 GB
limit, and no step ran out of memory during the run.

## 12. Verdicts

**Does fusion work once fixed?** **Not to the study's standard.** Lossless storage, a light KD
term and free α were all run. No FUSE option on any eligible task came within 2 % of CREATE
while keeping every old task within 1 %, and the selected config (F-c) kept both limits on only
one of six seed pairs (`fusion_viable` = false). Fusion does preserve most of a task: within 0.013 of
CREATE on xsum (CIs contain 0) and 0.78–0.84 vs 0.91 on NER. But it costs the old
adapter up to 28 % (xsum into seed-sum). The training-free uniform average (F-e) was as good as any
trained config at a quarter to a fifth of the memory, so the trained coefficients and KD added
nothing measurable here.

**Does the learned utility network beat `probe_rule` and `fixed_rule`?** **The pre-registered one
did not; the mid-run revision did on these four tasks, but not by learning to rank.** The first fit
(`amaun_v0`) was correct on 1 of 4 tasks, against 2 of 4 for `probe_rule` and `fixed_rule`. It
fused into seed-sum on three tasks, with 28 % forgetting. The ridge revision (D-029, chosen after
seeing the first fit) was correct on 4 of 4, but its threshold was high enough that it never used
the utility to fuse. It succeeded through the reuse gate and its CREATE default. Out of fold,
neither model beat predicting the mean. With 30 noisy pairs from six tasks, the utility network
did not learn a usable ranking.

**Does AMAUN beat `always_separate` when CREATE is not free?** **No.** With a cap of 6 in Phase 5,
`amaun` was ahead by a statistically clear but tiny +0.012 [+0.007, +0.019], because one task
(sentiment-B) had a good free reuse. It was correct on only 1 of 4, because no policy plans for the
cap. In the sequential stream, the realistic setting, it was worse: mean final score 0.378 against
0.419 under the cap and 0.464 against 0.510 without it, while saving three adapters. Without a cap
it matched `always_separate`'s quality on the four tasks (−0.002 [−0.009, +0.005]) with one adapter
fewer.

## 13. Limitations

- **Tier and scale.** `local_smoke`: Qwen2.5-1.5B, one seed (1234), one L4. No seed-to-seed
  variance is measured; CIs reflect TEST-example sampling only. Four policy-eligible tasks; pooled
  scores mix metrics.
- **Changes made during the run** (all recorded in `docs/decision_register.md`):
  - D-026, merged-weight batched evaluation: it overrides the prompt's "no speed optimisations"
    rule, and long greedy outputs are not bit-identical to the one-at-a-time path. Every option was
    scored the same way.
  - D-027, ELI5 gain retention: decided before any ELI5 VALIDATION or TEST row was read.
  - D-028, the Phase 4 crash fix.
  - D-029, the utility revision: chosen after seeing the first fit. `amaun_v0` is kept so the
    pre-registered model's result stands beside it.
- **Selection near-ties.** F-c won the Phase 2 fallback by 0.1 points of forgetting over F-e; with
  F-e, `fusion_viable` would have been true. Not re-run: that would be choosing on hindsight.
- **Labels and probes are noisy.** Supervision labels come from 100–150 VALIDATION examples, and
  their ratios centre on the 0.98 line. SUPPORT probes use 16 examples and saturate on easy tasks
  (sentiment-B: backbone and four candidates all scored 1.0). The loss proxy disagreed with the
  measured label on every beneficial pair.
- **Utility model.** 30 pairs from six tasks; out-of-fold error above the mean predictor for every
  model; both D-029 settings at the edge of their grids.
- **Policy design.** No policy accounts for the adapter cap (CREATE → BACKBONE, D-018). The
  `amaun` order picks BACKBONE whenever no existing adapter beats the backbone on SUPPORT, without
  comparing it to CREATE; that cost the stream qa and sentiment.
- **ELI5 is non-discriminating.** Its dedicated adapter scores below the backbone on rescaled
  BERTScore, so the "judgement call" task tests nothing in the policy tables.
- **Data changes.** FinQA TRAIN 5 656 instead of 5 700 (repeated rows, D-021). Training rows that
  repeat a TEST row were dropped (dialog 12, NER 29; D-025).
- **Left out to fit the budget** (D-024): more seeds; ablations A3 (fixed α) and A8 (global α);
  300-step fusion; GSM8K; `probe_rule` and the other baselines in the stream.
- **Tests.** The dummy-tier suite (433 tests) passes. Some test files the spec listed were not
  written as such: `test_full_study.py` (R1–R3 against the naive loop, resume), the fusion, data and
  state groups. Their behaviour was exercised by full dummy-tier runs (including a kill-and-resume
  and a surgery-and-resume), not by those named tests.

## Acceptance checklist (`docs/l4_next_run_prompt.md`)

| Item | Status | Where |
|---|---|---|
| F1 `keep_components` used; would-be SVD error logged | ✓ | §5 (SVD err 0.25–0.47) |
| F2 λ ∈ {0, 0.1} run | ✓ | §5 F-b, F-a / F-c |
| F3 `free` α and uniform average run | ✓ | §5 F-c, F-e |
| F4 cost fields; cap-6 table | ✓ | §8, §9 |
| F5 new task and forgetting judged separately; inconclusive flagged | ✓ | §8 |
| F6 ELI5 on rescaled BERTScore; eligibility applied | ✓ (gain retention, D-027) | §8 |
| F7 BACKBONE in every choice set | ✓ | §8 |
| F8 measured-metric labels on 100–150 examples; proxy agreement | ✓ (20/30) | §6 |
| F9 per-example scores; bootstrap CIs | ✓ (one seed) | §8, §9 |
| F10 NER and duplicate tasks included | ✓ | §3, §8 |
| F11 30 supervision pairs; beneficial fraction | ✓ (33 %) | §6 |
| F12 sequential stream, 4 runs | ✓ | §10 |
| F13 FinQA validation = 250 | ✓ (TRAIN 5 656) | §3 |
| F14 deterministic mode; repeatability spread | ✓ (spread 0) | §1, §11 |
| F15 preflight memory check with the new storage | ✓ (20.4 GiB) | §11 |
| F16 measured targets, leave-one-task-out CV, τ from out-of-fold, fallback | ✓ | §7 |
| B1 run-1 splits reproduced; duplicates outside pools; leakage clean | ✓ | §3 |
| B2 fits the budget; R1–R3 applied and tested | fits ✓ (14.2 h); R1–R3 applied ✓; dedicated naive-loop test not written | §2, §13 |
| B3 every policy's FUSE choice measured | ✓ | §8 |
| G1 decision correct / shortfall / cost excess | ✓ | §9 |
| G3 `amaun` decision order as specified | ✓ | §9, §10 |
| G6–G7 new task specs, generation limits, greedy | ✓ (batched, D-026) | §1 |
| Fusion config chosen on VALIDATION by the written rule; F-0 reported | ✓ | §5 |
| `probe_rule` and `fixed_rule` both reported | ✓ | §9 |
| Left-out items in the decision register | ✓ (D-024) | §13 |
| Preflight projection ≤ 32 h; projected vs actual | ✓ (29.06 h) | §2 |
| Every dummy-tier test listed in Part A passes | partial: suite passes; some listed test files not written | §13 |

## Where everything is

| | |
|---|---|
| Run outputs | `results/local_smoke/l4_full_study/` (`decision_study/*.json`, `predictions/`, `fusion_logs/`, `metrics.jsonl`, `decisions.jsonl`) |
| Pre-D-029 Phase 4 outputs | `results/local_smoke/l4_full_study/utility_network.pre_d029/`, `decision_study/utility_supervision.json.pre_d029`, `decision_study/checkpoint/state.json.pre_d029` |
| Adapters | `adapters/local_smoke/l4_full_study/` (seeds; `_candidates/` holds the five dedicated evaluation-task adapters) |
| Logs and checksums | `results/local_smoke/l4_full_study_vm_logs/` (`study*.log`, `MANIFEST.sha256`) |
| Decisions | `docs/decision_register.md` D-017 to D-029 |
