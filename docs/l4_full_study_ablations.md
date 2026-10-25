# L4 full decision study — ablations

Ablations from `docs/experiment_matrix.md` that the L4 full study
(`docs/l4_full_study_results.md`) did not include, run afterwards against that run's own
measurements (decision D-030). The run itself was only read. Outputs:
`results/local_smoke/l4_full_study_ablations/ablations/`. Config:
`configs/experiments/l4_full_study_ablations.yaml`; code: `scripts/run_ablations.py`,
`src/amaun/continual/ablations.py`.

**Tier `local_smoke`, one seed: descriptive, not reportable.** Everything inherits the run's
criteria, its D-029 utility-model protocol, its evaluation path and its chosen fusion config (F-c);
nothing was re-selected. These results are separate from the run's verdicts.

| Ablation | Question (experiment matrix) | How it was run | Result in one line |
|---|---|---|---|
| A1 | does heterogeneous evidence help? (semantic features only) | offline replay | same decisions as the run |
| A6 | does adapter history help? (no historical features) | offline replay | identical to the run, as predicted |
| A7 | does cost information change selection? (no resource features) | offline replay | same decisions as the run |
| A12 | does the semantic task representation matter? (`token_stats`) | offline replay | same decisions as the run |
| A3 | does learnable α help? (α fixed) | 6 Phase 2 pairs, L4 | slightly worse than learned α (mean Rel. 0.950 vs 0.957; forgetting 1.9 % vs 1.1 %) |
| A8 | is layer-level α needed? (one global α) | 6 Phase 2 pairs, L4 | slightly better than layer-wise α (0.963; forgetting 0.95 %) |

Not run: A5 (the study has no repository-compression mechanism to switch off), the Phase 6 stream
for any variant (its decisions depend on live features of a changing repository), more seeds.

## 1. Identity checks

Both ran before anything was scored. Either one failing would have stopped the job.

| Check | Mac (CPU) | L4 VM |
|---|---|---|
| Replaying the run's frozen models (ridge for `amaun` and `amaun_train_first`, the first-fit network for `amaun_v0`) from the logged gate verdicts, probes, prescreen and features reproduces the logged utilities and decisions | ✓ max utility difference 1.2 × 10⁻⁷ (float32), 0 of 15 decisions differ | ✓ difference 0.0, 0 differ |
| Re-scoring the run's logged decisions with the shared scoring code reproduces every stored option, oracle, policy row and pooled table, both caps | ✓ 0 differences | ✓ 0 differences |

So any difference an ablation shows comes from the ablation, not from the replay or scoring
machinery. (The scoring code was moved out of the study unchanged into
`amaun.continual.study_scoring`; the second check is the proof of that on the run's data.)

## 2. Feature ablations: A1, A6, A7, A12

Each variant changes only the utility model's input features. It is refitted with the run's
D-029 protocol on the run's 30 Phase 3 supervision pairs: the small network and ridge are tuned
by an inner leave-one-task-out loop, compared by nested CV, the winner is picked by the fixed rule,
and τ_u is taken from its out-of-fold utilities. Then `amaun` and `amaun_train_first` re-decide
each Phase 5 task. A12 recomputes `d_task_sq`, `d_task_l2` and `cos_task` from support-set word
statistics of the new task and of each candidate's task; every other feature is unchanged.

### Utility models

| Variant | Input features | After dropping constants | Selected | OOF decision accuracy (small net / ridge) | OOF primary-score MSE (small net / ridge) | τ_u |
|---|---|---|---|---|---|---|
| Run (D-029) | 54 | 32 | ridge (λ 100) | 0.000 / 0.333 | 2.527 / 3.686 | 0.903 |
| A1 semantic only | 6 | 6 | small net (20 epochs) | 0.333 / 0.333 | 2.017 / 2.325 | 0.130 |
| A6 no historical | 46 | 32 | ridge (λ 100) | 0.000 / 0.333 | 2.527 / 3.686 | 0.903 |
| A7 no resource | 47 | 30 | small net (20 epochs) | **0.500** / 0.333 | 2.548 / 3.039 | 0.368 |
| A12 `token_stats` | 54 | 32 | ridge (λ 100) | 0.000 / 0.333 | 2.371 / 3.310 | 0.929 |

OOF = out of fold, over the six seed tasks; MSE in z units (predicting the mean scores 1.0; chance
decision accuracy is 1 in 5). A1 chose the small net on the rule's second criterion (lower MSE at
equal accuracy).

- **A6 is the run's model to every digit.** All 8 historical features were constant on the 30
  pairs, and the D-029 protocol already dropped constant features, so removing the family removes
  nothing the model used.
- **A7 gives the best out-of-fold ranking seen in the study:** 3 of 6 held-out tasks (qa, qa-B,
  sentiment), against at most 2 of 6 for every other model. Every ridge fit got only the two
  same-task duplicates (qa, qa-B); the small network got 0 of 6 with the run's, A6's and A12's
  features. A7 drops the two resource features that varied, `Comm_i` and `Latency_i`.
- **Semantic features alone (A1) predict no worse** than all 32: equal decision accuracy and lower
  error than the run's model.
- Every model is still worse than predicting the mean on the primary head.

### Phase 5 decisions and outcomes

**All 4 variants make exactly the run's 10 decisions** (5 tasks × `amaun`, `amaun_train_first`):

| Policy | FinQA | sentiment-B | xsum | ELI5* | NER | Correct, no cap | Correct, cap 6 |
|---|---|---|---|---|---|---|---|
| `amaun` (run, A1, A6, A7, A12) | CREATE | REUSE seed-qa-B | CREATE | CREATE | CREATE | 4/4 | 1/4 |
| `amaun_train_first` (run, A1, A6, A7, A12) | CREATE | BACKBONE | CREATE | CREATE | CREATE | 3/4 | 0/4 |

\* ELI5 is non-discriminating and outside the policy tables, as in the run. Shortfall, cost excess,
forgetting and the CIs against `always_separate` are therefore identical to the run's tables.

**Why nothing changed.** The best candidate's utility never reaches the variant's own τ_u, so the
decision order falls through to CREATE. The one exception is sentiment-B, which the reuse gate
decides before the utility is consulted (step 1):

| Variant | τ_u | FinQA | sentiment-B | xsum | ELI5 | NER |
|---|---|---|---|---|---|---|
| Run | 0.903 | 0.009 → CREATE | gate → REUSE | −0.756 → CREATE | −1.017 → CREATE | 0.428 → CREATE |
| A1 | 0.130 | −0.128 → CREATE | gate → REUSE | −0.077 → CREATE | 0.106 → CREATE | −0.012 → CREATE |
| A6 | 0.903 | 0.009 → CREATE | gate → REUSE | −0.756 → CREATE | −1.017 → CREATE | 0.428 → CREATE |
| A7 | 0.368 | −0.600 → CREATE | gate → REUSE | −0.138 → CREATE | −0.384 → CREATE | 0.057 → CREATE |
| A12 | 0.929 | 0.507 → CREATE | gate → REUSE | −0.760 → CREATE | −1.363 → CREATE | 0.559 → CREATE |

(Cells: the best candidate's predicted utility → `amaun`'s decision.) The closest call is A1 on
ELI5, 0.106 against 0.130. Since no variant fused, no FUSE option needed measuring beyond what
the run had (the list of unmeasured FUSE choices was empty).

**Reading.** On these tasks the learned utility decides nothing that the CREATE default does not.
Changing its evidence (A1, A7), its history (A6) or its task representation (A12) changes how well
it ranks candidates out of fold, but not one Phase 5 decision. These ablations therefore cannot
separate the feature families here. That needs evaluation tasks where some candidate's utility
clears τ_u, so the utility actually selects a FUSE option.

## 3. Fusion ablations: A3, A8

Both are variants of the run's chosen fusion config F-c (lossless storage, free α starting at 1.0,
KD λ = 0.1, 100 steps), measured exactly as the run's Phase 2 measured its grid: the six ordered
pairs of sum / qa / sentiment, on VALIDATION (100–150 examples per task). "Rel." is fused ÷
standalone on the new task; a pair is within both limits when Rel. ≥ 0.98 and the old task drops
by at most 1 %.

| Config | α | Mean Rel. | Mean old drop | Pairs within both |
|---|---|---|---|---|
| F-c (the run's choice) | learned, one per layer and component (392) | 0.957 | 1.09 % | 1/6 |
| **A3** | fixed at 1.0 (not learned); the fused LoRA still trains | 0.950 | 1.87 % | 1/6 |
| **A8** | learned, one per component (2) | **0.963** | **0.95 %** | 1/6 |
| F-e (run; uniform average, no training) | fixed at 0.5 | 0.991 | 1.20 % | 3/6 |

| Pair (new→old) | A3 Rel. | A3 old drop | A8 Rel. | A8 old drop | F-c Rel. | F-c old drop |
|---|---|---|---|---|---|---|
| sum→qa | 0.901 | 1.4 % | 0.922 | 0.3 % | 0.910 | 0.0 % |
| sum→sentiment | 0.888 | 0.0 % | 0.897 | 0.0 % | 0.910 | 0.0 % |
| qa→sum | 0.970 | 7.4 % | 0.980 | 2.0 % | 0.966 | 3.1 % |
| qa→sentiment | 0.946 | 1.4 % | 0.979 | 1.4 % | 0.968 | 1.4 % |
| sentiment→sum | **1.007** | **0.0 %** ✓ | 1.000 | 1.7 % | 0.993 | 1.8 % |
| sentiment→qa | 0.986 | 1.0 % (just over) | **1.000** | **0.3 %** ✓ | 0.993 | 0.2 % ✓ |

- **A3 works as intended:** α stayed at exactly 1.0 for both components in all six pairs. Fixing α
  cost a little: mean Rel. −0.007 and mean forgetting +0.8 points, mostly from qa→sum (7.4 %
  against 3.1 %).
- **A8 learned clearly different weights per pair:** the old component 0.59–0.89 and the new
  0.69–1.36 (the new adapter weighted up when sum was the new task). It matched or beat F-c on
  five of the six pairs for the new task, and forgot less on average.
- **Memory and storage were unchanged**: peak 13.3–15.7 GiB (F-c: 13.3–15.8), 148 MB per fused adapter.
- **Neither learning α nor learning it per layer made a clear difference here.** The differences
  are a few hundredths on six pairs, smaller than the spread across pairs, and A3, A8 and F-c each
  kept both limits on just one pair. The training-free average (F-e) is still the best on both
  counts (3 of 6 pairs).
- **Had A8 been in the run's grid**, it would have been the only eligible config with a mean old
  drop at or below 1 %, so the written rule would have chosen it over F-c. `fusion_viable` would
  still have been false (one pair within both limits; it needs two). This is reported, not acted
  on: the run's choice stands.

## 4. Ablation coverage after this analysis

A2, A4 and A9–A11 were covered by the run itself: `fixed_rule` and `probe_rule` against the
learned utility; KD off in fusion config F-b; the reuse-only, fuse-only and create-always
policies. A1, A6, A7 and A12 were replayed here and left every Phase 5 decision unchanged, because
the utility never cleared its threshold. A6 and A7 could not be tested in substance in any case:
with no adapter history, every historical feature and the repository-level resource features
were constant across the supervision pairs, so a real test needs supervision gathered across
changing repository states (a long stream with reuse and fusion events). On the six Phase 2
pairs, fixing α (A3) cost a little and one global α (A8) did a little better than layer-wise α;
neither difference is clear, and the training-free average F-e remained the strongest fusion
config. A5 and the stream variants were not run, and every result rests on one seed.

## 5. Limitations

- One seed, `local_smoke` tier, 30 supervision pairs from six tasks, four policy-eligible
  evaluation tasks. Pooled scores mix metrics.
- The feature ablations are replays of Phase 5 decisions only. A variant could behave differently
  in the Phase 6 stream, whose repository states the replay cannot reconstruct.
- The utility models remain worse than the mean predictor out of fold; the differences between
  variants in the utility-model table rest on six held-out tasks.
- A3 and A8 are measured on the six Phase 2 pairs on VALIDATION (100–150 examples per task), not on
  TEST, and not across Phase 5's FUSE options.
- The ablations are post-run analyses. They were declared in the experiment matrix before the run
  and use its fixed machinery, but they were run after its results were known.

## Where everything is

| | |
|---|---|
| Identity checks | `ablations/identity_checks.json` |
| Feature variants (models, replayed decisions and utilities, per-task and pooled tables, A12 embeddings) | `ablations/feature_variants.json` |
| A3 / A8 rows and summary | `ablations/fusion_variants.json` (plus `fusion_logs/`, `predictions/`) |
| Part 2 checkpoint (seeded from the run's caches) | `decision_study/checkpoint/state.json` |
| Logs | `~/ablations_part1.log`, `~/ablations_part2.log` on the VM |

All paths are under `results/local_smoke/l4_full_study_ablations/` unless noted.
