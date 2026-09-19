# AMAUN — Master Implementation Prompt

**Adaptive Multi-Criteria Adapter Utility Network**
*A Federated Continual Learning Framework for Small Language Models using Intelligent Adapter Management and Learnable LoRA Fusion*

Use this document as a single master prompt/spec to hand to an engineer, a coding agent, or an LLM to implement the AMAUN system end-to-end. It consolidates the full methodology, architecture, notation, equations, algorithms, dataset/task assignment, dual-backbone (smoke-test vs. actual-training) configuration, and implementation/experimentation checklist into one actionable reference. **No numerical results are included anywhere below — every number-shaped item is marked `[TO BE FIXED EXPERIMENTALLY]`.**

---

## 0. Role and Objective

You are implementing AMAUN, an end-to-end adapter lifecycle management framework for Federated Continual Learning (FCL) with a **frozen Small Language Model (SLM) backbone**.

**Core design principle:** a newly arriving task must not automatically create a permanent adapter. Instead:

1. Train a **temporary LoRA adapter** for the new task.
2. Evaluate it against adapters already in the local repository using a **multi-criteria feature representation** (not a single similarity score).
3. Feed that evidence to a **learned utility network** that predicts whether merging is beneficial.
4. If beneficial → perform **layer-wise learnable LoRA fusion + knowledge-distillation (KD) guided optimization**.
5. If not beneficial → keep the temporary adapter as a new repository entry.
6. Periodically run **repository compression** (same utility mechanism) and **federated aggregation** across clients.

This separates "how related are two adapters?" (feature extraction) from "is merging them actually beneficial?" (utility prediction), turning adapter management into a **decision problem**, not a similarity-ranking problem.

---

## 0.1 Backbone & Run-Mode Configuration (dual-tier: smoke test vs. actual)

This project runs on **two distinct backbone tiers**, selected by a single `run_mode` config flag. Every module, log line, checkpoint path, and results artifact **must** be tagged with the active `run_mode` so that smoke-test runs and real experimental runs are never mixed, overwritten, or accidentally compared as if equivalent.

| `run_mode` | Backbone (`f_θ`) | Purpose | Notes |
|---|---|---|---|
| `local_smoke` | **Qwen 1.5B** | Fast local iteration: verify the full pipeline (Modules 1–5 + federated loop) runs end-to-end without errors, on a laptop/dev box, with small data slices and few steps | Not used for any reported metric. Purely a correctness/regression harness. |
| `actual` | **Qwen 7B** | Real experimental training/evaluation runs whose results are meant to be logged, plotted, and reported per Section 12 | Only `actual` runs may populate the final results/plots described in Section 12.3. |

**Implementation requirements:**

- `run_mode` is a required top-level config field (`enum: [local_smoke, actual]`), with **no default** — it must be explicitly set per invocation, to prevent accidentally logging a smoke test as a real result or vice versa.
- The backbone identifier (`model_name_or_path`), tokenizer, and any backbone-specific settings (max sequence length, attention implementation, dtype/precision, device placement) are **derived from `run_mode`** via a config lookup table, not hardcoded inline in training/eval scripts:
  ```
  BACKBONE_REGISTRY = {
    "local_smoke": { "model": "Qwen/Qwen2.5-1.5B" (or the exact Qwen 1.5B checkpoint chosen), ... },
    "actual":      { "model": "Qwen/Qwen2.5-7B" (or the exact Qwen 7B checkpoint chosen), ... },
  }
  ```
  (Pin the exact Qwen checkpoint tags used, once chosen, in the config — do not leave them ambiguous at run time.)
- All other AMAUN mechanics (temporary LoRA training, feature extraction, utility network, fusion+KD, repository compression, federated aggregation) are **backbone-agnostic** and must run unmodified under either `run_mode` — the only thing that changes is which frozen backbone is loaded and how large/real the data slice is.
- `local_smoke` runs should default to reduced data volume, reduced epochs/steps, and reduced client counts (all still config-driven, never hardcoded as magic numbers baked into logic) purely to keep iteration fast — but must still exercise every module in the pipeline (temp adapter training → feature extraction → utility prediction → decision → fusion/KD or new-adapter path → repository compression trigger → federated aggregation) at least once per smoke run.
- `actual` runs use the full dataset volumes specified in Section 0.2 and the full experimental protocol in Section 12.

### 0.1.1 Results storage convention (mandatory separation)

All artifacts (checkpoints, logs, metrics, plots, feature caches, utility-network training data, ablation outputs) **must** be written under a path/namespace that encodes the run mode, so smoke and real results can never collide:

```
results/
├── local_smoke/
│   ├── <experiment_id>/
│   │   ├── config_used.json
│   │   ├── repository_snapshots/
│   │   ├── utility_network/
│   │   ├── fusion_logs/           # α_l trajectories, KD losses
│   │   ├── metrics.jsonl          # task-wise metrics vs. task index
│   │   └── federated_rounds/
│   └── ...
└── actual/
    ├── <experiment_id>/
    │   ├── config_used.json
    │   ├── repository_snapshots/
    │   ├── utility_network/
    │   ├── fusion_logs/
    │   ├── metrics.jsonl
    │   └── federated_rounds/
    └── ...
```

- The top-level split is **exactly** `results/local_smoke/` vs `results/actual/` — no other top-level result roots.
- Every run must serialize its full resolved config (including `run_mode`, backbone id, all hyperparameters from Section 10) alongside its outputs (`config_used.json`) so any result can be traced back to exactly what produced it.
- Reporting/plotting hooks (Section 12.3) must accept a `run_mode` filter and **must default to `actual`** — `local_smoke` data should never silently leak into a results table or plot meant for the paper/report. If a smoke result is ever shown (e.g., for debugging a plot), it must be visibly labeled `[SMOKE TEST — not a reported result]`.
- Baseline and ablation runs (Sections 12.4, 12.5) follow the identical `results/<run_mode>/<experiment_id>/` convention, with `<experiment_id>` encoding the method/ablation id (e.g. `A3_fixed_alpha`, `baseline_similarity_selection`).
- Random seeds (Section 10, "Randomness" row) must be logged per run in `config_used.json` regardless of `run_mode`.

### 0.2 Adapter / Task / Dataset Assignment Table

This table defines the sequential-task curriculum used to populate the local adapter repository `R_c` during both `local_smoke` and `actual` runs. The **same task/dataset identities** are used in both modes; only data volume and step counts differ (Section 0.2.1).

| Adapter | Task | Dataset | Recommended amount (actual / full-scale) |
|---|---|---|---|
| LoRA-1 | Summarization | CNN/DailyMail | 30k–50k train |
| LoRA-2 | Question Answering | SQuAD v1.1 | 30k–50k train |
| LoRA-3 | Instruction Following | FLAN Collection / FLAN-style data | 30k–50k train |
| LoRA-4 | Dialogue Generation | DailyDialog | ~8k–10k train |
| LoRA-5 | Long-form Generation | ELI5 | 20k–30k train |
| LoRA-6 | Domain-specific Generation | FinQA / financial QA-generation data | 10k–20k train |

**Implementation notes on this table:**

- This is the canonical **task arrival curriculum** referenced throughout Module 1 (`D_t` for `t = 1..6`, in this order unless the experimental protocol in Section 12.2 specifies a different/randomized task order for a given run — task order itself is a config field, not hardcoded).
- Each adapter/task row must map to its own `D_t` train split **and** its own held-out validation/representative split (per Section 4.4/6.4/12.2) — never reuse a task's train split as its own validation data.
- "Recommended amount" (30k–50k, ~8k–10k, 20k–30k, 10k–20k) is the **`actual`-mode** target data volume; it is a range, not a fixed number, and the exact value used per run must be recorded in `config_used.json` (do not silently pick a value and leave it undocumented).
- The table as given has six rows (LoRA-1 through LoRA-6). Treat the task list as **extensible** — the config schema for the task curriculum must accept additional `(adapter_id, task_name, dataset_name, data_volume)` entries without code changes, in case the curriculum is later extended beyond LoRA-6.
- Client partitioning (which client(s) see which subset of LoRA-1..6, IID vs. non-IID split across clients) is governed separately by Section 12.2 and must remain a configurable field, not fixed by this table.

#### 0.2.1 `local_smoke` data-volume policy

For `run_mode = local_smoke`, every row in the table above is subsampled to a small fixed fraction (config field, e.g. a `smoke_fraction` or an explicit `smoke_max_examples_per_task` cap) purely to validate pipeline correctness quickly:

- Use a small slice of each dataset (train **and** val/representative splits) sufficient to run at least one full optimizer step and one full evaluation pass per task, per adapter, per module (temp-adapter training, feature extraction's reference-minibatch protocol, utility-network forward/backward pass, fusion+KD optimization, compression trigger, federated round).
- Do **not** hardcode the smoke sample count as a magic number scattered across scripts — define it once in config and reference it everywhere data loading happens.
- Smoke runs must still instantiate all six LoRA-1..6 tasks (or however many the curriculum config lists) in sequence, so that repository growth, the merge-or-create decision, and compression logic are all actually exercised, not skipped.
- Smoke-run metrics are diagnostic only (did it run, did loss decrease at all, are shapes/dtypes correct, does the α_l fusion path execute) — they are never compared numerically against `actual`-mode results, and per Section 0.1.1 must be stored only under `results/local_smoke/`.

---

## 1. System Architecture (Figure 1 equivalent)

Implement a closed continual-learning loop per client, connected to a federated server:

```
Client i: Sequential Task Arrival
        │
        ▼
Frozen SLM + Temporary LoRA Training ──► Adapter Repository (A1...An)
        │                                        │
        ▼                                        ▼
  Multi-Criteria Feature Extraction ◄─────────────┘
        │
        ▼
  AMAUN Utility Network (MLP + multi-head prediction)
        │
        ▼
  Utility Ranking + Decision (Merge / New Adapter)
        │
        ├── Merge ──► Layer-wise Learnable Fusion (α_l init = 0.5)
        │                     │
        │                     ▼
        │             KD-guided Post-Merge Training  (L = L_new + λ·L_KD)
        │                     │
        └── New Adapter ──────┴──────► Compressed Local Repository
                                              │
                                              ▼
                                   Federated Server: Adapter Aggregation
                                   (periodic communication only; raw data never leaves client)
```

**End-to-end control flow** (implement exactly this pipeline):

```
Task arrival → Temporary adapter → Feature extraction → Utility prediction
→ Decision → Fusion/KD → Repository update → Compression → Federated aggregation
```

This loop repeats for every sequential task, per client. The repository is a continually evolving structure, not a static set of task-specific adapters. This holds identically for both `local_smoke` and `actual` runs — only the backbone (Section 0.1) and data volume (Section 0.2.1) differ.

---

## 2. Design Principles and Notation

### 2.1 Core constraint

The SLM backbone is frozen at all times. Only LoRA adapter parameters are trained.

```
θ_(k+1) = θ_k = θ_0                      (backbone frozen throughout lifecycle)
W_l^adapt = W_l^0 + ΔW_l                 (adapted layer = frozen layer + LoRA update)
ΔW_l = B_l A_l,  B_l ∈ R^(d_out×r), A_l ∈ R^(r×d_in),  r ≪ min(d_in, d_out)
A_temp = TrainLoRA(f_θ, D_t; r, η, E)
```

### 2.2 Full symbol table — implement all of these as first-class variables/fields

| Symbol | Meaning |
|---|---|
| `f_θ` | Frozen SLM backbone, parameters θ (Qwen 1.5B under `local_smoke`, Qwen 7B under `actual` — see Section 0.1) |
| `t` | Current task arriving at a client |
| `A_temp` | Temporary LoRA adapter trained for the current task |
| `A_i` | Existing adapter i in the local repository |
| `R_c` | Local adapter repository for client c |
| `r` | LoRA rank |
| `B_l, A_l` | LoRA low-rank matrices at layer l |
| `ΔW_l` | Effective LoRA update `B_l A_l` |
| `α_l` | Learnable layer-wise fusion coefficient |
| `z` | Concatenated multi-criteria feature vector for one candidate pair |
| `U` | Predicted utility of merging temp adapter with a candidate |
| `λ` | KD loss weight |
| `τ_u` | Minimum utility threshold for accepting a merge |
| `T` | KD temperature |
| `R_max` | Configured repository memory budget |
| `η` | Optimization learning rate |

`r, η, E, λ, T, τ_u, R_max`, and all network hyperparameters are implementation parameters — **do not hardcode arbitrary numeric values**; expose them as config fields to be tuned in the experimental phase. These are shared across `run_mode`s but may take different (still config-driven, still logged) values for `local_smoke` vs. `actual`.

---

## 3. Module 1 — Sequential Task Processing & Temporary LoRA Generation

**Purpose:** convert an incoming task into a candidate (non-permanent) adapter.

```
R_c^0 = ∅                                        # repository starts empty
R_c^k = {A_1, ..., A_n}                          # after k accepted events
A_temp = LoRATrain(D_t | θ_0)                    # train on current task only, backbone frozen
A_temp ∉ R_c  until Decision(A_temp, R_c) is completed
```

**Processing sequence to implement:**

1. Receive task `D_t` at client c (per the curriculum in Section 0.2, subsampled per Section 0.2.1 if `run_mode = local_smoke`).
2. Load frozen SLM backbone (resolved from `run_mode` via `BACKBONE_REGISTRY`, Section 0.1) + client's current repository metadata.
3. Attach a temporary LoRA parameterization to the frozen backbone.
4. Optimize only LoRA parameters on `D_t`.
5. Compute/store the temporary adapter's task-side statistics needed by Module 2 (training loss trajectory, final adapter state).
6. Pass `A_temp` and repository candidates to Module 2.

**Implementation notes:**

- Keep `A_temp` in a scratch/candidate store — never write it into the permanent repository before the decision step.
- If `R_c` is empty, skip evaluation entirely: insert `A_temp` directly (see Algorithm 1, line 2–4).

---

## 4. Module 2 — Multi-Criteria Adapter Feature Extraction

**Purpose:** build a rich, heterogeneous evidence vector per candidate pair `(A_temp, A_i)`. **No single similarity score is ever used as the decision rule — cosine similarity is explicitly not the selection rule.**

```
P_t = {(A_temp, A_i) | A_i ∈ R_c}                # one candidate pair per existing adapter
z_i = [z_i^sem ; z_i^par ; z_i^grad ; z_i^perf ; z_i^cl ; z_i^hist ; z_i^res]   # 7 concatenated families
```

Implement all seven feature families:

### 4.1 Semantic features
```
e_task = Encoder(D_t),  e_old = Encoder(D_rep_i)          # fixed encoder, identical across methods
d_task(i) = ||e_t - e_i||_2^2                              # Euclidean, NOT cosine
d_hidden(i) = (1/N) Σ_n ||h_t(x_n) - h_i(x_n)||_2^2        # hidden-state discrepancy
D_out(i) = (1/N) Σ_n KL( p_i(y|x_n) || p_temp(y|x_n) )     # output-distribution divergence
```

### 4.2 Parameter-structure features (per layer, then aggregated)
```
d_F(l,i)    = || ΔW_temp_l - ΔW_i_l ||_F                   # Frobenius distance
d_2(l,i)    = || vec(ΔW_temp_l) - vec(ΔW_i_l) ||_2         # vectorized L2 distance
v_l^i       = Var(vec(ΔW_i_l))                             # layer-wise parameter variance
d_rank(l,i) = | rank(ΔW_temp_l) - rank(ΔW_i_l) |           # rank discrepancy
d_sv(l,i)   = || σ(ΔW_temp_l) - σ(ΔW_i_l) ||_2             # singular-value distance
```

### 4.3 Gradient / optimization features (same reference minibatch protocol for both adapters)
```
g_t = ∇_φtemp L_t                                          # grad of task loss wrt temp params
g_i = ∇_φi L_rep_i                                         # grad of representative loss wrt stored adapter
a_grad(i) = (g_t^T g_i) / (||g_t||_2 ||g_i||_2 + ε)        # gradient alignment (feature, not a threshold)
m_g(i) = || g_t - g_i ||_2                                 # gradient discrepancy
v_g(i) = Var(g_t - g_i)                                    # gradient variance
```

### 4.4 Performance-quality features (on held-out validation data)
```
Acc_i    = Correct_i / N_val
L_val_i  = (1/N_val) Σ_n CE(y_n, p_i(y|x_n))
C_i      = (1/N_val) Σ_n max_y p_i(y|x_n)                  # mean max confidence
H_i      = -(1/N_val) Σ_n Σ_y p_i(y|x_n) log p_i(y|x_n)    # mean predictive entropy
ECE_i    = Σ_b (|B_b|/N_val) |acc(B_b) - conf(B_b)|        # expected calibration error, predefined bins
```

### 4.5 Continual-learning features (forgetting-focused)
```
F_i  = (1/M) Σ_j max(0, Acc_j,pre - Acc_j,post)            # average forgetting over prior tasks
KR_i = 1 - F_i                                             # knowledge retention
P_i  = Acc_new,post - Acc_new,pre                          # plasticity on incoming task
S_i  = 1 - F_i + ρ · P_i                                   # stability-plasticity descriptor (ρ = config)
```

### 4.6 Historical & resource features
```
Age_i       = t_current - t_created,i
Freq_i      = N_use,i / N_events
Merge_i     = N_merge,i
PerfHist_i  = (1/K) Σ_k Acc_i,k
Mem_i       = |φ_i| · b                                    # bytes = param_count * bytes_per_param
Comm_i      = Mem_i + |meta_i|
RepoCost    = Σ_(A_j ∈ R_c) Mem_j
Latency_i   = T_infer(A_i)
```

### 4.7 Feature normalization (train-only statistics — critical for no test leakage)
```
z'_{i,j} = (z_{i,j} - μ_j) / (σ_j + ε)          # standardize each scalar feature
z_i = [z'_{i,1}, z'_{i,2}, ..., z'_{i,d}]^T      # final d-dim feature vector
```

`μ_j, σ_j` must be computed only from the utility-learning training split and reused unchanged for validation/test. **Never re-estimate from the test sequence.** This constraint holds independently for `local_smoke` and `actual` runs — normalization statistics from one `run_mode` must never be reused in the other, since backbone and data volume differ.

**Feature family summary table to reflect in code as named sub-vectors:**

| Family | Concrete features | Granularity |
|---|---|---|
| Semantic | task repr. distance; hidden-state discrepancy; output-distribution divergence | pair / layer-aware |
| Parameter | Frobenius dist.; L2 dist.; variance; rank diff.; singular-value dist. | layer → adapter |
| Gradient | alignment; discrepancy; variance; norms | layer / param group |
| Performance | accuracy; loss; confidence; entropy; calibration error | adapter / task |
| Continual learning | forgetting; retention; plasticity; stability | historical task set |
| Historical | age; usage freq.; merge count; last-use; historical performance | adapter |
| Resource | adapter size; repo size; comm payload; memory; inference latency | adapter / repository |

---

## 5. Module 3 — AMAUN Utility Prediction & Merge-or-Create Decision

**Purpose:** turn the heterogeneous feature vector into a learned decision signal, not a hand-tuned rule.

### 5.1 Utility network (MLP, multi-head)
```
h1 = ReLU(W1 z_i + b1)
h2 = ReLU(W2 h1 + b2)
o_i = W_o h2 + b_o                                   # raw multi-objective output vector
û_i = [ âcc_i, F̂_i, ĉ_i, M̂em_i, Ĉomm_i, L̂_i ]^T      # predicted: post-integration acc, forgetting,
                                                       # confidence, memory, comm cost, long-term loss
U_i = w^T ψ(û_i)                                     # scalar utility via monotonic transform ψ + learned weights w
```

Hidden width/depth, activation, dropout, optimizer, LR, epochs, and target construction are `[TO BE FIXED EXPERIMENTALLY]` — expose as config, do not hardcode invented values.

### 5.2 Decision rule
```
i* = argmax_{i ∈ R_c} U_i
Merge(A_temp, A_i*)  if  U_i* ≥ τ_u
CreateNew(A_temp)    if  max_i U_i < τ_u
```

### 5.3 Why this differs from a similarity-threshold pipeline (assert these properties in tests/design docs)

- Input is a multi-criteria feature vector, not a single similarity scalar.
- Network learns nonlinear interactions across all 7 families.
- Selection is by expected integration utility, not largest pairwise similarity.
- Minimum utility threshold prevents merging unrelated adapters due to one favorable metric.
- Same utility framework is reused for repository compression (Module 5).

### 5.4 Algorithm 1 — AMAUN Adapter Selection and Merge-or-Create

```
Require: frozen SLM f_θ, current task D_t, local repository R_c, utility model g_ω, threshold τ_u

1. Train temporary LoRA adapter A_temp on D_t (θ frozen)
2. if R_c is empty:
3.     insert A_temp into R_c; terminate this task event
4. end if
5. for each adapter A_i in R_c:
6.     construct candidate pair (A_temp, A_i)
7.     extract semantic, parameter, gradient, performance, continual, historical, resource features
8.     normalize features using training-derived statistics
9.     compute predicted utility U_i = g_ω(z_i)
10. end for
11. select i* = argmax_i U_i
12. if U_i* < τ_u:
13.     insert A_temp as a new repository adapter
14. else:
15.     pass (A_i*, A_temp) to Module 4 (fusion + KD)
16.     replace A_i* with the optimized merged adapter; update metadata
17. end if
18. if repository exceeds R_max:
19.     invoke repository compression (Module 5)
20. end if
21. return updated repository R_c
```

**Key invariant:** adapter creation, selection, and fusion are not independent — selection determines whether fusion is justified, and utility is the single shared decision signal across all three.

---

## 6. Module 4 — Layer-Wise Learnable LoRA Fusion + KD Optimization

Runs only when Module 3 selects an existing adapter (`A_old`) to merge with `A_temp`.

### 6.1 Layer-wise fusion (effective-update fusion, not separate A/B averaging)
```
α_l^(0) = 0.5,  for l = 1..L                          # equal contribution at init
ΔW_merge_l = α_l · ΔW_old_l + (1 - α_l) · ΔW_temp_l    # fuse EFFECTIVE updates, not A/B separately
W_merge_l = W_l^0 + ΔW_merge_l
```

**Implementation warning to preserve exactly:** averaging A matrices and B matrices independently is **not** equivalent to averaging the effective update `ΔW = BA`. Always fuse at the `ΔW` level, then explicitly decide whether to store the fused adapter as a dense effective update or re-factorize it back into low rank — document this choice in code comments/config.

### 6.2 Constrained (bounded) fusion coefficient
```
α_l = sigmoid(a_l)                        # unconstrained a_l mapped to (0,1)
a_l ← a_l - η_α · ∂L_total/∂a_l           # updated via backprop through merged adapter
```

### 6.3 New-task objective
```
L_new = -(1/N_t) Σ_n Σ_y 1[y_n = y] log p_merge(y | x_n)
```

### 6.4 Knowledge-distillation objective (teacher = pre-merge OLD adapter, not the frozen backbone)
```
q_old(y|x)   = softmax( s_old(x)   / T )         # teacher
q_merge(y|x) = softmax( s_merge(x) / T )         # student
L_KD = T^2 · (1/N_r) Σ_n KL( q_old(·|x_n) || q_merge(·|x_n) )
```

Preservation target = the behavior of the knowledge that would otherwise be overwritten. Teacher distributions from `A_old` must be captured **before** fusion begins.

### 6.5 Combined objective and joint optimization
```
L_total = L_new + λ · L_KD
φ_merge ← φ_merge - η_φ · ∇_φmerge L_total       # update LoRA params
a       ← a       - η_α · ∇_a L_total            # update fusion coefficients jointly
```

### 6.6 Layer-wise interpretation (log this for analysis/interpretability)

- Larger `α_l` → layer retained more of the old adapter's behavior.
- Smaller `α_l` → layer favored the new/temporary adapter.
- This pattern must **emerge from the joint objective** — never manually assigned post-init.

### 6.7 Algorithm 2 — Layer-Wise Learnable LoRA Fusion with KD

```
Require: A_old, A_temp, frozen f_θ, representative retention data D_rep, current-task data D_t

1. For every layer l, compute ΔW_old_l and ΔW_temp_l
2. Initialize α_l = 0.5 for all layers
3. Construct ΔW_merge_l = α_l·ΔW_old_l + (1-α_l)·ΔW_temp_l
4. Construct merged model: W_merge_l = W_l^0 + ΔW_merge_l
5. Run A_old on D_rep; store teacher distributions q_old
6. repeat:
7.     run merged adapter on D_rep and D_t
8.     compute L_new (on D_t) and L_KD (on D_rep)
9.     L_total = L_new + λ L_KD
10.    backprop: update LoRA params AND every layer's α (via a_l)
11. until post-merge stopping criterion is met
12. return optimized merged adapter and learned vector [α_1, ..., α_L]
```

---

## 7. Module 5 — Repository Lifecycle Management & Federated Aggregation

### 7.1 Repository update rules
```
R'_c = (R_c \ {A_i*}) ∪ {A_merge}       # successful merge replaces selected adapter
R'_c = R_c ∪ {A_temp}                   # rejected merge → retain temp adapter as new entry
Mem(R_c) = Σ_(A_i ∈ R_c) Mem(A_i)       # total repository memory
```

### 7.2 Repository compression (triggered by memory budget, uses the SAME utility mechanism)
```
Compress if Mem(R_c) ≥ R_max
(i*, j*) = argmax_{i≠j} U_merge(A_i, A_j)          # highest-utility merge candidate among stored adapters
R_c ← (R_c \ {A_i*, A_j*}) ∪ {A_merge_ij}          # replace pair with one merged adapter
```

**Do not** implement compression as age/usage-based deletion. It must reuse the utility network so the framework has a single decision principle across creation, merging, and compression.

### 7.3 Federated aggregation
```
C_k = { c | client c reports compatible adapter state k }
A_global_k = Σ_{c ∈ C_k} ( n_c / Σ_{j ∈ C_k} n_j ) · A_c,k        # FedAvg-style weighted aggregation
Payload_c = |A_c| + |meta_c|                                       # only adapter + lightweight metadata transmitted
```

**Critical constraints:**

- The backbone `θ` is never transmitted (shared + frozen).
- The server must determine compatibility among adapter states before aggregating — never blindly average unrelated adapters (this would defeat the purpose of the whole compatibility/utility mechanism).

---

## 8. Computational & Communication Cost Model (implement as profiling utilities, not hardcoded constants)

```
N_candidates = |R_c| = n
C_select  ≈ n · (C_sem + C_par + C_grad + C_perf + C_hist + C_res) + C_AMAUN
C_fusion  = O(L · C_layer)
Comm_full = |θ| · b                                     # conceptual full-model payload (never actually sent)
Comm_LoRA = Σ_l |φ_l| · b + |meta|                       # actual LoRA-only payload + metadata
```

**Implication for implementation:** feature extraction may run on a representative reference subset rather than the full historical datasets to bound overhead — expose subset size as config, and treat this as a genuine cost/accuracy tradeoff to measure, not something to hide. Profile and log this cost model separately for `local_smoke` (Qwen 1.5B) and `actual` (Qwen 7B), since backbone size directly changes `C_layer`, `Comm_LoRA`, and `Latency_i`.

---

## 9. Utility-Network Training Protocol (bootstrapping AMAUN's own supervision)

The utility network needs training targets that do not exist a priori — they must be generated from controlled candidate-integration observations:

```
y_i = [ Acc_i^post, F_i^post, C_i^post, Mem_i^post, Comm_i^post, L_i^post ]^T   # observed post-integration outcomes
L_U = Σ_i || g_ω(z_i) - y_i ||_2^2                                              # multi-output regression objective
ω* = argmin_ω L_U(ω)
```

**Implementation requirements:**

- For each historical candidate pair, actually perform the merge/fusion evaluation and record the real resulting outcomes (task performance, forgetting, confidence, resource change) — this is the supervision signal.
- The simple L2 regression loss may later be replaced by a task-appropriate multi-objective loss, but whatever is chosen must be reported/logged explicitly.
- The utility model must never be trained using the final held-out test sequence (leakage prevention).
- Utility-network weights `ω` trained under `local_smoke` and `actual` are separate artifacts (different backbone → different feature distributions) and must never be cross-loaded between run modes.

---

## 10. Full Implementation-Level Specification Table

Build a config schema with (at minimum) these fields. None should be hardcoded with invented numeric values — mark all as tunable/experimental.

| Component | Required implementation | Status |
|---|---|---|
| `run_mode` | `local_smoke` (Qwen 1.5B) or `actual` (Qwen 7B); no default, must be explicit | **Required, new field** |
| Backbone SLM | Frozen during all local + fusion ops; resolved from `run_mode` via `BACKBONE_REGISTRY` | Model choice = experimental config (per tier) |
| LoRA adaptation | Train only low-rank params; frozen backbone | rank r = experimental config |
| Temporary adapter | Train on current task; kept outside repo until decision | epochs/LR = experimental config |
| Task curriculum | LoRA-1..6 (or extended) task/dataset table (Section 0.2) | data volume = experimental config; subsampled under `local_smoke` (Section 0.2.1) |
| Feature extraction | Compute all 7 families on fixed reference protocol | dimensionality follows implementation |
| Feature normalization | Train-set stats only; reused on val/test; never shared across `run_mode`s | μ, σ learned from utility-training data |
| AMAUN network | MLP/nonlinear predictor, z_i → predicted outcomes | architecture = experimental config |
| Utility rule | Rank U_i, compare max utility vs τ_u | τ_u selected via validation |
| Fusion init | One trainable α_l per layer | α_l = 0.5 (fixed init, not tunable) |
| Fusion constraint | α_l = sigmoid(a_l) for bounded contribution | no separate numeric setting |
| KD | Old selected adapter = teacher; merged = student | T, λ = experimental config |
| Post-merge training | Joint optimize LoRA + fusion params | η_φ, η_α, stopping rule = experimental |
| Repository compression | Trigger by memory budget; high-utility merge pair | R_max = experimental config |
| Federated aggregation | Aggregate only compatible adapters; frozen backbone | participation/round schedule = experimental |
| Results storage | `results/<run_mode>/<experiment_id>/...` (Section 0.1.1) | mandatory path convention, no defaults leaking modes |
| Randomness | Fixed seeds for all comparative runs | seed protocol reported with final results, per `run_mode` |

---

## 11. Stage-by-Stage Feature Computation Protocol (pipeline stages to implement as discrete functions/modules)

| Stage | Inputs observed | Computation | Output |
|---|---|---|---|
| Task arrival | D_t | Register task identity, schema, split | Task metadata |
| Temporary training | LoRA params | Train A_temp; log loss trajectory | A_temp + training stats |
| Candidate retrieval | R_c | Enumerate A_1..A_n, read metadata | Candidate list |
| Semantic pass | Repr. inputs/representations | Task/hidden/output discrepancies | z_sem |
| Parameter pass | ΔW_temp_l, ΔW_i_l | Layer-wise norm/distance/variance/rank/SV stats | z_par |
| Gradient pass | Representative gradients | Alignment/discrepancy/variance | z_grad |
| Performance pass | Validation predictions | Accuracy, loss, confidence, entropy, calibration | z_perf |
| Continual pass | Historical task scores | Pre/post comparison, retention | z_cl |
| History pass | Repository metadata | Age, usage, merges, historical performance | z_hist |
| Resource pass | Adapter/repo system state | Memory, payload, repo size, latency | z_res |
| Utility pass | z_i | Normalize + evaluate AMAUN | U_i / predicted outcomes |

---

## 12. Experimental Protocol to Build Around (do not fabricate any of these — build the harness only)

### 12.1 Experimental setup — must be configurable & logged

`run_mode` (Section 0.1), SLM choice/tokenizer (resolved from `run_mode`), LoRA insertion locations, LoRA rank, scaling, dropout, optimizer, learning rates, batch size, sequence length, local epochs, number of clients, task order (from the curriculum in Section 0.2), client-task assignment, federated round schedule, representative-data size for KD, utility-network architecture, utility training protocol, utility threshold selection procedure, repository memory budget, compression trigger, random seeds.

### 12.2 Dataset handling

- Treat data as a continual sequence, not one static pool.
- For each dataset (LoRA-1..6 in Section 0.2, or any curriculum extension): task definitions, number of sequential tasks, class/label structure, client partitioning rule, train/val/test split, non-IID construction (if used), task arrival order, representative-data protocol for KD, and whether all clients see the same sequence or different subsets.
- Every task needs its own current-task training split and a validation/representative split; the representative split must not leak test information.
- Under `local_smoke`, apply the subsampling policy of Section 0.2.1 uniformly to train/val/representative splits alike.

### 12.3 Results reporting plan (build logging/plotting hooks for these — do not populate with fake numbers)

- Task-wise accuracy / precision / recall / F1 across the continual sequence.
- Average accuracy vs. task index.
- Forgetting vs. task index.
- Repository size / adapters retained vs. task index.
- Communication payload vs. federated round.
- Learned layer-wise α_l values after fusion.
- All reporting/plotting hooks read from `results/actual/` by default (Section 0.1.1); `results/local_smoke/` is available for the same hooks only in an explicit debug mode, always visibly labeled as smoke-test output.

### 12.4 Baseline comparison framework — implement adapters/wrappers for each baseline class

| Method category | Adapter management | Fusion coefficient | Forgetting control |
|---|---|---|---|
| Task-specific adapters | New adapter per task | N/A | Unbounded growth |
| Fixed fusion | Merge selected adapters | Fixed/manual | Usually indirect |
| Simple adapter averaging | Aggregate adapter states | Uniform/fixed | Depends on storage policy |
| Similarity-based selection | Select via similarity criterion | Method-dependent | Potentially reduced |
| AMAUN | Utility-based merge/create + compression | Learned layer-wise α_l | KD-guided, explicitly managed |

All baselines must use identical task sequences, client partitions, backbone (i.e., run under the same `run_mode`), evaluation splits, and federated budgets as the AMAUN run they are compared against.

### 12.5 Ablation studies to implement as toggles in config

| ID | Removed/changed component | Question answered |
|---|---|---|
| A1 | Remove multi-criteria features → reduced feature set | Does heterogeneous evidence help? |
| A2 | Remove AMAUN → fixed selection rule | Does learned utility help vs. fixed rule? |
| A3 | Replace learnable α_l with fixed fusion | Does layer-wise learnability help? |
| A4 | Remove KD term | How much does behavior preservation contribute? |
| A5 | Remove repository compression | Cost of unmanaged adapter growth? |
| A6 | Remove historical features | Does adapter history help long-term decisions? |
| A7 | Remove resource features | Does system cost info change selection? |
| A8 | One global α instead of layer-wise α_l | Is layer-level flexibility necessary? |

Ablations should be smoke-tested under `local_smoke` first (to verify the toggle actually changes pipeline behavior/wires correctly) before being run under `actual` for reportable results.

### 12.6 Hyperparameter tuning surface (validation-only; test sequence untouched until final)

LoRA rank r, LoRA scaling/α_LoRA, LoRA dropout, local learning rate η, local epochs E, AMAUN hidden-layer dims, AMAUN regularization, utility threshold τ_u, fusion learning rate η_α, KD temperature T, KD loss weight λ, repository memory budget R_max. Tune on `actual`-mode validation data only; `local_smoke` is not part of the tuning surface (it exists purely for pipeline correctness, not for hyperparameter search).

---

## 13. Novelty / Contribution Framing (for docs, papers, or README — do not overclaim individual components)

Position novelty at the architecture/mechanism level, not on individual ingredients (LoRA, KD, FedAvg, weighted fusion are not independently novel). The contribution is their integration into:

1. **Multi-criteria adapter utility formulation** — expected-utility decision problem, not single-metric similarity.
2. **Merge-or-create lifecycle decision** — can retain new adapter when nothing meets the utility bar.
3. **Layer-wise learnable fusion** — one trainable α per transformer layer.
4. **KD-guided joint optimization** — old adapter as behavioral teacher.
5. **Unified repository compression** — reuses the same utility principle instead of an ad hoc heuristic.
6. **Federated adapter lifecycle** — LoRA-only communication with a permanently frozen backbone.

---

## 14. Implementation Checklist (Appendix A — track as issues/tasks)

- [ ] Add and enforce `run_mode` config field (`local_smoke` / `actual`), with backbone resolution via `BACKBONE_REGISTRY` (Qwen 1.5B / Qwen 7B) and **no default value**.
- [ ] Implement `results/<run_mode>/<experiment_id>/...` storage convention; verify smoke and actual outputs cannot collide or be silently mixed.
- [ ] Wire the LoRA-1..6 task/dataset curriculum (Section 0.2) as a config-driven, extensible list.
- [ ] Implement `local_smoke` subsampling policy (Section 0.2.1) applied uniformly across train/val/representative splits.
- [ ] Freeze backbone; verify no backbone gradients are ever updated, under both `run_mode`s.
- [ ] Implement temporary LoRA training + repository isolation (candidate never enters repo pre-decision).
- [ ] Implement candidate enumeration over all local adapters.
- [ ] Implement all 7 feature families (semantic, parameter, gradient, performance, continual, historical, resource).
- [ ] Implement training-only feature normalization; persist μ/σ separately per `run_mode`.
- [ ] Implement AMAUN utility predictor + its training-target generation pipeline.
- [ ] Implement utility ranking + merge-or-create criterion (τ_u).
- [ ] Implement effective-update layer-wise fusion with α_l init = 0.5.
- [ ] Implement bounded α_l = sigmoid(a_l) parameterization.
- [ ] Implement old-adapter teacher inference + merged-adapter student inference for KD.
- [ ] Jointly optimize LoRA parameters and fusion parameters.
- [ ] Implement replace/insert logic per decision outcome.
- [ ] Implement memory-triggered repository compression (reusing utility network).
- [ ] Implement compatible-adapter grouping + weighted aggregation at federated server.
- [ ] Log task-wise metrics, forgetting, repository size, communication payload, fusion coefficients, and runtime — tagged by `run_mode`.
- [ ] Run a full `local_smoke` (Qwen 1.5B) pass end-to-end through every module above and confirm no errors, before touching `actual`.
- [ ] Only after the `local_smoke` pipeline is stable end-to-end: run `actual` (Qwen 7B) baseline, ablation, and hyperparameter-tuning experiments, storing results under `results/actual/`.

---

## 15. Figure & Equation Cross-Reference Map (for docs/paper writing)

- Figure 1 → overall architecture (Section 1 above).
- Eqs. (1)–(8) → adapter/task state definitions (Section 2–3).
- Eqs. (9)–(43) → candidate feature construction (Section 4).
- Eqs. (44)–(51) → utility prediction & decision (Section 5).
- Eqs. (52)–(63) → adaptive fusion & KD optimization (Section 6).
- Eqs. (64)–(73) → repository & federated operations (Section 7).
- Eqs. (74)–(81) → utility training & complexity/communication analysis (Sections 8–9).

---

## 16. Hard Constraints for Any Agent Implementing or Extending This Report

1. Never fabricate numerical results (accuracy, F1, latency, forgetting, communication cost, etc.). Every number belongs to the post-implementation evaluation phase.
2. Never use cosine similarity (or any single metric) as the final decision rule — it may exist only as one feature among many.
3. Never average LoRA A/B matrices independently — always fuse at the effective ΔW level.
4. Never let the backbone train or be transmitted — it is frozen and local-only forever, under both `run_mode`s.
5. Never estimate normalization statistics from the test sequence.
6. Never train the utility network on the final test sequence.
7. Never replace the utility-based compression with an age/usage heuristic — reuse the same mechanism.
8. Always initialize α_l = 0.5 per layer before any post-merge optimization.
9. Always treat the pre-merge old adapter (not the frozen backbone) as the KD teacher.
10. Always keep the temporary adapter out of the permanent repository until the decision step completes.
11. **Always tag every artifact (checkpoint, log, metric, plot) with its `run_mode`** and write it under `results/local_smoke/` or `results/actual/` accordingly — never let a smoke-test run populate, overwrite, or be mistaken for an `actual` result.
12. **Never load Qwen 1.5B weights/checkpoints into an `actual`-mode run, or Qwen 7B weights/checkpoints into a `local_smoke`-mode run** — the backbone tier is fixed per `run_mode` via `BACKBONE_REGISTRY` and must not be overridden ad hoc in code.