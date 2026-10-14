# AMAUN — System Specification

Engineering contract, module by module. Each module lists: purpose, inputs, outputs,
equations (from `idea.md`), invariants, config keys, and the test that verifies it.
Implementation agents must not deviate. Ambiguities go to `docs/decision_register.md`.

Notation follows `idea.md` §2.2.

---

## M0 — Configuration and run-mode resolution

**Module:** `src/amaun/config.py`

**Purpose.** Single source of truth for every tunable value. No numeric literal that affects
an experiment may appear anywhere else in the codebase.

**Contract.**
- `RunMode` enum: `dummy | local_smoke | actual`. **No default.**
- `BACKBONE_REGISTRY: dict[RunMode, BackboneSpec]` is the *only* place a checkpoint id
  appears. `BackboneSpec` = `(model_id, revision, dtype, max_seq_len, attn_impl, quantization,
  device_preference, is_reportable)`. `revision` is a 40-hex commit sha (D-001 requires the
  checkpoint tag to be pinned once chosen); `None` means unpinned-and-floating and is
  acceptable only for the `dummy` tier, which downloads nothing.
- `resolve(cfg) -> ResolvedConfig` performs: run-mode lookup, device feasibility check,
  quantisation availability check, cross-field validation, and produces a stable
  `config_hash` (sha256 over canonical JSON of the resolved tree).
- `ResolvedConfig.to_config_used()` emits `config_used.json` including `run_mode`, backbone
  id, every hyperparameter, git commit, package versions, hostname, device, and seed.

**Invariants.**
- `I0.1` Loading a config without `run_mode` raises `MissingRunModeError`.
- `I0.2` `run_mode=dummy` sets `is_reportable=False`; any attempt to write into
  `results/actual/` from a non-`actual` run raises `RunModeArtifactError`.
- `I0.3` Requesting 4-bit quantisation on a non-CUDA device raises
  `QuantizationUnavailableError` (never a silent downgrade — D-008).
- `I0.4` Two runs with the same resolved config produce the same `config_hash`.

**Config keys.** the entire tree; see `configs/`.
**Tests.** `tests/test_config.py`

---

## M1 — Frozen backbone

**Module:** `src/amaun/models/backbone.py`

**Purpose.** Load `f_θ` once, frozen, and guarantee it stays frozen.

**Contract.**
- `load_backbone(resolved) -> (model, tokenizer)`.
- `dummy` tier constructs a `Qwen2Config` with tiny dims **locally** (no network) so tests
  run offline and fast.
- `freeze_backbone(model)` sets `requires_grad=False` on all parameters and returns the
  count frozen.
- `assert_backbone_frozen(model, named_adapter_params)` raises `BackboneNotFrozenError` if
  any non-adapter parameter has `requires_grad=True`, and is called **before every
  optimiser step** in every trainer (cheap: cached parameter-name set).
- `backbone_fingerprint(model)` returns a hash of a deterministic sample of backbone weights,
  recorded before and after every training phase to prove the weights did not move.
- Non-`dummy` tiers load from a **local snapshot**: `model.cache_dir`, `model.revision`, and
  `model.local_files_only` are passed to both `from_pretrained` calls, so a tier can run with
  the network disabled once `scripts/fetch_backbone.py` has populated the cache. That script
  writes `backbone_cache.json` (`repo_id`, resolved sha, file list, bytes,
  `verified_offline_load`) and refuses `run_mode=dummy`.

**Invariant.** `I1.5` If `BackboneSpec.revision` is set, the resolved snapshot commit must
equal it; a mismatch raises rather than loading a different checkpoint.

**Config keys.** `model.cache_dir`, `model.local_files_only`, `model.revision`, `model.model_id`,
`model.max_seq_len`, `model.quantization`, `model.device`.

**Equations.** `θ_{k+1} = θ_k = θ_0`; `W_l^adapt = W_l^0 + ΔW_l`.

**Invariants.**
- `I1.1` No backbone parameter ever has `requires_grad=True`.
- `I1.2` `backbone_fingerprint` is bit-identical before and after any adapter training.
- `I1.3` Gradient checkpointing / k-bit preparation must not re-enable backbone grads
  (a known PEFT footgun — explicitly re-asserted after those calls).
- `I1.4` At most one full-precision backbone instance is resident at a time
  (`BackboneCache` is a single-slot cache keyed by run mode; a second distinct request
  evicts rather than duplicates).

**Tests.** `tests/test_backbone_frozen.py`, `tests/test_no_backbone_duplication.py`

---

## M2 — Data architecture and split isolation

**Modules:** `src/amaun/data/{registry,splits,preprocess,loaders,leakage}.py`

**Purpose.** Produce four disjoint splits per task, and make leakage structurally impossible.

**Contract.**
- `TaskSpec` = `(task_id, task_name, task_type, hf_id, revision, config_name, license,
  primary_score, prompt_template, max_input_tokens, max_target_tokens, volumes)`.
- `SplitName` enum: `TRAIN | VALIDATION | SUPPORT | TEST`.
- `build_splits(task_spec, seed, volumes) -> TaskSplits` derives SUPPORT from the *training
  pool*, never from the test pool, using a seeded deterministic partition of stable example
  ids (`sha256(prompt + target)` — not row index, so it is stable across dataset shuffles).
- `SplitAccessGuard`: every consumer declares the splits it is permitted to read; a read
  of an undeclared split raises `SplitAccessViolation`. The manager's guard permits
  `{SUPPORT}` only; the evaluator's permits `{TEST}` only.
- `synthetic` provider generates deterministic pseudo-data per task type so tests and
  `dummy` runs need no network.
- Unpinned datasets raise `UnpinnedDatasetError` rather than guessing. "Pinned" means a
  specific revision: every `hf_revision` is a 40-hex commit sha, or `refs/convert/parquet`
  recorded together with that ref's resolved sha (D-006).
- **Row adapters** (`row_adapters.py`) map one upstream row to
  `list[(prompt fields, target)]`. The adapter is named per task in config
  (`datasets.tasks.<id>.row_adapter`) with **no default**; unset or unknown raises
  `UnresolvedRowAdapterError` (D-015). `passthrough` selects `input_fields` directly and
  reduces list/dict targets by the explicit `list_target_pick` option. `TaskSpec.input_fields`
  means *prompt-template slots*; upstream column names are the adapter's concern, and by
  convention the two coincide wherever the upstream schema permits.
- **Snapshots** (`snapshots.py`) cache the *capped* sample of raw upstream rows under
  `data.cache_dir` as `<task>/<split>-<key>.jsonl` + `.manifest.json`. The key hashes every
  content-determining input (`hf_id`, `hf_revision`, `hf_config`, split, size, `skip`,
  `sampling`, `shuffle_buffer_size`, row adapter + options, required fields), so a stale
  snapshot can never be silently reused. A cache hit performs **no network call**, which is
  what makes an offline re-run possible.
- **Volume caps come only from config.** Under `provider: huggingface`,
  `data.volumes.<task>.{train_pool,test,test_split}` are mandatory; there is no code default
  (M0). `data.sampling` and `data.shuffle_buffer_size` are likewise mandatory with no code
  default; per D-016 the configured policy is `shuffled_buffer`, and the buffer must satisfy
  `≥ max(train_pool, test)` or the shuffle is a no-op. The shuffle seed derives from
  `data.seed` and the task id, and is identical across both loads of a task.
- **Single-split corpora.** When the TEST pool is drawn from the same upstream split as the
  training pool (e.g. `databricks/databricks-dolly-15k`, which has only `train`),
  `data.volumes.<task>.test_offset` is **required** and is applied as a streaming `.skip(n)`.
  HF streaming does not support slice syntax (`split="train[12000:]"` raises `ValueError:
  Bad split`), so an offset is the mechanism.

**Invariants.**
- `I2.1` `hash(SUPPORT) ∩ hash(TEST) = ∅`, likewise for every other split pair.
- `I2.2` SUPPORT ⊂ training pool; no example in SUPPORT derives from the upstream test split.
- `I2.3` Split assignment is a pure function of `(task_id, example_hash, seed, ratios)`.
- `I2.4` The manager cannot obtain a `TEST` iterator — enforced by type, then by guard.
- `I2.5` Under `local_smoke`, subsampling applies uniformly to all four splits with one
  config key (`data.smoke_max_examples_per_split`), defined once.
- `I2.6` When the TEST pool shares an upstream split with the training pool,
  `test_offset ≥ rows_consumed_by_the_training_pool`, checked against the **measured** row
  count (which differs from the example count whenever a row adapter expands rows). Violation
  raises `DataError`. Independently, `assert_disjoint` still catches any resulting content
  overlap — the offset is the fix, the hash check is the proof.
- `I2.7` A snapshot is reused only when its manifest matches the requested spec exactly; a
  cache hit makes no network call.
- `I2.7a` Under `sampling: shuffled_buffer` the stream is shuffled **before** it is skipped,
  with a seed derived from `(data.seed, task_id)`, so two loads of one task see the same row
  order. Without this, `I2.6`'s offset is not a disjointness argument.
- `I2.8` Only `unpinned_dataset` (D-006) and `unresolved_row_adapter` (D-015) may cause a task
  to be skipped. Any other failure propagates, so a transient network error can never
  masquerade as an open decision.

**Config keys.** `data.provider`, `data.volumes.<task>.{train_pool,test,test_split,test_offset}`,
`data.split_ratios.*`, `data.smoke_max_examples_per_split`, `data.cache_dir`, `data.seed`,
`data.sampling`, `data.shuffle_buffer_size`, `data.offline_snapshots`,
`datasets.tasks.<id>.{hf_id,hf_revision,hf_config,row_adapter,row_adapter_options}`.

**Tests.** `tests/test_split_isolation.py`, `tests/test_data_registry.py`,
`tests/test_row_adapters.py`, `tests/test_data_snapshots.py`,
`tests/test_single_split_carving.py`, `tests/test_prepare_data_blocked.py`

---

## M3 — LoRA adapters and the delta abstraction

**Modules:** `src/amaun/adapters/{lora,delta,metadata,repository}.py`

**Purpose.** Train, store, load, switch, and reason about adapters as *effective updates*.

**Contract.**
- `LoRASpec` = `(r, alpha, dropout, target_modules, bias, task_type)`.
- `AdapterDelta` is the canonical representation: a mapping
  `layer_key -> (A: [r, d_in], B: [d_out, r], scaling: float)` with
  `delta(layer_key) = scaling · B @ A` materialised **lazily** and never stored densely
  unless `fusion.storage=dense_delta`.
- `AdapterMetadata` (persisted as JSON next to the weights) carries every field the brief
  §6 requires: `adapter_id, task_name, task_type, dataset, dataset_revision,
  training_configuration, creation_time, model_version, backbone_fingerprint, lora_rank,
  target_modules, performance_metrics, task_embedding, resource_cost, parameter_count,
  storage_bytes, provenance` where `provenance` records `origin ∈ {trained, fused,
  aggregated}`, parent adapter ids, and `tasks_served: [task_id]` (required by D-014).
- `AdapterRepository` supports `register, retrieve, list, reuse, fuse, create, evaluate,
  archive, compress, snapshot, load_snapshot`. Archive is soft (moves to
  `archive/`, never deletes) so growth history is auditable.
- **Candidate isolation:** `A_temp` lives in a `CandidateStore` on a separate path. The
  repository refuses to register an adapter whose id is currently held by the candidate
  store unless a `Decision` object authorising it is passed
  (`register(adapter, authorised_by=decision)`).

**Invariants.**
- `I3.1` `save → load` round-trips `AdapterDelta` bit-exactly.
- `I3.2` A temporary adapter cannot enter `R_c` without a `Decision`.
- `I3.3` `backbone_fingerprint` in metadata must match the live backbone at load time, else
  `IncompatibleAdapterError` (this is also the federated compatibility primitive).
- `I3.4` `parameter_count` and `storage_bytes` are measured, never estimated from formulas.
- `I3.5` Adapter switching leaves no residual delta: applying then removing an adapter
  restores the exact base forward output.

**Tests.** `tests/test_adapter_io.py`, `tests/test_adapter_switching.py`,
`tests/test_repository.py`, `tests/test_candidate_isolation.py`

---

## M4 — LoRA / QLoRA training

**Module:** `src/amaun/training/lora_trainer.py`

**Purpose.** Train exactly one adapter on exactly one task's TRAIN split, from `θ₀`.

**Contract.**
- `train_adapter(task, lora_spec, train_cfg, resolved) -> (AdapterDelta, TrainingRecord)`.
- **Independence guarantee:** the trainer takes a *fresh* PEFT injection on a backbone whose
  fingerprint is verified pre- and post-training, a fresh optimiser, and a per-adapter
  derived seed `seed_i = derive_seed(base_seed, adapter_id)`. No state crosses adapters.
- Loss is next-token CE on the *response* tokens only (prompt tokens masked with `-100`),
  because five of six tasks are generation and scoring prompt tokens would corrupt the
  objective.
- Supports gradient accumulation, gradient checkpointing, bf16/fp16, 4-bit (CUDA only),
  configurable `max_seq_len`, cosine/linear schedules, early stopping on VALIDATION.
- `TrainingRecord` logs loss trajectory (needed by M5's `A_temp` statistics), steps, wall
  time, peak memory, tokens seen, and the resolved training config.

**Invariants.**
- `I4.1` Optimiser param groups contain **only** adapter parameters.
- `I4.2` `assert_backbone_frozen` passes before every step.
- `I4.3` Training adapter *k* with seed `s` yields an identical delta regardless of whether
  adapters `1..k-1` were trained first in the same process (the independence test).
- `I4.4` Prompt tokens contribute zero loss.

**Config keys.** `training.*` (lr, epochs/steps, batch, grad_accum, warmup, scheduler,
weight_decay, max_grad_norm, precision, gradient_checkpointing, early_stopping).

**Tests.** `tests/test_lora_training.py`, `tests/test_adapter_independence.py`,
`tests/test_loss_masking.py`

---

## M5 — Task representation

**Module:** `src/amaun/manager/task_representation.py`

**Purpose.** Represent an incoming task from its SUPPORT set only.

**Contract.** `TaskRepresentation` = `(embedding: np.ndarray, stats: dict, descriptor: str)`.
Configurable encoders, all reading SUPPORT only:

| `task_repr.encoder` | Mechanism |
|---|---|
| `backbone_mean_hidden` | mean last-hidden-state of the frozen backbone over SUPPORT prompts (no extra model — default recommendation) |
| `sentence_encoder` | an external fixed sentence encoder (adds a dependency and a second model in memory) |
| `token_stats` | vocabulary/length/type distributional descriptor only (encoder-free ablation) |
| `hybrid` | concatenation of the above |

`stats` carries support-set statistics: n, input/target length distribution, type-token
ratio, vocabulary coverage, task-type one-hot, and label-structure descriptors where they exist.

**Invariants.** `I5.1` the encoder is identical across all compared methods (`idea.md` §4.1
"fixed encoder, identical across methods"). `I5.2` representation is a pure function of
(SUPPORT, encoder cfg, seed). `I5.3` no TEST access (guard-enforced).

**Tests.** `tests/test_task_representation.py`

---

## M6 — Multi-criteria feature extraction (7 families)

**Modules:** `src/amaun/features/{semantic,parameter,gradient,performance,continual,historical,resource,extractor,normalizer}.py`

**Purpose.** Build `z_i` for each candidate pair `(A_temp, A_i)`. **Never a single scalar.**

Each family is an independently toggleable `FeatureFamily` with a stable, named, ordered
output vector so ablations A1/A6/A7 are exact zero-code-change toggles and feature indices
remain interpretable.

### 6.1 Semantic (`z_sem`) — `idea.md` §4.1
`d_task(i) = ||e_t − e_i||₂²` (**Euclidean, not cosine**);
`d_hidden(i) = (1/N) Σ ||h_t(x_n) − h_i(x_n)||₂²`;
`D_out(i) = (1/N) Σ KL(p_i(y|x_n) ‖ p_temp(y|x_n))`.
Cosine similarity **may** be logged as one additional feature but is never the rule.

### 6.2 Parameter (`z_par`) — `idea.md` §4.2
Per layer `l`: `d_F`, `d_2`, `v_l`, `d_rank`, `d_sv`; aggregated over layers by
`features.parameter.aggregation ∈ {mean, max, mean+std, per_layer}`. `rank` uses a
configurable tolerance; `σ(·)` is the singular-value spectrum of the effective `ΔW`.
Computed on `ΔW = BA` (never on `A`/`B` separately).

### 6.3 Gradient (`z_grad`) — `idea.md` §4.3
A single **fixed reference minibatch protocol** shared by both adapters of a pair:
`a_grad = (g_t·g_i)/(‖g_t‖‖g_i‖+ε)`, `m_g = ‖g_t − g_i‖₂`, `v_g = Var(g_t − g_i)`, plus norms.
`g_i` is the gradient of the *representative* loss for adapter `i`; `g_t` of the task loss.
Config: `features.gradient.reference_batch_size`, `.seed`, `.microbatches`.

### 6.4 Performance (`z_perf`) — `idea.md` §4.4, on VALIDATION/SUPPORT only
`primary_score` (D-009, replacing `Acc`), `L_val`, mean max confidence `C`, mean predictive
entropy `H`, `ECE` with `features.performance.ece_bins` fixed bins.

### 6.5 Continual (`z_cl`) — `idea.md` §4.5
`F_i = (1/M) Σ_j max(0, s_{j,pre} − s_{j,post})`, `KR = 1 − F`,
`P = s_new,post − s_new,pre`, `S = 1 − F + ρ·P` with `ρ = features.continual.rho`.

### 6.6 Historical (`z_hist`) — `idea.md` §4.6
`Age, Freq, Merge_count, PerfHist, last_use` from repository metadata.

### 6.7 Resource (`z_res`) — `idea.md` §4.6
`Mem_i = |φ_i|·b` (measured), `Comm_i = Mem_i + |meta_i|`, `RepoCost = Σ Mem_j`,
`Latency_i = T_infer(A_i)` measured under a fixed profiling protocol.

### 6.8 Normalisation — `idea.md` §4.7 (leakage-critical)
`z' = (z − μ)/(σ + ε)`. `FeatureNormalizer` persists `μ, σ`, the fitting split name, the
run mode, and the feature-name order. `transform()` raises if `fit()` was called on
anything other than the utility-training split, or if the run mode differs.

**Invariants.**
- `I6.1` No decision function receives a single scalar similarity.
- `I6.2` `μ, σ` come only from the utility-training split; never re-estimated on test.
- `I6.3` Normaliser artefacts are namespaced by run mode and refuse cross-loading.
- `I6.4` Disabling a family changes `z`'s dimension and is recorded in the run log
  (so an ablation cannot silently be a no-op).
- `I6.5` Feature vector order is stable and stored with the normaliser.
- `I6.6` Every feature is finite; non-finite values are replaced by a configured sentinel
  **and counted** in the run log.

**Tests.** `tests/test_features_*.py`, `tests/test_normalizer_leakage.py`

---

## M7 — Utility network and the REUSE/FUSE/CREATE decision

**Modules:** `src/amaun/manager/{utility_network,decision,retrieval,targets,threshold}.py`

### 7.1 Network — `idea.md` §5.1
```
h1 = ReLU(W1 z + b1);  h2 = ReLU(W2 h1 + b2);  o = W_o h2 + b_o
û  = [ŝ_post, F̂, ĉ, M̂em, Ĉomm, L̂]ᵀ        # 6 predicted post-integration outcomes
U  = wᵀ ψ(û)                               # ψ = monotonic transform, w = learned weights
```
`ψ` orients every head so that larger is better (costs and forgetting are negated/inverted)
and rescales to comparable ranges; `w` is learned, non-negative (softplus-parameterised) so
utility is monotone in each oriented outcome — otherwise the network could "prefer" more
forgetting. Width, depth, dropout, activation, optimiser, LR, epochs: config only.

### 7.2 Retrieval
`top_k(R_c, z) -> [candidate]` supports `retrieval.mode ∈ {single, top_k}` and
`retrieval.k`. Ranking is by predicted utility `U_i`, never by raw similarity.

### 7.3 Decision — `idea.md` §5.2 generalised to three outcomes (D-003)
```
i*  = argmax_i U_i
REUSE  if  U_i* ≥ τ_reuse  and  the reuse pre-screen meets the support-set bar
FUSE   if  U_i* ≥ τ_u      (top-k set passed to M8)
CREATE if  max_i U_i < τ_u
```
Both decision orders of D-003 are implemented (`manager.decision_order`). Every decision
emits a `Decision` record: outcome, ranked candidates, all `U_i`, all `û_i`, the feature
vector hash, the thresholds, the config hash, and the reason string.

### 7.4 Supervision — `idea.md` §9
`y_i = [s_post, F_post, C_post, Mem_post, Comm_post, L_post]ᵀ` obtained by **actually
performing** the candidate integration and measuring the outcome;
`L_U = Σ_i ‖g_ω(z_i) − y_i‖₂²`. Sampling/proxy policy per D-010.

**Invariants.**
- `I7.1` The utility network never sees a TEST-derived feature or target.
- `I7.2` `τ_u`/`τ_reuse` are selected only via the declared source (D-005) and logged.
- `I7.3` Given identical inputs the decision is deterministic.
- `I7.4` `U` is monotone non-decreasing in each oriented outcome head.
- `I7.5` `CREATE` is chosen when `R_c = ∅`, without invoking the network (`idea.md` §5.4 L2–4).
- `I7.6` Decision records are append-only JSONL.

**Tests.** `tests/test_utility_network.py`, `tests/test_decision.py`,
`tests/test_utility_no_test_leakage.py`, `tests/test_threshold_selection.py`

---

## M8 — Layer-wise learnable fusion + KD

**Modules:** `src/amaun/fusion/{layerwise,fused_module,refactorize}.py`,
`src/amaun/training/kd.py`

### 8.1 Fusion — `idea.md` §6.1–6.2, generalised per D-004
```
ΔW_merge_l = Σ_{k=1..m} α_{l,k} ΔW_{l,k},     Σ_k α_{l,k} = 1,  α_{l,k} ∈ (0,1)
α_l = softmax(a_l),  a_l ⁽⁰⁾ = 0  →  α_{l,k}⁽⁰⁾ = 1/m  (= 0.5 for m = 2)
m = 2 reduces exactly to α_l = sigmoid(a_l), ΔW = α ΔW_old + (1−α) ΔW_temp
```
`FusedLoRALinear` wraps the frozen base `Linear` and computes
`y = x W₀ᵀ + Σ_k α_{l,k} · scale_k · (x A_kᵀ) B_kᵀ` — mathematically identical to applying
the fused dense `ΔW_merge_l`, but **never materialising a `d_out × d_in` matrix**, which is
what makes this tractable at 7B. `refactorize.py` provides the dense/SVD/component storage
options of D-013 and reports per-layer reconstruction error.

### 8.2 KD — `idea.md` §6.4 (teacher = pre-merge OLD adapter)
```
q_old  = softmax(s_old / T)          # captured BEFORE fusion starts, on D_rep
q_merge= softmax(s_merge / T)
L_KD   = T² · (1/N_r) Σ KL(q_old ‖ q_merge)
L_total= L_new + λ · L_KD
```
Teacher logits are captured once and cached (optionally top-p truncated to bound memory,
with the truncation recorded). KD is computed on response positions only, masked identically
to the task loss.

### 8.3 Joint optimisation — `idea.md` §6.5
`φ_merge ← φ_merge − η_φ ∇L_total` and `a ← a − η_α ∇L_total`, in the same backward pass,
with separate LR groups.

**Invariants.**
- `I8.1` `α` at step 0 equals `0.5` for every layer when `m = 2` (asserted numerically).
- `I8.2` `Σ_k α_{l,k} = 1` at every step (softmax) and `α ∈ (0,1)`.
- `I8.3` Fusion output equals the dense `ΔW_merge` application to within float tolerance
  (verified against an explicitly materialised reference on a tiny model).
- `I8.4` Source adapters in `R_c` are unchanged by fusion unless the experiment declares
  otherwise; verified by delta hashing before/after.
- `I8.5` `a_l` receives non-zero gradient (a test that would catch a detached α — the single
  most likely silent bug in this module).
- `I8.6` Teacher logits are captured before any fusion parameter update; asserted by a
  step-order flag.
- `I8.7` Trainable-parameter count under fusion is reported: `|φ| + L·m` (D-014 / brief
  falsification question 8).
- `I8.8` `λ = 0` reproduces the no-KD arm exactly.

**Tests.** `tests/test_fusion_math.py`, `tests/test_fusion_alpha_init.py`,
`tests/test_fusion_grad_flow.py`, `tests/test_kd.py`, `tests/test_fusion_frozen_sources.py`

---

## M9 — Repository lifecycle and compression

**Module:** `src/amaun/adapters/repository.py`, `src/amaun/manager/compression.py`

`idea.md` §7.1–7.2:
```
R'_c = (R_c \ {A_i*}) ∪ {A_merge}      # FUSE
R'_c = R_c ∪ {A_temp}                  # CREATE
R'_c = R_c                             # REUSE
Mem(R_c) = Σ Mem(A_i)
compress while Mem(R_c) ≥ R_max:  (i*,j*) = argmax_{i≠j} U_merge(A_i,A_j)
                                  R_c ← (R_c\{A_i*,A_j*}) ∪ {A_merge_ij}
```

**Invariants.** `I9.1` compression uses the utility network, never age/usage
(`idea.md` §16.7) — a heuristic variant exists **only** as the `B_compress_heuristic`
baseline and is registered as such. `I9.2` compression terminates (`|R_c|` strictly
decreases each iteration; a floor `R_min` prevents collapse to one adapter).
`I9.3` every lifecycle transition is journaled to `repository_events.jsonl`.
`I9.4` `tasks_served` accumulates on merge, so D-014 accounting is always available.

**Tests.** `tests/test_compression.py`

---

## M10 — Continual-learning runner and metrics

**Modules:** `src/amaun/continual/{runner,metrics}.py`

`runner` executes the §5 pipeline over the configured task sequence for one client, emitting
one `TaskEvent` record per arrival. Metrics, computed from the standard performance matrix
`S[i][j]` = score on task `j` after learning task `i`:

```
AvgPerf_T = (1/T) Σ_j S[T][j]
Forgetting = (1/(T−1)) Σ_{j<T} max(0, max_{i<T} S[i][j] − S[T][j])
BWT = (1/(T−1)) Σ_{j<T} ( S[T][j] − S[j][j] )
FWT = (1/(T−1)) Σ_{j>1} ( S[j−1][j] − b_j )      # b_j = zero-shot/random baseline on task j
```

**Invariants.** `I10.1` `S` is filled only by the evaluator reading TEST after the decision
for row `i` is frozen. `I10.2` `b_j` (the FWT reference) is measured once, with the frozen
backbone and no adapters, and cached. `I10.3` the matrix is persisted so metrics can be
recomputed without re-running. `I10.4` metric definitions live in one module, used by every
arm including baselines.

**Tests.** `tests/test_cl_metrics.py`

---

## M11 — Federated layer

**Modules:** `src/amaun/federated/{client,server,aggregation,compatibility,comm}.py`

Four **distinct** axes, never conflated (brief §13): `TASK`, `CLIENT`, `ADAPTER`, `ROUND`.

`idea.md` §7.3:
```
C_k = { c : client c reports compatible adapter state k }
A_global_k = Σ_{c∈C_k} (n_c / Σ_{j∈C_k} n_j) · A_c,k
Payload_c  = |A_c| + |meta_c|
```

**Compatibility** (`compatibility.py`) is explicit and gated, never assumed: identical
backbone fingerprint, identical `target_modules`, identical rank (or a declared rank-aligning
policy), and a declared `compatibility.key ∈ {task_id, adapter_lineage, utility_cluster}`.
Aggregation of a group with `|C_k| = 1` is a no-op, not an error.

**Aggregation of LoRA factors is the subtle part.** Averaging `A` and `B` independently is
forbidden (`idea.md` §16.3). Implemented options:
`aggregation.method ∈ {delta_fedavg, svd_refactor_fedavg, component_concat}` where
`delta_fedavg` averages the *effective deltas* and `svd_refactor_fedavg` projects the
averaged delta back to rank `r`; both record reconstruction error.

**Invariants.** `I11.1` the backbone is never serialised into a payload — asserted by
inspecting payload keys. `I11.2` aggregation weights sum to 1 within tolerance.
`I11.3` `n_c` is the client's example count for that adapter's task(s), logged.
`I11.4` communication cost is *measured* from serialised payload bytes, not estimated.
`I11.5` incompatible adapters are never averaged.

**Tests.** `tests/test_federated_aggregation.py`, `tests/test_comm_accounting.py`

---

## M12 — Evaluation

**Modules:** `src/amaun/evaluation/{task_metrics,cl_metrics,efficiency,evaluator,stats}.py`

Task metrics: ROUGE-1/2/L (+optional BERTScore) for summarization; EM/F1 for QA; generation
quality + optional LLM-judge for instruction following; generation metrics for dialogue;
generation quality + perplexity for long-form; domain metrics for finance. Each task's
`primary_score` (D-009) is the single scalar used by decisions and CL metrics.

Efficiency: adapter count, tasks-served-per-adapter, trainable params, stored params,
storage bytes, communication bytes, training wall time, inference overhead.

`stats.py`: mean ± std over seeds, paired bootstrap CI over per-task scores, effect size.
**Single-seed differences are never reported as gains** (D-012).

**Invariants.** `I12.1` the evaluator is the only component with TEST access.
`I12.2` generation decoding config is identical across arms and logged.
`I12.3` metric implementations are shared by AMAUN and every baseline.

**Tests.** `tests/test_task_metrics.py`, `tests/test_stats.py`, `tests/test_evaluator_guard.py`

---

## M13 — Baselines

**Module:** `src/amaun/baselines/*.py`. See `docs/experiment_matrix.md` for ids.
Implemented: `B1` sequential full fine-tuning, `B2` separate LoRA per task,
`B3` naive uniform LoRA averaging, `B4` similarity-based selection,
`B5` fixed-coefficient fusion, `B6` compression-by-heuristic.
Registered-but-unimplemented (D-007): `LoRM`, `FedALoRA`, `FedZKD` — raise
`BaselineRequiresCitationError`.

**Invariant `I13.1` (budget parity).** A `BudgetLedger` records, per arm: optimiser steps,
tokens seen, examples seen, trainable parameters, wall time, and tuning trials. The
experiment runner **refuses to emit a comparison** between arms whose ledgers differ by more
than `experiment.budget_tolerance`, unless the run explicitly declares
`allow_unequal_budget: true` with a reason that is printed in the results.
This is the mechanical enforcement of brief §14's "do not give AMAUN additional
information" and falsification question 14.

**Tests.** `tests/test_baselines.py`, `tests/test_budget_parity.py`

---

## M14 — Experiment tracking and reproducibility

**Modules:** `src/amaun/utils/{tracking,provenance,seed,paths}.py`

- `results/<run_mode>/<experiment_id>/` and nothing else (`idea.md` §0.1.1), containing
  `config_used.json`, `provenance.json`, `metrics.jsonl`, `decisions.jsonl`,
  `repository_events.jsonl`, `fusion_logs/`, `federated_rounds/`, `utility_network/`,
  `repository_snapshots/`, `budget_ledger.json`, `performance_matrix.json`.
- `provenance.json`: git commit, dirty flag, diff hash, python/torch/transformers/peft
  versions, platform, device, CPU/GPU model, seed, hostname, UTC start/end.
- All records are JSONL with a `schema_version`.

**Invariants.** `I14.1` `run_mode` appears in every record. `I14.2` a run with a dirty git
tree is recorded as dirty and, under `actual`, requires
`experiment.allow_dirty_git: true`. `I14.3` reporting hooks default to `run_mode=actual`
and label anything else `[SMOKE TEST — not a reported result]`.

**Tests.** `tests/test_tracking_paths.py`, `tests/test_provenance.py`

---

## Cross-cutting: the ablation toggle contract

Every ablation in `docs/experiment_matrix.md` is a config toggle that provably changes
behaviour. `tests/test_ablation_toggles.py` asserts, for each id, that the toggle alters an
observable (feature dimension, decision distribution, α trajectory, loss terms present, or
repository trajectory) on the `dummy` tier — so an ablation can never silently be a no-op.
