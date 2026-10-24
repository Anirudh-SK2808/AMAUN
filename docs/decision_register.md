# AMAUN — Decision Register

Every research decision that `idea.md` or the master brief leaves open, or where the two
sources **conflict**, is recorded here. Nothing in this register may be resolved by an
implementation agent. Status is one of:

- `OPEN` — needs an explicit human decision. Code must expose all options with **no default**.
- `RESOLVED` — a human picked an option; the choice, the date, and the decider are recorded.
- `PROVISIONAL` — implemented behind a config flag so work can proceed, but the scientific
  choice is still open and **must not appear in a reported result** until resolved.

> Authority order when sources disagree: (1) explicit live user instruction,
> (2) `idea.md`, (3) the master brief, (4) agent inference. Agent inference never wins.

---

## D-001 — Backbone tier and checkpoint pinning

**Status:** `PROVISIONAL`

**Conflict.** The master brief §3 names `Qwen2.5-7B-Instruct` as *the* model for all six
initial adapters. `idea.md` §0.1 defines a two-tier scheme: `local_smoke` → "Qwen 1.5B",
`actual` → "Qwen 7B", with base (non-Instruct) names in the illustrative registry and an
explicit instruction to *pin the exact checkpoint tags once chosen*.

**Alternatives.**

| Option | Description | Trade-off |
|---|---|---|
| **O1 (implemented, provisional)** | Three tiers: `dummy` → locally-constructed tiny random Qwen2 config (no download, for tests); `local_smoke` → `Qwen/Qwen2.5-1.5B-Instruct`; `actual` → `Qwen/Qwen2.5-7B-Instruct` | Satisfies both sources. Adds a third tier beyond `idea.md`, needed because CI/unit tests cannot download 1.5B weights. |
| O2 | Two tiers exactly as `idea.md`, base (non-Instruct) checkpoints | Non-Instruct base is a worse fit: all six tasks are prompt/response formatted, so an instruct-tuned backbone is the honest baseline. |
| O3 | Single tier, 7B-Instruct only | Matches brief §3 literally, but makes local iteration impossible on the 24 GB dev box and violates `idea.md` §14's "smoke first" gate. |

**Recommendation:** O1. The `dummy` tier is a *test fixture*, not a research tier: it is
hard-blocked from producing any reportable metric (see `src/amaun/config.py`
`RunMode.is_reportable`).

**Blast radius:** every result table. Reported numbers may only come from `actual`.

**Decision:** _pending user sign-off_.

---

## D-002 — Repository initialisation: parallel bootstrap vs. empty sequential

**Status:** `OPEN` — **blocks the headline experiment's interpretation.**

**Conflict — this is the most consequential one.**

- Master brief §1 requires an **initial adapter repository of exactly six** LoRA adapters,
  each trained **independently** from the same frozen `θ₀` (explicit fan-out diagram,
  explicit prohibition on `Base → A1 → A2 → …`), and only *then* future tasks arriving
  incrementally through REUSE / FUSE / CREATE.
- `idea.md` §3 requires `R_c⁰ = ∅` and streams LoRA-1..6 **through** the merge-or-create
  decision. Under that protocol the repository may never contain six adapters — merging
  is the expected outcome, so the final count is an experimental *observable*, not a given.

These protocols answer different questions and produce different headline tables.

**Note on a false conflict:** "independently trained" is *not* in tension with sequential
arrival. Because the backbone is frozen for the entire lifecycle (`θ_{k+1} = θ_k = θ_0`),
every temporary adapter `A_temp` is trained from `θ₀` by construction. Sequential *arrival*
does not create sequential *training*. The real difference is only whether the six seed
adapters bypass the decision module.

**Alternatives.**

| Option | Protocol | What the headline table then means |
|---|---|---|
| **O1** | `bootstrap_six`: Phase A trains 6 adapters independently from `θ₀`, inserting all six into `R_c` unconditionally. Phase B streams new tasks `T₇…` through REUSE/FUSE/CREATE. | Measures decision quality **against a populated, diverse repository**. Directly matches the brief. Requires ≥2 genuinely new tasks beyond the six to have anything to decide about. |
| O2 | `empty_sequential`: exactly `idea.md` §3. | Measures **repository growth control from scratch**. Adapter-count-vs-task-index is the primary result. But with only 6 maximally-dissimilar tasks, CREATE is the a-priori correct answer every time, so the decision module is barely exercised. |
| O3 | Both, as separate registered experiments (`main_bootstrap`, `main_sequential`). | Costs ~2× compute. Strongest paper: O1 shows decision quality, O2 shows growth control. |

**Recommendation:** O3, with O1 as the primary experiment. O2 alone cannot support the
research question, because the core question asks whether the system can *choose* among
reuse/fuse/create — and a six-task curriculum of maximally distinct task types offers
almost no genuine reuse opportunity.

**Consequence if O1 is chosen:** the curriculum **must be extended past LoRA-6** with tasks
that plausibly overlap existing adapters (e.g. XSum → overlaps Summarization;
a second extractive QA set → overlaps QA; a second dialogue set → overlaps Dialogue).
`idea.md` §0.2 already declares the curriculum extensible for exactly this reason.
Without such tasks, a REUSE or FUSE decision can never be *correct*, and the experiment
cannot falsify H2/H3. **This is a required specification addition, not an optimisation.**

**Implemented as:** `configs/experiments/*.yaml` → `curriculum.repository_init ∈
{bootstrap_six, empty_sequential}`, **no default**.

**Decision:** _pending user sign-off_.

---

## D-003 — Two-way (MERGE / CREATE) vs three-way (REUSE / FUSE / CREATE) decision

**Status:** `OPEN` — **methodology addition beyond `idea.md`.**

**Conflict.** The brief §9 mandates three outcomes. `idea.md` §5.2 defines two:
`Merge(A_temp, A_i*)` if `U_i* ≥ τ_u`, else `CreateNew(A_temp)`. Under `idea.md` a
temporary adapter is *always* trained first, so there is no outcome in which the system
avoids paying training cost by simply reusing an existing adapter.

**Why this matters scientifically.** REUSE is the only outcome that reduces both adapter
count *and* training cost. If it is absent, the framework's efficiency claim rests solely
on merging, and the brief's falsification question 7 ("is REUSE actually reducing adapter
growth?") is unanswerable.

**Alternatives.**

| Option | Mechanism | Cost |
|---|---|---|
| **O1** | Three-way head. A cheap **pre-screen** runs on the support/calibration set *before* any temporary training: if some existing adapter (or fused set) already meets a utility bar, emit REUSE and train nothing. Otherwise train `A_temp` and run the `idea.md` decision, which yields FUSE or CREATE. | Cheapest. Deviates from `idea.md` §5.4 line 1 ("train `A_temp` first, always"). |
| O2 | Keep `idea.md`'s always-train-first order; add REUSE as a third utility-head outcome where `A_temp` is *discarded* and `R_c` is unchanged. | Faithful to `idea.md`; no training-cost saving, only storage saving. Honest but weaker efficiency claim. |
| O3 | Two-way only, as `idea.md`. Report REUSE as out of scope. | Contradicts the brief and the stated research question. |

**Recommendation:** O1 **and** O2 as configurable decision orders
(`manager.decision_order ∈ {prescreen_then_train, train_then_decide}`), reporting both.
O1's training-cost saving is a headline efficiency result; O2 isolates the storage effect
without the confound of differing training budgets. Reporting both also directly answers
falsification question 14 (is the gain from the mechanism or from differing compute?).

**Decision (2026-10-22, user instruction "add a step for reuse before training"):** O1 is
implemented as a stage-1 REUSE gate (`src/amaun/manager/reuse_gate.py`). With
`manager.decision_order: prescreen_then_train` and a fitted gate, REUSE is decided from temp-free
SUPPORT features **before** `A_temp` is trained; if the gate does not fire, the `idea.md` order
runs (train `A_temp` -> utility network -> FUSE | CREATE) with post-training REUSE disabled. O2
(`train_then_decide`) remains selectable, and the L4 decision study reports both orders on the
same measured options (`amaun` vs `amaun_train_first`). Because `prescreen_then_train` is the
default in `configs/utility/default.yaml`, every arm whose bootstrap fits a gate now skips
training when the gate fires; before this change the setting was accepted but ignored.

---

## D-004 — Fusion arity: pairwise vs top-k

**Status:** `PROVISIONAL` (safe generalisation — reduces exactly to `idea.md`)

**Conflict.** Brief §10: `ΔW_l = Σ_k α_{l,k} ΔW_{l,k}` over top-k adapters.
`idea.md` §6.1: strictly pairwise, `ΔW_merge_l = α_l·ΔW_old_l + (1-α_l)·ΔW_temp_l`,
with `α_l = sigmoid(a_l)` and `α_l⁽⁰⁾ = 0.5`.

**Resolution (mathematically exact).** Parameterise `m` contributing deltas per layer with
per-layer logits `a_l ∈ R^m` and `α_l = softmax(a_l)`, initialising all logits to `0` so
every `α_{l,k}⁽⁰⁾ = 1/m`. For `m = 2` this gives `α⁽⁰⁾ = (0.5, 0.5)` and
`softmax([a₁,a₂])₁ = sigmoid(a₁ - a₂)`, i.e. **the `idea.md` pairwise sigmoid form is the
exact `m = 2` case** of the general rule, including its initialisation. Normalisation
(`Σ_k α_{l,k} = 1`) and boundedness (`α ∈ (0,1)`) are preserved for all `m`.

`configs/fusion/*.yaml` → `coefficient.parameterisation ∈ {softmax_k, sigmoid_pair}` with
`sigmoid_pair` available for exact `idea.md` reproduction, plus
`coefficient.granularity ∈ {layerwise, global}` for ablation A8.

**Decision:** provisionally accepted as a strict generalisation. Flag if you disagree.

---

## D-005 — What "validation" selects `τ_u`

**Status:** `OPEN`

`idea.md` §10 says `τ_u` is "selected via validation" without saying which validation.
Selecting `τ_u` on the same task stream that is later reported is selection on test data
(brief development rule 4, `idea.md` §16.6).

**Alternatives.** (a) A disjoint **decision-development task stream** — tasks reserved
purely for tuning the manager, never evaluated. (b) Per-task support/calibration sets only,
with `τ_u` chosen by cross-validation over *historical* task events. (c) Fixed `τ_u` from
the first `k` task events, frozen thereafter.

**Recommendation:** (a) primary, (b) as a cheaper fallback; both must be declared per run.
Implemented as `manager.threshold.selection ∈ {dev_stream, historical_cv, fixed}` with
`no default`. Test splits are hard-blocked from the selection code path.

**Decision:** _pending user sign-off_.

---

## D-006 — Dataset pinning, availability, and licensing

**Status:** `RESOLVED` — 2026-09-22.

**Decided by:** the user, on 2026-09-22, **relayed to the planning agent by the coordinating
orchestrator agent rather than typed directly into this session.** That provenance is recorded
here deliberately: an agent relay is not itself authority. If the attribution is wrong, revert
this entry to `OPEN` *before any `actual` run* — these two rows determine what Adapters 3 and 5
**are**, and the blast radius is every result table.

`idea.md` §0.2 and brief §4 name six datasets loosely. Final pinned table (schemas and commit
shas verified against the live HF API on 2026-09-22):

| Task | HF id | Revision | License | Note |
|---|---|---|---|---|
| Summarization | `abisee/cnn_dailymail` | commit sha; `hf_config: 3.0.0` | Apache-2.0 (scripts), news content under fair-use research terms | `3.0.0` is a **config name, not a git ref** — the repo has only `main`/`script` branches |
| QA | `rajpurkar/squad` | commit sha; `hf_config: plain_text` | CC BY-SA 4.0 | share-alike applies to derivatives |
| Instruction following | **`databricks/databricks-dolly-15k`** (B1) | commit sha | CC BY-SA 3.0 | share-alike reaches derivatives. **One split only** (`train`, 15 011 rows) → TEST must be offset-carved, see D-015 and `specs/local-smoke-data-bootstrap/design.md` §4 |
| Dialogue | `li2017dailydialog/daily_dialog` | `refs/convert/parquet` + its resolved sha | CC BY-NC-SA 4.0 | **non-commercial only**, must be stated in any release. `main` holds only a loading script, which `datasets` ≥ 4 cannot execute; the parquet conversion is the *same* dataset, mechanically converted |
| Long-form generation | **`Pavithree/eli5`** (B2) | commit sha | **none declared on the mirror**; content Reddit-derived | `eli5_category` was rejected: its legacy script-loading format no longer loads. The undeclared license is a **residual publication risk** — a data statement must disclose it |
| Domain-specific | `dreamerdeo/finqa` | commit sha | MIT (per source repo) | re-verified 2026-09-22. No `context` column; see D-015 (b) |

**B1 — resolved to `databricks/databricks-dolly-15k`.** Rejected alternatives were
`Muennighoff/flan` and `SirNeural/flan_v2` (much larger, mixture composition harder to state
precisely). Consequence: Adapter 3 is an *instruction-following* adapter over a 15 k
human-authored set, not a FLAN-mixture adapter; the paper must say so rather than "FLAN-style".
Real schema is `instruction`, `context`, `response`, `category` — **not** the
`instruction`/`input`/`output` originally assumed in `configs/datasets/curriculum.yaml`.
`context` is empty for many rows; an empty value is valid input, not a missing field.

**B2 — resolved to `Pavithree/eli5`.** A community mirror of the original ELI5 with proper
`train`/`validation`/`test` splits (216 147 / 3 020 / 10 000). Real schema is `title`,
`selftext`, `document`, `subreddit`, `url`, `answers{a_id,score,text}` — **not** the
`question`/`answer` originally assumed. Which answer becomes the target is a live sub-decision
(D-015 (c)).

**Remaining consequence:** "pinned" in this repo means *a specific revision*. Five of the six
rows previously carried no revision at all; `specs/local-smoke-data-bootstrap/` pins all six to
a 40-hex commit sha and adds a test that enforces the format, so "pinned but floating on
`main`" cannot recur.

---

## D-015 — Row→example adaptation for multi-field / list-valued upstream schemas

**Status:** `RESOLVED` — 2026-09-22. All three sub-parts decided as recommended:

| Sub-part | Chosen | Config |
|---|---|---|
| (a) `dialog` turn segmentation | **`dialog_last_turn`** | `datasets.tasks.dialog.row_adapter: dialog_last_turn` |
| (b) `domain` context composition | **`finqa_pre_table_post`**, `table_format: markdown` | `datasets.tasks.domain.row_adapter: finqa_pre_table_post` + `row_adapter_options.table_format: markdown` |
| (c) `longform` target selection | **`highest_score`** | `datasets.tasks.longform.row_adapter_options.list_target_pick: highest_score` |

**Decided by:** the user, on 2026-09-22, **relayed by the coordinating orchestrator agent**
rather than typed into the planning session — the same provenance caveat as D-006 applies. If
the attribution is wrong, revert to `OPEN` before any `actual` run: (a) and (b) change what
Adapters 4 and 6 *are*, and (c) changes Adapter 5's targets.

**Consequences to state in the paper.** `dialog` yields one example per dialogue (~11 k train),
not one per turn prefix, so its volume is comparable to the other tasks. `domain`'s context is
FinQA's original `pre_text` + markdown-serialised `table` + `post_text`, which is long against
`max_input_tokens: 768` — truncation rates must be measured, not assumed. `longform`'s target
is the highest-scored ELI5 answer, **not** `answers.text[0]`; any earlier synthetic-data run is
therefore not comparable at the target level.

The **mechanism** stays as designed: the rule is named per task in config with no code default,
and an unset or unknown name still raises `UnresolvedRowAdapterError`. Resolving the decision
selects a value; it does not remove the guard.

The alternatives considered, retained for the record:

Three of the pinned datasets do not expose the fields the task contract needs, so a rule must
map an upstream row to `(prompt fields, target)`. Each rule changes **what the adapter learns**,
so none may be chosen by an agent. Every option is reachable by config
(`datasets.tasks.<id>.row_adapter` + `row_adapter_options`) with **no default**; an unset value
raises `UnresolvedRowAdapterError` and the task is skipped and reported, never guessed.

**(a) `dialog` — DailyDialog turn segmentation.** Upstream row is `dialog: list[str]`
(+ `act`, `emotion`); the task contract wants `history` and `response`.

| Option | Rule | Trade-off |
|---|---|---|
| **`dialog_last_turn`** | `history` = all but the last utterance; `target` = last utterance | One example per dialogue (~11 k train). The **only** reading consistent with the already-configured prompt template ("Continue the conversation naturally… Response:") |
| `dialog_all_prefixes` | one example per prefix of ≥ 1 turn | ~7× more examples, shifts the context-length distribution, and changes the effective training volume relative to every other task |

*Recommendation:* `dialog_last_turn` — lowest risk, matches the existing template, keeps
per-task volumes comparable.

**(b) `domain` — FinQA context composition.** Upstream row has `pre_text`, `post_text`,
`table` (list of rows), `question`, `answer`; the task contract wants `context`.

| Option | Rule | Trade-off |
|---|---|---|
| **`finqa_pre_table_post`** | `context` = `pre_text` + serialised `table` + `post_text`; `table_format ∈ {markdown, pipe_rows}` | Faithful to FinQA's original input. Table serialisation format is itself a choice, and tables inflate input length against `max_input_tokens: 768` |
| `finqa_text_only` | table dropped | Shorter inputs, but many FinQA answers are unanswerable without the table — the task becomes materially easier/incoherent |

*Recommendation:* `finqa_pre_table_post` with `table_format: markdown`.

**(c) `longform` — which ELI5 answer is the target.** `answers` is
`{a_id: [...], score: [...], text: [...]}`. Today's provider silently takes `text[0]`.

| Option | Rule | Trade-off |
|---|---|---|
| `first` | `text[0]` | Status quo; arbitrary, and the ordering is not documented upstream |
| **`highest_score`** | `text[argmax(score)]` | Uses the signal the dataset provides; a higher-quality, better-motivated target |

*Recommendation (chosen):* `highest_score`. Exposed as `row_adapter_options.list_target_pick`,
set explicitly in config so the choice is never hidden in code.

---

## D-016 — Sub-corpus sampling policy under a volume cap

**Status:** `RESOLVED` — 2026-09-22. Chosen: **`shuffled_buffer`**
(`data.sampling: shuffled_buffer`), for every tier including `local_smoke`.

**Decided by:** the user, on 2026-09-22, relayed by the coordinating orchestrator agent — same
provenance caveat as D-006/D-015.

**Two conditions the choice imposes**, both now part of the contract:

1. `data.shuffle_buffer_size` must be set explicitly (still **no default**) and must be
   `≥ max(train_pool, test)` for the task, or the shuffle window is smaller than the sample and
   the shuffle does nothing. It is logged in `config_used.json` and in every snapshot manifest,
   because the buffer size determines how near-random the sample actually is.
2. The shuffle seed is derived deterministically from `data.seed` and must be **identical across
   both loads of the same task**. `skip`-based TEST carving from a single-split corpus (`instr`,
   invariant `I2.6`) is a disjointness guarantee only if the two streams are shuffled into the
   same order; a per-call random seed would silently break it. `assert_disjoint` remains the
   independent check.

The alternatives considered, retained for the record:

`idea.md` §0.2 gives `actual` volumes as ranges and brief §4 forbids blindly processing entire
datasets, so every load is capped. Neither source says **which** rows the cap selects. The
existing provider streams and stops, i.e. it takes the *head* of the file — a systematically
non-random sample (CNN/DailyMail is date-ordered; SQuAD is grouped by article title).

| Option | Mechanism | Trade-off |
|---|---|---|
| `head_of_stream` | stream and stop | Cheapest; status quo. Biased in any corpus with meaningful row order — not defensible for a reported number |
| **`shuffled_buffer`** | `IterableDataset.shuffle(buffer_size)` then take | Near-random at bounded memory; the buffer size is itself a knob and must be logged |
| `full_then_seeded_subsample` | download the corpus, then sample with the run seed | Unbiased, but downloads whole corpora — directly against brief §4 |

*Recommendation (chosen):* `shuffled_buffer`. Implemented as `data.sampling` with **no default**
at base level plus `data.shuffle_buffer_size`, so a run that omits either still fails loudly —
resolving the decision supplies the value, it does not remove the guard.
`head_of_stream` stays implemented and selectable, but is no longer used by any configured arm.

---

## D-007 — Named literature baselines (LoRM, FedALoRA, FedZKD)

**Status:** `OPEN`

The brief §14 names LoRM, FedALoRA, FedZKD and "an appropriate federated continual-learning
baseline". `idea.md` §12.4 instead specifies four *method classes*
(task-specific adapters, fixed fusion, simple averaging, similarity-based selection).

**I have not verified the LoRM / FedALoRA / FedZKD references and will not reconstruct a
method from its name.** Guessing an algorithm and labelling it with a published method's
name would fabricate a baseline — prohibited by brief development rule 1 and `idea.md` §16.1.

**Implemented now:** the four reproducible classes from `idea.md` §12.4, plus sequential
full fine-tuning and separate-LoRA-per-task from brief §14 — all six are fully specified by
the two source documents and are implementable faithfully.

**Not implemented:** LoRM, FedALoRA, FedZKD. Each has a registry entry that raises
`BaselineRequiresCitationError` with the exact information needed (paper, official code,
hyperparameters, budget) to implement it fairly.

**Decision needed:** supply the three references (or drop them from the comparison set).

---

## D-008 — QLoRA is unavailable on the development machine

**Status:** `RESOLVED` (technical fact, not a research choice) — 2026-09-19, agent-verified

`bitsandbytes` 4-bit quantisation requires CUDA. The dev box is Apple Silicon (MPS, 24 GB,
no CUDA). Therefore:

- `actual` tier (7B, QLoRA) **must** run on a CUDA host. The code supports it and the
  config validates it, but it cannot be executed here.
- `local_smoke` (1.5B) runs on MPS in bf16 **without** quantisation.
- `dummy` runs on CPU.

`config.resolve()` raises `QuantizationUnavailableError` rather than silently downgrading
to bf16, because a silent downgrade would change the reported training recipe.

---

## D-009 — "Accuracy" is undefined for generative tasks

**Status:** `PROVISIONAL`

`idea.md` §4.4/§4.5 use `Acc_i`, and §12.3 asks for accuracy/precision/recall/F1. Five of
the six tasks are open-ended generation, where accuracy does not exist.

**Resolution:** each task declares a single `primary_score ∈ [0,1]` (higher is better) in
its dataset config — ROUGE-L for summarization, token-F1 for QA, etc. All
continual-learning metrics (`avg_performance`, `forgetting`, `FWT`, `BWT`) and all utility
targets are computed over `primary_score`, never over "accuracy". The term "accuracy" is
banned from result tables unless the task genuinely has one.

Auxiliary metrics are still logged per task; only the scalar used for *decisions* is
constrained. Per-task `primary_score` choice is in `configs/datasets/*.yaml`.

---

## D-010 — Utility-network supervision is O(n²) and needs a sampling policy

**Status:** `PROVISIONAL`

`idea.md` §9 requires *actually performing* the merge for each historical candidate pair to
observe `y_i`. With `n` adapters and `T` task events this is `O(T·n)` full fusion+KD runs
for the bootstrap, and `O(n²)` for compression supervision — the dominant cost of the whole
project at 7B scale.

**Implemented policy** (all config-driven, all logged):
`utility.supervision.max_pairs_per_event`, `utility.supervision.pair_sampling ∈
{all, random, stratified_by_feature}`, `utility.supervision.proxy_steps` (a truncated
fusion+KD run used as a cheap proxy target), and
`utility.supervision.proxy_validation_frac` (fraction of sampled pairs also run to full
convergence, to measure proxy bias).

**Open question for the user:** is a truncated-proxy target acceptable as the supervision
signal, given that its bias must then be measured and reported? If not, the bootstrap must
be run at full convergence and the budget grows by roughly `proxy_steps⁻¹`.

---

## D-011 — Federated × continual factorisation

**Status:** `OPEN`

`idea.md` §7.3 and brief §13 both forbid assuming `one client = one task`, but neither
specifies the actual partition. The four axes — TASK, CLIENT, ADAPTER, ROUND — must be
pinned per experiment.

**Alternatives:** (a) every client sees the same task sequence, disjoint data shards (IID
across clients, sequential across tasks) — isolates the continual axis;
(b) non-IID: clients see overlapping but different task subsets — the realistic FCL setting,
confounds the two axes; (c) label/domain-skewed Dirichlet partition per task.

**Recommendation:** (a) as the controlled primary experiment, (b) as the realistic
secondary. Implemented as `federated.partition ∈ {iid_shards, disjoint_task_subsets,
dirichlet}` with `federated.dirichlet_alpha`, **no default**.

**Decision:** _pending user sign-off._

---

## D-012 — Statistical protocol

**Status:** `OPEN`

Neither source specifies seed count or a significance test, yet brief falsification
question 12 demands statistical meaningfulness.

**Recommendation:** ≥3 seeds for every reported arm (5 preferred); report mean ± std;
compare arms with a paired bootstrap over per-task scores (paired by task and seed); report
effect size alongside any p-value; never report a single-seed number as a gain.
At 7B with QLoRA this multiplies `actual` compute by the seed count — a real budget
decision, hence `OPEN`.

Implemented as `experiment.seeds: [int]` (list, no default) and
`src/amaun/evaluation/stats.py`.

---

## D-013 — Storage form of a fused adapter

**Status:** `PROVISIONAL`

`idea.md` §6.1 explicitly requires documenting whether a fused adapter is stored as a dense
effective update or re-factorised to low rank, but does not choose.

**Implemented:** `fusion.storage ∈ {dense_delta, refactorize_svd, keep_components}`.
`refactorize_svd` records the relative Frobenius reconstruction error per layer in the
fusion log, so the approximation cost is measured rather than assumed. `keep_components`
stores the components plus `α`, which is exact but grows linearly in `m`.

**Recommendation:** `refactorize_svd` at `rank = r` for the reported runs (keeps adapter
size constant, making the storage claim honest), with the error reported.

**l4_full_study (2026-10-23):** run 1 measured the truncation error at 0.36–0.48 mean, 0.58 max
per layer, so this run fuses with `keep_components` (exact, stored bytes recorded as they are)
and logs the error `refactorize_svd` *would* have had. F-0 keeps `refactorize_svd` as the
run-1 reference. The storage claim is therefore measured, not assumed, and is worse than
`refactorize_svd` would suggest.

---

## D-014 — Adapter-count accounting under fusion

**Status:** `OPEN`

If FUSE replaces `A_i*` with `A_merge` (`idea.md` §7.1), adapter count stays flat while the
*merged adapter now serves two tasks*. Reporting "adapter count" alone can therefore make
fusion look free. The honest accounting must report, per task index: adapter count,
**tasks-served-per-adapter**, total stored parameters, and per-task performance — so that a
count reduction bought by performance loss is visible.

**Recommendation:** all four reported jointly; adapter count never reported alone.

---

## D-017 — Installed fused adapters ran at `alpha/r` times their value (finding)

**Status:** `RESOLVED` 2026-10-23 (code defect; fixed, not a methodology choice)

`fuse_to_delta` folds every scaling into `B` and returns `LayerDelta(scaling=1.0)`, so the
fused update is `B·A`. `load_delta_into` copied `A` and `B` into the PEFT LoRA layer without
looking at `delta.scaling`, and PEFT multiplies by its own `lora_alpha / r = 2.0`. Every fused
adapter in run 1 (`results/local_smoke/l4_decision_study/`) was therefore **evaluated, loss-measured
and probed at twice its intended update**: with α ≈ 0.5 / 0.5 the installed update was close to
`ΔW_old + ΔW_new`, not their average. Trained adapters were unaffected (their scaling equals the
layer's). Run 1's fusion numbers must be read with this in mind; the run-1 code path is not
re-run except as F-0, which uses the corrected installer.

**Fix:** `load_delta_into` installs `B · (delta.scaling / layer.scaling)`, so the installed update
is exactly `delta.scaling · B · A`. It also installs factors of any rank (needed by
`keep_components`, D-013), and `reset_lora_parameters` restores the spec rank before training.
Tested in `tests/test_full_study_fusion.py`.

---

## D-018 — Acceptability, oracle, decision quality, and the adapter cap (l4_full_study)

**Status:** `PROVISIONAL` (the definitions are fixed by `docs/l4_next_run_prompt.md` items 5–7;
the cap fallback below is an agent choice the prompt does not make)

- **Acceptable:** new ≥ 0.98 × CREATE and every old task's relative drop ≤ 1 %, on point
  estimates. **Inconclusive:** the paired bootstrap 95 % CI of (option − 0.98 × CREATE) contains 0.
- **Oracle:** cheapest acceptable option — fewest adapters after, then least training seconds, then
  higher new-task score; CREATE if none is acceptable.
- **Per policy:** `decision_correct`, `quality_shortfall = max(0, 0.98·CREATE − chosen)`,
  `cost_excess` (adapters, seconds vs the oracle), `forgetting` (max relative old drop). Run 1's
  `average_performance`, `oracle_option` and `regret_vs_oracle` remain as secondary fields.
- **Cap (`max_adapters`):** at the cap CREATE is infeasible. *Choice made here:* every policy's
  CREATE becomes **BACKBONE** ("no room: serve the task with the frozen backbone"), the same rule
  for every policy so the comparison stays fair. The capped oracle is the cheapest acceptable
  *feasible* option; if none, the feasible option with the highest new-task score among those with
  ≤ 1 % forgetting (BACKBONE always qualifies). *Alternative not taken:* a policy-specific fallback
  (e.g. `amaun` → FUSE with its best-U candidate), which would mix the decision rule with the cap
  rule. Switch by changing `study_criteria.cap_fallback`.

---

## D-019 — Measured labels and the utility network's supervision (l4_full_study)

**Status:** `PROVISIONAL` (prompt items 9, 12–14)

- Labels use the measured task metric on VALIDATION: 150 generations for qa / qa-B / sentiment,
  100 for sum / instr / dialog (`decision_study.label_examples`). `beneficial` =
  fused/standalone ≥ 0.98 and max relative old drop ≤ 1 %. The loss-proxy label is kept as
  `beneficial_loss_proxy` and agreement is reported.
- Head targets: `primary_score` = fused / standalone new-task metric; `forgetting` = max relative
  old drop; `memory` = `communication` = stored MB; `confidence` and `long_term_loss` keep run 1's
  loss-based definitions (measured, not proxied, but not task metrics — the prompt names only the
  first four). Every head is z-scored on the fit split.
- Leave-one-new-task-out CV (6 folds); τ_u from out-of-fold utilities; refit on all pairs.
- Fit only with ≥ 20 pairs and 20–80 % beneficial; otherwise `amaun` and `amaun_train_first` use
  `probe_rule` and the reason is written to `utility_supervision.json`.
- Reuse-gate target: relative measured gap `s(A_cand)/s(A_dedicated) − 1`, tolerance 0.02.

---

## D-020 — The `amaun` decision order, BACKBONE, and the simple policies (l4_full_study)

**Status:** `PROVISIONAL` (prompt items 8, 15–17)

`amaun`: (1) the reuse gate fires → REUSE; else backbone probe score ≥ 0.98 × best candidate's →
BACKBONE; (2) max U ≥ τ_u → FUSE with the best-U candidate; (3) CREATE.
`amaun_train_first` (*interpretation made here*): the same order, where step (1)'s REUSE comes
from the post-training evidence instead of the pre-training gate — U ≥ τ_reuse and the support
pre-screen (candidate support loss ≤ A_temp's) — and step (1)'s BACKBONE check is unchanged.
Probes are 16 SUPPORT examples: teacher-forced loss and greedy-generation primary metric.
`probe_rule`, REUSE-only (best probe score), FUSE-only (best probe loss gain), `always_merge`
(nearest `semantic.d_task_sq`) and `fixed_rule` (run-1 B4) as in the prompt.

---

## D-021 — Tasks added for l4_full_study

**Status:** `PROVISIONAL` (prompt items 19–25; revisions resolved 2026-10-23 from the Hub API,
to be re-verified on the VM)

- `xsum`: EdinburghNLP/xsum @ `7d4d486c2f8ef850b1a11aead99b894ff3dd7da9` (parquet on `main`);
  TEST from upstream `test`. License declared `unknown` on the card (BBC articles; research use).
- `ner`: eriktks/conll2003, `main` holds only a loading script, so pinned to
  `refs/convert/parquet` @ `ce85b39f9dd99f552d0739d456814e95fb6a39b0` (the D-006 FR-2.6
  mechanism), `hf_config: null` (the conversion exposes only `default`; verified on the VM
  2026-10-23: 14 041 train / 3 453 test rows, tag order as `tag_names`). Row adapter `conll_entities`: sentence = tokens joined by spaces; target = one
  `TYPE: span` line per entity in order, `none` when the sentence has none; `-DOCSTART-` rows
  emit nothing. *Metric definition chosen here:* `entity_f1` is exact (type, span) F1 **per
  sentence, averaged over sentences** (an empty prediction for an empty reference scores 1), so
  per-example bootstrap CIs are possible; corpus micro F1 is reported as `entity_f1_micro`.
- `qa-B`, `sentiment-B`: the original's shuffled stream after `train_offset: 9000`, minus every
  content hash in the original's pool; TEST identical to the original's.
- FinQA (`domain`): exact-count split **5 656** / 250 / 250 / 300 (ratios cannot fill
  TRAIN + 250 + 250 from so small a pool). *Revised 2026-10-23 on the VM:* the prompt's 5 700 /
  250 / 250 needs 6 200 rows, but of FinQA's 6 203 usable rows 47 repeat another row exactly
  (6 156 distinct), and exact-count mode keeps one copy of each. TRAIN was cut to every distinct
  row left after VALIDATION and SUPPORT (5 656 = 6 156 − 500); VALIDATION stays 250 (item F13).
  Training: 353 optimiser steps instead of 356.
- ELI5 (`longform`): primary metric rescaled BERTScore-F1 — roberta-large @
  `722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, no idf, baseline F = 0.83122575
  (bert_score 0.3.13 `rescale_baseline/en/roberta-large.tsv`). BERTScore was **not** implemented
  before this change (the prompt's "already implemented" was incorrect: `task_metrics.py` listed it
  as unimplemented); `amaun.evaluation.bertscore` re-implements bert_score's greedy matching.
- Per-task `max_new_tokens` = `max_target_tokens` (the global cap is raised to 512).

---

## D-022 — Evidence that run 1's splits are reproduced

**Status:** `RESOLVED` 2026-10-23 (finding + check)

Run 1's `decision_study/data.json` records snapshot keys and split sizes but **no per-row hashes**,
so "row hashes match run 1" cannot be checked literally. Preflight therefore checks the strongest
evidence run 1 recorded: (a) the data config of sum / qa / sentiment equals run 1's
`config_used.json`; (b) every load's snapshot key equals run 1's; (c) split sizes are equal;
(d) each seed adapter's TRAIN token count (prompt + target after truncation, the quantity run 1
logged as `tokens_seen` over exactly one epoch) equals run 1's `initial_adapters.json`. This run
writes per-split row digests (`split_digests`) into its own `data.json`, so later runs can compare
rows exactly.

---

## D-023 — Feasibility rules and determinism (l4_full_study)

**Status:** `PROVISIONAL` (prompt items 11, 26)

R1 (standalone scores once), R2 (a stream CREATE reuses the adapter trained for that task — same
adapter id, hence the same derived seed), R3 (re-evaluate only changed tasks, then all at the end).
Evaluations are cached by (adapter content hash, task, split, example count), so R1 and R3 are
equalities by construction; dummy-tier tests compare them with the naive computation.
`deterministic: true` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and strict deterministic algorithms
before the backbone loads; an op without a deterministic kernel raises in preflight rather than
silently running nondeterministically.

---

## D-024 — Left out of l4_full_study to fit ~30 h (future work)

**Status:** `OPEN` (deferred by `docs/l4_next_run_prompt.md`)

More seeds (the run has one, 1234; no seed-to-seed variance is measured); ablations A3 (fixed α)
and A8 (global α) inside the run (both remain selectable: `parameterisation: free` with
`learning_rate: 0`, `granularity: global`); 300-step fusion; GSM8K; `probe_rule` and the other
baselines in the sequential stream (they are compared in Phase 5).

---

## D-025 — Training-pool rows that repeat a TEST row of the same task

**Status:** `RESOLVED` 2026-10-23 (user decision, after `--prepare-only` failed on the VM)

Some corpora repeat rows across their upstream splits. DailyDialog has 9 dialogues in both its
`train` split (dialog's training pool) and its `validation` split (dialog's TEST), so the
disjointness check (invariant I2.1) stopped `prepare_data`. Decision: for **every** task, a
training-pool row whose content hash (prompt + target) equals one of the same task's TEST rows is
dropped from the pool before the split. TEST is never changed. The split is assigned per row from
its own content, so every other row keeps its split; the TRAIN cap may then take a few different
rows to fill 6 000. The number dropped is recorded per task as
`provenance.excluded_pool_rows_in_test`. sum, qa and sentiment had no such rows in run 1 (it
passed the same check), so their splits are unchanged and D-022's run-1 check still applies.
Alternative considered: replacing DailyDialog; rejected because other corpora (e.g. CoNLL-2003)
can repeat short rows across splits too. Test: `tests/test_pool_test_overlap.py`.

---

## D-026 — Merged-weight, batched evaluation (overrides prompt rule 7)

**Status:** `RESOLVED` 2026-10-23 (user decision, after the first L4 preflight projected 78.4 h)

**Why.** The first launch stopped in Phase 0: projected 78.4 h against the 32 h budget (69 h on
this host's probe costs alone), ~58 h of it greedy generation. Measured on the L4: the backbone
decodes at 35 ms/token, but with the LoRA branch installed at 78 ms/token (the KV cache is in
use; the cost is the adapter's extra matmuls on every adapted module at every token), and the
evaluator generated one example at a time. `docs/l4_next_run_prompt.md` rule 7 ("no speed
optimisations") is overridden for evaluation only.

**Decision.**
1. `evaluation.merged_weights: true` — every generating evaluation of the full study (TEST,
   VALIDATION labels, eligibility, SUPPORT probes, the Phase 7 audit) runs with the adapter merged
   into the base weights, `W + scaling * B @ A` summed in fp32 and stored in bf16, with the LoRA
   branch disabled (`amaun.adapters.lora.MergedWeights`). Base weights are restored by copying
   back a pristine CPU snapshot, never by subtraction, so the backbone is bit-for-bit unchanged
   (the run's fingerprint check proves it). Loss-only paths (features, validation loss, fusion,
   training) keep the LoRA branch.
2. `generation.batch_size: 4` — 4 prompts per `generate` call, left-padded, grouped by prompt
   length, predictions returned in example order. Default 1 elsewhere (run 1's path).

**Evidence** (L4, run-1 seed adapters, VALIDATION, current path vs merged vs merged + batch):

| Task | Path | Time | Score | Outputs identical to current |
|---|---|---|---|---|
| sum (32, ROUGE-L) | current | 130 s | 0.2379 | — |
| | merged | 64 s | 0.2443 | 17/32 |
| | merged + batch 4 | 23 s | 0.2370 | 15/32 |
| | merged + batch 8 | 13 s | 0.2226 | 15/32 |
| qa (64, token F1) | current | 27 s | 0.8645 | — |
| | merged / + batch 4 / + batch 8 | 13 / 7 / 5 s | 0.8645 | 64/64 each |

Short outputs are unchanged; long greedy outputs can diverge after one rounding difference, with
the score moving by about as much as the current path's own rounding sensitivity. Batch 8 moved
the sum score by −0.015 and was not adopted. Every option of the run is evaluated the same way,
so within-run comparisons stay like for like; run 1's stored numbers are not comparable at the
level of individual outputs (F-0 re-runs run 1's setting under this run's evaluation).

**Projection fixes made at the same time** (no measured value changes): generation is timed on
the evaluator's own path (merged, batch 4, expected decode length = the longest target in a
batch); run 1's per-example generation costs are not used, as they belong to the old path;
fusion step cost is `(3 steps − 1 step) / 2`, so first-step set-up is not counted per step; the
`always_separate` stream runs add no generation (every evaluation is cached from phases 1 and 5);
old-task TEST of FUSE options is spread over the seed tasks at their own costs instead of all at
`sum`'s. Phase 3's repeat of Phase 2's pairs is still counted: fusion is not seeded, so its
result (and the cache key) can differ between repetitions.

**Latent bug fixed on the way.** `_set_rank` (resizing LoRA factors for a rank-2r fused delta)
created its new parameters as inference tensors when called from `_validation_loss`, which runs
under `torch.inference_mode`; every later in-place install then failed. Run 1's path never hit it
because evaluation installed each fused delta first, outside inference mode. The resize now
always creates normal tensors.

**Memory fix found at the first passing preflight.** Phase 0 loads the BERTScore encoder after
its memory checks, and the encoder then stayed on the GPU for the whole run: roberta-large in
fp32 (~1.3 GiB) on top of FinQA's 20.4 GiB fusion peak exceeds the 22.5 GB limit. It is now held
on CPU and moved to the GPU only inside each ELI5 scoring call (same computation, same values).
The run was stopped early in Phase 1 and resumed with the fix (`--resume`; config hash unchanged).

Tests: `tests/test_merged_evaluation.py`, `tests/test_bertscore.py`.

---

## D-027 — ELI5 is judged by gain retention (rescaled BERTScore has no meaningful zero)

**Status:** `RESOLVED` 2026-10-24 (user decision; made before Phase 5 read any ELI5 VALIDATION or
TEST row)

**Finding.** Rescaled BERTScore-F1 subtracts roberta-large's baseline (F = 0.831), so ordinary
ELI5 answers score below zero. Measured on the L4 on 16 ELI5 **TRAIN** rows: backbone mean
−0.106 (14/16 negative), `seed-sum` −0.084 (11/16). Every relative rule of the study is a ratio,
and ratios are undefined on a scale without a meaningful zero; on negative scores they flip:
"≥ 0.98 × CREATE" demands beating CREATE, eligibility becomes a sign test, a drop from a
negative score is always 0. Every other task's primary metric has a true zero (ROUGE, F1,
accuracy, exact match) and is unaffected.

**Decision.** Metrics in `study_criteria.INTERVAL_SCALE_METRICS` (`bertscore_f1`) are judged by
**gain retention** over the zero-shot score that the BACKBONE option measures on the same TEST
examples:

- equivalence bar = zero-shot + 0.98 × (CREATE − zero-shot): an option must keep 98 % of CREATE's
  gain; the paired CI uses the per-example bar; quality shortfall is measured from the bar;
- forgetting of ELI5 (Phase 6 stream) = share of the adapter's gain lost (`gain_drop`).

The rule follows from the metric, not from a new config key, so the config hash is unchanged
and the run resumes. Two comparisons have no CREATE score to anchor a gain and are made
**sign-safe** instead (identical for every non-negative score): eligibility divides by
|zero-shot|; the SUPPORT probe comparisons (`probe_rule`, `amaun`'s BACKBONE step) use
`reference − 2 % × |reference|`. Options considered: sign-safe ratios everywhere (still depends on
the scale's zero), raw BERTScore or ROUGE-L as primary (config change → full restart; raw is
compressed, ROUGE-L was non-discriminating in run 1), a fixed metric-unit margin or a
non-inferiority test (a new threshold to choose). Limitation: ELI5's "within 2 %" is 2 % of a
gain, not of a score, so it is stricter than for the other tasks; reported as such.

Tests: `tests/test_study_criteria.py` (D-027 block).

---

## D-028 — Phase 4 crash: utility-CV fold split name (run resumed)

**Status:** `RESOLVED` 2026-10-24 (bug fix; no methodology change)

Phase 3 gave 30 pairs, 33 % beneficial — inside the fit gate, so the network was fitted for the
first time (the dummy tier's supervision is one-class and always fell back). Each leave-one-out
fold fitted its normaliser under the name `utility_train_minus_<task>`, which the normaliser's
leakage guard rejects; the run stopped at the start of Phase 4 (2026-10-23 23:12 UTC) with
Phases 0–3 checkpointed. Each fold now fits under its parent split's name (`utility_train`; the
held-out task is recorded per fold). Checked by `tests/test_utility_cv.py` (reproduces the crash
without the fix), a replay of Phase 4 on the run's real Phase 3 measurements, and a dummy study
with the fit gate forced open (Phases 4–6 with a fitted network). The run resumed with
`--resume`; the gap is recorded in the checkpoint's `resumes`.

---

## Summary

| ID | Topic | Status | Blocks |
|---|---|---|---|
| D-001 | Backbone tiers | PROVISIONAL | reported results |
| D-002 | bootstrap_six vs empty_sequential | **OPEN** | **headline experiment** |
| D-003 | 3-way vs 2-way decision | O1 IMPLEMENTED 2026-10-22 (O2 retained) | report both orders |
| D-004 | top-k fusion | PROVISIONAL | — |
| D-005 | τ_u selection source | **OPEN** | leakage validity |
| D-006 | dataset pinning (B1 → dolly-15k, B2 → Pavithree/eli5) | RESOLVED 2026-09-22 (relayed attribution — see entry) | — |
| D-007 | LoRM / FedALoRA / FedZKD citations | **OPEN** | baseline completeness |
| D-008 | QLoRA needs CUDA | RESOLVED | local `actual` runs |
| D-009 | primary_score not "accuracy" | PROVISIONAL | — |
| D-010 | utility supervision cost | PROVISIONAL | budget |
| D-011 | federated partition | **OPEN** | federated experiments |
| D-012 | seeds / significance | **OPEN** | any claimed gain |
| D-013 | fused-adapter storage | PROVISIONAL | — |
| D-014 | adapter-count accounting | **OPEN** | efficiency claim |
| D-015 | row→example adaptation (a `dialog_last_turn`, b `finqa_pre_table_post`/markdown, c `highest_score`) | RESOLVED 2026-09-22 (relayed attribution — see entry) | — |
| D-016 | sub-corpus sampling (`shuffled_buffer`) | RESOLVED 2026-09-22 (relayed attribution — see entry) | — |
| D-017 | fused adapters installed at `alpha/r` × their value (run 1) | RESOLVED 2026-10-23 (fixed) | reading run-1 fusion numbers |
| D-018 | acceptability / oracle / cap fallback | PROVISIONAL | l4_full_study tables |
| D-019 | measured labels, utility CV, fallback | PROVISIONAL | l4_full_study utility claims |
| D-020 | `amaun` order, BACKBONE, simple policies | PROVISIONAL | — |
| D-021 | xsum, ner, duplicates, FinQA counts, BERTScore | PROVISIONAL (verify revisions on VM) | l4_full_study run |
| D-022 | run-1 split evidence (no row hashes in run 1) | RESOLVED 2026-10-23 | — |
| D-023 | R1–R3, determinism | PROVISIONAL | — |
| D-024 | items left out of l4_full_study | **OPEN** (future work) | — |
| D-025 | training-pool rows repeating a TEST row: dropped from the pool | RESOLVED 2026-10-23 | — |
| D-026 | merged-weight, batched (4) evaluation; projection fixes | RESOLVED 2026-10-23 | — |
| D-027 | ELI5 judged by gain retention over zero-shot; sign-safe probes and eligibility | RESOLVED 2026-10-24 | — |
| D-028 | Phase 4 crash (utility-CV fold split name); run resumed | RESOLVED 2026-10-24 | — |
