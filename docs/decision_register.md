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
