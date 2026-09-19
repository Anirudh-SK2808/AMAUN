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

**Decision:** _pending user sign-off_.

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

**Status:** `OPEN` — **two concrete blockers.**

`idea.md` §0.2 and brief §4 name six datasets loosely. Pinned proposals:

| Task | Proposed HF id | Revision | License | Issue |
|---|---|---|---|---|
| Summarization | `abisee/cnn_dailymail` | `3.0.0` | Apache-2.0 (code), news content under fair-use research terms | usable |
| QA | `rajpurkar/squad` | v1.1 | CC BY-SA 4.0 | usable; share-alike applies to derivatives |
| Instruction following | **unpinned** — "FLAN-style" is not a dataset | — | — | **BLOCKER B1** |
| Dialogue | `li2017dailydialog/daily_dialog` | — | CC BY-NC-SA 4.0 | **non-commercial only**; fine for research, must be stated in any release |
| Long-form generation | `eli5` | — | Reddit-derived | **BLOCKER B2** — the original `eli5` HF dataset was withdrawn over Reddit API/licensing; it is not reliably loadable |
| Domain-specific | `ibm-research/finqa` (or `dreamerdeo/finqa`) | — | MIT (per repo) | usable; license must be re-verified at download time |

**B1 — "FLAN-style" must be pinned.** Candidates: `Muennighoff/flan`,
`SirNeural/flan_v2`, or a licensed subset of `databricks/databricks-dolly-15k` (CC BY-SA
3.0). These differ substantially in mixture composition, which changes what Adapter 3
*is*. Cannot be chosen by an agent.

**B2 — ELI5 substitute needed.** Candidates: `eli5_category` (still Reddit-derived),
`kilt_tasks`/`eli5` via KILT, or replace long-form generation with
`allenai/wildchat`-style long responses or `cnn_dailymail`-disjoint long-form writing.
Changing this changes what "long-form generation" means in the paper.

**Until B1/B2 are resol**, `configs/datasets/*.yaml` carries `hf_id: null` for those two
rows and the loader raises a typed `UnpinnedDatasetError`. The offline `synthetic` provider
covers tests and dummy-tier smoke runs so no work is blocked.

**Decision:** _pending user sign-off on B1 and B2._

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
| D-003 | 3-way vs 2-way decision | **OPEN** | **research question scope** |
| D-004 | top-k fusion | PROVISIONAL | — |
| D-005 | τ_u selection source | **OPEN** | leakage validity |
| D-006 | dataset pinning (B1 FLAN, B2 ELI5) | **OPEN** | **real data runs** |
| D-007 | LoRM / FedALoRA / FedZKD citations | **OPEN** | baseline completeness |
| D-008 | QLoRA needs CUDA | RESOLVED | local `actual` runs |
| D-009 | primary_score not "accuracy" | PROVISIONAL | — |
| D-010 | utility supervision cost | PROVISIONAL | budget |
| D-011 | federated partition | **OPEN** | federated experiments |
| D-012 | seeds / significance | **OPEN** | any claimed gain |
| D-013 | fused-adapter storage | PROVISIONAL | — |
| D-014 | adapter-count accounting | **OPEN** | efficiency claim |
