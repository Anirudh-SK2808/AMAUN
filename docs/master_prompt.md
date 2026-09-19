# AMAUN — Master Prompt (consolidated, authoritative)

**AMAUN — Adaptive Multi-criteria Adapter Utility Network**
A federated continual-learning framework for small/medium language models that manages
LoRA adapters dynamically.

This document is the single self-contained statement of what is being built and what may
never be done. It consolidates `idea.md` (the research master spec) with the engineering
brief. Where they conflict, the conflict is recorded in `docs/decision_register.md` and is
**not** resolved here.

---

## 1. Research question

> Can a language model dynamically determine whether an incoming NLP task should reuse an
> existing LoRA adapter, fuse multiple existing adapters, or create a new adapter, while
> controlling adapter growth and mitigating continual-learning degradation?

## 2. Falsifiable hypotheses

Each hypothesis names the experiment that can refute it. A hypothesis with no refuting
experiment is not admissible.

| ID | Hypothesis | Refuted if | Experiment |
|---|---|---|---|
| **H1** | A learned multi-criteria utility predictor selects better integration targets than any single-similarity rule. | AMAUN's average `primary_score` does not exceed `baseline_similarity_selection` beyond seed noise. | `main_*` vs `B4` |
| **H2** | Layer-wise *learnable* fusion coefficients outperform fixed/uniform fusion. | A3 (fixed α) and A8 (global α) match full AMAUN within seed noise. | `A3`, `A8` |
| **H3** | The three-way REUSE/FUSE/CREATE decision controls adapter growth at no significant performance cost. | AMAUN stores fewer adapters but loses `primary_score`, or stores no fewer than `B2` (separate LoRA per task). | `main_*` vs `B2`, `A9`, `A10`, `A11` |
| **H4** | KD against the pre-merge old adapter reduces forgetting caused by fusion. | A4 (no KD) shows equal or lower forgetting. | `A4` |
| **H5** | Utility-based repository compression retains more capability per byte than age/usage heuristics. | A5 / heuristic compression matches utility-based compression. | `A5`, `B_compress_heuristic` |
| **H6** | LoRA-only federated aggregation of *compatible* adapters preserves the continual metrics while transmitting ≪ the backbone. | Compatibility-gated aggregation matches naive all-adapter averaging. | `F1` vs `F2` |

**Non-hypothesis.** AMAUN does not claim LoRA, KD, FedAvg, or weighted fusion as novel
components. The claim is mechanism-level integration (see `docs/methodology.md` §Novelty).

## 3. Hard constraints (inviolable — from `idea.md` §16 plus brief §25)

1. **Never fabricate** numerical results, datasets, metrics, or citations. Every
   number-shaped item not produced by a real run is literally `[TO BE FIXED EXPERIMENTALLY]`.
2. **Never use cosine similarity — or any single metric — as the decision rule.** It may
   exist only as one feature among many.
3. **Never average LoRA `A`/`B` factors independently.** Fuse at the effective
   `ΔW = BA` level.
4. **The backbone is frozen forever and never transmitted.** `θ_{k+1} = θ_k = θ_0`.
5. **Never estimate normalisation statistics from the test sequence.**
6. **Never train the utility network on the final test sequence.**
7. **Never replace utility-based compression with an age/usage heuristic.**
8. **Always initialise `α_l = 0.5`** (equal contribution) before post-merge optimisation.
9. **The KD teacher is the pre-merge old adapter**, never the frozen backbone.
10. **The temporary adapter stays out of the permanent repository** until the decision completes.
11. **Every artefact is tagged with `run_mode`** and written under `results/<run_mode>/<experiment_id>/`.
12. **Never cross-load checkpoints between run modes.**
13. **Never use test data for any selection, tuning, or normalisation decision.**
14. **Never silently change the research methodology.** Ambiguity → `docs/decision_register.md`.
15. **Do not give AMAUN information unavailable to the baselines.**

## 4. Run modes

`run_mode` is required, has **no default**, and determines the backbone (D-001):

| `run_mode` | Backbone | Purpose | Reportable |
|---|---|---|---|
| `dummy` | locally-built tiny random Qwen2 (no download) | unit tests, CI, wiring checks | **no — hard-blocked** |
| `local_smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | end-to-end pipeline validation on the dev box | **no** |
| `actual` | `Qwen/Qwen2.5-7B-Instruct` | real experiments | **yes, only these** |

## 5. Pipeline (implement exactly this control flow)

```
task arrival
   → support/calibration set only
   → task representation
   → candidate enumeration over R_c
   → 7-family feature extraction per candidate pair
   → train-split normalisation
   → utility prediction  U_i = g_ω(z_i)
   → decision:  REUSE | FUSE | CREATE
        REUSE  → select existing adapter; train nothing; R_c unchanged
        FUSE   → top-k α-fusion at ΔW level + KD-guided joint optimisation
        CREATE → keep/train a new LoRA adapter on the task's TRAIN split
   → repository update
   → compression if Mem(R_c) ≥ R_max   (same utility mechanism)
   → federated aggregation on round boundaries (compatible adapters only)
   → evaluation on TEST (never before this point)
```

## 6. Data contract

Every task maintains four disjoint splits: `TRAIN`, `VALIDATION`, `SUPPORT`, `TEST`.

- The manager may read **`SUPPORT` only**.
- Training a new adapter may read **`TRAIN`** (and `VALIDATION` for early stopping).
- KD retention data (`D_rep`) is drawn from prior tasks' `TRAIN`/`VALIDATION`, never `TEST`.
- Utility-network supervision may read `TRAIN`/`VALIDATION`/`SUPPORT`, never `TEST`.
- `TEST` is read exactly once per arm, by the evaluator, after all decisions are frozen.

Disjointness is enforced in code by content hashing, not by trusting the split function
(`src/amaun/data/leakage.py`), and is asserted by `tests/test_split_isolation.py`.

## 7. Non-goals

- Not a serving system; inference overhead is measured, not optimised.
- Not a new PEFT method; LoRA is used as given.
- Pruning/deletion is **not** a mandatory research component (brief §6) — only
  utility-based compression is.
- No GGUF/llama.cpp path: training representation is Transformers + PEFT.

## 8. Deliverable gate

Nothing is "done" until `docs/acceptance_criteria.md` is fully checked, Agent 3 has filed
an independent audit, and every CRITICAL/HIGH finding is resolved.
