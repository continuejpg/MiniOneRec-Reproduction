# MiniOneRec — Authoritative Experiment Record

**Status.** This file is the single authoritative record of the experiments run in this
repository. Every headline number below is traceable to a prediction artifact in
`runs/` and was re-derived by machine from that artifact, not copied from an earlier
document. Where a number could not be traced, it is marked as such rather than filled in.

**Verification command.** `python analysis/audit_predictions.py` re-scores every
prediction artifact against its own split; `python analysis/verify_summary_claims.py`
machine-checks the headline metrics and the intervention-coverage claim in §10.

---

## 1. Experiment scope

Reproduction of the MiniOneRec pipeline (Semantic ID → SFT → GRPO → constrained beam-search
generative recommendation) on **Amazon Industrial_and_Scientific**, plus a controlled
intervention on the Semantic-ID assignment and a compact sequential baseline.

| Line of work | What it answers | Status |
|---|---|---|
| Clean SFT baseline | How strong is constrained SID generation after one SFT pass? | complete |
| GRPO benchmark | Does ranking-GRPO on top of SFT improve broad Top-K quality? | complete |
| GRPO efficiency | Can the GRPO training pipeline be made much cheaper at equal quality? | complete |
| Compact SASRec baseline | How does a non-generative sequential model compare? | complete |
| SID observational analysis | How does quality vary with popularity and with target–history SID prefix affinity? | complete |
| Shuffled-SID controlled intervention | Does the item↔SID assignment itself carry the signal? | complete |

**Out of scope / not attempted:** multi-seed repeats, SOTA chasing, heavy SASRec tuning,
Steam or Office_Products migration, new reward functions, new RL algorithms.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | single NVIDIA RTX 4090 (24 GB) |
| Python | 3.10.8 (`/root/miniconda3`) |
| torch | 2.6.0+cu118 |
| transformers | 4.57.1 |
| trl | 0.24.0 |
| base model | Qwen2.5-0.5B (`vocab_size` 151 665) |
| SID vocabulary added | **560 tokens** → `vocab_size` **152 225** |
| SID format | `<a_K><b_K><c_K>` — 3 levels, one atomic token each |

---

## 3. Correctness fixes

Changes made to the upstream code before the recorded runs. All are in the repository
history; none alter the evaluation protocol.

### 3.1 Training reproducibility / correctness

- **Zero-advantage logging** added to the GRPO trainer so the group-relative-advantage
  degeneracy can be measured instead of guessed.
- **Namespaced `sample_id`** and `task_type` so that sample identity survives DataFrame
  sampling (the upstream `seq:{local_idx}` form silently renumbered rows).
- **Train/valid split identity** folded into the sequential sample IDs; train/eval
  sample-ID collisions removed.
- **Reward binding by `sample_id`**, fail-loud, replacing prompt-text matching.
- **Shared `build_recommendation_prompt()`** so the SFT and evaluation prompt text cannot
  drift apart.

### 3.2 Evaluation-protocol fix

The **formal** evaluation prompt differs from an earlier, pre-unification evaluation
(see §13). The pre-unification artifacts are retained only as legacy/debug evidence and
are excluded from every table in this document.

### 3.3 Documented non-findings

- The custom cosine scheduler defined in `sft.py` is **dead code**: the `optimizers=`
  argument that would install it is commented out, so SFT trains with the
  `TrainingArguments` default **linear** schedule. Confirmed by the logged LR
  trajectory, which decays smoothly and linearly to 0.
- Rescaling probabilities over legal SID tokens did **not** remove the KL outliers (§12).

---

## 4. Frozen data and protocols

### 4.1 Data

| Item | Value |
|---|---|
| Category | Amazon `Industrial_and_Scientific` |
| Catalogue items | **3 686** |
| Unique Semantic IDs | **3 670** (15 collision groups, 31 items involved) |
| Train rows | **36 259** |
| Valid rows | **4 532** |
| Test rows | **4 533** |

### 4.2 SFT training recipe (identical for the clean and shuffled arms)

| Parameter | Value |
|---|---|
| epochs | 2 |
| effective batch size | 64 (`micro_batch_size` 16 × `grad_accum` 4) |
| learning rate | 3e-4, **linear** schedule, `warmup_steps` 20 |
| `cutoff_len` | 512 |
| precision | bf16 |
| optimiser | `adamw_torch`, `weight_decay` 0, `max_grad_norm` 1.0 |
| seed | 42 |
| `freeze_LLM` / `train_from_scratch` | False / False |
| eval / save | every 0.05 epoch, `save_total_limit` 1, best model at end |
| train sub-datasets | `SidSFTDataset` + `SidItemFeatDataset` + `FusionSeqRecDataset` |
| resulting train rows | **79 834** (verified from the run log) |
| steps per epoch | 1 247 → **2 496 total** |

### 4.3 Formal evaluation protocol

| Parameter | Value |
|---|---|
| test samples | **4 533** |
| `num_beams` | 20 |
| `num_return_sequences` | 20 (= `num_beams`, hard-coded) |
| `length_penalty` | **0** |
| `max_new_tokens` | 256 (an upper bound; a SID is 3 tokens + EOS) |
| decoding | **deterministic beam search** (`do_sample` unset → False) |
| constraint | constrained SID decoding via a prefix→allowed-token dictionary built from the info file |
| cutoffs | **K = [1, 3, 5, 10, 20]** |
| metric code | `calc.py`, unmodified |
| hit rule | first exact SID match; `HR@k` credits rank `< k` |

### 4.4 GRPO frozen subset (17 516 rows)

Frozen and verified so that the 0.25-epoch comparison is data-matched:

| Sub-dataset | Rows |
|---|---|
| seq_rec | 10 000 |
| title / description | 6 516 |
| seqtitle | 1 000 |
| **total** | **17 516** |

Verified: bound IDs 17 516, missing IDs 0, train/eval overlap 0.

---

## 5. Clean SFT baseline

**Artifact:** `runs/eval_clean_sft/test_beam20.json` (4 533 samples × 20 candidates)

| K | HR | NDCG |
|---|---|---|
| 1 | 0.06926980 | 0.06926980 |
| 3 | 0.10103684 | 0.08789727 |
| 5 | 0.12111185 | 0.09613706 |
| 10 | 0.15376131 | 0.10663781 |
| **20** | **0.19832341** | **0.11786798** |

- Training: 2 496 steps, final `train_loss` 0.7712 (mean over all logged steps).
- Decoding audit: **100.0000 %** of the 90 660 emitted candidates are legal catalogue
  SIDs, **0** duplicate candidates, every candidate is exactly one 3-token SID.

> **Do not confuse these with the legacy numbers.** An earlier evaluation of the same
> checkpoint under a pre-unification prompt reported HR@20 = 0.15376131 and
> NDCG@20 = 0.08471693 (`runs/eval_industrial/`). That is the *legacy* protocol and is
> retained only as debug evidence (§13). In particular, **0.15376131 is the legacy HR@20,
> not the unified HR@10** — an earlier version of this file mixed the two.

---

## 6. GRPO benchmark

All GRPO arms start from the clean SFT checkpoint. Every row below was re-scored from its
own prediction artifact with `calc.py`.

| Arm | Checkpoint | HR@1 | HR@3 | HR@5 | HR@10 | **HR@20** | **NDCG@20** |
|---|---|---|---|---|---|---|---|
| Clean SFT | `industrial_sft/final_checkpoint` | 0.06926980 | 0.10103684 | 0.12111185 | 0.15376131 | **0.19832341** | **0.11786798** |
| GRPO 0.25 ep (original) | `grpo_short025/checkpoint-4379` | 0.06794617 | 0.09463931 | 0.11361129 | 0.14251048 | **0.17626296** | **0.10885475** |
| GRPO 0.25 ep (optimized) | `grpo_fast025/checkpoint-4379` | 0.06750496 | 0.09596294 | 0.11030223 | 0.14096625 | **0.17538054** | **0.10824989** |
| GRPO 1.5 ep | `grpo_baseline/checkpoint-26274` | 0.07059343 | 0.09309508 | 0.10566953 | 0.13214207 | **0.16170307** | **0.10470676** |
| GRPO 2.0 ep | `grpo_baseline/final_checkpoint` | 0.07037282 | 0.09309508 | 0.10434591 | 0.12971542 | **0.16236488** | **0.10447064** |

### 6.1 Checkpoint ↔ epoch labels (corrected)

| Checkpoint | Trained fraction | Evidence |
|---|---|---|
| `checkpoint-4379` (`grpo_short025`, `grpo_fast025`) | **0.25 epoch** | `train.log` max `epoch` = 0.25 over 4 379 steps |
| `checkpoint-26274` (`grpo_baseline`) | **1.5 epoch** | intermediate checkpoint of the 2-epoch run |
| `grpo_baseline/final_checkpoint` | **2.0 epoch** | `train.log` max `epoch` = 2.0 over 35 032 steps |

An earlier version of this file labelled `checkpoint-4379` as 1.5 epoch and pointed the
1.5-epoch row at the wrong artifact. **`checkpoint-4379` is 0.25 epoch.**

### 6.2 Conclusion

> **Under the fixed 17.5k GRPO subset, the ranking reward and the current training
> configuration, GRPO did not improve broad Top-K quality over SFT.** Every GRPO arm
> scores below the SFT baseline on HR@20 and NDCG@20.

This is **not** a statement that GRPO is ineffective for recommendation in general. No
claim is made about other subsets, rewards, reward weights, algorithms, or datasets —
none of those were varied.

---

## 7. GRPO efficiency optimization

Source of the gain:

1. remove repeated full-validation evaluation during training
2. remove redundant intermediate checkpoint saves
3. final offline evaluation protocol left **unchanged**

| Metric | Original 0.25 ep | Optimized 0.25 ep | Change |
|---|---|---|---|
| wall-clock | 6 111.5734 s (101.9 min) | 1 598.1889 s (26.6 min) | **−73.8 %** |
| throughput | 0.717 step/s | 2.74 step/s | **3.82×** |
| peak VRAM | 7.628 GB | 7.628 GB | unchanged |
| HR@20 | 0.17626296 | 0.17538054 | −0.088 pp |
| NDCG@20 | 0.10885475 | 0.10824989 | −0.060 pp |

All six numbers are read from `grpo_short025/train.log`, `grpo_fast025/train.log`, the two
prediction artifacts, and the per-step `peak_vram_gb` series in
`runs/_EXTRACTED_metrics/*.txt` (max = 7.627975 GB for both arms).

**Wording.** The quality change is **approximately preserved**, not lossless: HR@20 moves
by −0.088 pp and NDCG@20 by −0.060 pp. This is a **single-seed** engineering comparison;
disabling stochastic during-training evaluation can change RNG consumption, so identical
optimization trajectories are **not** assumed.

---

## 8. Compact SASRec baseline

**Artifacts:** `runs/sasrec_baseline/d_h64_neg100_metrics.json` (stored grid) and an
independent per-sample reconstruction from `d_h64_neg100.pt`, which reproduces the stored
numbers to floating-point precision.

Configuration: hidden 64, 1 head, sequence length 10, dropout 0.3, 200 epochs,
negative-sampling objective with 100 negatives, seed 42 — the best of a 4-config sweep.

**Formal Protocol-B (SID-level Top-20) test results:**

| K | HR | NDCG |
|---|---|---|
| 1 | 0.04213545 | 0.04213545 |
| 3 | 0.04588573 | 0.04435719 |
| 5 | 0.04742996 | 0.04499325 |
| 10 | 0.05537172 | 0.04756201 |
| **20** | **0.07147584** | **0.05156672** |

**Clean SFT HR@20 ÷ compact SASRec HR@20 = 0.19832341 / 0.07147584 ≈ 2.77×.**

### 8.1 Required caveats (these bound the claim)

1. **Compact baseline.** This is a small single-block SASRec, not a heavily tuned one;
   it does not represent the achievable ceiling of the SASRec family.
2. **Shared metric, different mechanisms.** Both are scored with the same SID-level Top-K
   metric, but their candidate generation and scoring differ fundamentally: MiniOneRec
   emits a constrained 3-token SID sequence with beam search; SASRec scores the full
   catalogue with a dot product and then maps items to SIDs.
3. **Uncontrolled variable.** The SFT arm is trained with SID↔title / SID↔description
   alignment auxiliary tasks; SASRec has no such objective. This is **not** controlled for.

> Therefore the correct statement is that clean SFT attains roughly 2.77× the HR@20 of this
> compact SASRec under a shared SID-level metric. **It is not a claim of a fully
> controlled, like-for-like defeat of SASRec.**

---

## 9. SID observational analysis

All results in this section are **observational**. No causal claim is made. Sources:
`analysis/results/sid_value_analysis.json`, `sid_prefix_control.json`,
`sid_prefix_matched_control.json`.

### 9.1 Popularity stratification

Strata are defined by how often the target item appears as a next-item label in **train**.
The `f = 0` stratum is called **interaction-unseen** — *not* cold-start, which would imply
a different (content-based) notion.

HR@20 by stratum (test n = 4 533):

| Stratum | n | SASRec | Clean SFT | GRPO 0.25 orig | GRPO 1.5 ep | GRPO 2.0 ep |
|---|---|---|---|---|---|---|
| interaction-unseen (f=0) | 156 | 0.00000 | 0.00000 | 0.01923 | 0.05769 | 0.05769 |
| f=1 | 211 | 0.00000 | 0.01896 | 0.03791 | 0.07109 | 0.06635 |
| f=2 | 312 | 0.00320 | 0.02244 | 0.04167 | 0.05449 | 0.05449 |
| f=3-5 | 940 | 0.00213 | 0.03298 | 0.03830 | 0.05000 | 0.04681 |
| f=6-20 | 1 630 | 0.00491 | 0.15890 | 0.16626 | 0.13436 | 0.14172 |
| f>20 | 1 284 | 0.24377 | 0.46573 | 0.36449 | 0.33178 | 0.32788 |

**Correct summary.** The bulk of SFT's *absolute* gain over compact SASRec sits in the
**medium and head** strata (f=6-20: +15.4 pp; f>20: +22.2 pp), while SASRec is essentially
inoperative in the sparse strata. GRPO's changes are a **popularity-dependent
redistribution**: the longer GRPO runs improve the interaction-sparse strata relative to
SFT (f=0: 0.00000 → 0.05769; f=1: 0.01896 → 0.07109) but lose substantially in the
medium/head (f>20: 0.46573 → 0.32788), which nets out to a lower overall Top-K.

**Not supported:** "SFT's advantage is concentrated in the tail." The tail is where SFT is
close to *zero* in absolute terms; the absolute gain is in the medium/head.

### 9.2 Target–history SID prefix affinity

Affinity is measured as the **token-level longest common prefix (LCP)** between the target
SID and any history SID. *(An earlier revision compared SID **characters** instead of
tokens, which made `<a_2>` look like a prefix of `<a_223>`; that bug produced empty
depth-0/1/2 strata and has been fixed.)*

HR@20 by LCP depth:

| Depth | n | freq median | SASRec | Clean SFT | GRPO 0.25 orig | GRPO 1.5 ep | GRPO 2.0 ep |
|---|---|---|---|---|---|---|---|
| 0 (no shared prefix) | 2 711 | 8 | 0.03762 | 0.04832 | 0.03098 | 0.01512 | 0.01291 |
| 1 (`<a_x>`) | 1 041 | 7 | 0.03074 | 0.15370 | 0.11527 | 0.08453 | 0.08742 |
| 2 (`<a_x><b_y>`) | 566 | 10 | 0.07951 | 0.70141 | 0.67491 | 0.69435 | 0.70495 |
| 3 (target in history) | 215 | 491 | 0.67442 | 0.98140 | 0.99070 | 0.98140 | 0.98140 |

**Finding.** Deeper target–history SID prefix affinity is **strongly associated** with
higher HR@20. Because the `f=0` stratum is a 491-median-frequency repeat-consumption case,
depth 2 → 3 is a qualitatively different regime and is listed separately rather than read
as a trend.

### 9.3 Matched branch-size control

To test whether §9.2 is merely an artefact of "a small catalogue branch is easy to hit",
every sample is given **one common control variable**:

> `target_prefix2_support` = number of unique catalogue SIDs sharing the target SID's
> first two tokens — used for **every** depth level.

HR@20 inside the same support bucket:

| `target_prefix2_support` | Depth | n | freq median | Clean SFT | GRPO 0.25 orig | GRPO 1.5 ep | GRPO 2.0 ep |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 312 | 8.0 | 0.06021 | 0.04878 | 0.01982 | 0.01829 |
| 1 | 1 | 309 | 8.0 | 0.11327 | 0.09385 | 0.05502 | 0.06149 |
| 1 | 2 | **0** | — | — | — | — | — |
| 2-5 | 0 | 1 111 | 8.0 | 0.03870 | 0.01440 | 0.00810 | 0.00720 |
| 2-5 | 1 | 498 | 7.0 | 0.21084 | 0.15663 | 0.10442 | 0.10442 |
| 2-5 | 2 | 110 | 7.5 | 0.61818 | 0.60909 | 0.59091 | 0.60000 |
| 6-20 | 0 | 227 | 10.0 | 0.02643 | 0.00441 | 0.00441 | 0.00441 |
| 6-20 | 1 | 147 | 6.0 | 0.09524 | 0.06122 | 0.06803 | 0.07483 |
| 6-20 | 2 | 184 | 9.0 | 0.65217 | 0.58696 | 0.57065 | 0.60326 |
| 21-100 | 0 | 61 | 23.0 | 0.04918 | 0.04918 | 0.08197 | 0.03279 |
| 21-100 | 1 | 87 | 10.0 | 0.06897 | 0.04598 | 0.10345 | 0.10345 |
| 21-100 | 2 | 272 | 17.5 | 0.76838 | 0.76103 | 0.81985 | 0.81618 |
| >100 | any | **0** | — | — | — | — | — |

**Finding.** After matching on the common control variable, depth 2 still far exceeds
depths 0 and 1 in every bucket where it exists (e.g. support 2-5: 0.61818 vs 0.21084 for
SFT). A frequency-band × support × depth triple split (cells with n ≥ 30) shows the same
ordering throughout. So **"small trie branch alone" does not fully explain the effect.**

**Two structural facts, not defects.** (a) `depth = 2` cannot occur in the
`support = 1` bucket — a branch containing only the target cannot share two tokens with
any history item. (b) No test sample has `support > 100`. Both cells are reported as empty
rather than filled.

**Caveat.** This remains **observational evidence, not causal proof.** Depth correlates
with item and user-history properties that are not controlled here.

---

## 10. Shuffled-SID controlled intervention

### 10.1 Design

The intervention destroys the **item↔SID assignment** while holding the **SID codebook**
fixed. Built by `analysis/build_shuffled_sid_strict.py`, seed 42, deterministic.

| Property | Value |
|---|---|
| catalogue items | 3 686 |
| unique SIDs | 3 670 |
| SID tokens | 560 |
| collision-involved items | **31 intentionally frozen** at their original SIDs (not shuffled) |
| singleton items permuted | **3 655** |
| permutation unit | whole 3-token SID codes, within train-frequency strata |
| bucketing | 0 / 1 / 2 / 3-5 / 6-20 / >20 by train next-item frequency |

**Intervention strength**

| Metric | Value |
|---|---|
| singleton exact-SID retention | **0.0000 %** |
| all-item exact-SID retention | **0.8410 %** (= the 31 intentionally frozen items) |
| prefix-1 self retention (singleton) | 2.4077 % |
| prefix-2 self retention (singleton) | 0.0547 % |
| **pair-identity retention, depth 1 (all)** | **2.8155 %** (Jaccard 1.4279 %) |
| **pair-identity retention, depth 2 (all)** | **0.6885 %** (Jaccard 0.3454 %) |
| pair-identity retention, depth 2 (singleton) | **0.0494 %** (Jaccard 0.0247 %) |

Pair-identity retention is `|BEFORE_pairs ∩ AFTER_pairs| / |BEFORE_pairs|`, where a pair is
an unordered item pair whose SIDs share a prefix. *(An earlier revision reported
`|AFTER_pairs| / |BEFORE_pairs|` over prefix-group counts, which is identically 100 %
because a within-bucket permutation preserves every group's SID set. That definition
measured group-size preservation, not pair identity, and has been replaced.)*

**Held invariant** (verified, 18/18 checks pass in
`analysis/results/shuffled_sid_preflight.json`): exact SID set, exact SID multiset, SID
token vocabulary, trie prefix sets at depth 1/2/3, collision group membership, popularity
strata, user and item identity, titles and descriptions, sequence lengths, and the SFT
recipe.

> **The 31 frozen items are retained, not removed.** They appear in the shuffled train,
> valid and test files exactly as in the clean ones, with their original SIDs; they are
> trained on and evaluated like any other item. "Frozen" means only that their
> item↔SID assignment was deliberately left unshuffled, which is what keeps the collision
> structure and the exact SID multiset intact. They are therefore part of the evaluation,
> not excluded from it — but they carry no treatment, so a null result cannot be
> attributed to them and a positive result cannot be explained by them.

### 10.2 Intervention coverage

| Quantity | Value |
|---|---|
| test examples | 4 533 |
| **prompts changed** | **98.81 %** (54 rows unchanged) |
| **targets changed** | **93.01 %** (317 rows unchanged) |

**Machine-verified:** the 317 rows whose target SID is unchanged are **exactly** the rows
whose target item is one of the 31 intentionally frozen collision-involved items
(`unchanged_output_rows == frozen_rows`, with `|set difference| = 0` in both directions).
So the unchanged targets correspond to intentionally frozen collision-affected examples.
The 54 unchanged **prompts** are *not* the same set (only 23 overlap) and this document
makes no claim about them.

### 10.3 Result

| K | Clean SFT HR | Shuffled-SID HR | Clean SFT NDCG | Shuffled-SID NDCG |
|---|---|---|---|---|
| 1 | 0.06926980 | 0.06132804 | 0.06926980 | 0.06132804 |
| 3 | 0.10103684 | 0.08140304 | 0.08789727 | 0.07315633 |
| 5 | 0.12111185 | 0.08824178 | 0.09613706 | 0.07599527 |
| 10 | 0.15376131 | 0.09816898 | 0.10663781 | 0.07921379 |
| **20** | **0.19832341** | **0.10765497** | **0.11786798** | **0.08157358** |

**Headline**

| Metric | Clean SFT | Shuffled-SID | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| **HR@20** | 0.19832341 | 0.10765497 | **−0.09066844** | **≈ −45.72 %** |
| **NDCG@20** | 0.11786798 | 0.08157358 | **−0.03629440** | **≈ −30.79 %** |

**Decoding health** (`analysis/results/shuffled_eval_provenance.json`, 18/18 checks pass):
LegalRate@20 = **100 %**, DuplicateRate@20 = **0 %**, unique predictions per sample = **20**
for all 4 533 rows.

**Interpretation.** Under a controlled intervention that holds the SID codebook, token
vocabulary, trie topology, popularity strata and SFT recipe essentially fixed while
altering the item↔SID assignment, Top-K recommendation quality drops sharply. This
**provides strong controlled-intervention evidence that semantic SID assignment
contributes substantially to MiniOneRec SFT performance.**

**It does not prove Semantic ID is the sole cause.** The intervention is a single
manipulation at one scale, on one category, with a single seed; the 31 collision-involved
items were intentionally frozen rather than shuffled, so for those items the item↔SID
assignment carries no treatment; and the training run carries the resume caveat in §13.

---

## 11. Final comparison table

Formal test split, K = [1, 3, 5, 10, 20]. **Two different decoding/scoring regimes are
shown, and the shared quantity is only the SID-level Top-K metric:**

**Generative runs** (rows 2-7 — clean SFT, all GRPO arms, shuffled-SID SFT)

- `beam = 20`
- `length_penalty = 0`
- constrained SID decoding (the trie restricts every step to legal SID tokens)
- `num_return_sequences = 20`; 4 533 test samples

**Compact SASRec, Protocol B** (row 1)

- item-level Top-20 scoring (full-catalogue dot product over 3 686 items)
- the 20 scored items are then mapped to SID
- therefore it shares the **same SID-level Top-K hit rule and NDCG cutoff** as the
  generative runs, which is what makes the comparison meaningful
- but its **candidate generation and scoring mechanism differs fundamentally**: no beam
  search, no length penalty, no constrained SID decoding, and no SID-sequence generation

| # | System | Regime | Split | HR@1 | HR@5 | HR@10 | **HR@20** | **NDCG@20** |
|---|---|---|---|---|---|---|---|---|
| 1 | Compact SASRec (Protocol B) | item-level Top-20 → SID | clean | 0.04213545 | 0.04742996 | 0.05537172 | **0.07147584** | **0.05156672** |
| 2 | **Clean SFT** | beam 20, constrained SID | clean | 0.06926980 | 0.12111185 | 0.15376131 | **0.19832341** | **0.11786798** |
| 3 | GRPO 0.25 ep original | beam 20, constrained SID | clean | 0.06794617 | 0.11361129 | 0.14251048 | **0.17626296** | **0.10885475** |
| 4 | GRPO 0.25 ep optimized | beam 20, constrained SID | clean | 0.06750496 | 0.11030223 | 0.14096625 | **0.17538054** | **0.10824989** |
| 5 | GRPO 1.5 ep | beam 20, constrained SID | clean | 0.07059343 | 0.10566953 | 0.13214207 | **0.16170307** | **0.10470676** |
| 6 | GRPO 2.0 ep | beam 20, constrained SID | clean | 0.07037282 | 0.10434591 | 0.12971542 | **0.16236488** | **0.10447064** |
| 7 | **Shuffled-SID SFT** | beam 20, constrained SID | shuffled | 0.06132804 | 0.08824178 | 0.09816898 | **0.10765497** | **0.08157358** |

Rows 1-6 are scored on the clean test split; row 7 uses the shuffled test split because its
target SIDs are the shuffled ones.

**Provenance differs by regime, and was verified separately:**

- **Rows 2-7 (generative):** re-scored from their own `test_beam20.json` /
  `eval_beam20.json` artifacts with the unmodified `calc.py`; every headline value matched
  its recorded metric to `|diff| = 0.000e+00`. See
  `analysis/results/shuffled_eval_provenance.json` and `analysis/audit_predictions.py`.
- **Row 1 (compact SASRec):** `calc.py` does not apply to it. Its numbers come from
  `runs/sasrec_baseline/d_h64_neg100_metrics.json` (`results.test.protocol_B_top20_sid_level`)
  and were independently cross-checked by re-running the trained
  `d_h64_neg100.pt` checkpoint to reconstruct the 4 533 × 20 candidates and recomputing the
  grid, which reproduced the stored values to floating-point precision (see §8).

**Ordering.** Clean SFT > every GRPO arm > shuffled-SID SFT > compact SASRec on both
HR@20 and NDCG@20. Because row 1 comes from a different regime, the ordering statement is
about the shared SID-level Top-K metric, not about a like-for-like decoder comparison
(the caveats in §8.1 apply).

---

## 12. Failure diagnostics: zero-advantage groups and KL

Diagnostics extracted from the per-step records in `runs/_EXTRACTED_metrics/*.txt`.
These are **observations**, not fixes, and they are not used to explain recommendation
quality.

| Run | steps logged | zero-advantage group ratio | KL median | KL p95 | KL p99 | KL max |
|---|---|---|---|---|---|---|
| GRPO 0.25 ep original | 4 379 | **69.856 %** | 0.0413 | 1.132 | 36.800 | 1.047e8 |
| GRPO 0.25 ep optimized | 4 379 | **70.153 %** | 0.0417 | 0.9997 | 28.914 | 1.047e8 |
| GRPO 2.0 ep | 35 032 | **69.277 %** | 0.0411 | 2.117 | 173.083 | 1.761e11 |
| GRPO smoke (debug) | 34 300 | 77.187 % | 0.0398 | 2.444 | 172.797 | 7.678e10 |

**Observed**

- A high zero-advantage ratio (≈70 %) is present from early training: for roughly seven of
  every ten groups the reward is constant, so the group-relative advantage is zero and the
  step contributes no learning signal.
- The KL distribution is strongly heavy-tailed, with maxima spanning 8-11 orders of
  magnitude above the median.
- Extreme KL was **not** primarily caused by singleton / forced trie branches.
- Re-normalising probabilities over legal SID tokens did **not** remove the KL outliers.

Consistent with §6, these observations describe the reward configuration as run; they do
not support any general claim about GRPO.

---

## 13. Provenance and caveats

### 13.1 Prediction sources

| Label | Artifact | Split |
|---|---|---|
| Clean SFT | `runs/eval_clean_sft/test_beam20.json` | clean |
| GRPO 0.25 ep original | `runs/grpo_short025/eval_beam20.json` | clean |
| GRPO 0.25 ep optimized | `runs/grpo_fast025/eval_beam20.json` | clean |
| GRPO 1.5 ep | `runs/eval_grpo_step26274/test_beam20.json` | clean |
| GRPO 2.0 ep | `runs/eval_grpo_baseline/test_beam20.json` | clean |
| Shuffled-SID SFT | `runs/eval_shuffled_sid/test_beam20.json` | shuffled |
| Compact SASRec | `runs/sasrec_baseline/d_h64_neg100_metrics.json` | clean |

### 13.2 Legacy / debug artifacts — excluded from every table above

| Artifact | Why excluded |
|---|---|
| `runs/eval_industrial/final_result_Industrial_and_Scientific.json` | evaluated **before** prompt unification; HR@20 = 0.15376131, NDCG@20 = 0.08471693. Legacy protocol, not comparable. |
| `runs/_EXTRACTED_metrics/grpo_smoke_metrics.txt` | debug smoke run; diagnostics only (§12). |

### 13.3 Shuffled-SFT training resume caveat

| Item | Value |
|---|---|
| completed steps | 2 496 |
| interrupted after | ≈ step 2 180 |
| `resume_from_checkpoint` | `checkpoint-2125` |
| retrained steps | **2 125 – 2 180** |

Consequences, stated plainly:

- clean vs shuffled are **not** a bitwise-identical optimization trajectory. Final LR
  reconverged to the identical value 1.21163166e-07 and the loss trajectory is continuous
  across the resume, but a material-divergence-free comparison is not assumed.
- The resume summary reports `train_loss = 0.060042…`. **This is a resumed-segment
  statistic and must not be used as the full training loss.** The reconstructed
  all-step means are **shuffled ≈ 0.802715** and **clean ≈ 0.771231**.
- Training loss is recorded here for provenance only. It is **not** used to explain
  recommendation quality.

### 13.4 `--info_file` correction (superseded conclusion)

An earlier revision of the evaluation audit asserted that the shuffled evaluation **must**
replace `--info_file`, "otherwise constrained decoding would restore the original
item↔SID mapping and undo the intervention." **That claim was wrong and is withdrawn.**

`evaluate.py` uses `--info_file` for exactly one purpose: reading field 0 (the SID string)
and building the constrained-decoding trie. The item_id field is never read, and the title
field is tokenised into a dictionary that is built and never consumed. The trie is a
function of the **SID codebook alone**, and the strict shuffle preserves that codebook
exactly. Measured for the two info files (`analysis/verify_info_equivalence_final.py`):

| Quantity | Result |
|---|---|
| legal SID sequence set | identical |
| trie prefix keys | 9 684 both, key sets identical |
| allowed-token **sets** per prefix | 0 differences |
| EOS termination nodes | 3 670 both, identical |
| decoding masks | **byte-identical** (same MD5 under the index assignment at `LogitProcessor.py:68`) |

The formal shuffled evaluation therefore defaults to the **original** info file
(`INFO_MODE=original` in `scripts/eval_shuffled_sid.sh`), keeping the evaluation constraint
space unchanged. The shuffled info file is retained only as a consistency-audit artifact.

*(121 trie keys differ in the **order** of their allowed-token list. That ordering is set
by Python set iteration over insertion history; the sole consumer is
`mask[b, prefix_allowed_tokens] = 0`, an index assignment, so the order cannot affect the
result. Verified: masks built from the two orders are byte-identical.)*

### 13.5 Other standing caveats

- **Single seed** for every arm. No variance estimates; differences are not tested for
  significance.
- **One category, one dataset.** No claim generalises to other categories.
- **Compact SASRec** is a 4-config sweep, not a tuned baseline (§8.1).
- **The GRPO subset is fixed at 17 516 rows**, so GRPO sees much less data than SFT. The
  §6 conclusion is scoped to that subset.
- **Observational analyses (§9)** are correlational; §9.3 narrows but does not eliminate
  confounding.
- **Beam-restricted metric.** HR@20 is bounded by beam width 20; this is a beam-restricted
  metric, not a full-catalogue recall.

---

## 14. Reproducibility artifacts

### 14.1 Scripts

| Path | Purpose |
|---|---|
| `scripts/sft_shuffled_sid.sh` | shuffled-SID SFT (identical recipe to clean SFT, 4 paths swapped) |
| `scripts/eval_shuffled_sid.sh` | shuffled-SID beam-20 evaluation (`INFO_MODE=original` by default) |
| `baselines/sasrec_baseline.py`, `baselines/sasrec_sweep.sh` | compact SASRec + sweep driver |
| `sft.py`, `evaluate.py`, `calc.py`, `data.py` | pipeline (evaluation code unmodified) |

### 14.2 Analysis

| Path | Purpose |
|---|---|
| `analysis/audit_predictions.py` | re-scores every prediction artifact against its own split |
| `analysis/verify_summary_claims.py` | machine-checks the headline metrics and §10.2 |
| `analysis/audit_shuffled_eval.py` | shuffled-eval provenance audit (18 checks) |
| `analysis/build_shuffled_sid_strict.py` | builds the strict shuffled dataset |
| `analysis/build_shuffled_info.py` | builds the shuffled info file (audit artifact) |
| `analysis/verify_info_equivalence_final.py` | proves trie equivalence (§13.4) |
| `analysis/analyze_sid_value.py`, `analyze_sid_prefix_control.py`, `analyze_sid_prefix_matched_control.py`, `analyze_pair_identity_retention.py` | §9 and §10.1 analyses |
| `analysis/preflight_shuffled_sft.sh`, `preflight_eval_shuffled_sid.sh` | static preflights |

### 14.3 Results

| Path | Contents |
|---|---|
| `analysis/results/sid_value_analysis.json` | popularity-stratified metrics |
| `analysis/results/sid_prefix_control.json` | prefix-affinity analysis |
| `analysis/results/sid_prefix_matched_control.json` | matched branch-size control |
| `analysis/results/shuffled_sid_preflight.json` | intervention invariants + strength |
| `analysis/results/shuffled_eval_provenance.json` | shuffled-eval provenance + deltas |
| `analysis/results/clean_vs_shuffled_sft_recipe.md` | training recipe diff |
| `analysis/results/clean_vs_shuffled_eval_recipe.md` | evaluation protocol diff |

### 14.4 Method notes worth keeping

- **`calc.py` NDCG scale.** `calc.py` accumulates `1/log(minID + 2)` with the **natural**
  logarithm and converts to the log2 scale only at print time by dividing by
  `1/log(2)`. Taking the intermediate sum at face value gives 0.17004756 instead of the
  correct 0.11786798.
- **`length_penalty`.** The value reaching `model.generate()` is the CLI value
  (**0** for the formal runs). The inner helper's `length_penalty=1.0` default is dead:
  the call site always passes the value explicitly.
- **`checkpoint-4379` is 0.25 epoch**, and `checkpoint-26274` is 1.5 epoch. See §6.1.
