<div align="center">

<img src="./assets/logo.png" width="420em" ></img>

**MiniOneRec — generative recommendation reproduction, Semantic-ID intervention analysis, and single-GPU training/inference optimization. All on one RTX 4090.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%204090-orange.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Upstream%20Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Upstream Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Upstream Huggingface</a> | <a href="notes/experiment_summary.md">📊 Full Experiment Record</a>

</div>

> **Relationship to upstream.** This is an independent reproduction and analysis of
> [MiniOneRec](https://github.com/AkaliKong/MiniOneRec), not the upstream project.
> All upstream code, attribution, citation and license information is preserved at the
> bottom of this file — see [Upstream framework](#-upstream-framework-attribution).

---

## 1. Project Overview

MiniOneRec turns recommendation into **constrained sequence generation**: every item is
compressed into a 3-token Semantic ID, an LLM is fine-tuned to emit the next item's SID,
and a prefix trie guarantees that every beam is a legal item.

This repository does three things on top of that pipeline, all on **a single RTX 4090**:

| | |
|---|---|
| **Reproduce and fix** | Reproduced the SFT → GRPO → constrained-decoding pipeline end to end, and fixed correctness bugs in the reward/target binding that made RL results unattributable. |
| **Optimize** | Cut GRPO wall-clock by **73.8 %** (throughput **3.82×**) with approximately preserved quality. |
| **Diagnose** | Added a traditional sequential baseline, and ran a **controlled intervention on the Semantic-ID assignment** to measure how much of the SFT result the SID correspondence actually carries. |

**What this repository is not:** it is not a SOTA reproduction and not a paper summary.
It is a *diagnostic study* on a small-scale setup, with every headline number traced back
to a prediction artifact. Negative results are kept and explained rather than dropped.

---

## 2. Main Results

Formal protocol, test split, `K = [1, 3, 5, 10, 20]`, single seed. Rows 1-8 are scored on
the clean split; the shuffled rows use the shuffled split because their target SIDs are the
shuffled ones.

| # | System | Regime | HR@20 | NDCG@20 |
|---|---|---|---|---|
| 1 | Compact SASRec (Protocol B) | item-level Top-20 to SID | 0.07147584 | 0.05156672 |
| 2 | **Clean SFT** | beam 20, constrained SID | **0.19832341** | **0.11786798** |
| 3 | GRPO 0.25 ep, original | beam 20, constrained SID | 0.17626296 | 0.10885475 |
| 4 | GRPO 0.25 ep, optimized | beam 20, constrained SID | 0.17538054 | 0.10824989 |
| 5 | GRPO 1.5 ep | beam 20, constrained SID | 0.16170307 | 0.10470676 |
| 6 | GRPO 2.0 ep | beam 20, constrained SID | 0.16236488 | 0.10447064 |
| 7 | Shuffled Full SFT | beam 20, constrained SID | 0.10765497 | 0.08157358 |
| 8 | Seq-only Clean | beam 20, constrained SID | 0.17802780 | 0.10842814 |
| 9 | Seq-only Shuffled | beam 20, constrained SID | 0.10125745 | 0.07683363 |

Row 1 shares only the **SID-level Top-K metric** with the generative rows: its candidate
generation and scoring mechanism differ fundamentally (no beam search, no length penalty,
no constrained SID decoding), and it is a 4-config sweep rather than a tuned baseline. Clean
SFT reaches 2.7747x its HR@20. See [Caveats](#14-caveats).

Inference cost for these operating points is in
[Inference Benchmark](#9-inference-benchmark).

---

## 3. Key Results

### ① Training correctness — the reward was bound to the wrong thing

The upstream RL stage matched rewards to samples by **prompt text**. After any DataFrame
sampling, prompts can repeat or reorder, so a reward could be credited to a different
sample than the one that produced it — making the RL results uninterpretable.

**Fix:** stable `sample_id` → target attribution (namespaced by task type and train/valid
split), with **fail-loud validation** instead of a prompt fallback. Verified on the frozen
GRPO subset:

```
bound IDs = 17 516   missing IDs = 0   train/eval overlap = 0
```

### ② Clean SFT vs compact SASRec

| Model | HR@20 | NDCG@20 |
|---|---|---|
| Clean SFT (beam 20, constrained SID decoding) | **19.832 %** (`0.19832341`) | **11.787 %** (`0.11786798`) |
| Compact SASRec (item-level Top-20 → SID) | 7.148 % (`0.07147584`) | 5.157 % (`0.05156672`) |

Clean SFT reaches **≈ 2.77×** the HR@20 of this compact SASRec
(`0.19832341 / 0.07147584 = 2.7747`).

> **These two share only the SID-level Top-K metric, not the mechanism.** The generative
> model emits a constrained 3-token SID sequence with beam search; SASRec scores the full
> catalogue with a dot product and then maps items to SIDs. SASRec here is a compact
> 4-config sweep, not a tuned baseline, and the SFT arm additionally trains SID↔title /
> SID↔description alignment objectives that SASRec has no counterpart for — an
> **uncontrolled** variable. See [Caveats](#14-caveats).

### ③ GRPO training efficiency

Optimization = remove repeated full-validation during training + remove redundant
intermediate checkpoint saves. **The final offline evaluation protocol is unchanged.**

| Metric | Original 0.25 ep | Optimized 0.25 ep | Change |
|---|---|---|---|
| wall-clock | 6 111.5734 s (101.9 min) | 1 598.1889 s (26.6 min) | **−73.8 %** |
| throughput | 0.717 step/s | 2.74 step/s | **3.82×** |
| peak VRAM | 7.628 GB | 7.628 GB | unchanged |
| HR@20 | 17.626 % (`0.17626296`) | 17.538 % (`0.17538054`) | −0.088 pp |
| NDCG@20 | 10.885 % (`0.10885475`) | 10.825 % (`0.10824989`) | −0.060 pp |

Quality is **approximately preserved**, not lossless. This is a **single-seed** engineering
comparison; disabling stochastic during-training evaluation can change RNG consumption, so
identical optimization trajectories are **not** assumed.

### ④ Semantic-ID controlled intervention

A strict intervention that destroys the **item↔SID assignment** while holding the SID
codebook fixed:

| Model | HR@20 | NDCG@20 |
|---|---|---|
| Clean SFT | 19.832 % (`0.19832341`) | 11.787 % (`0.11786798`) |
| Shuffled-SID SFT | 10.765 % (`0.10765497`) | 8.157 % (`0.08157358`) |
| **Relative change** | **−45.72 %** | **−30.79 %** |

The intervention holds the **SID codebook, token vocabulary, trie topology, collision
structure, popularity strata and SFT recipe** essentially fixed, changing only which item
owns which SID. Absolute ΔHR@20 = **−0.09066844**, ΔNDCG@20 = **−0.03629440**.

### (5) Sequence-only SID ablation

The same shuffled-SID intervention, but with the two explicit text-auxiliary supervision
tasks removed: only `SidSFTDataset` (history SID -> target SID) is kept, while
`SidItemFeatDataset` and `FusionSeqRecDataset` are not instantiated. What is removed is the
**explicit text auxiliary supervision**, not all textual semantics — the SIDs themselves are
still produced by upstream text quantisation.

| Model | HR@20 | NDCG@20 |
|---|---|---|
| Seq-only, clean | 17.803 % (`0.17802780`) | 10.843 % (`0.10842814`) |
| Seq-only, shuffled | 10.126 % (`0.10125745`) | 7.683 % (`0.07683363`) |
| **Relative change** | **−43.12 %** | **−29.14 %** |

The shuffled-SID effect remains large without the auxiliary tasks, so the full-SFT
intervention effect is not mainly dependent on those supervision paths. See
[Semantic ID Analysis](#8-semantic-id-analysis) for the side-by-side comparison.

### (6) Inference quality–latency benchmark

Zero-shot cost of the *same* protocol at different beam widths and batch sizes, measured
with the repository's own `evaluate.py` (see [Inference Benchmark](#9-inference-benchmark)).

| Operating point | HR@20 | throughput | P50 latency | peak VRAM |
|---|---|---|---|---|
| beam 20, batch 8 (**balanced default**) | 0.19832341 | 27.49 samples/s | 300.17 ms | 2.03 GB |
| beam 50, batch 8 | 0.19942643 | 12.60 samples/s | 649.25 ms | 3.68 GB |

Doubling the beam buys **+0.0011 HR@20 absolute (+0.11 pp)** for **−54 % throughput** and
**1.81× VRAM**.

---

### (6b) Backbone scaling sanity check — Qwen2.5-0.5B vs 1.5B

The full Clean vs. strict shuffled-SID protocol was repeated on a **3.12x larger backbone**
(same Qwen2.5 family, same tokenizer, same formal recipe; only the backbone changed).

| Backbone | Clean HR@20 | Shuffled HR@20 | HR drop | Clean NDCG@20 | Shuffled NDCG@20 | NDCG drop |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.19832341 | 0.10765497 | -45.7175% | 0.11786798 | 0.08157358 | -30.7924% |
| Qwen2.5-1.5B | 0.19964703 | 0.11625855 | -41.7680% | 0.11877667 | 0.08510443 | -28.3492% |

0.5B -> 1.5B Clean: **HR@20 +0.1324 pp**, **NDCG@20 +0.0909 pp**.
0.5B -> 1.5B Shuffled: **HR@20 +0.8604 pp**, **NDCG@20 +0.3531 pp**.

> Scaling Qwen2.5 from 0.5B to 1.5B produced only a small change in Clean Top-K
> performance, while the large degradation under the strict shuffled-SID intervention
> persisted.

> This is a single-seed, single-category scaling sanity check; it is not evidence of
> statistical significance or universal backbone invariance.

The Clean gain is small and is **not** uniform across cutoffs: on 1.5B, HR@3 and HR@5 are
slightly *lower* than on 0.5B while HR@1 and HR@10 are higher. Clean and shuffled rows are
scored on different splits (a shuffled target SID is not comparable to a clean one), so the
drop columns are the meaningful comparison, not the cross-column differences.

0.5B remains the **project's main experimental backbone**; 1.5B is an extension check, not a
second full replication. This step adds no significance claim and no causality claim.

---

## 4. What I Changed

### Training-correctness fixes

| Issue | Fix |
|---|---|
| Reward matched to samples by **prompt text** | Stable `sample_id` → target binding; fail-loud, no prompt fallback |
| `seq:{local_idx}` IDs renumbered rows after DataFrame sampling | Namespaced, stable sample IDs |
| Train and eval sample IDs could collide | Split identity folded into the ID |
| Reward binding broke when subset datasets became Python lists | Fixed binding for the list conversion path |
| SFT and evaluation prompt text could silently drift apart | Shared `build_recommendation_prompt()` |
| GRPO degeneracy was invisible | Added per-step **zero-advantage group ratio** and KL logging |

### Reproducibility and audit tooling (added)

- `analysis/audit_predictions.py` — re-scores **every** prediction artifact against its own
  split and matches it to its official metric.
- `analysis/verify_summary_claims.py` — machine-checks the headline metrics and the
  intervention-coverage claim.
- `analysis/audit_shuffled_eval.py`, `analysis/verify_info_equivalence_final.py` —
  provenance and decoding-equivalence audits for the intervention run.
- Static preflight scripts for **both** the training and the evaluation step; they refuse
  to start on any inconsistency.

### Protocol corrections found while auditing

- **`checkpoint-4379` is 0.25 epoch**, not 1.5 epoch; `checkpoint-26274` is 1.5 epoch.
- `calc.py` accumulates NDCG with the **natural** logarithm and converts to the log2
  scale only at print time, so the intermediate accumulator is not the reported metric.
  Taking it at face value overstates NDCG substantially. Exact figures in the
  [detailed record](notes/experiment_summary.md#194-method-notes-worth-keeping).
- The `length_penalty` reaching `model.generate()` is the CLI value (**0**); an inner
  helper's `1.0` default is dead code.
- A "prefix-pair retention = 100 %" metric was **wrong** (it measured group-size
  preservation, not pair identity) and was replaced.

---

## 5. Experimental Pipeline

```
   Amazon Industrial_and_Scientific  (3 686 items / 3 670 unique SIDs)
                 │
                 │  SID construction is taken from the upstream pre-computed
                 │  index (RQ-based).  This repository starts from that index.
                 ▼
   Semantic-ID index  ──►  560 SID tokens added  ──►  vocab 151 665 → 152 225
                 │
                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  SFT        Qwen2.5-0.5B, 2 epochs                          │
   │             79 834 train rows, effective batch 64            │
   │             1 247 steps/epoch → 2 496 steps total            │
   │             LR 3e-4, linear schedule, warmup 20, bf16        │
   └──────────────────────────────────────────────────────────────┘
                 │
                 ├──────────────►  Clean SFT baseline   (HR@20 19.832 %)
                 │
                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  GRPO on a frozen 17 516-row subset                          │
   │     seq_rec 10 000 | title/desc 6 516 | seqtitle 1 000       │
   │     group-relative advantage + KL penalty, constrained        │
   │     beam rollout                                             │
   └──────────────────────────────────────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Evaluation   beam 20 · length_penalty 0 ·                     │
   │               deterministic beam search · constrained SID      │
   │               decoding · K = [1, 3, 5, 10, 20] · 4 533 samples │
   └──────────────────────────────────────────────────────────────┘

   Interventional arm (same recipe, different data):
   Semantic-ID assignment shuffled  ──►  SFT  ──►  same evaluation
```

**Decoding health of the formal runs:** LegalRate@20 = **100 %**, DuplicateRate@20 = **0 %**,
exactly 20 unique candidates per sample.

---

## 6. Semantic-ID Controlled Intervention

### Design

The intervention destroys the **item↔SID assignment** while holding the **SID codebook**
fixed, so that any measured drop is attributable to the correspondence and not to a
smaller or differently-shaped code space.

| Property | Value |
|---|---|
| catalogue items | 3 686 |
| unique SIDs | 3 670 |
| SID tokens | 560 |
| collision-involved items | **31 intentionally frozen** (not shuffled) |
| singleton items permuted | **3 655** |
| permutation unit | whole 3-token SID codes, within train-frequency strata |
| strata | 0 / 1 / 2 / 3-5 / 6-20 / >20 by train next-item frequency |
| seed | 42, deterministic |

> **The 31 frozen items are retained, not removed.** They stay in the shuffled train,
> valid and test files with their original SIDs and are trained on and evaluated like any
> other item. "Frozen" means only that their item↔SID assignment was left unshuffled —
> which is exactly what keeps the collision structure and the exact SID multiset intact.

### Intervention strength

| Metric | Value |
|---|---|
| singleton exact-SID retention | **0.0000 %** |
| all-item exact-SID retention | 0.8410 % (= the 31 frozen items) |
| pair-identity retention, prefix depth 1 | 2.8155 % |
| pair-identity retention, prefix depth 2 | 0.6885 % (singleton 0.0494 %) |
| prompts changed | 98.81 % |
| targets changed | 93.01 % |

The 317 unchanged targets are **exactly** the rows whose target item is one of the frozen
collision items (machine-verified, set difference 0 in both directions).

### Held invariant

Verified by 18/18 checks: exact SID set, exact SID multiset, SID token vocabulary, trie
prefix sets at depth 1/2/3, collision group membership, popularity strata, user and item
identity, titles and descriptions, sequence lengths, and the SFT recipe.

### Result

| Model | HR@20 | NDCG@20 |
|---|---|---|
| Clean SFT | 19.832 % (`0.19832341`) | 11.787 % (`0.11786798`) |
| Shuffled-SID SFT | 10.765 % (`0.10765497`) | 8.157 % (`0.08157358`) |
| **Relative change** | **−45.72 %** | **−30.79 %** |

This is **controlled-intervention evidence on the complete MiniOneRec SFT recipe**. The
experiment supports that **semantic SID assignment contributes substantially to the full
MiniOneRec SFT recipe, end to end.** A separate sequence-only ablation (see
[Semantic ID Analysis](#8-semantic-id-analysis)) removes the two explicit text
auxiliary supervision tasks and finds the shuffled-SID effect remains large
(-43.12 % HR@20, -29.14 % NDCG@20 relative), so this effect is not mainly carried by
those auxiliary paths. Neither experiment **proves** that Semantic ID is the sole
cause. See
[Caveats](#14-caveats) for exactly what the design does and does not control.

---

## 7. GRPO Training Efficiency

**What was removed**

1. repeated full-validation evaluation during training
2. redundant intermediate checkpoint saves

**What was deliberately left untouched:** the final offline evaluation protocol.

| Metric | Original 0.25 ep | Optimized 0.25 ep | Change |
|---|---|---|---|
| wall-clock | 6 111.5734 s (101.9 min) | 1 598.1889 s (26.6 min) | **−73.8 %** |
| throughput | 0.717 step/s | 2.74 step/s | **3.82×** |
| peak VRAM | 7.628 GB | 7.628 GB | unchanged |
| HR@20 | 0.17626296 | 0.17538054 | −0.088 pp |
| NDCG@20 | 0.10885475 | 0.10824989 | −0.060 pp |

Quality is **approximately preserved**. Single seed, so no variance estimate is claimed.

> **Failure diagnostics are deliberately not on this page.** The GRPO arms exhibit a
> high zero-advantage group ratio (~70 %) from early training and a strongly heavy-tailed
> KL distribution. Those observations, and the scope of the GRPO conclusion, live in
> [the detailed experiment record](notes/experiment_summary.md#12-failure-diagnostics-zero-advantage-groups-and-kl).

---

## 8. Semantic ID Analysis

Controlled intervention evidence on the **item↔SID assignment**. The treatment changes only
which item owns which Semantic ID; the SID codebook and set, the token vocabulary, the trie
prefix structure at every depth, the resulting allowed-token decoding masks, the popularity
stratification, item/user identity, sequence lengths and the SFT recipe are all held fixed.
The 31 collision items are frozen and carry no treatment.

| Arm | HR@20 clean → shuffled | relative Δ | NDCG@20 clean → shuffled | relative Δ |
|---|---|---|---|---|
| Full SFT | 0.19832341 → 0.10765497 | **−45.72 %** | 0.11786798 → 0.08157358 | **−30.79 %** |
| Sequence-only | 0.17802780 → 0.10125745 | **−43.12 %** | 0.10842814 → 0.07683363 | **−29.14 %** |

This is **controlled intervention evidence**, not proof of a causal semantic hierarchy. The
two arms also differ in training-set composition (79 834 vs 36 259 samples), so the seq-only
comparison is not a single-variable isolation. See [Caveats](#14-caveats).

---

## 9. Inference Benchmark

Formal protocol, unmodified `calc.py`, ORIGINAL INFO constrained decoding. The benchmark
imports the repository's own `evaluate.py`, so it cannot drift from the formal
recommendation protocol.

**Quality sweep** — 4 533 test rows, batch 8:

| beam | HR@20 | NDCG@20 | throughput | P50 | P95 | peak VRAM |
|---|---|---|---|---|---|---|
| 5 | — | — | 61.35 samples/s | 130.85 ms | 135.73 ms | 1.21 GB |
| 10 | — | — | 45.37 samples/s | 178.49 ms | 184.46 ms | 1.48 GB |
| **20** | **0.19832341** | **0.11786798** | 27.49 samples/s | 300.17 ms | 308.54 ms | 2.03 GB |
| 50 | 0.19942643 | 0.11818527 | 12.60 samples/s | 649.25 ms | 668.67 ms | 3.68 GB |

A beam-k run emits only k candidates, so beam 5 and beam 10 cannot report @20. Those cells
are **deliberately em-dashed, not zero**. The beam-20 / batch-8 cell reproduces the formal
clean-SFT reference exactly (`reference_match = True`).

**Efficiency sweep** — beam 20, fixed first-512 workload:

| batch | throughput | batch P50 | batch P95 | peak VRAM | peak device used |
|---|---|---|---|---|---|
| 1 | 11.53 samples/s | 85.63 ms | 90.66 ms | 1.07 GB | 5.53 GB |
| **8** | 27.28 samples/s | 300.14 ms | 305.54 ms | 2.03 GB | 5.53 GB |
| 32 | 30.33 samples/s | 1 041.73 ms | 1 049.10 ms | 5.34 GB | 6.99 GB |

For batch 1 the single-request latency is that same measurement
(P50 85.63 ms / P95 90.66 ms). `peak VRAM` is PyTorch peak *allocated* memory over the
measured generation window; `peak device used` is a separately named device-level proxy that
includes the CUDA context — they are not the same quantity.

### Balanced operating point

**`beam = 20, batch = 8`.**

- Raising the beam to 50 costs **−54 % throughput** and **1.81× VRAM** for a marginal
  Top-20 gain (**+0.11 pp** HR@20).
- Raising the batch to 32 gains only **+11 % throughput** for **3.47× batch latency** and
  **2.63× peak VRAM**.

---

## 10. Reproducibility & Engineering

| Practice | What it means here |
|---|---|
| **Fail-loud preflight** | Every formal stage has a static preflight that refuses to start on any inconsistency, instead of proceeding and producing unattributable numbers. |
| **Stable sample IDs** | The RL reward is bound to `sample_id` (namespaced by task and split) with fail-loud validation, not to prompt text — the bug that made upstream RL results unattributable. |
| **Shared prompts** | One prompt construction path across arms, so clean vs. intervention differ only in the intervention variable. |
| **Artifact hashing** | Every headline number traces to a prediction artifact with a recorded SHA256; archives are hash-verified on both server and workstation. |
| **Byte-identical reference reproduction** | The restored formal checkpoint regenerates the historical beam-20 prediction artifact byte for byte (identical SHA256). |
| **Portable launchers** | Entrypoints resolve paths from their own location, so they run from any working directory and on any checkout. |
| **Atomic benchmark results** | Results are written per configuration via temp-file + `fsync` + `os.replace`, so a later failure cannot lose earlier configurations. |
| **No second protocol** | The benchmark imports the repository's `evaluate.py`; an AST guard asserts the protocol classes are not re-implemented. |

---

## 11. Baselines & Evaluation

### Formal evaluation protocol

| Parameter | Value |
|---|---|
| test samples | 4 533 |
| `num_beams` | 20 |
| `num_return_sequences` | 20 |
| `length_penalty` | 0 |
| `max_new_tokens` | 256 (upper bound; a SID is 3 tokens + EOS) |
| decoding | deterministic beam search (`do_sample` unset → False) |
| constraint | constrained SID decoding via a prefix→allowed-token dictionary |
| cutoffs | K = [1, 3, 5, 10, 20] |
| metric code | `calc.py`, unmodified |

### Baselines

| Baseline | Configuration | HR@20 | NDCG@20 |
|---|---|---|---|
| **Compact SASRec** | hidden 64, 1 head, seq len 10, dropout 0.3, 200 epochs, 100 negatives — best of a 4-config sweep | 7.148 % | 5.157 % |

SASRec is evaluated under **Protocol B**: item-level Top-20 scoring over the full
catalogue, then mapped to SIDs, so that it shares the **SID-level Top-K hit rule and NDCG
cutoff** with the generative runs. Its candidate generation and scoring mechanism still
differs fundamentally.

### Other arms measured

GRPO at 0.25 / 1.5 / 2.0 epochs (all below the clean SFT baseline), a legacy
pre-prompt-unification evaluation (**excluded** from all result tables), and observational
analyses of quality vs item popularity and vs target–history SID prefix affinity
(**correlational, not causal**). Full grids, per-stratum breakdowns and the matched
branch-size control are in [the detailed experiment record](notes/experiment_summary.md).

---

## 12. Reproduction

### Environment

| Item | Value |
|---|---|
| GPU | single NVIDIA RTX 4090 (24 GB) |
| Python | 3.10.8 |
| torch | 2.6.0+cu118 |
| transformers | 4.57.1 |
| trl | 0.24.0 |
| base model | Qwen2.5-0.5B |

```bash
pip install -r requirements.txt
```

### Clean SFT

```bash
bash sft_full.sh          # 2 epochs, effective batch 64, LR 3e-4, cutoff 512, seed 42
```

### Shuffled-SID intervention arm

```bash
# 1. build the strict shuffled dataset (seed 42, deterministic)
python analysis/build_shuffled_sid_strict.py

# 2. static preflight -- refuses to start on any inconsistency
bash analysis/preflight_shuffled_sft.sh

# 3. SFT on the shuffled data (identical recipe, 4 paths swapped)
bash scripts/sft_shuffled_sid.sh

# 4. evaluation (formal protocol; INFO_MODE=original keeps the decoding
#    constraint space unchanged)
bash analysis/preflight_eval_shuffled_sid.sh
bash scripts/eval_shuffled_sid.sh

# 5. provenance audit
python analysis/audit_shuffled_eval.py
```

### Verification (no GPU needed)

```bash
python analysis/audit_predictions.py      # re-score every prediction artifact
python analysis/verify_summary_claims.py  # machine-check the headline numbers
python analysis/verify_clean_eval_metrics.py
```

### Upstream full pipeline

The upstream SID-construction path (RQ-VAE / RQ-Kmeans / RQ-Kmeans+ on Amazon item
embeddings) is unchanged and documented under
[Upstream framework](#-upstream-framework-attribution) below.

---

## 13. Repository Structure

```
├── README.md                        this file
├── notes/
│   └── experiment_summary.md        ⭐ authoritative experiment record (14 sections)
├── scripts/
│   ├── sft_shuffled_sid.sh          intervention training launcher
│   └── eval_shuffled_sid.sh         intervention evaluation launcher
├── baselines/
│   ├── sasrec_baseline.py           compact SASRec, dual-protocol evaluation
│   └── sasrec_sweep.sh              4-config sweep driver
├── analysis/
│   ├── audit_predictions.py         re-score every prediction artifact
│   ├── verify_summary_claims.py     machine-check headline numbers
│   ├── audit_shuffled_eval.py       intervention provenance audit (18 checks)
│   ├── build_shuffled_sid_strict.py build the strict shuffled dataset
│   ├── build_shuffled_info.py       build the shuffled info file (audit artifact)
│   ├── verify_info_equivalence_final.py   prove trie/decoding equivalence
│   ├── analyze_sid_value.py                 popularity-stratified analysis
│   ├── analyze_sid_prefix_control.py        prefix-affinity analysis
│   ├── analyze_sid_prefix_matched_control.py  matched branch-size control
│   ├── analyze_pair_identity_retention.py   pair-identity retention
│   ├── preflight_shuffled_sft.sh            training preflight
│   ├── preflight_eval_shuffled_sid.sh       evaluation preflight
│   └── results/                     small authoritative JSON / Markdown results
├── sft.py  rl.py  minionerec_trainer.py  data.py       upstream pipeline
├── evaluate.py  calc.py  LogitProcessor.py             upstream evaluation
└── rq/                                                 upstream SID construction
```

---

## 14. Caveats

**Scope of the intervention claim.** The **controlled intervention variable is the
item↔SID assignment.** Everything the constrained decoder depends on is held fixed: the
legal SID codebook and set, the token vocabulary, the trie prefix structure at every depth,
and therefore the allowed-token decoding masks — which were separately verified to be
**byte-identical** between the original and shuffled info files, so the intervention
introduces no change in decoding behaviour. Also unchanged: popularity stratification,
item and user identity, titles and descriptions, sequence lengths, and the SFT training
recipe.

What necessarily differs is what training *learns* from that assignment. Retraining on the
shuffled mapping naturally produces **different model parameters**, and the metadata↔SID
auxiliary supervision is **deterministically remapped together with the SID assignment**,
because that supervision is derived from the same index.

So the experiment supports: **semantic SID assignment contributes substantially to the
full MiniOneRec SFT recipe, end to end.** A separate sequence-only ablation
(see [Semantic ID Analysis](#8-semantic-id-analysis)) removes the two explicit text
auxiliary supervision tasks and finds the shuffled-SID effect remains large
(−43.12 % HR@20, −29.14 % NDCG@20 relative), so this effect is not mainly carried by those
auxiliary paths. Neither experiment **proves** that Semantic ID is the sole cause.

**The 31 frozen collision items carry no treatment.** They are retained in train, valid
and test and are evaluated normally, but their item↔SID assignment was intentionally not
shuffled. A null result cannot be attributed to them and a positive result cannot be
explained by them.

**SASRec comparison.** Compact 4-config sweep, not a tuned baseline. The shared quantity is
the SID-level Top-K metric only; generation/scoring regimes differ. The SFT arm trains
SID↔title and SID↔description alignment objectives that SASRec has no counterpart for —
an uncontrolled variable. This is **not** a claim of a fully controlled, like-for-like
defeat of SASRec.

**Single seed everywhere.** No variance estimates; differences are not tested for
significance. One category, one dataset.

**GRPO scope.** The GRPO subset is frozen at 17 516 rows, much smaller than SFT's 79 834,
so the GRPO conclusion is scoped to *that subset, that reward and that configuration*.
It is **not** a general statement that GRPO is ineffective for recommendation.

**Beam-restricted metric.** HR@20 is bounded by beam width 20 — a beam-restricted metric,
not full-catalogue recall.

**Shuffled-arm training was resumed.** The run completed all 2 496 steps but was
interrupted after ≈ step 2 180 and resumed from `checkpoint-2125`, so steps 2 125–2 180
were retrained. The arms are **not** a bitwise-identical optimization trajectory. The
`train_loss` printed by the resume segment is a resumed-segment statistic and must not be
used as the full training loss.

**Observational analyses are correlational.** Popularity and prefix-affinity findings carry
no causal claim; the matched branch-size control narrows but does not eliminate confounding.

**The seq-only arm is not a single-variable isolation.** It removes the explicit text
auxiliary supervision, but it also trains on far fewer samples (36 259 vs 79 834), so the
full-vs-seq-only comparison changes more than one thing. It also does **not** remove text
from the system: the Semantic IDs remain text-quantised upstream.

**Batch-dependent prediction divergence was observed** in the inference benchmark, while
aggregate HR/NDCG remained nearly stable (batch 1 vs 8: 0.78 % exact prediction match;
batch 1 vs 32: 0.98 %; subset HR@20 identical at 0.18359375 across all three). No root cause
is claimed — this is recorded as an observation, not attributed to floating-point
non-determinism.

**Backbone scale.** Qwen2.5-0.5B is the current main backbone; no larger-backbone result is
reported here.

---

## 15. Detailed Experiment Record

📊 **[`notes/experiment_summary.md`](notes/experiment_summary.md)** is the **single
authoritative record** for every number on this page.

It contains, in full:

1. Experiment scope
2. Environment
3. Correctness fixes
4. Frozen data / protocols
5. Clean SFT baseline
6. **GRPO benchmark** — all arms, checkpoint↔epoch labels, scoped conclusion
7. **GRPO efficiency optimization**
8. **Compact SASRec baseline** — full grid + required caveats
9. **SID observational analysis** — popularity strata, token-level LCP, matched control
10. **Shuffled-SID controlled intervention**
11. **Final comparison table** — all 7 systems
12. **Failure diagnostics** — zero-advantage ratio, KL heavy tail
13. **Provenance and caveats** — including corrections of superseded conclusions
14. **Sequence-only SID ablation** — clean vs. shuffled, and the full-SFT comparison
15. **Reference reproduction** — restored checkpoint under the formal protocol
16. **Inference quality–latency benchmark** — protocol identity, quality and efficiency sweeps
17. **Batch-invariance caveat**
18. **Final artifact provenance** — archives, hashes, benchmark method notes
19. **Reproducibility artifacts**

Every headline number there is traceable to a prediction artifact and was re-derived by
machine from it. Superseded conclusions (an earlier `--info_file` claim, an earlier
pair-retention definition, a mislabelled checkpoint) are corrected **in place** rather
than silently removed.
---

## 15b. RL Post-training Extension (R1 / R2)

Two RL reward designs were tried after the GRPO benchmark above. **Both are negative
results and are documented in full** in
[`docs/rl_posttraining_results.md`](docs/rl_posttraining_results.md).

| experiment | change | outcome |
|---|---|---|
| **R1** ReRe-style rank reward | replaced the `ranking` reward with ReRe Eq.7-9 (rank-aware, normalised per group) | `zero_advantage_group_ratio` fell to **0.0** on every step, yet test HR@20 fell to **0.05912199** and the output distribution collapsed (**99** distinct top-1 predictions, most-common share 33.6 %). |
| **R2** reachability-guided GRPO | exact-match 0/1 reward + a route cache that puts the first GT SID token in the prompt for samples the frozen SFT could not reach (h=1); supervised prefix CE (coef 0.1) keeps the hinted level trained | `zero_advantage_group_ratio` improved to **0.4750** (from ~0.70), yet validation HR@20 fell to **0.15600177** vs SFT **0.18314210** and old GRPO **0.17122683**. |

**Conclusion: neither method beats the clean SFT baseline. SFT remains the best model.**
Reducing the zero-advantage group ratio was **not** the bottleneck - old GRPO carried
~70 % zero-advantage groups and still outscored R2.

> **Mechanism notes are hypotheses, not proven causes.** R1's collapse is *consistent
> with* ReRe Eq.9's within-group normalisation rewarding preservation of the frozen
> policy's own ordering, but no control run isolates it. R2 changed the reward **and**
> the prompt routing at once and has **no ablation separating them**, so the cause of
> its regression is **UNKNOWN**. All runs are single-seed.

### R2 infrastructure worth reusing

- `LogitProcessor.py` gained an optional **`count_0`** parameter. Its first trie key was
  hardcoded to `sent[-3:]`, which slides onto a hinted prefix and forces EOS; counting the
  hinted tokens as already-generated fixes it. **Default 0 leaves the NORMAL path
  bit-for-bit unchanged.**
- `rl_router.py` builds an offline route cache from the frozen SFT's h=0 rollout over the
  **train split only** (17,516 entries, 4,818 NORMAL / 12,698 HARD). In formal mode `rl.py`
  **aborts** if the cache does not cover every training `sample_id`; it will not silently
  shrink the training set.
- New pure-function modules: `rere_reward.py`, `rl_reward.py`, `rl_prefix_ce.py`.
- New launchers: `patches/grpo_rere_rank025.sh`, `patches/grpo_r21.sh`,
  `patches/eval_grpo_r21_valid.sh`.

These are marked **experimental** and are **not** wired to any default configuration.

---

## 16. Next Step

The Qwen2.5-1.5B backbone-scaling sanity check described in
[Key Results (6b)](#6b-backbone-scaling-sanity-check--qwen25-05b-vs-15b) is complete.

> Future work: multi-seed / multi-category validation and broader backbone scaling.

---

# 🧩 Upstream framework & attribution

**MiniOneRec** is the first fully open-source **generative recommendation** framework,
providing an end-to-end workflow spanning **SID construction**, **supervised fine-tuning
(SFT)**, and recommendation-oriented **reinforcement learning (RL)**.

> **This repository is an independent reproduction and analysis of MiniOneRec, not the
> upstream project.** All upstream code is used unmodified except where listed in
> [§3 What I Changed](#4-what-i-changed). The upstream framework documentation, SID
> construction recipes and full pipeline walk-through live in the upstream repository
> linked below and are not duplicated here.

| | |
|---|---|
| **Upstream repository** | https://github.com/AkaliKong/MiniOneRec |
| **Upstream paper** | [arXiv:2510.24431](https://arxiv.org/abs/2510.24431) — *MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation* |
| **Upstream models** | [Huggingface](https://huggingface.co/kkknight/MiniOneRec) · [Modelscope](https://modelscope.cn/models/k925238839/MiniOneRec) |
| **License** | Apache-2.0 (see [`LICENSE`](LICENSE)) |
| **Institutions** | LDS · AlphaLab · NExT (USTC) |

### Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects.
We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

### Citation

If you find the upstream code, paper or models helpful, please consider citing them 📝 and
staring the upstream repository ⭐️

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}
```

---

<div align="center">
Upstream framework by the MiniOneRec authors. Reproduction and analysis in this
repository by the repository owner.
</div>
