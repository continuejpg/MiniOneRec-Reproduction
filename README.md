<div align="center">

<img src="./assets/logo.png" width="420em" ></img>

**MiniOneRec reproduction and analysis on a single RTX 4090, with training-correctness fixes, GRPO efficiency optimization, traditional recommender baselines, and controlled Semantic-ID intervention experiments.**

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

## 2. Key Results

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
> **uncontrolled** variable. See [Caveats](#10-caveats).

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

---

## 3. What I Changed

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
  [detailed record](notes/experiment_summary.md#144-method-notes-worth-keeping).
- The `length_penalty` reaching `model.generate()` is the CLI value (**0**); an inner
  helper's `1.0` default is dead code.
- A "prefix-pair retention = 100 %" metric was **wrong** (it measured group-size
  preservation, not pair identity) and was replaced.

---

## 4. Experimental Pipeline

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

## 5. Semantic-ID Controlled Intervention

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
MiniOneRec SFT recipe, end to end**. It does **not** isolate the sequence-only semantic
pathway, and it does **not** show that Semantic ID is the sole cause. See
[Caveats](#10-caveats) for exactly what the design does and does not control.

---

## 6. GRPO Training Efficiency

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

## 7. Baselines & Evaluation

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

## 8. Reproduction

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

## 9. Repository Structure

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

## 10. Caveats

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
full MiniOneRec SFT recipe, end to end.** It does **not** isolate the sequence-only
semantic pathway, because the metadata↔SID auxiliary heads were retrained under the same
remapped supervision and cannot be separated from it by this design. It does **not** prove
that Semantic ID is the sole cause.

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

---

## 11. Detailed Experiment Record

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
14. **Reproducibility artifacts**

Every headline number there is traceable to a prediction artifact and was re-derived by
machine from it. Superseded conclusions (an earlier `--info_file` claim, an earlier
pair-retention definition, a mislabelled checkpoint) are corrected **in place** rather
than silently removed.
---

# 🧩 Upstream framework & attribution

**MiniOneRec** is the first fully open-source **generative recommendation** framework,
providing an end-to-end workflow spanning **SID construction**, **supervised fine-tuning
(SFT)**, and recommendation-oriented **reinforcement learning (RL)**.

> **This repository is an independent reproduction and analysis of MiniOneRec, not the
> upstream project.** All upstream code is used unmodified except where listed in
> [§3 What I Changed](#3-what-i-changed). The upstream framework documentation, SID
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
