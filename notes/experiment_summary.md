# MiniOneRec Experiment Summary

## Evaluation protocol

- Dataset: Amazon Industrial_and_Scientific
- Test samples: 4533
- num_beams: 20
- max_new_tokens: 256
- length_penalty: 0
- seed: 42
- Semantic-ID constrained decoding
- Metrics: HR@K / NDCG@K, K = [1, 3, 5, 10, 20]

## Clean SFT

NDCG:
[0.06926980, 0.08789727, 0.09613706, 0.10663781, 0.11786798]

HR:
[0.06926980, 0.10103684, 0.12111185, 0.15376131, 0.19832341]

## GRPO 0.25 epoch - original training pipeline

Train runtime:
6111.5734 s (~101.9 min)

Train throughput:
0.717 steps/s

Peak VRAM:
7.628 GB

NDCG:
[0.06794617, 0.08343013, 0.09109820, 0.10028147, 0.10885475]

HR:
[0.06794617, 0.09463931, 0.11361129, 0.14251048, 0.17626296]

## GRPO 0.25 epoch - optimized training pipeline

Optimization:
- Disable repeated full-validation evaluation during training
- Disable redundant intermediate checkpoints
- Keep final offline evaluation protocol unchanged

Train runtime:
1598.1889 s (~26.6 min)

Train throughput:
2.74 steps/s

Peak VRAM:
7.628 GB

NDCG:
[0.06750496, 0.08387134, 0.08969890, 0.09953889, 0.10824989]

HR:
[0.06750496, 0.09596294, 0.11030223, 0.14096625, 0.17538054]

Engineering result (single-seed comparison):
- Training algorithm, frozen data subset, reward, LR, beta, generation count, and epoch budget were kept fixed
- Wall-clock training time reduced by ~73.8%
- Training throughput improved by ~3.82x
- Peak VRAM essentially unchanged
- HR@20 changed from 17.626% to 17.538% (-0.088 pp)
- NDCG@20 changed from 10.885% to 10.825% (-0.060 pp)

## GRPO 1.5 epoch

NDCG:
[0.07059343, 0.08369281, 0.08883764, 0.09727708, 0.10470676]

HR:
[0.07059343, 0.09309508, 0.10566953, 0.13214207, 0.16170307]

## GRPO 2.0 epoch

Train runtime:
16999.7476 s (~4h43m)

Train throughput:
2.061 steps/s

Peak VRAM:
7.628 GB

NDCG:
[0.07037282, 0.08346697, 0.08805141, 0.09617803, 0.10447064]

HR:
[0.07037282, 0.09309508, 0.10434591, 0.12971542, 0.16236488]

## Training diagnostics

GRPO 0.25 epoch - original pipeline (short025):
- zero-advantage group ratio: 69.856%
- KL median: 0.0413
- KL p95: 1.132
- KL p99: 36.800
- KL max: 1.047e8

GRPO 2.0 epoch:
- zero-advantage group ratio: 69.277%
- KL median: 0.0411
- KL p95: 2.117
- KL p99: 173.002
- KL max: 1.761e11

Observed:
- High zero-advantage ratio appears from early training.
- KL distribution is strongly heavy-tailed.
- Extreme KL was not primarily caused by singleton/forced trie branches.
- Re-normalizing probabilities over legal Semantic-ID tokens did not remove KL outliers.
- These findings are diagnostic observations, not algorithmic fixes.

## Data / reward pipeline fixes

- Replaced prompt-text reward binding with stable sample_id -> target binding.
- Fixed unstable seq:{local_idx} IDs after DataFrame sampling.
- Added train/valid split identity into sequential sample IDs.
- Removed train/eval sample-ID collisions.
- Fixed reward binding when subset datasets are converted to Python lists.
- Verified fixed GRPO train set:
  - seq_rec: 10000
  - title/description: 6516
  - seqtitle: 1000
  - total: 17516
- Verified:
  - train bound IDs: 17516
  - missing IDs: 0
  - train/eval overlap: 0

## Main conclusion

The upstream ranking-GRPO setup did not outperform the clean SFT baseline on overall Top-K recommendation quality.

The strongest engineering result is the training-pipeline optimization:
repeated full-set evaluation/checkpoint overhead was removed, reducing 0.25-epoch GRPO wall-clock time from ~102 min to ~27 min while approximately preserving the original GRPO evaluation metrics. This is a single-seed engineering comparison; disabling stochastic during-training evaluation may change RNG consumption, so identical optimization trajectories are not assumed.
