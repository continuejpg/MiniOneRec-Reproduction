#!/bin/bash
# =====================================================================
# grpo_baseline.sh -- fixed small-scale GRPO baseline.
# Run on the AutoDL SERVER from /root/autodl-tmp/code, AFTER:
#   1) frozen subset files exist under splits/
#   2) python patches/preflight_grpo.py passes
#
# Data (frozen, identical for every future run):
#   seq_rec       : 10,000  (splits/grpo_seq_10k.json)
#   title2sid+desc:  6,516  (full, fixed)
#   seqtitle2sid  :  1,000  (splits/grpo_seqtitle_1k.json)
#   -> 17,516 samples
#
# Reward is FIXED to upstream --reward_type ranking (= rule_reward + ndcg_rule_reward,
# equal weight 1.0 each). Not to be changed in this phase.
# =====================================================================
set -e
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

D=/root/autodl-tmp
cat=Industrial_and_Scientific
CKPT=$D/runs/industrial_sft/final_checkpoint
OUT=$D/runs/grpo_baseline
mkdir -p "$OUT"

echo "=== GRPO baseline (fixed subsets) ==="
date -Is | tee "$OUT/started_at.txt"
echo "policy/ref : $CKPT"
echo "output     : $OUT"
echo "disk before:"
df -h /root/autodl-tmp | tail -1 | tee -a "$OUT/started_at.txt"

nohup accelerate launch --num_processes 1 \
    rl.py \
    --model_path              "$CKPT" \
    --train_file              "$D/code/data/Amazon/train/${cat}_5_2016-10-2018-11.csv" \
    --eval_file               "$D/code/data/Amazon/valid/${cat}_5_2016-10-2018-11.csv" \
    --info_file               "$D/code/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    --category                "$cat" \
    --sid_index_path          "$D/code/data/Amazon/index/${cat}.index.json" \
    --item_meta_path          "$D/code/data/Amazon/index/${cat}.item.json" \
    --subset_dir              "$D/code/splits" \
    --subset_seq              grpo_seq_10k.json \
    --subset_seqtitle         grpo_seqtitle_1k.json \
    --output_dir              "$OUT" \
    --reward_type             ranking \
    --num_generations         16 \
    --beam_search             True \
    --train_batch_size        16 \
    --eval_batch_size         16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs        2 \
    --learning_rate           1e-5 \
    --beta                    1e-3 \
    --temperature             1.0 \
    --save_steps_frac         0.25 \
    --sample_train            False \
    --seqtitle_sample         10000 \
    --dynamic_sampling        False \
    --mask_all_zero           False \
    --sync_ref_model          True \
    --test_during_training    False \
    --add_gt                  False \
    --dapo                    False \
    --gspo                    False \
    > "$OUT/train.log" 2>&1 &

PID=$!
echo "PID: $PID" | tee -a "$OUT/started_at.txt"
echo "$PID" > "$OUT/pid.txt"
echo
echo "launched in background. monitor with:"
echo "  tail -f $OUT/train.log"
echo "  grep -oE \"'[a-z_/]*zero_adv[a-z_/]*': [0-9.eE+-]+|'reward': [0-9.eE+-]+|'reward_std': [0-9.eE+-]+|'kl': [0-9.eE+-]+|'peak_vram_gb': [0-9.eE+-]+\" $OUT/train.log | tail -20"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "to stop: kill \$(cat $OUT/pid.txt)"
