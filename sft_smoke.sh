#!/bin/bash
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

D=/root/autodl-tmp
cat=Industrial_and_Scientific

torchrun --nproc_per_node 1 \
    sft.py \
    --base_model       $D/models/Qwen2.5-0.5B \
    --train_file       $D/code/data/Amazon/train/${cat}_5_2016-10-2018-11.csv \
    --eval_file        $D/code/data/Amazon/valid/${cat}_5_2016-10-2018-11.csv \
    --output_dir       $D/runs/industrial_sft_smoke \
    --category         $cat \
    --sid_index_path   $D/code/data/Amazon/index/${cat}.index.json \
    --item_meta_path   $D/code/data/Amazon/index/${cat}.item.json \
    --sample           100 \
    --num_epochs       1 \
    --batch_size       8 \
    --micro_batch_size 1 \
    --learning_rate    3e-4 \
    --cutoff_len       512 \
    --seed             42 \
    --train_from_scratch False \
    --freeze_LLM       False \
    2>&1 | tee $D/runs/industrial_sft_smoke/train.log
