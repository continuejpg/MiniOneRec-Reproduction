#!/bin/bash
# SASRec configuration sweep -- find a fairly-trained baseline, not a straw man.
cd /root/autodl-tmp/code
PY=/root/miniconda3/bin/python
B=baselines/sasrec_baseline.py
L=/root/autodl-tmp/runs/sasrec_sweep.log
: > "$L"

run () {
  echo "" | tee -a "$L"
  echo "############################ $1 ############################" | tee -a "$L"
  shift
  $PY -u $B "$@" 2>/dev/null | grep -vE "^\s*$" | tail -32 | tee -a "$L"
}

run "A: hidden=64  heads=1  full-softmax  100ep"  --tag a_h64_full    --hidden 64  --heads 1 --objective full --epochs 100
run "B: hidden=64  heads=2  full-softmax  200ep"  --tag b_h64_full2   --hidden 64  --heads 2 --objective full --epochs 200
run "C: hidden=128 heads=2  full-softmax  200ep"  --tag c_h128_full   --hidden 128 --heads 2 --objective full --epochs 200
run "D: hidden=64  heads=1  neg=100       200ep"  --tag d_h64_neg100  --hidden 64  --heads 1 --objective neg --neg 100 --epochs 200

echo "" | tee -a "$L"
echo "############################ SWEEP DONE ############################" | tee -a "$L"
