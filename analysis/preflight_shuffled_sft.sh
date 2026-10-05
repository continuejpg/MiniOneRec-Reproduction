#!/bin/bash
# Static preflight for scripts/sft_shuffled_sid.sh -- NO GPU training is started.
S=/root/autodl-tmp/code/scripts/sft_shuffled_sid.sh
D=/root/autodl-tmp
cat=Industrial_and_Scientific
fail=0

echo "=============================================================================="
echo "STATIC PREFLIGHT: sft_shuffled_sid.sh   (no training is launched)"
echo "=============================================================================="

echo
echo "--- 1. shell syntax (bash -n) ---"
if bash -n "$S"; then echo "  PASS  syntax OK"; else echo "  FAIL  syntax error"; fail=1; fi

echo
echo "--- 2. line endings / shebang ---"
if grep -q $'\r' "$S"; then echo "  FAIL  CRLF present"; fail=1; else echo "  PASS  LF only"; fi
head -1 "$S" | grep -q '^#!/bin/bash' && echo "  PASS  shebang ok" || { echo "  FAIL shebang"; fail=1; }

echo
echo "--- 3. no CRLF, executable bit ---"
[ -x "$S" ] && echo "  PASS  executable" || { echo "  FAIL not executable"; fail=1; }

echo
echo "--- 4. every path the script references must exist ---"
for p in \
  $D/models/Qwen2.5-0.5B \
  $D/code/sft.py \
  $D/code/data/Amazon/index/${cat}.item.json \
  $D/code/analysis/shuffled_sid/${cat}.index.json \
  $D/code/analysis/shuffled_sid/train.csv \
  $D/code/analysis/shuffled_sid/valid.csv \
  $D/code/data/Amazon/index/${cat}.index.json
do
  if [ -e "$p" ]; then printf '  PASS  %s\n' "$p"; else printf '  FAIL  %s\n' "$p"; fail=1; fi
done

echo
echo "--- 5. shuffled data readable + row counts + column check ---"
/root/miniconda3/bin/python - <<'PY' || fail=1
import csv, json, os, sys
SH = "/root/autodl-tmp/code/analysis/shuffled_sid"
CAT = "Industrial_and_Scientific"
INDEX_NAME = f"{CAT}.index.json"
ok = True

for name, expect in [("train.csv", 36259), ("valid.csv", 4532)]:
    with open(f"{SH}/{name}", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = {"user_id", "history_item_sid", "item_id", "item_sid",
            "history_item_id", "item_title", "history_item_title"}
    have = set(rows[0].keys())
    missing = need - have
    status = "PASS" if (len(rows) == expect and not missing) else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"  {status}  {name}: rows={len(rows)} (expect {expect}) "
          f"missing_cols={sorted(missing) if missing else 'none'}")

ipath = f"{SH}/{INDEX_NAME}"
if not os.path.exists(ipath):
    print(f"  FAIL  index file missing: {ipath}")
    sys.exit(1)
idx = json.load(open(ipath, encoding="utf-8"))
print(f"  PASS  {INDEX_NAME}: items={len(idx)} "
      f"tokens={len({t for v in idx.values() for t in v})}")

# reproduce what TokenExtender will compute from --sid_index_path
rebuilt = os.path.join(os.path.dirname(ipath),
                       os.path.basename(ipath).split(".")[0] + ".index.json")
if os.path.abspath(rebuilt) != os.path.abspath(ipath):
    print(f"  FAIL  TokenExtender would open {rebuilt}, not {ipath}")
    ok = False
else:
    print(f"  PASS  TokenExtender resolved path == --sid_index_path ({os.path.basename(ipath)})")

# cross-check: item_sid in the CSV matches the index for a sample of rows
with open(f"{SH}/train.csv", encoding="utf-8") as f:
    r = list(csv.DictReader(f))
bad = sum(1 for x in r if x["item_sid"] != "".join(idx[x["item_id"]]))
print(f"  {'PASS' if bad == 0 else 'FAIL'}  train.csv item_sid consistent with index "
      f"({bad} mismatches over {len(r)} rows)")
if bad:
    ok = False
sys.exit(0 if ok else 1)
PY

echo
echo "--- 6. output_dir must not collide with an existing clean run ---"
OUT=$D/runs/industrial_sft_shuffled_sid
if [ -e "$OUT/final_checkpoint/model.safetensors" ]; then
  echo "  FAIL  $OUT already holds a finished run"; fail=1
else
  echo "  PASS  $OUT is free"
fi
if [ -e "$D/runs/industrial_sft/final_checkpoint/model.safetensors" ]; then
  echo "  PASS  clean run intact: $D/runs/industrial_sft"
else
  echo "  FAIL  clean run missing"; fail=1
fi

echo
echo "--- 7. the only differing arguments vs sft_full.sh (after shell expansion) ---"
/root/miniconda3/bin/python - <<'PY'
import re
D = "/root/autodl-tmp"
cat = "Industrial_and_Scientific"
# variables defined by each script
ENV = {
    "$D": D, "${cat}": cat, "${D}": D,
    "$SHUF_DIR": f"{D}/code/analysis/shuffled_sid",
    "$SHUF_INDEX": f"{D}/code/analysis/shuffled_sid/{cat}.index.json",
    "$SHUF_TRAIN": f"{D}/code/analysis/shuffled_sid/train.csv",
    "$SHUF_VALID": f"{D}/code/analysis/shuffled_sid/valid.csv",
    "$ITEM_META": f"{D}/code/data/Amazon/index/{cat}.item.json",
    "$OUT": f"{D}/runs/industrial_sft_shuffled_sid",
    "$CLEAN_RUN": f"{D}/runs/industrial_sft",
}

def expand(v):
    for k in sorted(ENV, key=len, reverse=True):
        v = v.replace(k, ENV[k])
    return v

def args(p):
    t = open(p, encoding="utf-8").read()
    # only the argument block inside the torchrun invocation
    m = re.search(r"torchrun.*?(?=\n\s*2>&1|\Z)", t, re.S)
    body = m.group(0) if m else t
    return {k: expand(v) for k, v in re.findall(r"--([a-z_]+)\s+(\S+)", body)}

a = args(f"{D}/code/sft_full.sh")
b = args(f"{D}/code/scripts/sft_shuffled_sid.sh")
print(f"  {'argument':20s} {'same?':6s} clean")
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "(absent)"), b.get(k, "(absent)")
    tag = "same" if va == vb else "DIFF"
    print(f"  {k:20s} {tag:6s} {va}")
    if tag == "DIFF":
        print(f"  {'':20s} {'->':6s} {vb}")

diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
expected = ["eval_file", "output_dir", "sid_index_path", "train_file"]
print(f"\n  differing arguments : {diff}")
print(f"  expected exactly    : {expected}")
print(f"  VERDICT: {'MATCH -- only the 4 intended arguments differ' if diff == expected else 'MISMATCH'}")
PY

echo
echo "--- 8. confirm no training process is running ---"
if pgrep -af "sft.py|torchrun" | grep -v pgrep > /dev/null; then
  echo "  WARNING  an sft.py/torchrun process is already running:"
  pgrep -af "sft.py|torchrun" | grep -v pgrep
else
  echo "  PASS  no sft.py / torchrun process running"
fi

echo
echo "--- 9. GPU state (informational only; nothing is launched) ---"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

echo
echo "=============================================================================="
if [ "$fail" -eq 0 ]; then echo "PREFLIGHT: ALL PASS"; else echo "PREFLIGHT: FAILURES PRESENT"; fi
echo "=============================================================================="
exit $fail
