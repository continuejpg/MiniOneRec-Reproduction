#!/usr/bin/env python3
"""
audit_predictions.py -- read-only audit of every beam-20 prediction file.

For each candidate prediction file:
  - compute global HR@1/5/10/20 and NDCG@20 over all test rows using the SAME
    first-exact-SID-match rule as calc.py
  - print path, n, beam width
  - match against the known official metrics to identify which run it came from

Prints what is found. Trains nothing. Modifies no existing file.
"""
import ast
import csv
import glob
import json
import math
import os

ROOT = "/root/autodl-tmp"
RUNS = f"{ROOT}/runs"
CODE = f"{ROOT}/code"
TEST_CSV = f"{CODE}/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"

# Known official results (user-supplied ground truth)
KNOWN = [
    ("SFT                      ", 0.19832341, 0.11786798),
    ("GRPO 1.5ep               ", 0.16170307, 0.10470676),
    ("GRPO 2.0ep               ", 0.16236488, 0.10447064),
    ("GRPO original 0.25ep     ", 0.17626296, 0.10885475),
    ("GRPO fast 0.25ep         ", 0.17538054, 0.10824989),
]

TOPK = [1, 5, 10, 20]


def ncdcg(rank):
    return 1.0 / math.log2(rank + 2)


def metrics(preds, targets):
    hr = {k: 0 for k in TOPK}
    nd = 0.0
    for p, t in zip(preds, targets):
        r = -1
        for i, x in enumerate(p):
            if x == t:
                r = i
                break
        for k in TOPK:
            if 0 <= r < k:
                hr[k] += 1
        if 0 <= r < 20:
            nd += ncdcg(r)
    n = len(targets)
    return {f"HR@{k}": hr[k] / n for k in TOPK}, nd / n


def main():
    with open(TEST_CSV, encoding="utf-8") as f:
        test = list(csv.DictReader(f))
    targets = [r["item_sid"] for r in test]
    print(f"test rows = {len(test)}")

    # ---------------------------------------------------------------- discover
    pats = [
        f"{RUNS}/*/test_beam20.json",
        f"{RUNS}/*/eval_beam20.json",
        f"{RUNS}/*/*/test_beam20.json",
        f"{RUNS}/eval_*/final_result_*.json",
    ]
    found = []
    for p in pats:
        found += glob.glob(p)
    found = sorted(set(found))

    print(f"\ndiscovered {len(found)} candidate prediction files\n")
    rows = []
    for p in found:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {p}: {type(e).__name__}")
            continue
        if not isinstance(d, list) or not d or "predict" not in d[0]:
            print(f"  SKIP {p}: no 'predict' field")
            continue
        preds = [x["predict"] for x in d]
        n = len(preds)
        beam = len(preds[0]) if n else 0
        note = ""
        if n != len(test):
            note = f"!! n={n} != test {len(test)}"
        hr, nd = metrics(preds, targets)

        # closest known match
        best, bestd = None, 1e9
        for label, khr, knd in KNOWN:
            dist = abs(hr["HR@20"] - khr) + abs(nd - knd)
            if dist < bestd:
                best, bestd = label.strip(), dist
        match = f"{best}  (d={bestd:.2e})" if bestd < 5e-4 else f"(no match, closest {best} d={bestd:.2e})"

        rows.append((os.path.relpath(p, ROOT), n, beam,
                     hr["HR@1"], hr["HR@5"], hr["HR@10"], hr["HR@20"], nd, match, note))

    hdr = ["path", "n", "beam", "HR@1", "HR@5", "HR@10", "HR@20", "NDCG@20", "matches official", "note"]
    print("=" * 150)
    print("PER-FILE RECOMPUTED GLOBAL METRICS (all test samples, first-exact-SID rule)")
    print("=" * 150)
    for r in rows:
        print(f"\npath      : {r[0]}")
        print(f"n / beam  : {r[1]} / {r[2]}   {r[9]}")
        print(f"HR@1      : {r[3]*100:.5f}%")
        print(f"HR@5      : {r[4]*100:.5f}%")
        print(f"HR@10     : {r[5]*100:.5f}%")
        print(f"HR@20     : {r[6]*100:.5f}%   (as fraction {r[6]:.8f})")
        print(f"NDCG@20   : {r[7]*100:.5f}%   (as fraction {r[7]:.8f})")
        print(f"identity  : {r[8]}")

    print()
    print("=" * 150)
    print("OFFICIAL REFERENCE")
    print("=" * 150)
    for label, khr, knd in KNOWN:
        print(f"  {label} HR@20={khr:.8f}  NDCG@20={knd:.8f}")

    # ---------------------------------------------------------------- checkpoint inventory
    print()
    print("=" * 150)
    print("CHECKPOINT / ARTIFACT INVENTORY (to establish epoch identity)")
    print("=" * 150)
    for d in sorted(glob.glob(f"{RUNS}/*/")):
        name = os.path.basename(d.rstrip("/"))
        ck = sorted(glob.glob(f"{d}checkpoint-*"))
        has_final = os.path.isdir(f"{d}final_checkpoint")
        logs = glob.glob(f"{d}*.log")
        extra = [os.path.basename(x) for x in glob.glob(f"{d}*.json") if "tokenizer" not in x
                 and os.path.basename(x) not in ("config.json", "generation_config.json")]
        print(f"\n{name}/")
        print(f"   checkpoint-*      : {[os.path.basename(c) for c in ck] or '-'}")
        print(f"   final_checkpoint  : {has_final}")
        print(f"   logs              : {[os.path.basename(l) for l in logs] or '-'}")
        print(f"   json              : {extra or '-'}")


if __name__ == "__main__":
    main()
