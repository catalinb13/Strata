#!/usr/bin/env python3
"""gap/tune/report.py - summarize the tuning campaign's results.

Reads gap/tune/results.jsonl (one JSON per arm per rep) and prints, for every
label, the median decode tok/s, prefill tok/s, spec acceptance, window cost and
memory, with the BASE anchor highlighted so drift is visible.

    python3 gap/tune/report.py [label ...]
"""
import json
import sys
from statistics import median

PATH = "/Users/pc/src/Strata-fork/gap/tune/results.jsonl"
KEYS = ["decode_tok", "prefill_tok", "spec_accept_frac", "spec_tpr",
        "win_T2_ms", "win_T1_ms", "resident_gb", "cap_gb"]


def load():
    rows = []
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def agg(rows, label):
    rs = [r for r in rows if r["label"] == label]
    if not rs:
        return None
    out = {"n": len(rs)}
    for k in KEYS:
        vals = [r[k] for r in rs if k in r]
        if vals:
            out[k] = round(median(vals), 3)
    if any(r.get("over_cap") for r in rs):
        out["over_cap"] = True
    return out


def main():
    want = sys.argv[1:]
    rows = load()
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    if want:
        labels = [l for l in labels if l in want]
    print(f"{'label':<16} {'n':>2} {'decode':>7} {'prefill':>8} {'spec':>6} {'tpr':>5} "
          f"{'T2ms':>6} {'T1ms':>6} {'resid':>6} {'cap':>6}  flags")
    for l in labels:
        a = agg(rows, l)
        if not a:
            continue
        flags = []
        if a.get("over_cap"):
            flags.append("OVER-CAP")
        print(f"{l:<16} {a['n']:>2} {a.get('decode_tok','-'):>7} {a.get('prefill_tok','-'):>8} "
              f"{a.get('spec_accept_frac','-'):>6} {a.get('spec_tpr','-'):>5} "
              f"{a.get('win_T2_ms','-'):>6} {a.get('win_T1_ms','-'):>6} "
              f"{a.get('resident_gb','-'):>6} {a.get('cap_gb','-'):>6}  {' '.join(flags)}")


if __name__ == "__main__":
    main()