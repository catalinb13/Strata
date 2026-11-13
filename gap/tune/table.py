#!/usr/bin/env python3
"""gap/tune/table.py - medians per arm from results2.jsonl (window ms is the kernel metric,
tokens-per-round is the drafter metric; tok/s is both).  Usage: table.py [group ...]"""
import json
import statistics
import sys

GROUPS = sys.argv[1:]
rows = [json.loads(l) for l in open("/Users/pc/src/Strata-fork/gap/tune/results2.jsonl")]
seen = {}
for r in rows:
    g = r.get("group")
    if GROUPS and g not in GROUPS:
        continue
    seen.setdefault((g, r["label"]), []).append(r)


def m(rs, k):
    xs = [r[k] for r in rs if r.get(k) is not None]
    return statistics.median(xs) if xs else None


print(f"{'group':8s} {'arm':10s} {'n':>2s} {'dec':>6s} {'win_ms':>7s} {'T':>2s} {'tpr':>5s} {'acc':>6s} {'pre':>6s} {'resGB':>6s}")
for (g, lab), rs in seen.items():
    win = None
    t = None
    for tt in (2, 3, 4, 6, 8):
        v = m(rs, f"win_T{tt}_ms")
        if v:
            win, t = v, tt
            break
    d = m(rs, "decode_tok")
    print(f"{g:8s} {lab:10s} {len(rs):2d} {d or 0:6.2f} {win or 0:7.2f} {t or 0:2d} "
          f"{m(rs,'spec_tpr') or 0:5.2f} {m(rs,'spec_accept_frac') or 0:6.3f} {m(rs,'prefill_tok') or 0:6.1f} "
          f"{m(rs,'resident_gb') or 0:6.2f}")
