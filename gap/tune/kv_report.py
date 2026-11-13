#!/usr/bin/env python3
"""gap/tune/kv_report.py - paired int8-vs-fp16 comparison from gap/tune/kv_ab/*.json."""
import glob
import json
import os
from statistics import median

OUTDIR = "/Users/pc/src/Strata-fork/gap/tune/kv_ab"


def load():
    d = {}
    for p in glob.glob(f"{OUTDIR}/*.json"):
        r = json.load(open(p))
        lab = r.get("label", os.path.basename(p)[:-5])
        if "__" not in lab:
            continue
        knob, kv = lab.rsplit("__", 1)
        d.setdefault(knob, {})[kv] = r
    return d


def med(rs, key):
    vals = [r[key] for r in rs if key in r]
    return round(median(vals), 2) if vals else None


def main():
    d = load()
    order = ["BASE", "spec3", "lmt12", "lmt16", "sfind", "chain2",
             "cut_ev", "gdn_scan1", "gdn_scan4", "moe_tile", "gemv_nr", "fuse0",
             "readsp", "io8", "no_nfchk", "no_nanck", "no_tail"]
    knobs = [k for k in order if k in d] + [k for k in d if k not in order]
    print(f"{'knob':<10} {'int8':>7} {'fp16':>7} {'d%':>6} {'acc8':>6} {'acc16':>6} "
          f"{'T2_8':>6} {'T2_16':>6} {'res8':>6} {'res16':>6}")
    for k in knobs:
        a = d[k].get("int8", {})
        b = d[k].get("fp16", {})
        da, db = med([a], "decode_tok"), med([b], "decode_tok")
        dp = f"{100*(db-da)/da:+.1f}" if da and db else "-"
        print(f"{k:<10} {da!s:>7} {db!s:>7} {dp:>6} "
              f"{med([a],'spec_accept_frac')!s:>6} {med([b],'spec_accept_frac')!s:>6} "
              f"{med([a],'win_T2_ms')!s:>6} {med([b],'win_T2_ms')!s:>6} "
              f"{med([a],'resident_gb')!s:>6} {med([b],'resident_gb')!s:>6}")


if __name__ == "__main__":
    main()