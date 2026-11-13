#!/usr/bin/env python3
"""gap/tune/fp16_vs_int8.py - paired FP16-vs-int8 decode sweep (2026-11-13).

Each knob runs int8 then fp16 back to back on the same warm host, so the pair
is a clean A/B.  Arena (32) and context (204800) are held at the live config's
values by design -- only the KV knob and the KV-independent decode knobs move.

Each arm is written to gap/tune/kv_ab/<label>__<kv>.json the moment it lands,
so progress is visible without waiting on a buffer.

    python3 gap/tune/fp16_vs_int8.py spec|kern|safe|all
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/Users/pc/src/Strata-fork"
BENCH = f"{ROOT}/gap/tune/bench.py"
PY = f"{ROOT}/.venv/bin/python"
OUTDIR = f"{ROOT}/gap/tune/kv_ab"

# (label, env-over dict, extra args)
FAMILIES = {
    "spec": [
        ("BASE",   {}, []),
        ("spec3",  {}, ["--spec", "3"]),
        ("lmt12",  {}, ["--lookup-max-t", "12"]),
        ("lmt16",  {}, ["--lookup-max-t", "16"]),
        ("sfind",  {}, ["--spec-min-p", "0.6"]),
        ("chain2", {}, ["--lookup-chain", "2"]),
    ],
    "kern": [
        ("BASE",     {}, []),
        ("cut_ev",   {"STRATA_METAL_CUT": "event"}, []),
        ("gdn_scan1",{"STRATA_METAL_GDN_SCAN": "scan_1"}, []),
        ("gdn_scan4",{"STRATA_METAL_GDN_SCAN": "scan_4"}, []),
        ("moe_tile", {"STRATA_METAL_MOE_TILE": "8,8,8,8"}, []),
        ("gemv_nr",  {"STRATA_METAL_GEMV_NR": "4"}, []),
        ("fuse0",    {"STRATA_METAL_FUSE": "0"}, []),
        ("readsp",   {"STRATA_METAL_READ_SPLIT": "2"}, []),
        ("io8",      {"STRATA_METAL_IO_THREADS": "8"}, []),
    ],
    "safe": [
        ("BASE",     {}, []),
        ("no_nfchk", {"STRATA_METAL_NFCHECK": "0"}, []),
        ("no_nanck", {"STRATA_NAN_CKPT_RESUME": "0"}, []),
        ("no_tail",  {"STRATA_ROUTE_TAIL_SKIP": "1"}, []),
    ],
}


def run_one(label, kv, env_over, extra, max_new=192, tokens="t357"):
    out = f"{OUTDIR}/{label}__{kv}.json"
    if os.path.exists(out):
        rec = json.load(open(out))
        if "decode_tok" in rec:
            print(f"[{time.strftime('%H:%M:%S')}] {label:<10} {kv:<5} "
                  f"{rec['decode_tok']} tok/s (cached)", flush=True)
            return rec
    cmd = [PY, BENCH, f"{label}__{kv}", "--max-new", str(max_new), "--tokens",
           f"{ROOT}/gap/tokens/{tokens}.txt", "--arg=--kv", f"--arg={kv}"]
    for k, v in env_over.items():
        cmd += ["--env", f"{k}={v}"]
    for a in extra:
        cmd += [f"--arg={a}"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    dt = time.time() - t0
    rec = None
    for line in reversed((r.stdout + r.stderr).strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    if rec is None:
        rec = {"label": f"{label}__{kv}", "error": (r.stdout + r.stderr)[-400:]}
    rec["wall_s"] = round(dt, 1)
    json.dump(rec, open(out, "w"))
    if "decode_tok" in rec:
        print(f"[{time.strftime('%H:%M:%S')}] {label:<10} {kv:<5} "
              f"{rec['decode_tok']} tok/s  {dt:>5.0f}s", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {label:<10} {kv:<5} "
              f"FAIL rc={r.returncode} {dt:>5.0f}s :: {rec.get('error','')[:120]}", flush=True)
    return rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUTDIR, exist_ok=True)
    names = list(FAMILIES) if which == "all" else [which]
    for name in names:
        print(f"=== family {name} ===", flush=True)
        for label, env_over, extra in FAMILIES[name]:
            run_one(label, "int8", env_over, extra)
            run_one(label, "fp16", env_over, extra)
    print("done", flush=True)


if __name__ == "__main__":
    main()