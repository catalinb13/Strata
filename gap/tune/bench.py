#!/usr/bin/env python3
"""gap/tune/bench.py - one CLI arm of the UD-IQ4_XS tuning campaign (2026-11-13).

A single engine invocation, measured the way the engine itself reports it
(decode tok/s from the "decode N tokens in T ms" line, spec from the
"speculation" line, window cost from "stream windows T=N", footprint from the
"resident footprint after the fill" line).  No server restart, no session
pollution: the CLI reads its own throwaway conversation.

Usage:
    python3 gap/tune/bench.py <label> [--tokens FILE] [--max-new N]
                              [--ctx N] [--reps N]
                              [--env K=V ...] [--arg --flag value ...]
                              [--bin PATH]

"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/Users/pc/src/Strata-fork"
PACK = "/Users/pc/src/Strata-data/packs/ud-iq4_xs-huihui"
NATIVE = "/Users/pc/src/Strata-data/metal/ud-iq4_xs-huihui-metal.gguf"
PLE = "/Users/pc/models/Qwen3.8-Flash-Next-UD-IQ4_XS-abliterated/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf"
PROFILE = f"{ROOT}/data/expert-profile.bin"
MTP = "/Users/pc/src/Strata-data/mtp/rt"
BIN = f"{ROOT}/build-metal/strata"
OUT = f"{ROOT}/gap/tune/results.jsonl"
PAGE = 16384

# the live config's env block (2026-11-13): the baseline every arm is compared to
BASE_ENV = {
    "STRATA_METAL_STREAM": "1",
    "STRATA_METAL_PROMPT_CHUNK": "4096",
    "STRATA_METAL_PROMPT_MOE": "mpp",
    "STRATA_METAL_GDN_SCAN": "scan_v",
    "STRATA_METAL_ARENA_GB": "32",
    "STRATA_METAL_LEAVE_SHARE": "0.55",
    "STRATA_METAL_CUT_EARLY": "0",
    "STRATA_METAL_NFCHECK": "1",
    "STRATA_NAN_CKPT_RESUME": "1",
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_METAL_GDNMAX": "0",
}
# the live config's args (2026-11-13)
BASE_ARGS = [
    "--expert-cache", "2048", "--prefill", "auto", "--mtp", MTP,
    "--spec", "2", "--max-context", "204800", "--lookup-max-t", "8",
    "--suffix-draft", "1", "--mmap-experts",
]

RE = {
    "decode": re.compile(r"decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s"),
    "prefill": re.compile(r"prefill\s+(\d+) tokens in (\d+) chunks, ([\d.]+) ms \(([\d.]+) tok/s\)"),
    "spec": re.compile(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) \(([\d.]+)\), ([\d.]+) tokens per round"),
    "suffix": re.compile(r"suffix drafts\s+(\d+) windows, drafts accepted (\d+) of (\d+)"),
    "win": re.compile(r"stream windows T=(\d+)\s+(\d+) \(([^)]+)\): ([\d.]+) ms each \(GPU[^)]*([\d.]+) ms\)"),
    "footprint": re.compile(r"resident footprint after the fill: the process holds ([\d.]+) GB \(([\d.]+) GB resident\), planned ([\d.]+) GB, cap ([\d.]+) GB"),
    "arena": re.compile(r"expert arena ([\d.]+) GB, (\d+) slots"),
    "ec": re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB"),
    "overcap": re.compile(r"OVER THE CAP"),
    "err": re.compile(r"(#879|nonfinite|Traceback|engine stopped|no function|not found|abort|SIGSEGV)"),
    "mtp": re.compile(r"mtp\s+([\d.]+) ms/round"),
    "tiers": re.compile(r"expert tiers\s+decode: RAM (\d+) blobs, files (\d+) blobs, ([\d.]+) MB read"),
}


def memsnap():
    free = spec = wired = 0.0
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Pages free" in line:
            free = float(line.split()[-1].strip("."))
        elif "Pages speculative" in line:
            spec = float(line.split()[-1].strip("."))
        elif "wired down" in line:
            wired = float(line.split()[-1].strip("."))
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return {"free_gb": round((free + spec) * PAGE / 2**30, 2),
            "wired_gb": round(wired * PAGE / 2**30, 2), "swap": swap}


def run_once(label, env, args, tokens, max_new, ctx, binp, unset=()):
    cmd = [binp, "--pack", PACK, "--native", NATIVE, "--ple-gguf", PLE,
           "--native-dense-gguf", NATIVE, "--expert-profile", PROFILE,
           "--tokens-file", tokens, "--max-new", str(max_new)]
    if "--max-context" not in args:
        cmd += ["--max-context", str(ctx)]
    cmd += args
    e = dict(os.environ)
    e.update(env)
    for k in unset:
        e.pop(k, None)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=1800)
    wall = time.time() - t0
    body = r.stdout + r.stderr
    rec = {"label": label, "rc": r.returncode, "wall_s": round(wall, 1), "cmd": " ".join(cmd)}

    def last(pat, key, cast=float, idx=0):
        m = None
        for m in RE[pat].finditer(body):
            pass
        if m:
            rec[key] = cast(m.group(idx + 1))

    last("decode", "decode_tok", float, 2)
    last("decode", "decode_n", int, 0)
    last("decode", "decode_ms", float, 1)
    last("prefill", "prefill_tok", float, 3)
    last("prefill", "prefill_n", int, 0)
    last("prefill", "prefill_ms", float, 2)
    last("spec", "spec_rounds", int, 0)
    last("spec", "spec_accept_frac", float, 4)
    last("spec", "spec_tpr", float, 5)
    last("suffix", "suffix_windows", int, 0)
    last("suffix", "suffix_accept", int, 1)
    last("suffix", "suffix_total", int, 2)
    last("footprint", "holds_gb", float, 0)
    last("footprint", "resident_gb", float, 1)
    last("footprint", "cap_gb", float, 3)
    last("arena", "arena_gb", float, 0)
    last("arena", "arena_slots", int, 1)
    last("ec", "ec_slots", int, 0)
    last("ec", "ec_gib", float, 1)
    last("mtp", "mtp_ms", float, 0)
    if RE["overcap"].search(body):
        rec["over_cap"] = True
    e = RE["err"].findall(body)
    if e:
        rec["errors"] = list(dict.fromkeys(e))[:4]
    # windows: keep the T=1 and T=2 means
    for m in RE["win"].finditer(body):
        t = int(m.group(1))
        rec[f"win_T{t}_ms"] = float(m.group(4))
        rec[f"win_T{t}_gpu_ms"] = float(m.group(5))
        rec[f"win_T{t}_n"] = int(m.group(2))
    return rec, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--tokens", default=f"{ROOT}/gap/tokens/t2048.txt")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--unset", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--no-base-env", action="store_true")
    ap.add_argument("--no-base-args", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    env = {} if a.no_base_env else dict(BASE_ENV)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    for k in a.unset:
        env.pop(k, None)
    args = [] if a.no_base_args else list(BASE_ARGS)
    args += a.arg

    recs = []
    for rep in range(a.reps):
        rec, body = run_once(a.label, env, args, a.tokens, a.max_new, a.ctx, a.bin, a.unset)
        rec["rep"] = rep
        rec["mem"] = memsnap()
        recs.append(rec)
        if not a.quiet:
            print(json.dumps(rec), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if not a.quiet:
        # human summary
        r = recs[-1]
        print(f"\n== {a.label} ==")
        print(f"  decode   {r.get('decode_tok','?')} tok/s over {r.get('decode_n','?')} tok")
        print(f"  prefill  {r.get('prefill_tok','?')} tok/s over {r.get('prefill_n','?')} tok")
        print(f"  spec     {r.get('spec_accept_frac','?')} accept, {r.get('spec_tpr','?')} tok/round")
        if "win_T2_ms" in r:
            print(f"  win T2   {r['win_T2_ms']:.2f} ms (GPU {r.get('win_T2_gpu_ms','?')}), {r.get('win_T2_n','?')} windows")
        if "win_T1_ms" in r:
            print(f"  win T1   {r['win_T1_ms']:.2f} ms")
        print(f"  memory   holds {r.get('holds_gb','?')} GB, resident {r.get('resident_gb','?')} GB, cap {r.get('cap_gb','?')} GB"
              + ("  OVER CAP" if r.get("over_cap") else ""))
        print(f"  free now {r['mem']['free_gb']} GB, swap {r['mem']['swap']}")


if __name__ == "__main__":
    main()