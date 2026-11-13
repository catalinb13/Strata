#!/usr/bin/env python3
"""gap/tune/campaign.py - the UD-IQ4_XS tuning campaign driver (2026-11-13).

Runs a list of arms back to back through gap/tune/bench.py, each arm a fresh
engine invocation, and writes one JSON line per arm to gap/tune/results.jsonl.
The BASE anchor is interleaved around each group so drift is visible.

Each arm is ~60 s (30 s arena read + decode), so a whole family of 10 arms is
~10 minutes.  Short by design: only the promising leads get a longer sample.

Arm tuple: (label, env dict, unset tuple, extra args, use_base_args, max_new)
  use_base_args=False -> the arm supplies the whole --arg list itself (DFlash2).

Usage:
    python3 gap/tune/campaign.py spec      # the named campaign
    python3 gap/tune/campaign.py --list
"""
import os
import subprocess
import sys
import time

ROOT = "/Users/pc/src/Strata-fork"
BENCH = f"{ROOT}/gap/tune/bench.py"
PY = f"{ROOT}/.venv/bin/python"
DF2 = "/Users/pc/src/Strata-data/dflash2/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack"
MTP = "/Users/pc/src/Strata-data/mtp/rt"

# the common native-pack arg set DFlash2 arms build on
NATIVE_ARGS = ["--expert-cache", "2048", "--prefill", "auto", "--spec", "2",
               "--lookup-max-t", "8", "--suffix-draft", "1", "--mmap-experts"]

CAMPAIGNS = {
    # Speculative decoding: the cheapest wins if acceptance holds.
    "spec": [
        ("BASE",      {}, (), [], True, 192),
        ("spec4",     {}, (), ["--spec", "4"], True, 192),
        ("spec6",     {}, (), ["--spec", "6"], True, 192),
        ("spec8",     {}, (), ["--spec", "8"], True, 192),
        ("lmt12",     {}, (), ["--lookup-max-t", "12"], True, 192),
        ("lmt16",     {}, (), ["--lookup-max-t", "16"], True, 192),
        ("spec4_l12", {}, (), ["--spec", "4", "--lookup-max-t", "12"], True, 192),
        ("sfind",     {}, (), ["--spec-min-p", "0.6"], True, 192),
        ("chain",     {}, (), ["--lookup-chain", "2"], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
    # Memory: the browser-headroom constraint.
    "mem": [
        ("BASE",      {}, (), [], True, 192),
        ("ar28",      {"STRATA_METAL_ARENA_GB": "28"}, (), [], True, 192),
        ("ar30",      {"STRATA_METAL_ARENA_GB": "30"}, (), [], True, 192),
        ("ar36",      {"STRATA_METAL_ARENA_GB": "36"}, (), [], True, 192),
        ("ar40",      {"STRATA_METAL_ARENA_GB": "40"}, (), [], True, 192),
        ("ec3072",    {}, (), ["--expert-cache", "3072"], True, 192),
        ("ec4096",    {}, (), ["--expert-cache", "4096"], True, 192),
        ("ec1024",    {}, (), ["--expert-cache", "1024"], True, 192),
        ("ls070",     {"STRATA_METAL_LEAVE_SHARE": "0.70"}, (), [], True, 192),
        ("nowire",    {"STRATA_METAL_WIRE_ARENA": "0"}, (), [], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
    # Kernel knobs that are byte-identical or near-identical.
    "kern": [
        ("BASE",      {}, (), [], True, 192),
        ("cut_ev",    {"STRATA_METAL_CUT": "event"}, (), [], True, 192),
        ("gdn_scan1", {"STRATA_METAL_GDN_SCAN": "scan_1"}, (), [], True, 192),
        ("gdn_scan2", {"STRATA_METAL_GDN_SCAN": "scan_2"}, (), [], True, 192),
        ("gdn_scan4", {"STRATA_METAL_GDN_SCAN": "scan_4"}, (), [], True, 192),
        ("moe_tile",  {"STRATA_METAL_MOE_TILE": "8,8,8,8"}, (), [], True, 192),
        ("gemv_nr",   {"STRATA_METAL_GEMV_NR": "4"}, (), [], True, 192),
        ("fuse0",     {"STRATA_METAL_FUSE": "0"}, (), [], True, 192),
        ("readsp",    {"STRATA_METAL_READ_SPLIT": "2"}, (), [], True, 192),
        ("io8",       {"STRATA_METAL_IO_THREADS": "8"}, (), [], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
    # Safety/diagnostic envs that may cost decode.
    "safe": [
        ("BASE",      {}, (), [], True, 192),
        ("no_nfchk",  {}, ("STRATA_METAL_NFCHECK",), [], True, 192),
        ("no_nanck",  {}, ("STRATA_NAN_CKPT_RESUME",), [], True, 192),
        ("no_tail",   {}, ("STRATA_ROUTE_TAIL_SKIP",), [], True, 192),
        ("gdnmax1",   {"STRATA_METAL_GDNMAX": "1"}, (), [], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
    # DFlash2 and MTP: the two drafter backends, plus the cheap MTP variants.
    "draft": [
        ("BASE",      {}, (), [], True, 192),
        ("df2",       {}, (), NATIVE_ARGS + ["--draft", "dflash2",
                                             "--draft-model", DF2,
                                             "--mtp", MTP,
                                             "--mtp-draft-vocab", MTP + "/draft_vocab.bin"], False, 192),
        ("df2_cap8",  {}, (), NATIVE_ARGS + ["--draft", "dflash2",
                                             "--draft-model", DF2,
                                             "--draft-cap", "8",
                                             "--draft-conf", "0.7",
                                             "--mtp", MTP], False, 192),
        ("mtp_q4",    {}, (), ["--mtp-q4", "proj"], True, 192),
        ("mtp_hnorm", {}, (), ["--mtp-hnorm", "stream"], True, 192),
        ("nopref",    {}, (), ["--no-prefill-cpu-share"], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
    # Context/KV: does max-context size cost decode, and do the KV encodings
    # change anything on Metal (the window MSL always stores int8)?
    "kv": [
        ("BASE",      {}, (), [], True, 192),
        ("kv_int8",   {}, (), ["--kv", "int8"], True, 192),
        ("kv_q4",     {}, (), ["--kv", "q4_0"], True, 192),
        ("kv_k8v4",   {}, (), ["--kv", "k8v4"], True, 192),
        ("kv_grow",   {}, (), ["--kv-grow"], True, 192),
        ("ctx8k",     {}, (), ["--max-context", "8192"], True, 192),
        ("ctx20k",    {}, (), ["--max-context", "20480"], True, 192),
        ("ctx80k",    {}, (), ["--max-context", "81920"], True, 192),
        ("BASE2",     {}, (), [], True, 192),
    ],
}


def run_arm(label, env_over, unset, extra, use_base, max_new, tokens="t2048"):
    cmd = [PY, BENCH, label, "--max-new", str(max_new), "--tokens",
           f"{ROOT}/gap/tokens/{tokens}.txt", "--quiet"]
    if not use_base:
        cmd.append("--no-base-args")
    for k, v in env_over.items():
        cmd += ["--env", f"{k}={v}"]
    for k in unset:
        cmd += ["--unset", k]
    for a in extra:
        cmd += [f"--arg={a}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode, (r.stdout + r.stderr).strip()[-200:]


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("campaigns:", ", ".join(CAMPAIGNS))
        return
    name = sys.argv[1]
    if name not in CAMPAIGNS:
        sys.exit(f"unknown campaign {name}; try --list")
    os.makedirs(f"{ROOT}/gap/tune/logs", exist_ok=True)
    arms = CAMPAIGNS[name]
    print(f"campaign {name}: {len(arms)} arms, ~{len(arms)} min", flush=True)
    for label, env_over, unset, extra, use_base, max_new in arms:
        t0 = time.time()
        rc, tail = run_arm(label, env_over, unset, extra, use_base, max_new)
        print(f"[{time.strftime('%H:%M:%S')}] {label:<12} rc={rc} {time.time()-t0:>5.0f}s {tail}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()