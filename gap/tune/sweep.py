#!/usr/bin/env python3
"""gap/tune/sweep.py - paired knob arms on the live UD-IQ4_XS config (2026-11-13).

Every arm runs the engine exactly the way the live server does (same env, same
args), with one knob changed.  Arms are grouped and each group is bracketed by a
BASE anchor so drift on the box cannot be mistaken for a win.

    python3 gap/tune/sweep.py <group> [--reps N] [--max-new N] [--tokens FILE]

Groups are the ARMS dict below.  Results append to gap/tune/results2.jsonl.
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402  (run_once / parsers / paths)

ROOT = bench.ROOT
MTP = bench.MTP
DF2 = "/Users/pc/src/Strata-data/dflash2/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack"
OUT2 = f"{ROOT}/gap/tune/results2.jsonl"

# The live server's env block, exactly (strata-ud-iq4_xs-huihui.json)
LIVE_ENV = {
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
# The live server's engine args, exactly (the parts a CLI run shares)
LIVE_ARGS = [
    "--expert-cache", "1024", "--prefill", "auto", "--mtp", MTP,
    "--spec", "2", "--max-context", "204800", "--lookup-max-t", "12",
    "--suffix-draft", "1", "--mmap-experts", "--kv", "fp16",
]


def A(env=None, unset=(), args=None, drop=(), tokens=None, max_new=None):
    """One arm: env overrides, env removals, extra args, args to strip."""
    return {"env": env or {}, "unset": list(unset), "args": args or [],
            "drop": list(drop), "tokens": tokens, "max_new": max_new}


def apply_args(base, arm):
    a = list(base)
    for d in arm["drop"]:
        i = 0
        while i < len(a):
            if a[i] == d:
                a.pop(i)
                if i < len(a) and not a[i].startswith("--"):
                    a.pop(i)
                continue
            i += 1
    return a + arm["args"]


GROUPS = {}


def group(name, *arms):
    GROUPS[name] = list(arms)


# ---------------------------------------------------------------- spec / drafter
group("spec",
      ("BASE", A()),
      ("spec3", A(args=["--spec", "3"])),
      ("spec4", A(args=["--spec", "4"])),
      ("spec1", A(args=["--spec", "1"])),
      ("mtp0", A(drop=["--mtp"])),
      ("suf0", A(drop=["--suffix-draft"])),
      ("lmt8", A(args=["--lookup-max-t", "8"])),
      ("lmt16", A(args=["--lookup-max-t", "16"])),
      ("base2", A()),
      )

# ------------------------------------------------- depth: window cost vs tokens
group("depth",
      ("BASE", A()),
      ("spec4", A(args=["--spec", "4"])),
      ("chain2", A(args=["--lookup-chain", "2"])),
      ("chain3", A(args=["--lookup-chain", "3"])),
      ("spec4c3", A(args=["--lookup-chain", "3", "--spec", "4"])),
      ("BASE2", A()),
      )

# ---------------------------------------------------- env knobs (cheap, one line each)
group("knobs",
      ("BASE", A()),
      ("f16", A(env={"STRATA_METAL_F16": "1"})),
      ("fastmath", A(env={"STRATA_METAL_MATH": "fast"})),
      ("nr2", A(env={"STRATA_METAL_GEMV_NR": "2"})),
      ("pref16", A(env={"STRATA_METAL_PREFETCH": "16"})),
      ("pref8", A(env={"STRATA_METAL_PREFETCH": "8"})),
      ("pref2", A(env={"STRATA_METAL_PREFETCH2": "6"})),
      ("ctab0", A(env={"STRATA_METAL_CTAB": "0"})),
      ("contig0", A(env={"STRATA_METAL_CONTIG": "0"})),
      ("nobar", A(env={"STRATA_METAL_NOBARRIER": "1"})),
      ("f16nr2", A(env={"STRATA_METAL_F16": "1", "STRATA_METAL_GEMV_NR": "2"})),
      ("BASE2", A()),
      )

# ------------------------------------------- prefetch / ctab focus (screen showed a window win)
group("pref",
      ("BASE", A()),
      ("pref0", A(env={"STRATA_METAL_PREFETCH": "0"})),
      ("pref4", A(env={"STRATA_METAL_PREFETCH": "4"})),
      ("pref8", A(env={"STRATA_METAL_PREFETCH": "8"})),
      ("pref8mru0", A(env={"STRATA_METAL_PREFETCH": "8", "STRATA_METAL_PREFETCH_MRU": "0"})),
      ("pref8ctab0", A(env={"STRATA_METAL_PREFETCH": "8", "STRATA_METAL_CTAB": "0"})),
      ("ctab0", A(env={"STRATA_METAL_CTAB": "0"})),
      ("BASE2", A()),
      )

# ------------------------------------------------- KV encoding at real context depth
group("kv",
      ("BASEv", A()),
      ("int8", A(args=["--kv", "int8"])),
      ("kvq4", A(args=["--kv", "q4_0"])),
      ("BASEv2", A()),
      )

group("longctx",
      ("fp16_16k", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=96)),
      ("int8_16k", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=96, args=["--kv", "int8"])),
      ("fp16b_16k", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=96)),
      ("int8b_16k", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=96, args=["--kv", "int8"])),
      )

# ------------------------------- the tile re-sweep itself: new picks vs the old ones
group("tiles",
      ("NEW", A()),
      ("OLD", A(env={"STRATA_METAL_M1TILES": "0"})),
      ("NEW2", A()),
      ("OLD2", A(env={"STRATA_METAL_M1TILES": "0"})),
      )

# ------------------------------------------------ expert kernel tiles (top class: moe_gu)
group("moe",
      ("BASE", A()),
      ("gu16", A(env={"STRATA_METAL_MOE_GU": "16,1,4"})),
      ("gu8", A(env={"STRATA_METAL_MOE_GU": "8,1,4"})),
      ("gu32x2", A(env={"STRATA_METAL_MOE_GU": "32,2,4"})),
      ("dn8", A(env={"STRATA_METAL_MOE_DN": "8,1,4"})),
      ("BASE2", A()),
      )

# ------------------------------------------------------------ the final combination
group("final",
      ("BASE", A()),
      ("pref8", A(env={"STRATA_METAL_PREFETCH": "8"})),
      ("ctab0", A(env={"STRATA_METAL_CTAB": "0"})),
      ("p8c0", A(env={"STRATA_METAL_PREFETCH": "8", "STRATA_METAL_CTAB": "0"})),
      ("ar34", A(env={"STRATA_METAL_ARENA_GB": "34"})),
      ("combo34", A(env={"STRATA_METAL_ARENA_GB": "34", "STRATA_METAL_PREFETCH": "8", "STRATA_METAL_CTAB": "0"})),
      ("BASE2", A()),
      )

group("df2x",
      ("MTPa", A()),
      ("DF2a", A(args=["--draft", "dflash2", "--draft-model", DF2, "--draft-fallback", "none", "--spec", "4", "--draft-conf", "0.7"])),
      ("MTPb", A()),
      ("DF2b", A(args=["--draft", "dflash2", "--draft-model", DF2, "--draft-fallback", "none", "--spec", "4", "--draft-conf", "0.7"])),
      ("MTPc", A()),
      ("DF2c", A(args=["--draft", "dflash2", "--draft-model", DF2, "--draft-fallback", "none", "--spec", "4", "--draft-conf", "0.7"])),
      )

group("f16b",
      ("BASE", A()),
      ("F16", A(env={"STRATA_METAL_F16": "1"})),
      ("BASE2", A()),
      ("F16b", A(env={"STRATA_METAL_F16": "1"})),
      ("F16_s3", A(env={"STRATA_METAL_F16": "1"}, args=["--spec", "3"])),
      )

group("prefill",
      ("base16k", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=32)),
      ("pipe1", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=32, args=["--prefill-pipe", "1"])),
      ("base16kb", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=32)),
      ("pipe1b", A(tokens=f"{ROOT}/gap/tokens/t16384.txt", max_new=32, args=["--prefill-pipe", "1"])),
      )

# ---------------------------------------------------------------- memory shape
group("mem",
      ("BASE", A()),
      ("ar33", A(env={"STRATA_METAL_ARENA_GB": "33"})),
      ("ar34", A(env={"STRATA_METAL_ARENA_GB": "34"})),
      ("ar35", A(env={"STRATA_METAL_ARENA_GB": "35"})),
      ("ar36", A(env={"STRATA_METAL_ARENA_GB": "36"})),
      ("ar36ec512", A(env={"STRATA_METAL_ARENA_GB": "36"}, args=["--expert-cache", "512"])),
      ("BASE2", A()),
      )

# ---------------------------------------------------------------- kernels / env
group("kern",
      ("BASE", A()),
      ("cut_ev", A(env={"STRATA_METAL_CUT": "event"})),
      ("cut_off", A(env={"STRATA_METAL_CUT": "0"})),
      ("nobar", A(env={"STRATA_METAL_NOBARRIER": "1"})),
      ("gemv8", A(env={"STRATA_METAL_GEMV_NR": "8"})),
      ("gemv2", A(env={"STRATA_METAL_GEMV_NR": "2"})),
      ("gdn1", A(env={"STRATA_METAL_GDN_SCAN": "scan_1"})),
      ("gdn2", A(env={"STRATA_METAL_GDN_SCAN": "scan_2"})),
      ("fuse0", A(env={"STRATA_METAL_FUSE": "0"})),
      ("base2", A()),
      )

# ---------------------------------------------------------------- small extras
group("extra",
      ("BASE", A()),
      ("mtp_ovl", A(env={"STRATA_METAL_MTP_OVERLAP": "1"})),
      ("mtp_chain", A(env={"STRATA_METAL_MTP_CHAIN": "1"})),
      ("io8", A(env={"STRATA_METAL_IO_THREADS": "8"})),
      ("spin", A(env={"STRATA_METAL_SPIN": "1"})),
      ("hopnorm", A(args=["--mtp-hnorm", "stream"])),
      ("q4head", A(args=["--mtp-q4", "head"])),
      ("chained", A(args=["--lookup-chain", "2"])),
      ("base2", A()),
      )


# ---------------------------------------------------------------- DF2 drafter
group("df2",
      ("BASE", A()),
      ("df2", A(args=["--draft", "dflash2", "--draft-model", DF2, "--draft-fallback", "none", "--spec", "6", "--draft-conf", "0.7"])),
      ("df2s4", A(args=["--draft", "dflash2", "--draft-model", DF2, "--draft-fallback", "none", "--spec", "4", "--draft-conf", "0.7"])),
      ("hopnorm", A(args=["--mtp-hnorm", "stream"])),
      ("q4head", A(args=["--mtp-q4", "head"])),
      ("BASE2", A()),
      )


def run(label, arm, reps, tokens, max_new):
    # one engine at a time: two 32 GB arenas plus two 65 GB maps sent the box into swap (2026-11-13)
    busy = subprocess.run(["pgrep", "-f", "build-metal/strata --pack"], capture_output=True, text=True).stdout.split()
    if busy:
        raise SystemExit(f"another engine is running (pid {', '.join(busy)}): stop it first")
    env = dict(LIVE_ENV)
    env.update(arm["env"])
    for k in arm["unset"]:
        env.pop(k, None)
    args = apply_args(LIVE_ARGS, arm)
    recs = []
    for rep in range(reps):
        rec, body = bench.run_once(label, env, args,
                                   arm["tokens"] or tokens,
                                   arm["max_new"] or max_new, 204800, bench.BIN)
        rec["rep"] = rep
        rec["group"] = CUR_GROUP
        rec["mem"] = bench.memsnap()
        recs.append(rec)
        with open(OUT2, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(f"{ROOT}/gap/tune/prof/{CUR_GROUP}_{label}_{rep}.log", "w") as f:
            f.write(body)
    return recs


def med(key, recs):
    xs = [r[key] for r in recs if r.get(key) is not None]
    return statistics.median(xs) if xs else None


def main():
    global CUR_GROUP
    if len(sys.argv) < 2:
        print("groups:", ", ".join(GROUPS))
        return
    CUR_GROUP = sys.argv[1]
    reps, tokens, max_new = 2, f"{ROOT}/gap/tokens/t2048.txt", 192
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--reps":
            reps = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--max-new":
            max_new = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--tokens":
            tokens = sys.argv[i + 1]; i += 2
        else:
            raise SystemExit(f"unknown arg {sys.argv[i]}")
    os.makedirs(f"{ROOT}/gap/tune/prof", exist_ok=True)
    anchors = {}
    print(f"{'arm':14s} {'dec':>6s} {'pre':>7s} {'acc':>6s} {'tpr':>5s} {'win2':>6s} {'resGB':>6s} {'note'}", flush=True)
    for label, arm in GROUPS[CUR_GROUP]:
        t0 = time.time()
        recs = run(label, arm, reps, tokens, max_new)
        d, pre = med("decode_tok", recs), med("prefill_tok", recs)
        acc, tpr = med("spec_accept_frac", recs), med("spec_tpr", recs)
        win2, res = med("win_T2_ms", recs), med("resident_gb", recs)
        note = ""
        errs = [r["errors"] for r in recs if r.get("errors")]
        if errs:
            note = str(errs[0])[:60]
        if label.startswith("BASE") and label not in anchors:
            anchors[label] = d
        ref = anchors.get("BASE") or d
        if d and ref and label != "BASE":
            note = f"{(d / ref - 1) * 100:+.1f}% vs BASE " + note
        print(f"{label:14s} {d or float('nan'):6.2f} {pre or float('nan'):7.1f} "
              f"{acc or float('nan'):6.3f} {tpr or float('nan'):5.2f} "
              f"{win2 or float('nan'):6.2f} {res or float('nan'):6.2f}  {note}  [{time.time() - t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
