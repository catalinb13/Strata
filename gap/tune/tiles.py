#!/usr/bin/env python3
"""gap/tune/tiles.py - best GEMV tile per hot dense shape at decode depth, this machine.

gemv_parity --sweep only tries tiles with the base's L, so the L=8/4/2 space stays
unexplored.  --one tries every instantiated plain tile and prints each one (a
"NOT BITWISE" suffix marks the ones that change the answer).  This runs it for the
model's actual shapes and prints the winner against what Window::gemv_tile picks.

    python3 gap/tune/tiles.py [--ntok N] [--top K]
"""
import re
import subprocess
import sys

BIN = "/Users/pc/src/Strata-fork/build-metal/gemv_parity"
# role, type, n_in, n_out, calls/window, today's tile (kprof)
SHAPES = [
    ("attn_qkv   ", "Q6_K", 2560, 10240, 35, "16,2,4,1"),
    ("attn_gate  ", "Q6_K", 2560, 6144, 35, "16,2,4,1"),
    ("attn_q     ", "Q6_K", 2560, 12288, 12, "16,2,4,1"),
    ("attn_output", "Q6_K", 6144, 2560, 12, "32,2,4,2"),
    ("ssm_out    ", "Q8_0", 6144, 2560, 36, "32,2,4,2"),
    ("lm_head    ", "Q6_K", 2560, 248320, 1, "16,2,4,1"),
    ("shexp_gu   ", "Q6_K", 2560, 640, 94, "32,2,2,1"),
    ("shexp_dn   ", "Q8_0", 640, 2560, 48, "8,1,4,1"),
    ("attn_v/k   ", "Q6_K", 2560, 512, 24, "32,2,2,1"),
    ("router     ", "F32", 2560, 512, 96, "32,1,4,1"),
]

ntok, top = 2, 3
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--ntok":
        ntok = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--top":
        top = int(sys.argv[i + 1]); i += 2
    else:
        raise SystemExit("unknown " + sys.argv[i])

ROW = re.compile(r"^\s+tile\s+([\d,]+):\s+([\d.]+) us\s+([\d.]+) GB/s(.*)$")
print(f"ntok {ntok}: best plain tile per shape (bitwise only); today = Window::gemv_tile's pick")
tot_today = tot_best = 0.0
for role, ty, nin, nout, calls, today in SHAPES:
    out = subprocess.run([BIN, "--one", ty, str(nin), str(nout), "--ntok", str(ntok), "-v"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        m = ROW.match(line)
        if m:
            rows.append((float(m.group(2)), float(m.group(3)), m.group(1), "NOT BITWISE" in m.group(4)))
    ok = [r for r in rows if not r[3]]
    ok.sort()
    if not ok:
        print(f"{role} {ty:7s} {nin:5d}->{nout:6d}: no bitwise row")
        continue
    best = ok[0]
    cur = [r for r in ok if r[2] == today]
    cms = cur[0][0] if cur else float("nan")
    if cms == cms:
        tot_today += cms * calls
        tot_best += best[0] * calls
    print(f"{role} {ty:7s} {nin:5d}->{nout:6d} x{calls:3d}: today {today:9s} {cms:7.2f}us | "
          f"best {best[2]:9s} {best[0]:7.2f}us {best[1]:6.1f}GB/s {(cms / best[0] - 1) * 100:+5.1f}%"
          f"  (next: " + ", ".join(f"{r[2]} {r[0]:.1f}" for r in ok[1:top]) + ")")
print(f"\nwindow weight: today {tot_today / 1000:.2f} ms vs best {tot_best / 1000:.2f} ms "
      f"({(tot_today / tot_best - 1) * 100:+.1f}%)")
