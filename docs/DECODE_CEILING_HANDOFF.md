# Handoff: where the decode and prefill time goes, and what is worth doing next

**Status: the analysis was read-only; the answers landed 2026-11-13.** §2 and §7 now carry the
measured class split for this pack, T1's two flags are refuted with numbers, T2 is resized to ~2%,
T3.1 was implemented (commit `ea40e4f5`: window 45.55 → 40.95 ms GPU, bitwise), T3.2 is refuted twice
and T3.3 is measured (spec stays 2). Everything still marked `[INFERENCE]` has not been measured.
The arms and their logs are in `gap/tune/` (`sweep.py`, `tiles.py`, `CAMPAIGN-2026-11-13.md`).

Audience: whoever picks up tuning on the 64 GB M1 Max next. Read §1, then §2, then pick a tier in §3.
§4 exists so that nobody re-runs the experiments that already failed, and §6 exists so that two
agents do not fight over one GPU.

### Two flags before anything else

1. **The config and the prose disagree — and the config is the truth here.** The live
   `strata-ud-iq4_xs-huihui.json` says arena **32 GB** / `LEAVE_SHARE=0.35` / `--expert-cache 1024`;
   `docs/DETAILS.md` describes this box as arena **38 GB** / `0.55` / `4096`. **Resolved 2026-11-13**:
   the running engine's startup line reads `planned resident footprint 46.6 GB = arena 32.0 + 14.6 held
   before it + 0.5 reserved for the n-gram table; cap 44.7 GB = … the RAM 68.7 GB less 24.1 GB left to
   macOS` — `24.1 = 0.35 × 68.7`, so the live box is **arena 32 / 0.35 / expert-cache 1024** and the
   `DETAILS.md` prose is the stale side. Every memory number below is scaled off arena 32.
2. **The per-class split in §2 was measured on the old Q4_K pack, not on this IQ4_XS pack.** The
   ordering of T1 and T3 rests on it. T0 is not a formality: if the split moved, the tiers reorder.
   Run the KPROF first (§5 R1) and correct §2 before sizing any work.

---

## 1. Current state (verify this first; the previous session may have moved it)

| | |
|---|---|
| Box | M1 Max, 64 GB, macOS 27. Single-tenant: one engine at a time, no other GPU work during timed arms |
| Engine | `build-metal/strata` from this fork (`--backend metal`); AGENTS.md says engine 0.1.59 — confirm against the server's startup line |
| Pack | `Strata-data/metal/ud-iq4_xs-huihui-metal.gguf`, 64.88 GB, 1223 tensors (llama.cpp-legal, 16 KiB aligned); PLE/n-gram shard `Strata-data/metal/.../00004-of-00004.gguf` (28.8 GB, stays on disk) |
| Run config | `strata-ud-iq4_xs-huihui.json` — `--expert-cache 1024`, `--spec 2`, `--lookup-max-t 12`, `--suffix-draft 1`, `--kv fp16`, `--conversation-cache-mib 8192` (6 slots, min 2048 tokens), `--mmap-experts`, `--max-context 204800` |
| Env | `STRATA_METAL_STREAM=1`, `PROMPT_CHUNK=4096`, `PROMPT_MOE=mpp`, `GDN_SCAN=scan_v`, `ARENA_GB=32`, `LEAVE_SHARE=0.35`, `CUT_EARLY=0`, `NFCHECK=1`, `GDNMAX=0`, `ROUTE_TAIL_SKIP=0`, `NAN_CKPT_RESUME=1` |
| Serving | `:8180`, self-hosted; `metal/tools/serve_huihui.sh {status\|stop\|start\|run CMD}` |

**Discrepancy to resolve before any A/B:** `docs/DETAILS.md` describes this box as *arena 38 GB,
`LEAVE_SHARE=0.55`, `--expert-cache 4096`*, while the live config file says *arena 32, 0.35,
expert-cache 1024*. One of the two is stale. Read the startup line, do not trust the prose.

Baseline to reproduce before changing anything (all measured on this box, this pack):

| Quantity | Value | Source |
|---|---|---|
| Steady decode, 12–18K depth | 27–30 tok/s (≈33.5 ms/token) | DETAILS.md, 2026-11-11 A/B |
| Decode at 111K depth | 24.7 tok/s | DETAILS.md |
| Cold prefill, chunk 4096, instruments off | 330–374 tok/s | DETAILS.md |
| Reuse / parking (the felt TTFT) | parked restore 8–10 ms; 108,665 tokens save 1.78 GB, restore 645 ms; `prompt 137995 = 137833 reused + 162 read in 3520 ms` | DETAILS.md, MAC.md |
| Process footprint | ≈50 GB of 64 GB, swap steady ≈2 GB | DETAILS.md |
| DRAM read ceiling | **358 GB/s** | `rt_bench bw`, 2026-10-28 |
| Packed-format decode ceiling | **~250–259 Gval/s** (IQ4_XS 259, Q4_K 259; IQ2/IQ3 table formats 150–260) | `expert_parity` benches, 2026-10-28 |
| Decode miss cost | **0.8 ms per miss**; 8.6 misses/window with read-ahead 12 (was 15.9) | gap/results/1018-* |
| Cut handshake (device flag) | window T=2 87.8 → 56.0 ms; decode 19.5–20.3 → 29.75–30.49 tok/s | gap/results/1018-1545 |

## 2. Where the time goes

Two ceilings and two charge models: **decode** pays dequant per (weight × token) and is ALU-bound;
**prefill** pays it per (weight × tile) and is matrix-unit-bound. The pack's format mix decides which
one binds, which is why the same pack can be well chosen for one phase and wrong for the other.

### Decode

Per token this pack reads **5.31 GB** — dense/output 4.13 GB (Q6_K 2.34, Q8_0 1.44, F32/BF16 0.35)
plus routed experts 1.18 GB (48 layers × 10 of 512 experts). That is **~6.8 G weight-values**
(4.3 dense + 2.47 experts). Derived from the file's own tensor table and `native_experts.txt`.

| Class | Bytes/token | Values/token | Rate now | Ceiling | Time |
|---|---:|---:|---:|---:|---:|
| Dense + output | 4.13 GB | ~4.3 G | 180 / 200 / 243 GB/s (qkv / hc / output) | 358 GB/s | ~15–17 ms |
| Routed experts | 1.18 GB | 2.47 G | ~103 GB/s = 194 Gval/s | 358 GB/s / 259 Gval/s | ~12 ms |
| Attention at depth, router, GDN, dispatch, cuts | — | — | ssm_out 65, attn_gate 69, router 37 GB/s | 150–230 by shape | ~4 ms |
| Decode misses | — | — | 8.6/window × 0.8 ms | — | ~6.9 ms |
| **Total** | **5.31 GB** | **6.8 G** | | **byte floor 14.8 ms → 67 tok/s** | **33.5 ms** |

**FLAGGED PARAGRAPH — ANSWERED 2026-11-13.** T0 was run: the split below was measured with
`STRATA_METAL_KPROF=1` on **this** IQ4_XS pack, and it moved a lot. The table that follows the
old Q4_K numbers is the one to use; the old one is kept only to show what changed.

### Measured split, this pack, 2026-11-13 (use this one)

Conditions: 2048-token prompt, 2-token MTP window, fp16 K/V, arena 32, `--expert-cache 1024`,
after the dense-GEMV tile re-sweep (commit `ea40e4f5`). Window GPU **40.95 ms** for 2 rows.

| Class | ms | share | rate now | DRAM ceiling | bytes/window |
|---|---:|---:|---|---:|---:|
| experts: gate+up (`moe_gu`) | 8.18 | 20.0% | **~295 GB/s** | 358 | 2.41 GB |
| experts: down (`moe_down`) | 3.55 | 8.7% | **~304 GB/s** | 358 | 1.08 GB |
| dense: attn_qkv / gate / q (Q6_K) | 7.24 | 17.7% | 203-224 GB/s | 358 | 1.52 GB |
| lm head (Q6_K 2560x248320) | 2.57 | 6.3% | 203 GB/s | 358 | 0.52 GB |
| ssm_out (Q8_0 6144x2560) | 2.39 | 5.8% | 237 GB/s | 358 | 0.57 GB |
| hyper-connection gr_down (x2) | 3.38 | 8.3% | **95 GB/s** | 358 | 0.32 GB |
| hyper-connection gr_up (x2) | 2.24 | 5.5% | **143 GB/s** | 358 | 0.32 GB |
| gdn_step (72 dispatches, elementwise) | 2.24 | 5.5% | — | — | — |
| router (`ffn_gate_inp` x95) | 1.88 | 4.6% | 127 GB/s | 358 | 0.24 GB |
| shared expert gate/up/down | 1.80 | 4.4% | 105-150 GB/s | 358 | 0.32 GB |
| qsa (attn/merge/topk/score) | 1.64 | 4.0% | — | — | — |
| **decode misses** | **0** | **0%** | `expert tiers decode: 0 blobs, 0.0 MB read` | — | 0 |

Two corrections to the old table, both material:

- **The routed experts are already near the byte floor (~300 GB/s of 358), not at 103 GB/s.** The old
  row and every plan sized off it (T2's "+8-12%", the "0.8 ms/miss" term) overstate the room. Repacking
  the Q5_0 down matrices to IQ4_XS cuts down-matrix bytes by 23%, i.e. ~0.25 GB of a 7.1 GB window:
  **~2% decode**, not 8-12%, because the down class is bandwidth-bound at 304 GB/s, not ALU-bound.
- **There are no decode misses on this box** (`--mmap-experts`, arena 32, `--expert-cache 1024`: the
  tier line reports 0 file reads). The 6.9 ms miss term in the old table is spent, not lost — it is
  what turned into the 3.4 ms of hyper-connection reads and the 1.9 ms router that the old split
  did not name.

The whole window moves **7.1 GB in 40.95 ms = 174 GB/s**, 49% of the 358 GB/s DRAM ceiling. Per class
the gap is: dense Q6_K ~200 vs 358, hyper-connection reads ~114 vs 358, experts ~300 vs 358. Those are
the three numbers that bound any further kernel work.

**The one structural fact to hold onto:** the 2-token window is *byte*-bound at ~50% of the DRAM
ceiling, not ALU-bound — so the mechanism that pays is fewer bytes per token, and the mechanism that
does not pay (measured twice, see §3 T1) is the fp16 unpack that was supposed to cut ALU.

### Prefill, the same treatment

Prefill is not decode with bigger batches. Every weight is read once per 16-token tile, so dequant is
charged per (weight × tile) and the matrix units are the ceiling — which is the opposite of decode,
where dequant is charged per (weight × token).

| | |
|---|---|
| Matrix units, `mpp` path | **6.2 TFLOP/s** = 3.1 T MAC/s (the K-loop path is 2.9 TFLOP/s) |
| Values per token | 6.8 G |
| Implied ceiling | 6.8 G ÷ 3.1 T ≈ 2.2 ms/token ≈ **456 tok/s** if every value ran on the matrix units `[INFERENCE, from the measured 6.2 TFLOP/s]` |
| Measured | **330–374 tok/s** cold, chunk 4096 — 72–82% of that |
| Per-chunk GPU time | a 4096-token chunk: 12243 → 8818 ms with `mpp` (−28%); 8818 ms for 4096 tokens is 464 tok/s of pure GPU time |
| Format sensitivity | near zero: the tile path measures 6.3–7.4 T "whatever the type" (strata-mlx, M4 Max). **Choose the pack for decode.** |

Where the missing 20% goes: building the fp16 slabs (each batch of experts decoded once per chunk),
sorting the (token, expert) pairs, attention, and the dense GEMVs. The tile shape is already at its
measured optimum — 16-token tiles, 64-column dequant; 32-token tiles and 128 columns each measured
~20% slower, and 256 columns produced wrong results.

The felt-latency half: a 10K system prompt costs ~27 s cold at 374 tok/s, but a *reused* one costs the
tail only — measured `prompt 137995 = 137833 reused + 162 read in 3520 ms`, parked restore 8–10 ms,
108,665 tokens restored from disk in 645 ms. For agent traffic the reuse path is the prefill lever
that matters; raw tok/s is the one that matters on the first turn of the day.

## 3. Plan, in the order that pays

### T0 — measure, no code (~30 min, do this first)

`KPROF` on this pack, T=1 and the spec-2 window, warmed conversation. Expected dense ~15 ms, experts
~12 ms, misses ~7 ms, misc ~4 ms. Any class that comes back far from that changes the ranking below.

### T1 — config only, today's engine

**Measured 2026-11-13 — both rows of the original plan are refuted on this box:**

| Change | Plan said | Measured | Verdict |
|---|---|---|---|
| `STRATA_METAL_F16=1` | +25-30% decode (window 56-60 → 40-41 ms) | window **57.8 vs 58.0 ms** (two arms each, before and after the tile re-sweep: 62.6/64.8 vs 56.1/59.8 old tiles, 58.2/57.4 vs 56.1/59.8 new tiles); tok/s parity | **rejected** — the fp16 unpack is neutral on Q6_K/Q8_0 and costs acceptance drift for nothing. The 40-41 ms window the plan remembered is what the *tile re-sweep* produced (T3.1), not F16 |
| `--kv int8` | no decode cost, gives back 2.5 GB | at a 16K-token prompt: window **63.7 vs 58.8 ms** (int8 8% slower), prefill 15% faster, 2 GB less resident | **rejected** — fp16 K/V is both exact and faster at depth; the "no decode cost" reading came from shallow prompts |
| arena / `--expert-cache` re-check | 32 vs 38 | arena 34 = window 55.5 vs 57.7 ms (**-3.9%**) at **+2 GB** resident; 28/30 materially slower (earlier campaign) | **not taken** — 4% is not worth 2 GB of browser headroom; it is the one-line change left for a user who wants it |

### T2 — the pack (DONE 2026-11-13: worth ~2%, not 8-12%)

**19 GB of the file — the down matrices of 33 of 48 layers — is Q5_0**, a split-plane 5-bit format
that is worse per value than IQ4_XS on both axes: 0.69 vs 0.53 B/val and `[INFERENCE]` ~180 vs 259
Gval/s. Repack those to IQ4_XS. While in the repacker, bake in the aligned split layout that
mlx-serve uses (all payloads first, scale heads after, rows padded to 16 bytes) so the decoders issue
`uint4` loads instead of byte gathers.

**Resized after the measured split:** the down class runs at ~304 GB/s of the 358 GB/s ceiling, so it
is byte-bound, not ALU-bound — the repack's whole gain is the 23% fewer bytes, of a class that is
8.7% of the window: **about +2% decode, not +8-12%**, plus 8 GB of file and ~15% more experts an arena
GB. No requantiser exists in this repo (no `repack`/`requantize` in `strata --help`, no tool under
`tools/`), so it is a new tool plus a 57 GB file plus `tools/quant_quality.py`, for 2%.

- Keep Q6_K dense exactly as it is: it sits near its byte floor, and Q8_0 dense is a wash (+30%
  bytes for +22% rate) `[INFERENCE from mlx-serve's per-type table]`.
- Gate: `tools/quant_quality.py` (28 prompts, ~6,000 positions; the repo's own bar is mean KL and
  top-1 against BF16), plus `tools/needle_bench.py`. Keep the current pack and its config for A/B.

### T3 — kernels

1. **DONE 2026-11-13 (commit `ea40e4f5`).** The dense loser tiles were the M2-Max picks: a per-shape
   sweep at 2 tokens (`gemv_parity --one`, all lanes-per-row, every pick bitwise) moved the window
   **45.55 → 40.95 ms GPU (-10.1%)**: `ssm_out` 80.6 → 63.7 us (+27%), `attn_output` 78.3 → 67.6
   (+16%), lm head 2284 → 2091 (+9%), `attn_q` +7%, `attn_gate` +6%, `attn_qkv` +1%, shared down +3%.
   The wide decoders are *slower* than plain on this GPU (101.8 vs 100.9 us on attn_qkv), so the picks
   run plain. Bitwise verified per shape; `force_check` event rule PASS, rate rule inconclusive at
   1429 tokens. What is left of this class: Q6_K at ~200-235 GB/s against 358, which is a decoder
   rewrite, not a tile.
2. **half2 unpack across the dense GEMVs — refuted, twice** (`STRATA_METAL_F16=1`, window 57.8 vs
   58.0 ms both before and after T3.1). It does not flatten the window's marginal row cost either:
   `--spec 3` under F16 measured **22.75 tok/s against 28.4** (window T3 not even produced).
3. **Re-measured the window slope, re-tested depth: no change.** `--spec 3` 82.2 ms / 2.17 tokens,
   `--spec 4` 99.2 ms / 2.61 tokens, against `--spec 2` 57.7 ms / 1.81 tokens — 37.9 and 38.0 ms per
   token against 31.9. `--lookup-chain 2/3` (= window 4/6 with high-precision drafts) measured
   27.8/28.3 against 28.3: it never fires on prose. **Spec stays 2.**
4. **Hyper-connection reads: 95 GB/s (gr_down) and 143 (gr_up), 13.8% of the window.** Two cheap
   retiles lost (16 simdgroups over 2 rows: 3.79 vs 3.38 ms; a 4-way split: 3.22 vs 3.39 ms and it
   changes the sum order). The remaining idea is a load-time transpose of the 10240x320 up matrix so
   the column read becomes a row read, plus a `uint4`-wide Q8_0 decode — a kernel + pack-layout change,
   worth at most 3.3 ms (8%).

### T4 — prefill (a separate campaign; ceiling ~456 tok/s, today 330–374)

Already banked, do not re-litigate without anchors: `mpp` (−28% chunk GPU time), chunk 4096 (+8.5% at
the cost of ~13% on the first ~100 decode tokens while the arena re-warms), 16×64 tiles, read-ahead.
`PROMPT_EXPERTS=v3` and `ROUTE=1` are M5-only / wrong on this chip — do not enable them.

What is left, in order:

| Change | Expected | Gate | Risk |
|---|---|---|---|
| `PROMPT_MOE_BATCH=128` | fewer slab rebuilds per chunk; buffers 5.1 → 5.8 GB | 4096- and 8192-token chunks, cold and warm | borrows more arena during a prompt, so the decode re-warm penalty grows |
| Persist the fp16 slab for experts that stay resident across chunks (a "resident slab" tier) | the slab is rebuilt every chunk today. mlx-serve amortises a bank at 3.5 ms per 270 M weights and beats the tile kernel above 24 (token, expert) pairs `[INFERENCE for our shapes]` | chunk N vs chunk N+1 on the same conversation, GPU time only | slab memory competes with the arena; opt-in and explicitly sized |
| Wider `mpp` coverage (the dense thirds, attention) | end-to-end is ~80% of the matmul phase, so at most ~1.2× | per-class timing with `STRATA_METAL_PROFILE=1` | the exact prompt path must stay reachable behind a flag |
| Length-gate `mpp` | today it *loses* on short chunks (356 tokens: 60.0 → 55.8 tok/s) | 27–400-token prompts, TTFT only | none, if gated by length |

Do not chase raw prefill tok/s alone. Measure both numbers and report both: tokens/s on a cold 4096
prompt, and **time to first token on a 27-token and a 10K-token prompt**, warm and cold. On this box
the reuse path already beats any plausible tok/s gain for a conversation that continues.

### T5 — one UX item

For agent traffic (10K-token system prompts), first-token latency is worth more than the fifth tok/s.
strata-mlx's `--keep-awake` (a small GPU addition per second, arrays wired, for N seconds after a
request) took short-prompt TTFT from 618–835 ms to 148–215 ms on an M4 Max. This engine has no
equivalent, and its own docs note the first ~100 decode tokens after a cold prompt pay ~13% while
the arena re-warms. A keep-warm timer is a small addition with a large felt benefit.

## 4. Ruled out, with the arithmetic

- **Switching to strata-mlx (or any MLX engine).** Same GPU, same two ceilings. The only public
  head-to-head on equal conditions (48 GB M4 Pro, same GGUF, MTP off both, cache reuse eliminated)
  put Metal 5–6% *ahead* on prefill by 2× and 5–6% behind on decode; MLX's visible decode lead there
  was entirely its drafter, which we also have. Also: strata-mlx already supports IQ4_XS
  (`strata_mlx/blocks.py`), but not Q5_0 — 19 GB of our file — and it reads GGUF tensors, so a switch
  means a Q5_0 decoder, loader work and a ~60 GB re-download (our original shards 1–3 are gone).
- **fp16 hot-expert tier for decode** (`docs/DECODE_2X_PLAN.md` C2/C3). 2 B/weight at 358 GB/s is
  179 Gval/s against the 194 we already get from packed experts: flat or worse. It pays for *prefill*,
  where `mpp` already does it.
- **Q8_0 experts.** −5.4 ms of arithmetic against ~+7 ms of extra miss stalls: doubling expert bytes
  halves arena slots and roughly doubles the 8.6 misses/window that cost 0.8 ms each.
- **Deeper speculation before T3.** Measured losers on this box and on the M5 Max.
- **Sub-4-bit packs** and **re-quantising to MLX-native affine 4-bit**: no ALU saving, and the second
  breaks parity for nothing.

## 5. Recipes

- **R1 — the class split (do this before anything in §3).** `STRATA_METAL_KPROF=1` (per-class
  serialized GPU time, printed at the 20th occurrence) on this pack, T=1 and the spec-2 window, warmed
  conversation; also `STRATA_METAL_PROFILE=1`, `STRATA_METAL_TRACE=1`. Expected dense ~15 ms, experts
  ~12 ms, misses ~7 ms, misc ~4 ms. Write the real numbers into §2 and correct the tiers if they moved.
- **R2 — the prefill split.** `STRATA_METAL_PROFILE=1` on a 4096-token cold prompt: chunk GPU time,
  slab build, pair sort, attention, dense. Plus TTFT on a 27-token and a 10K-token prompt, warm and
  cold, and `tools/needle_bench.py` for correctness at depth after any prompt-path change. Always say
  whether the page cache was warm — a 1,537-token prompt read at "273–287 tok/s" once turned out to be
  the page cache answering, not the disk (DETAILS.md).
- Benches: `build-metal/expert_parity --gu Q4_K --dn Q5_1 --bench [--sweep] [--gucfg L,R,S] [--dncfg L,R,S] [--layers N]`;
  `build-metal/gemv_parity --one ...`; `build-metal/rt_bench {bw|scatter|dispatch}`.
  **The bench must rotate through ~400 MB of weights** — a benchmark that reuses one hot weight lies
  (mlx-serve's point, and it matches our live-vs-bench 2× anomaly on the down kernel).
- DRAM: `mactop --headless --format json --count 20 -i 1000` → `soc_metrics.dram_bw_combined_gbs`
  (powermetrics lost its DRAM samplers on macOS 27).
- Quality: `tools/quant_quality.py`, `tools/needle_bench.py`, `tools/force_check.py`.
- Live A/B drive: `/tmp/bw/drive.py "prompt" max_tokens` (`/tmp/bw` is ephemeral — move anything still
  needed into the repo). Anchor brackets: alternate the control arm with every variant arm; this box
  slows under sustained load, so cross-arm comparisons from different sessions are worthless.
- Build: `cd build-metal && ninja strata` (~11 s). **MSL kernel edits compile at engine start, so a
  kernel change needs a server restart, not a rebuild.**
- No GNU `timeout` on this Mac. Scripts run `set -u`.

## 6. Rules of engagement

- One engine at a time. Two engines (parking and a serve window, or this fork and strata-mlx) cause a
  swap storm and the token-0 stall abort. `metal/tools/serve_huihui.sh stop` first.
- Leave the box with the server **started and health-checked** (`/v1/models` loaded, one live chat
  probe). An experiment window that ends with the server down is a bug in the script.
- Warm boot ~4 s to loaded, ~11 s to first answer with the page cache hot. Never time a cold arm
  against a warm one.
- Repo rules: every new path opt-in, **the default path stays byte-identical to the last release**,
  say how it was checked; PR titles prefixed `[Engine] …`; **never push the fork branch without the
  user's permission.**
- Do not `git checkout --` `window.cpp`.

## 7. Open questions for the next session

**Answered 2026-11-13 (the tuning session that landed commit `ea40e4f5`; tables in
`gap/tune/CAMPAIGN-2026-11-13.md`, arms reproduce with `python3 gap/tune/sweep.py <group>`):**

1. KPROF on the IQ4_XS pack (R1): **done — the split moved a lot and is written into §2.** Experts
   ~300 GB/s (near the ceiling), dense Q6_K ~200-235, hyper-connection reads 95-143, no decode misses.
2. Arena 32/0.35 or 38/0.55: **32 / 0.35, confirmed from the startup line** (§1 flag 1).
3. With `STRATA_METAL_F16=1`, does a verify row get cheaper? **No: F16 is window-neutral (57.8 vs
   58.0 ms) and `--spec 3` under it measures 22.75 tok/s against 28.4.** The 50 tok/s plateau is a
   byte wall at ~50% of the DRAM ceiling, not a step waiting for depth.
4. Does Q5_0 → IQ4_XS clear quality? **Not run — the value does not justify it at ~2% decode** (T2),
   given that no requantiser exists here and the pack is 57 GB to produce.
5. At 111K depth, how much is attention and how much arena churn? **Open.** Cheapest next run on this
   box: KPROF at 111K with `--kv fp16`, one arm, ~2 min.
6. Prefill split (R2): **open.** The machinery is there (`STRATA_METAL_PROFILE=1`), nobody has run it
   since the tile change.

**Still open and worth doing, in order:** (a) the hyper-connection transpose (§3 T3.4, ≤8%), (b) a
Q6_K dense decoder above 235 GB/s (§2's largest remaining byte class), (c) the keep-warm timer (§3 T5,
felt TTFT, no decode risk), (d) Q5 at 111K depth.

## Appendix A — this pack, by the numbers

Parsed from `ud-iq4_xs-huihui-metal.gguf` (1223 tensors) and `packs/ud-iq4_xs-huihui/native_experts.txt`:

| Type | Bytes | Where |
|---|---:|---|
| Q5_0 | 19.03 GB | down matrices, 33 of 48 layers |
| IQ3_XXS | 17.98 GB | gate/up, 28 layers |
| IQ4_XS | 15.49 GB | gate/up, 17 layers |
| IQ4_NL | 6.13 GB | down, 13 layers |
| Q6_K | 2.34 GB | dense |
| IQ2_S | 1.61 GB | gate/up, 3 layers |
| Q8_0 | 1.44 GB | dense |
| Q2_0 | 0.47 GB | down, 2 layers |
| F32 / BF16 | 0.35 GB | dense |

Experts total 60.38 GB; per-expert blob 2.17–2.87 MB; arena at 32 GB holds roughly 4.8k of 24.6k
expert slots at `--expert-cache 1024`.

## Appendix B — the two MLX projects, and what they are good for

Kept here so the next session does not have to re-do the reading.

- **`ddalcu/mlx-serve-gguf`** (MIT, Zig + MLX, experimental) — GGUF run as-is on Apple GPUs, 20 block
  formats, per-type Metal decoders that are bit-exact against ggml, and the metal source is plain text
  designed to be reused outside MLX. Useful as a **kernel donor and a per-type rate yardstick**: Q6_K
  and Q8_0 sit at their byte-count floors while 4-bit types do not; low-bit types are compute-bound;
  multi-row instances decode each weight once and dot it against up to 5 rows at 0.65× the cost per
  row; a materialised bf16 bank beats the tile kernel above 24 (token, expert) pairs. It does **not**
  support this model's architecture, so it is not an engine option.
- **`yibie/strata-mlx`** (MIT, Python + MLX) — the only MLX engine that runs Qwen3.8-Flash-Next from
  Strata's GGUF files. Its measured numbers are on an M4 Max, 128 GB, whole model resident: 68–111
  tok/s on Q2_0 with its drafter, 44–67 without, 570 tok/s prefill; a 4-token window at 1.45× a
  single-token step; per-layer prompt reading with read-ahead. Its kernel set covers IQ4_XS but not
  Q5_0. Its ideas that are portable and worth reading: per-row-multiplexed matvec, the 16-entry LUT
  for 2-bit formats (3 ops/weight against ~8), keep-awake for TTFT, and the buffer-pool cap (MLX's
  pool grows unbounded otherwise and took a 128 GB box to swap).
