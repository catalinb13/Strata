#!/usr/bin/env python3
"""metal/tools/filler_state_probe.py - decide WHERE the token-0 filler poisoning enters the engine.

Forensics on the quarantined 2026-11-09 sessions (auto-session.corrupted-20261109{b,c}.quarantine)
found the same footprint in both: the LAST 10-11 of the 36 GDN recurrent states contain NaN, while
layers 0..24 are finite, ple is clean, and the checkpoint's TOKEN list is clean too (no filler run
inside CKPT0).  NaN in the recurrent state is permanent: every later decay multiplies it in, so the
argmax returns token 0 (the filler) forever until the window is reset.  Two live theories remain and
this harness separates them:

  content  - specific transcript content drove an activation overflow during prefill/generation;
  state    - the engine's discard paths poison the state: the serve's repeat-guard discard, a client
             stream abort mid-reply (what happens every time omp parses a tool call and cancels the
             rest of the stream), or the checkpoint fold/restore bookkeeping itself.

Arms (each starts its own serve from strata-ud_iq4_xs-huihui.json with a scratch session dir; the
shared auto-session and the real packs are never touched):

  guard    fresh engine with STRATA_NAN_INJECT=1 (no real poison): the engine must report the
           injected window, the server must retry once and the reply must still come out clean.
           Proves the #879 non-finite detection is wired into the Metal serve loop (it was not:
           Verifier::run's Metal branch returned before note_nonfinite, so a poisoned GDN state
           streamed the sampler's token-0 "no candidate" fallback forever - the filler).
  control  fresh engine, three clean short turns.  Must be ok - the sanity gate.
  cancel   fresh engine, force a repeat-guard discard ("repeat mariner...") and a client stream
           abort mid-reply, then three clean probe turns.  A CORRUPT probe proves the discard
           path poisons the state - no filler content ever entered the conversation.
  persist  start WITH the quarantined corrupted session restored (copied into the scratch dir as
           auto-session), three clean probes.  SHOWN INCONCLUSIVE (2026-11-09 run): a single-message
           probe never prefix-matches the 86K session, so the engine re-reads from token 0 and the
           poisoned state is recomputed away before it can serve anything.  Kept for the negative
           control it does provide; use poison below for the real end-to-end test.
  poison   build a conversation, let auto-session save it, rewrite the saved GDN arrays offline with
           NaN (the quarantine's forensic signature: last 10 of 36 states), restart and CONTINUE the
           same conversation.  The prefix now matches, the poisoned state enters the compute, and the
           #879 scan must flag the window and self-heal the request.  Real poison, real reuse path.
  replay   fresh engine, replay the real omp transcript turn by turn (same history as the live
           11:23 episode; corrupt generations were never sent back by the client).  Corruption
           at the same place proves the trigger is CONTENT (it reproduces from a cold state).
           2026-11-09 run: 51 asks / 62.6K tokens, all clean, zero #879 lines - content alone under
           ~62K is not the trigger; the quarantines broke at ~85K.

Run it with sushi stopped (the arms need ~54 GiB free), from the repo root:

    .venv/bin/python metal/tools/filler_state_probe.py --arms guard    # first: proves the fix
    .venv/bin/python metal/tools/filler_state_probe.py --arms cancel,persist
    .venv/bin/python metal/tools/filler_state_probe.py --arms replay --to 75

Scrub policy honored: filler runs are never printed; '!' inside a run shows as <F>.
Results: gap/filler-state-probe-<ts>/results.jsonl + one engine log per arm.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "/Users/pc/src/Strata-fork"
PY = os.path.join(REPO, ".venv/bin/python")
BASE_CFG = os.path.join(REPO, "strata-ud-iq4_xs-huihui.json")
QUARANTINE = "/Users/pc/src/Strata-data/sessions/ud-iq4_xs-huihui/auto-session.corrupted-20261109c.quarantine"
SESSION = ("/Users/pc/.omp/agent/sessions/-src-Strata-fork/"
           "2026-11-09T09-23-10-598Z_01a1bf9e-6286-7367-8803-7bcb3b31266f.jsonl")
FILLER = chr(33)
PROBE = "Ignore everything above and answer this single request only. Reply with exactly one word: ocean"


def clean_len(s: str) -> int:
    return len(re.sub(r"\s", "", (s or "").replace(FILLER, "")))


def scrub(s: str) -> str:
    s = re.sub(FILLER + "{4,}", lambda m: "<F>" + str(len(m.group(0))), s or "")
    return s.replace("\n", " ")


def port_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3):
            return True
    except Exception:
        return False


def free_bytes() -> int:
    """Reclaimable RAM: vm_stat's free + purgeable + speculative pages.  Pages free alone is
    misleading on macOS (it ignores the huge file-backed cache the system hands out).  Wired
    pages stay excluded: that is GPU-resident memory another model holds."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    total = 0
    for key in ("Pages free", "Pages purgeable", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            total += int(m.group(1))
    return total * page


def build_cfg(tag: str, outdir: str, port: int, extra_env: dict | None = None) -> str:
    cfg = copy.deepcopy(json.load(open(BASE_CFG)))
    cfg["log"] = os.path.join(outdir, f"{tag}_engine.log")
    cfg["slot_save_path"] = os.path.join(outdir, f"sessions_{tag}")
    os.makedirs(cfg["slot_save_path"], exist_ok=True)
    cfg["port"] = port
    path = os.path.join(outdir, f"cfg_{tag}.json")
    if extra_env:
        cfg.setdefault("env", {}).update(extra_env)
    json.dump(cfg, open(path, "w"), indent=1)
    return path


def start_engine(cfg_path: str, log_path: str, port: int):
    log = open(log_path, "ab")
    log.write(f"\n[filler_state_probe] ---- starting at {time.strftime('%H:%M:%S')} ----\n".encode())
    log.flush()
    proc = subprocess.Popen([PY, "serve/server.py", "--engine", "strata", "--config", cfg_path,
                             "--port", str(port)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server.py exited early (rc {proc.returncode}); see {log_path}")
        if port_up(port):
            return proc
        time.sleep(2)
    stop_engine(proc, port)
    raise RuntimeError("engine not ready within 900 s; see " + log_path)


def stop_engine(proc, port: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + 30
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()
    for _ in range(15):
        if not port_up(port):
            break
        time.sleep(1)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in subprocess.run(["pgrep", "-x", "strata"], capture_output=True,
                                  text=True).stdout.split():
            try:
                os.kill(int(pid), sig)
            except ProcessLookupError:
                pass
        time.sleep(2)


def chat(port: int, model: str, messages: list, max_tokens: int = 64, timeout: int = 1800) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    return {"text": text, "dt": time.time() - t0, "usage": resp.get("usage", {}),
            "finish": ch.get("finish_reason")}


def stream_abort(port: int, model: str, messages: list, read_bytes: int = 6000) -> float:
    """Open a streaming completion and hang up mid-reply: the engine sees a client cancel.

    This is exactly what happens when a harness parses a tool call out of the stream and stops
    reading; the serve then drops the partial reply and the window rolls back.
    """
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 400, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    try:
        r.read(read_bytes)
    finally:
        r.close()
    return time.time() - t0


def log_grep(log_path: str, needle: str) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 200000))
            tail = f.read().decode(errors="replace")
        hits = [ln for ln in tail.splitlines() if needle in ln]
        return scrub(hits[-1])[:200] if hits else ""
    except OSError:
        return ""


class Rec:
    def __init__(self, outdir: str):
        self.f = open(os.path.join(outdir, "results.jsonl"), "a")

    def __call__(self, **kw):
        kw["t"] = time.strftime("%H:%M:%S")
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()
        tag = f"[{kw.get('arm','?')}] {kw.get('phase','?'):28s} {kw.get('verdict',''):8s}"
        extra = f" {kw.get('secs', 0):6.1f}s" if "secs" in kw else ""
        det = kw.get("detail", "")
        print(tag + extra + ("  " + det[:150] if det else ""), flush=True)


def verdict_of(text: str) -> str:
    return "CORRUPT" if clean_len(text) == 0 else "ok"


def log_hits(log_path: str, needle: str) -> int:
    """Count a needle in the log's tail (cheap diffing between phases)."""
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 400000))
            return f.read().decode(errors="replace").count(needle)
    except OSError:
        return 0


def probes(port: int, model: str, rec, arm: str, log_path: str, msgs_list: list[list], max_tokens=64):
    bad, caught = [], []
    for i, m in enumerate(msgs_list, 1):
        n0 = log_hits(log_path, "non-finite logits (#879)")
        r = chat(port, model, m, max_tokens=max_tokens)
        n1 = log_hits(log_path, "non-finite logits (#879)") - n0
        v = verdict_of(r["text"])
        if v == "ok" and n1:
            v = "POISON CAUGHT"   # the engine flagged a window, flushed and retried: the fix working
        rec(arm=arm, phase=f"probe {i}", verdict=v, secs=round(r["dt"], 1),
            finish=r["finish"], completion_tokens=r["usage"].get("completion_tokens"),
            prompt_tokens=r["usage"].get("prompt_tokens"), nf879=n1,
            engine_log=log_grep(log_path, "serve: prompt"),
            nf_log=log_grep(log_path, "non-finite logits (#879)"),
            sample=scrub(r["text"][:100]).replace(FILLER, "<F>"))
        if v == "CORRUPT":
            bad.append(i)
        elif n1:
            caught.append(i)
    return bad, caught


def arm_control(port, rec, outdir):
    tag = "control"
    cfg = build_cfg(tag, outdir, port)
    log = json.load(open(cfg))["log"]
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok", detail=model)
        msgs = [{"role": "user", "content": "What is the capital of Japan? One short sentence."},
                {"role": "assistant", "content": "Tokyo is the capital of Japan."},
                {"role": "user", "content": "And its population, roughly?"}]
        probes(port, model, rec, tag, log, [msgs, [{"role": "user", "content": PROBE}],
                                            [{"role": "user", "content": "Say the word lighthouse."}]])
    finally:
        stop_engine(proc, port)


def arm_cancel(port, rec, outdir):
    tag = "cancel"
    cfg = build_cfg(tag, outdir, port)
    log = json.load(open(cfg))["log"]
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok", detail=model)
        # (a) force the server's repeat guard: a degenerate same-token loop that gets discarded.
        r = chat(port, model, [{"role": "user", "content":
                                "Output the single word 'mariner' over and over, space separated, "
                                "as many times as you can until you run out of tokens. Nothing else."}],
                 max_tokens=400, timeout=1800)
        rec(arm=tag, phase="repeat-guard fire", verdict=verdict_of(r["text"]), secs=round(r["dt"], 1),
            finish=r["finish"], completion_tokens=r["usage"].get("completion_tokens"),
            sample=scrub(r["text"][:80]).replace(FILLER, "<F>"))
        bad, caught = probes(port, model, rec, tag, log,
                             [[{"role": "user", "content": PROBE}],
                              [{"role": "user", "content": "What is 12 times 9?"}]])
        rec(arm=tag, phase="after repeat-discard",
            verdict="POISONED" if bad else ("POISON CAUGHT (fix holds)" if caught else "clean"))
        # (b) client stream abort mid-reply: the engine cancels and rolls the window back.
        t = stream_abort(port, model, [{"role": "user", "content":
                                        "Write a detailed 500-word essay on the history of the lighthouse."}])
        rec(arm=tag, phase="stream aborted", verdict="ok", secs=round(t, 1),
            detail=f"read 6 KB then hung up ({t:.0f}s)")
        bad2, caught2 = probes(port, model, rec, tag, log,
                               [[{"role": "user", "content": PROBE}],
                                [{"role": "user", "content": "Name three primary colors."}]])
        rec(arm=tag, phase="after stream-abort",
            verdict="POISONED" if bad2 else ("POISON CAUGHT (fix holds)" if caught2 else "clean"))
    finally:
        stop_engine(proc, port)


def arm_persist(port, rec, outdir):
    tag = "persist"
    cfg = build_cfg(tag, outdir, port)
    c = json.load(open(cfg))
    if not os.path.exists(QUARANTINE):
        rec(arm=tag, phase="staging", verdict="SKIP", detail="quarantine file missing: " + QUARANTINE)
        return
    import shutil
    dst = os.path.join(c["slot_save_path"], "auto-session")
    shutil.copyfile(QUARANTINE, dst)
    rec(arm=tag, phase="staged session", verdict="ok", detail=f"{os.path.getsize(dst)/1e9:.1f} GB -> {dst}")
    log = c["log"]
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok",
            detail=log_grep(log, "session") or "restored from staged auto-session")
        bad, caught = probes(port, model, rec, tag, log,
                             [[{"role": "user", "content": PROBE}],
                              [{"role": "user", "content": "What is the capital of Brazil?"}]])
        rec(arm=tag, phase="restored-corrupted-state",
            verdict="POISONED" if bad else ("POISON CAUGHT (fix holds)" if caught else "clean"),
            detail="clean content on a restored poisoned session")
        # restart WITHOUT restoring (remove the file): if the model was only poisoned by the
        # restored state, a cold engine must answer clean again.
        stop_engine(proc, port)
        proc = None
        os.rename(dst, dst + ".used")
        proc = start_engine(cfg, log, port)
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        bad2, caught2 = probes(port, model, rec, tag, log, [[{"role": "user", "content": PROBE}]])
        rec(arm=tag, phase="cold after restore test", verdict="POISONED" if bad2 else "clean")
    finally:
        if proc:
            stop_engine(proc, port)


def load_turns(path: str) -> list:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from filler_replay import load_turns as lt, asks
    return lt(path), asks


def arm_replay(port, rec, outdir, to: int):
    tag = "replay"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from filler_replay import load_turns as lt, asks
    queue = asks(lt(SESSION))
    rec(arm=tag, phase="transcript", verdict="ok", detail=f"{len(queue)} asks from {os.path.basename(SESSION)}")
    cfg = build_cfg(tag, outdir, port)
    log = json.load(open(cfg))["log"]
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok", detail=model)
        for k, messages in enumerate(queue[:to], 1):
            t0 = time.time()
            try:
                r = chat(port, model, messages, max_tokens=400, timeout=2400)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                rec(arm=tag, phase=f"ask {k}", verdict="HTTP-ERR", detail=str(e))
                break
            v = verdict_of(r["text"])
            rec(arm=tag, phase=f"ask {k}", verdict=v, secs=round(time.time() - t0, 1),
                finish=r["finish"], prompt_tokens=r["usage"].get("prompt_tokens"),
                completion_tokens=r["usage"].get("completion_tokens"),
                engine_log=log_grep(log, "serve: prompt"),
                sample=scrub(r["text"][:80]).replace(FILLER, "<F>"))
            if v == "CORRUPT":
                bad, caught = probes(port, model, rec, tag, log, [[{"role": "user", "content": PROBE}]])
                rec(arm=tag, phase="replay verdict",
                    verdict="CONTENT TRIGGER" if (bad or caught) else "content-only",
                    detail=f"first corruption at ask {k}; sticky probe "
                           f"{'POISONED' if bad else ('CAUGHT' if caught else 'clean')}")
                break
        else:
            rec(arm=tag, phase="replay verdict", verdict="CLEAN REPLAY",
                detail=f"no corruption in {min(len(queue), to)} asks - not content alone")
    finally:
        stop_engine(proc, port)


def arm_guard(port, rec, outdir):
    """Prove the #879 wiring on Metal without any real poison: STRATA_NAN_INJECT=1 reports the
    first head window as non-finite.  With the detection wired into Verifier::run's Metal branch,
    the engine ends that request DONE nonfinite (no token of the window is emitted), the server
    retries once unseen, and the answer still arrives clean.  Before the wiring the injection was
    never counted: the branch returned before note_nonfinite, exactly as a poisoned window did."""
    tag = "guard"
    cfg = build_cfg(tag, outdir, port, extra_env={"STRATA_NAN_INJECT": "1"})
    log = json.load(open(cfg))["log"]
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok", detail=model)
        r = chat(port, model, [{"role": "user", "content": PROBE}], max_tokens=24)
        inj = log_grep(log, "STRATA_NAN_INJECT")
        srv = log_grep(log, "non-finite")
        v = ("FIRED" if inj and srv and verdict_of(r["text"]) == "ok"
             else "CORRUPT" if verdict_of(r["text"]) != "ok" else "NOT-FIRED")
        rec(arm=tag, phase="guard verdict", verdict=v, secs=round(r["dt"], 1),
            detail=f"engine: {inj[:90] or 'NO INJECT LINE'}; serve: {srv[:90] or 'NO NON-FINITE LINE'}; "
                   f"reply: {scrub(r['text'][:60]).replace(FILLER, '<F>')}")
    finally:
        stop_engine(proc, port)

# ---- the poison arm: rewrite a saved session's own bytes (STRSESS v1, conversation_file.cpp
# put_payload/put_checkpoint) so a real poisoned state re-enters the compute through restore+reuse.
GDN_STATE_BYTES = 3268608          # this pack: 36 states x (786432 gs + 30720 hs) f32
P1, P2, P3, P4, P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                      0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)
M64 = (1 << 64) - 1


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & M64


def _round1(acc: int, w: int) -> int:
    return _rotl((acc + w * P2) & M64, 31) * P1 & M64


def session_hash64(data: bytes, seed: int = 0) -> int:
    """Port of strata::core::SessionHasher: the trailer checksum the reader verifies (line 1232)."""
    lanes = [(seed + P1 + P2) & M64, (seed + P2) & M64, seed & M64, (seed - P1) & M64]
    total = len(data)
    nblk = total // 32
    for b in range(nblk):
        o = b * 32
        lanes = [_round1(lanes[i], int.from_bytes(data[o + 8 * i:o + 8 * i + 8], "little")) for i in range(4)]
    h = (_rotl(lanes[0], 1) + _rotl(lanes[1], 7) + _rotl(lanes[2], 12) + _rotl(lanes[3], 18)) & M64
    for l in lanes:
        h = (h ^ _round1(0, l)) & M64
        h = (h * P1 + P4) & M64
    h = (h + total) & M64
    pend, i = data[nblk * 32:], 0
    while i + 8 <= len(pend):
        h = (h ^ _round1(0, int.from_bytes(pend[i:i + 8], "little"))) & M64
        h = (_rotl(h, 27) * P1 + P4) & M64
        i += 8
    while i < len(pend):
        h = (h ^ pend[i] * P5) & M64
        h = _rotl(h, 11) * P1 & M64
        i += 1
    h ^= h >> 33; h = h * P2 & M64
    h ^= h >> 29; h = h * P3 & M64
    h ^= h >> 32
    return h


def walk_gdn_blobs(data: bytes):
    """Yield (offset, length) of every checkpoint's gdn array in payload order (live first)."""
    import struct
    assert data[:8] == b"STRSESS\x01", "not a STRSESS v1 file"
    at = 64 + 18 * 8 + 8 + 8 + 8        # geometry, layer_lo, layer_hi, cvec

    def ckpt():
        nonlocal at
        n_ids = struct.unpack_from("<Q", data, at)[0]; at += 8 + 4 * n_ids
        n_img = struct.unpack_from("<Q", data, at)[0]; at += 8 + 16 * n_img
        gdn_len = struct.unpack_from("<Q", data, at)[0]; gdn_at = at + 8; at += 8 + gdn_len
        for _ in range(4):              # ple, tails, dead, block_pos
            ln = struct.unpack_from("<Q", data, at)[0]; at += 8 + ln
        at += 8                         # used
        return gdn_at, gdn_len

    yield ckpt()
    n_ckpt = struct.unpack_from("<Q", data, at)[0]; at += 8
    for _ in range(n_ckpt):
        yield ckpt()


def poison_session(path: str, rec) -> bool:
    """NaN the last 10 of 36 GDN states of every checkpoint (the forensic signature of the real
    quarantines), then rewrite the payload checksum so the reader still accepts the file."""
    import struct
    data = bytearray(open(path, "rb").read())
    payload_len = struct.unpack_from("<Q", data, 32)[0]
    if len(data) != 64 + payload_len + 16:
        rec(arm="poison", phase="poison", verdict="SKIP", detail="session size vs header mismatch")
        return False
    blobs = 0
    for gdn_at, gdn_len in walk_gdn_blobs(bytes(data)):
        if gdn_len == 0 or gdn_len % GDN_STATE_BYTES:
            rec(arm="poison", phase="poison", verdict="SKIP",
                detail=f"gdn blob {gdn_len} not a whole number of {GDN_STATE_BYTES}-byte states")
            return False
        states = gdn_len // GDN_STATE_BYTES
        first = (states - min(10, states)) * GDN_STATE_BYTES
        data[gdn_at + first: gdn_at + gdn_len] = struct.pack("<I", 0x7FC00000) * ((gdn_len - first) // 4)
        blobs += 1
    struct.pack_into("<Q", data, 64 + payload_len, session_hash64(bytes(data[64:64 + payload_len])))
    open(path, "wb").write(bytes(data))
    rec(arm="poison", phase="poison", verdict="ok",
        detail=f"{blobs} gdn arrays poisoned (10/36 states NaN), checksum rewritten")
    return True


def arm_poison(port, rec, outdir):
    """Build a conversation, let auto-session save it, poison the saved GDN state offline, restart
    and CONTINUE the same conversation: the request prefix-matches, so the poisoned state re-enters
    the compute through reuse.  The #879 scan must flag the window, the engine flushes, the server
    retries unseen and the answer arrives clean (SELF-HEALED).  Before the fix this exact sequence
    streamed the token-0 filler: the live incident's mechanism, reproduced offline."""
    tag = "poison"
    cfg = build_cfg(tag, outdir, port)
    c = json.load(open(cfg))
    log = c["log"]
    dst = os.path.join(c["slot_save_path"], "auto-session")
    proc = start_engine(cfg, log, port)
    try:
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up", verdict="ok", detail=model)
        history = []
        for p in ("You are a lighthouse keeper writing a logbook. Describe your morning duties in about 150 words.",
                  "What animals visit a lighthouse at night? Answer in about 150 words.",
                  "Write a two-hundred-word logbook entry about a stormy night on watch."):
            history.append({"role": "user", "content": p})
            r = chat(port, model, history, max_tokens=400)
            history.append({"role": "assistant", "content": r["text"]})
        rec(arm=tag, phase="build turns", verdict="ok", detail=log_grep(log, "serve: prompt"))
        time.sleep(15)                    # auto-session lands ~10 s after the last request settles
        stop_engine(proc, port)
        proc = None
        if not os.path.exists(dst):
            rec(arm=tag, phase="saved", verdict="SKIP", detail="no auto-session written: " + dst)
            return
        if not poison_session(dst, rec):
            return
        proc = start_engine(cfg, log, port)
        model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                                 timeout=10))["data"][0]["id"]
        rec(arm=tag, phase="engine up (poisoned)", verdict="ok",
            detail=log_grep(log, "session") or "restored from the poisoned auto-session")
        history.append({"role": "user", "content": PROBE})
        r = chat(port, model, history, max_tokens=64)
        pl = log_grep(log, "serve: prompt")
        reused = int(re.search(r"(\d+) reused", pl).group(1)) if re.search(r"(\d+) reused", pl) else -1
        nf = log_grep(log, "non-finite logits (#879)")
        v = verdict_of(r["text"])
        if reused <= 0:
            verd = "REUSE FAILED - inconclusive"   # the poisoned state never entered the compute
        elif nf and v == "ok":
            verd = "SELF-HEALED (real poison)"
        elif v == "CORRUPT":
            verd = "FIX MISSES REAL POISON"
        else:
            verd = "NO FLAG"                       # poisoned reuse ran to a clean answer: NaN did not propagate
        rec(arm=tag, phase="poison verdict", verdict=verd, secs=round(r["dt"], 1),
            reused=reused, nf879=log_hits(log, "non-finite logits (#879)"),
            detail=f"{pl[:120]}; {nf[:120] or 'no #879 line'}; reply: {scrub(r['text'][:60]).replace(FILLER, '<F>')}")
    finally:
        if proc:
            stop_engine(proc, port)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="guard,control,cancel,poison,replay")
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--to", type=int, default=90, help="max asks for the replay arm")
    ap.add_argument("--force", action="store_true", help="run even with little free RAM")
    a = ap.parse_args()
    for req, why in ((BASE_CFG, "serve config"),
                     (QUARANTINE if "persist" in a.arms else "", "quarantined session"),
                     (SESSION if "replay" in a.arms else "", "transcript jsonl")):
        if req and not os.path.exists(req):
            print(f"missing {why}: {req}\n(a slot-switch script may have moved it; restore it first)")
            return 2
    if port_up(a.port):
        print(f"something already serves on {a.port}; stop it first (this harness starts its own)")
        return 2
    need = 54 * 1024**3
    have = free_bytes()
    if have < need and not a.force:
        print(f"only {have/1e9:.1f} GB free; the arms need ~{need/1e9:.0f} GB. "
              "Stop sushi (and anything else holding RAM) first, or pass --force.")
        return 2
    outdir = os.path.join(REPO, "gap", "filler-state-probe-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    rec = Rec(outdir)
    rec(arm="gate", phase="free RAM", verdict="ok", detail=f"{have/1e9:.1f} GB free")
    failed = None
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        rec(arm=arm, phase="START", verdict="")
        try:
            if arm == "guard":
                arm_guard(a.port, rec, outdir)
            elif arm == "control":
                arm_control(a.port, rec, outdir)
            elif arm == "cancel":
                arm_cancel(a.port, rec, outdir)
            elif arm == "persist":
                arm_persist(a.port, rec, outdir)
            elif arm == "poison":
                arm_poison(a.port, rec, outdir)
            elif arm == "replay":
                arm_replay(a.port, rec, outdir, a.to)
            else:
                print("unknown arm " + arm)
        except Exception as e:
            failed = arm
            rec(arm=arm, phase="ARM-ERROR", verdict="ERR", detail=str(e)[:300])
            subprocess.run(["pkill", "-x", "strata"], capture_output=True)
            time.sleep(5)
        if failed == arm:
            print(f"arm {arm} errored; later arms still run (engine is stopped between arms)")
            failed = None
        time.sleep(6)
    print(f"\nresults: {os.path.join(outdir, 'results.jsonl')}")
    print("Paste that file (and any *_engine.log tails you want kept) back here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
