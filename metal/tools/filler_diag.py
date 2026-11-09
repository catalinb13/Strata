#!/usr/bin/env python3
"""metal/tools/filler_diag.py - A/B diagnosis of the token-0 filler corruption on the
UD-IQ4_XS huihui slot (M1 Max, Metal serve path).

Run this with sushi STOPPED (the 80B needs the whole machine; the preflight refuses to
start while memory is short) and nothing serving on port 8180:

    .venv/bin/python metal/tools/filler_diag.py            # all four arms, ~45-60 min
    .venv/bin/python metal/tools/filler_diag.py --arms A   # baseline only, ~20 min

Arms (each gets its own scratch session directory; the shared auto-session is never
read or written, so a poisoned history cannot imitate its way into the results):

  A baseline  - the serve config as set, every phase: cold short prompt, cold ~21K
                prompt twice, a four-turn conversation that crosses the 2048-token
                parking threshold, then a restart that RESTORES the saved session.
  B no-mpp    - STRATA_METAL_PROMPT_MOE removed (the legacy prompt-expert path):
                does the mpp chunk path corrupt KV as the Q5_0 mm_expand bug did.
  C no-draft  - --spec / --lookup-max-t / --suffix-draft removed: does the verify or
                the suffix drafter feed the filler (upstream issue #5160 reads).
  D no-cache  - the conversation-cache flags and auto_session removed: does parking
                or checkpoint restore corrupt KV mid-conversation.

Every completion is scored: nearly-empty once the token-0 filler character (chr 33)
and whitespace are stripped means CORRUPT; otherwise the needle (or, for the last
turn, the parcel weight) must come back. Results print as a table and are written to
gap/filler-diag-<stamp>/results.txt and results.json; engine logs stay beside them.

Scrub policy honored: this file never contains a run of the filler character.
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
BASE_CFG = os.path.join(REPO, "strata-ud-iq4_xs-huihui.json")
PY = os.path.join(REPO, ".venv", "bin", "python")
FILLER = chr(33)

PARA = ("The archivist copied one ledger page each evening, reading the columns aloud so the "
        "paper would remember what the ink forgot. Shelves of almanacs lined the west wall, and "
        "between them a small stove warmed a kettle that nobody drank from. She noted weather, "
        "wage, and cargo, and at the foot of every page she drew a narrow horizontal rule, and "
        "above it the day of the month in old numerals, and once a margin note about rain.")


def clean_len(s: str) -> int:
    return len(re.sub(r"\s", "", (s or "").replace(FILLER, "")))


def scrub(s: str) -> str:
    return re.sub(FILLER + r"{2,}", "[FILLER-RUN]", s or "")


def long_prompt(needle: str) -> str:
    parts, total, n = [], 0, 0
    while total < 88000:
        n += 1
        para = f"Section {n}. " + PARA + f" The section number {n} was checked twice before filing."
        if n == 60:
            para += f" Somewhere in this document the code {needle} appears exactly once."
        parts.append(para)
        total += len(para)
    return "\n\n".join(parts) + f"\n\nRepeat the code verbatim, nothing else. The code is {needle}."


def port_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def free_bytes() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        ps = int(m.group(1))
    vals = {}
    for line in out.splitlines():
        mm = re.match(r"^(.*?)\s*:\s*(\d+)\.", line)
        if mm:
            vals[mm.group(1).strip()] = int(mm.group(2))
    # same accounting as metal/tools/serve_huihui.sh (macOS 27 labels, 16 KiB pages)
    return (vals.get("Pages free", 0) + vals.get("Pages speculative", 0)) * ps


def start_engine(cfg_path: str, log_path: str, port: int):
    log = open(log_path, "ab")
    log.write(f"\n[filler_diag] ---- starting {cfg_path} at {time.strftime('%H:%M:%S')} ----\n".encode())
    log.flush()
    proc = subprocess.Popen([PY, "serve/server.py", "--engine", "strata", "--config", cfg_path,
                             "--port", str(port)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server.py exited early (rc {proc.returncode}); see {log_path}")
        if port_up(port):
            return proc, log
        time.sleep(2)
    stop_engine(proc, port)
    raise RuntimeError("engine not ready within 600 s; see " + log_path)


def stop_engine(proc, port: int = 8180) -> None:
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
    for _ in range(10):
        if not port_up(port):
            break
        time.sleep(1)
    out = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout.split()
    for pid in out:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pid in subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def chat(port: int, model: str, messages: list, max_tokens: int = 48, timeout: int = 1500):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    dt = time.time() - t0
    msg = resp["choices"][0].get("message", {})
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    usage = resp.get("usage", {})
    return text, dt, usage


def tail_prompt_line(log_path: str) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 65536))
            tail = f.read().decode(errors="replace")
        for line in reversed(tail.splitlines()):
            if "serve: prompt" in line:
                return scrub(line)[:200]
    except OSError:
        pass
    return ""


def build_cfg(arm: str, outdir: str, port: int) -> str:
    cfg = copy.deepcopy(json.load(open(BASE_CFG)))
    cfg["log"] = os.path.join(outdir, f"{arm}_engine.log")
    cfg["slot_save_path"] = os.path.join(outdir, f"sessions_{arm}")
    os.makedirs(cfg["slot_save_path"], exist_ok=True)
    cfg["port"] = port
    if arm == "B":
        cfg.get("env", {}).pop("STRATA_METAL_PROMPT_MOE", None)
    elif arm == "C":
        cfg["args"] = strip_pairs(cfg["args"], ["--spec", "--lookup-max-t", "--suffix-draft"])
        cfg["drafting"] = "off"
    elif arm == "D":
        # the budget flag must be set to 0: without it the serve path sizes the cache from free RAM
        # (up to 8192 MiB), which is the very state under test (generate.cpp: "-1: sized from free RAM, 0: off")
        i = cfg["args"].index("--conversation-cache-mib")
        cfg["args"][i + 1] = "0"
        cfg["args"] = strip_pairs(cfg["args"], ["--conversation-cache-slots",
                                                "--conversation-cache-min-tokens", "--conversation-cache-min-free-mib"])
        cfg["auto_session"] = False
    path = os.path.join(outdir, f"cfg_{arm}.json")
    json.dump(cfg, open(path, "w"), indent=1)
    return path


def strip_pairs(args: list, flags) -> list:
    out = list(args)
    for f in flags:
        while f in out:
            i = out.index(f)
            del out[i:i + 2]
    return out


def run_arm(arm: str, outdir: str, port: int, results: list) -> None:
    cfg_path = build_cfg(arm, outdir, port)
    log_path = json.load(open(cfg_path))["log"]
    proc = None
    weights = [str(400 + 37 * i) for i in range(4)]

    def rec(phase, text, dt, usage, verdict):
        line = {"arm": arm, "phase": phase, "verdict": verdict, "secs": round(dt, 1),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "engine_log": tail_prompt_line(log_path),
                "sample": scrub((text or "")[:120]).replace(FILLER, "<F>")}
        results.append(line)
        print(f"  [{arm}] {phase:26s} {verdict:12s} {line['secs']:7.1f}s  "
              f"completion={line['completion_tokens']}  {line['engine_log'][:90]}", flush=True)

    try:
        proc, _ = start_engine(cfg_path, log_path, port)
        models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10))
        model = models["data"][0]["id"] if models.get("data") else ""
        print(f"  [{arm}] model {model!r}", flush=True)

        t, dt, u = chat(port, model, [{"role": "user", "content":
                                       "What is the capital of France? Answer in one short sentence."}])
        rec("P1 short", t, dt, u, "CORRUPT" if clean_len(t) < 3 else "ok")

        # max_tokens 240: with 48 the model's thinking is cut mid-sentence and the needle simply
        # never fits in the completion (the 2026-11-09 run showed exactly that at 48)
        t, dt, u = chat(port, model, [{"role": "user", "content": long_prompt("ZQX4471K7")}], max_tokens=240)
        rec("P2 cold 21K", t, dt, u, "CORRUPT" if clean_len(t) < 3 else
            ("ok" if "ZQX4471K7" in t else "NEEDLE-MISS"))

        t, dt, u = chat(port, model, [{"role": "user", "content": long_prompt("TRN9026Q4")}], max_tokens=240)
        rec("P3 cold 21K again", t, dt, u, "CORRUPT" if clean_len(t) < 3 else
            ("ok" if "TRN9026Q4" in t else "NEEDLE-MISS"))

        if arm == "A":
            msgs = []
            pad = ("A ledger line about the state of the harbor, kept every Tuesday, "
                   "with the tide table folded into the second drawer. ") * 14
            for i in range(4):
                msgs.append({"role": "user", "content":
                             f"Note {i + 1}: the parcel weighs {weights[i]} grams. {pad} "
                             "Reply with the single word noted."})
                t, dt, u = chat(port, model, msgs)
                msgs.append({"role": "assistant", "content": t[:200]})
                rec(f"P4 turn {i + 1}", t, dt, u, "CORRUPT" if clean_len(t) < 3 else "ok")

            # restart and let the engine restore the saved session; replay the whole history so
            # the request must reuse the restored (parked) prefix - the path under test
            stop_engine(proc, port)
            proc = None
            time.sleep(6)
            proc, _ = start_engine(cfg_path, log_path, port)
            t, dt, u = chat(port, model, msgs + [{"role": "user", "content":
                                                  "Repeat the weight of the last parcel you noted. One number."}])
            rec("P5 after restart", t, dt, u, "CORRUPT" if clean_len(t) < 3 else
                ("ok" if weights[-1] in t else "RECALL-MISS"))
    except Exception as e:  # noqa: BLE001 - one arm dying must not stop the run
        rec("ARM-ERROR " + type(e).__name__, "", 0.0, {}, str(e)[:120])
    finally:
        if proc is not None:
            stop_engine(proc, port)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--port", type=int, default=8180)
    args = ap.parse_args()

    if port_up(args.port):
        print(f"port {args.port} already answers - stop the serving engine first", file=sys.stderr)
        return 2
    avail = free_bytes()
    # the same floor switch_to_strata.sh waits for before wiring the 38 GB arena (40 GB)
    if avail < 40 * (1 << 30):
        print(f"only {avail / 2**30:.1f} GiB free: stop sushi first and let the pages come back "
              f"(metal/tools/switch_to_strata.sh does this; the harness waits for 40 GB)", file=sys.stderr)
        return 2

    outdir = os.path.join(REPO, "gap", "filler-diag-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(f"filler diagnosis in {outdir}; arms {args.arms}; expect 10-15 min per arm", flush=True)

    results: list = []
    try:
        for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
            print(f"== arm {arm} ==", flush=True)
            run_arm(arm, outdir, args.port, results)
            json.dump(results, open(os.path.join(outdir, "results.json"), "w"), indent=1)
    finally:
        with open(os.path.join(outdir, "results.txt"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {outdir}/results.json", flush=True)
        print("you can bring sushi back now and send this directory to the assistant", flush=True)
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in sorted({r["verdict"] for r in results})}
    print("\nverdict counts:", counts, flush=True)
    return 1 if any(r["verdict"] == "CORRUPT" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
