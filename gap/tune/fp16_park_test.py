#!/usr/bin/env python3
"""fp16 conversation PARKING test.

Session::park_live_() refuses a conversation shorter than PARK_MIN (1024 tokens,
src/split/src/session.cpp:18), so every earlier attempt at ~100 tokens re-read the
prompt instead of parking.  This test uses two ~1400-token conversations and
alternates A, B, A: A must be parked to make room for B, then restored.

    python3 gap/tune/fp16_park_test.py
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = "/Users/pc/src/Strata-fork"
PORT = 8198
LOG = "/tmp/fp16_park.log"
CFG = "/tmp/fp16_park_cfg.json"
PY = f"{ROOT}/.venv/bin/python"

# Two distinguishable ~1400-token bodies.  Varied sentences so the tokenizer does not collapse them.
SEA = " ".join(
    f"Sentence {i}: the sea moved slowly under a grey sky and the boats knocked together in the harbour "
    f"while the gulls circled the pier and the fog lifted off the water near the old lighthouse." for i in range(38))
HILL = " ".join(
    f"Line {i}: the mountain path climbed past broken walls and dry stone terraces where the goats grazed "
    f"and the wind carried the smell of pine across the valley toward the ruined chapel on the ridge." for i in range(38))


def start_server():
    subprocess.run([PY, f"{ROOT}/gap/tune/mk_cfg.py", CFG, "--port", str(PORT),
                    "--set", "--max-context", "16384", "--set", "--kv", "fp16",
                    "--set", "--expert-cache", "1024",
                    "--set", "--conversation-cache-mib", "4096",
                    "--set", "--conversation-cache-slots", "2",
                    "--set", "--conversation-cache-min-tokens", "512",
                    "--set", "--conversation-cache-min-free-mib", "512"], check=True)
    env = dict(os.environ, STRATA_PARK_DEBUG="1")
    log = open(LOG, "w")
    p = subprocess.Popen([PY, f"{ROOT}/serve/server.py", "--engine", "strata",
                          "--config", CFG, "--port", str(PORT)],
                         stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    for _ in range(120):
        r = subprocess.run(["curl", "-sf", "--max-time", "2",
                            f"http://127.0.0.1:{PORT}/v1/models"], capture_output=True, text=True)
        if "model" in r.stdout:
            return p
        if p.poll() is not None:
            print("server exited early; log tail:")
            print(open(LOG).read()[-1500:])
            sys.exit(1)
        time.sleep(2)
    print("server never became ready")
    sys.exit(1)


def ask(messages, tag, max_tokens=64):
    body = {"model": "x", "max_tokens": max_tokens, "messages": messages}
    t0 = time.time()
    r = subprocess.run(["curl", "-s", "--max-time", "600",
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        "-H", "Content-Type: application/json", "-d", json.dumps(body)],
                       capture_output=True, text=True)
    dt = time.time() - t0
    try:
        d = json.loads(r.stdout)
        m = d["choices"][0]["message"]
        print(f"{tag}: {dt:5.1f}s  {str(m.get('content'))[:80]!r}")
        return m
    except Exception:
        print(f"{tag}: FAILED ({dt:.1f}s) {r.stdout[:200] or r.stderr[:200]}")
        return None


def main():
    p = start_server()
    try:
        print(f"ready on {PORT}; parking needs >=1024 tokens per conversation")
        a1 = ask([{"role": "user", "content": SEA + "\n\nSummarise the passage in one short sentence."}], "A1")
        b1 = ask([{"role": "user", "content": HILL + "\n\nSummarise the passage in one short sentence."}], "B1 (should evict and park A1)")
        a2 = None
        if a1:
            a2 = ask([{"role": "user", "content": SEA + "\n\nSummarise the passage in one short sentence."},
                      {"role": "assistant", "content": a1.get("content") or ""},
                      {"role": "user", "content": "Name one colour, one word."}], "A2 (should RESTORE A1)")
    finally:
        p.send_signal(signal.SIGTERM)
        time.sleep(2)
        subprocess.run(["pkill", "-x", "strata"], capture_output=True)

    # the engine's own stderr goes to the config's "log" path, NOT to the server's stdout redirect
    eng = json.load(open(CFG)).get("log") or LOG
    log = open(eng).read()
    print(f"\n--- engine log tail ({eng}) ---")
    ev = [l for l in log.splitlines() if "conversation cache" in l]
    for l in ev[:12]:
        print(" ", l.strip()[:160])
    if not ev:
        print("  (none - parking did not run)")
    print("\n--- engine park debug ---")
    pd = [l for l in log.splitlines() if "strata metal park" in l]
    for l in pd[:8]:
        print(" ", l.strip()[:160])
    print("\n--- reuse ---")
    for l in log.splitlines():
        if re.search(r"prompt \d+ =|reused", l):
            print(" ", l.strip()[:160])
    print("\n--- failures ---")
    bad = [l for l in log.splitlines()
           if re.search(r"holds no live cells|Traceback|no function", l)]
    print("  " + ("; ".join(bad[:3]) if bad else "none"))


if __name__ == "__main__":
    main()
