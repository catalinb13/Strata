#!/usr/bin/env python3
"""metal/tools/bench_verify.py - the verify-window cost curve on the Mac engine.

Run between a `serve_huihui.sh stop` and its restart (the `run CMD` slot) with
STRATA_METAL_KPROF=1 in the config's env: it sends one long recitation turn per
conversation, so each draft-window class (T tokens with the head) runs 20+ times
and the window prints its GPU-time report to the engine log.  Pass the --spec the
engine was started with as the only argument (a label; it picks how deep the
suffix drafter goes, so which classes fill).

    HUIHUI_CFG=... metal/tools/serve_huihui.sh run python3 metal/tools/bench_verify.py 8

The answers are still correct up to the report; the report's serial replay makes
the remainder of that conversation wrong (it is discarded anyway: a fresh
throwaway conversation per call, far under the parking floor).
"""
import json
import sys
import time
import urllib.request

CFG = "/Users/pc/src/Strata-fork/strata-ud-iq4_xs-huihui.json"   # 2026-11-08: 8180 serves the UD-IQ4_XS pack
URL = "http://127.0.0.1:8180/v1/chat/completions"


def turn(messages, max_tokens=220):
    body = json.dumps({"model": "bench", "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], 1000.0 * (time.time() - t0)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "?"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    model = json.load(open(CFG))["model_name"]
    globals()["URL"] = URL  # keep
    para = ("The harbor slept under a thin crust of ice, and the fishing boats knocked their hulls "
            "against the quay in a slow rhythm that nobody was left to hear. Gulls wheeled over the "
            "breakwater, crying at a catch they would never reach, while the tide pulled the dark "
            "water out and back again over the stones. Somewhere behind the warehouse roofs a clock "
            "struck the hour and the sound carried flat and clear across the frozen bay.")
    # one throwaway conversation: a verbatim copy gives the suffix drafter a long literal
    # match every round, so the windows run at the full offered depth (up to --lookup-max-t)
    msgs = [{"role": "user", "content": "Repeat this paragraph verbatim again and again:\n\n" + para}]
    for k in range(3):
        text, ms = turn(msgs, max_tokens)
        print(f"bench_verify[{label}] turn {k}: {ms:.0f} ms client", flush=True)
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": "continue"})
    print(f"bench_verify[{label}] done", flush=True)


if __name__ == "__main__":
    main()
