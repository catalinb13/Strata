#!/usr/bin/env python3
"""metal/tools/filler_replay.py - replay a real omp session against the running serve.

Arm E of the filler diagnosis: the four synthetic arms of filler_diag.py stayed clean,
but the live 2026-11-09 UD-IQ4_XS session corrupted at turn 69 - a ~298-token tool
result on a reused ~85.6K-token prefix.  This script feeds that exact conversation
back, turn by turn, and stops at the first reply whose content and reasoning are
empty once the token-0 filler character and whitespace are stripped.

Run it against a serve started from the SAME config (run-ud_iq4_xs-huihui.sh) with
sushi stopped, from a shell:

    .venv/bin/python metal/tools/filler_replay.py                      # default session
    .venv/bin/python metal/tools/filler_replay.py --path <file.jsonl>  # another session
    .venv/bin/python metal/tools/filler_replay.py --to 40              # only the first 40 asks

The serve's own conversation cache and incremental reuse carry the replay, so the
token counts land close to the originals (a few hundred off where the system prompt
digests differently); the CORRUPT verdict needs no exact numbers.

Scrub policy honored: the filler run is never printed; only "<F>" counts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

FILLER = chr(33)


def clean_len(s: str) -> int:
    return len(re.sub(r"\s", "", (s or "").replace(FILLER, "")))


def load_turns(path: str) -> list:
    """The omp message list in order, skipping titles/model changes/usage rows."""
    msgs = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message") if isinstance(d.get("message"), dict) else None
        if m is None and d.get("type") == "custom_message":
            m = {"role": "user", "content": [{"type": "text", "text": str(d.get("content") or "")}]}
        if m is None:
            continue
        role = m.get("role")
        if role not in ("user", "assistant", "toolResult"):
            continue
        out: dict = {"role": "user" if role == "user" else role, "content": ""}
        if isinstance(m.get("content"), str):
            out["content"] = m["content"]
        elif role == "assistant":
            think, text, calls = [], [], []
            for c in m.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "thinking":
                    think.append(c.get("thinking") or c.get("text") or "")
                elif c.get("type") == "text":
                    text.append(c.get("text") or "")
                elif c.get("type") == "toolCall":
                    fn = c.get("function") if isinstance(c.get("function"), dict) else c
                    name = fn.get("name") or c.get("name") or "tool"
                    args = fn.get("arguments", c.get("arguments", {}))
                    calls.append({"id": c.get("id") or f"call_{len(calls)}", "type": "function",
                                  "function": {"name": name,
                                               "arguments": args if isinstance(args, str)
                                               else json.dumps(args, ensure_ascii=False)}})
            out = {"role": "assistant", "content": "".join(text)}
            if think:
                out["reasoning_content"] = "\n".join(think)
            if calls:
                out["tool_calls"] = calls
        elif role == "toolResult":
            parts = "".join(c.get("text", "") for c in (m.get("content") or []) if isinstance(c, dict))
            out = {"role": "tool", "tool_call_id": m.get("toolCallId") or "call_0", "content": parts}
        msgs.append(out)
    return msgs


def asks(msgs: list) -> list:
    """Request message-lists: everything up to each assistant entry that follows input."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant":
            out.append(msgs[:i])
    return out


def post(port: int, model: str, messages: list, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/Users/pc/.omp/agent/sessions/-src-Strata-fork/"
                    "2026-11-09T09-23-10-598Z_01a1bf9e-6286-7367-8803-7bcb3b31266f.jsonl")
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--model", default="qwen3.8-flash-next-ud-iq4_xs-huihui")
    ap.add_argument("--to", type=int, default=10**9, help="stop after this many asks")
    ap.add_argument("--max-tokens", type=int, default=400, help="short replies: the filler shows within them")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    turn_hist = load_turns(a.path)
    # the replay must not carry the discarded/corrupt generations: keep the history as the
    # CLIENT saw it (assistant replies are the ones recorded in the session - corrupt ones
    # were discarded by the harness and never entered the next request).
    queue = asks(turn_hist)
    print(f"replaying {len(queue)} asks from {a.path.split('/')[-1]}", flush=True)
    bad = 0
    for k, messages in enumerate(queue[:a.to], 1):
        t0 = time.time()
        try:
            resp = post(a.port, a.model, messages, a.max_tokens, a.timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"ask {k}: HTTP error {e}", flush=True)
            return 2
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        n_f = text.count(FILLER)
        corrupt = clean_len(text) == 0
        bad += corrupt
        print(f"ask {k}: prompt msgs={len(messages)} reply={len(text)}ch !={n_f} "
              f"finish={ch.get('finish_reason')} {time.time()-t0:.0f}s"
              + ("  <<< CORRUPT (token-0 filler)" if corrupt else ""), flush=True)
        if corrupt:
            print(f"first corrupt at ask {k}; stop - bisect with --to {k-1} and the "
                  "filler_diag arms (B no-mpp, C no-draft, D no-cache)", flush=True)
            return 1
    print(f"replay clean: {bad} corrupt of {min(len(queue), a.to)} asks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
