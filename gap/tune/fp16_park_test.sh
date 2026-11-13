#!/usr/bin/env bash
# Verify --kv fp16 with conversation PARKING.
# Parking happens when a conversation must give up its KV cells to another one, so the
# test needs TWO conversations that alternate: A, B, A.  The second A restores A's park.
# That is the path gated by window_pools() in src/core/conversation_snapshot.cpp.
set -u
cd /Users/pc/src/Strata-fork
PORT=8198
LOG=/tmp/fp16_park.log
CFG=/tmp/fp16_park_cfg.json
pkill -f "server.py.*--port $PORT" 2>/dev/null
pkill -x strata 2>/dev/null
sleep 1

.venv/bin/python gap/tune/mk_cfg.py "$CFG" --port $PORT \
  --set --max-context 16384 --set --kv fp16 --set --expert-cache 1024 \
  --set --conversation-cache-mib 4096 --set --conversation-cache-slots 1 \
  --set --conversation-cache-min-tokens 32 --set --conversation-cache-min-free-mib 512 \
  || { echo "cfg failed"; exit 1; }

STRATA_PARK_DEBUG=1 .venv/bin/python serve/server.py --engine strata --config "$CFG" --port $PORT > $LOG 2>&1 &
SPID=$!
echo "server pid $SPID; waiting..."
for i in $(seq 1 100); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" | grep -q model && break
  sleep 2
done
if ! curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" | grep -q model; then
  echo "FAIL: not ready"; tail -8 $LOG; kill $SPID 2>/dev/null; exit 1
fi
echo "ready."

# A and B alternate so each one must park its cells for the other.
.venv/bin/python - "$PORT" <<'PY'
import json, subprocess, sys
port = sys.argv[1]

def ask(messages, tag):
    body = {"model": "x", "max_tokens": 160, "messages": messages}
    r = subprocess.run(["curl", "-s", "--max-time", "300",
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps(body)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        m = d["choices"][0]["message"]
        print(f"{tag}: {str(m.get('content'))[:90]!r}")
        return m
    except Exception:
        print(f"{tag}: FAILED {r.stdout[:180] or r.stderr[:180]}")
        return None

a1 = ask([{"role": "user", "content": "Write a short paragraph about the sea."}], "A1")
b1 = ask([{"role": "user", "content": "Write a short paragraph about mountains."}], "B1")
if a1:
    a2 = ask([{"role": "user", "content": "Write a short paragraph about the sea."},
              {"role": "assistant", "content": a1.get("content") or ""},
              {"role": "user", "content": "Name one colour, one word."}], "A2 (restores A1)")
PY

echo "--- park / restore activity ---"
grep -aE "conversation cache|strata metal park" $LOG | head -12
grep -aiE "reused|holds no live cells" $LOG | tail -4
echo "--- errors ---"
grep -aiE "no function|Traceback|error" $LOG | tail -4
kill $SPID 2>/dev/null
sleep 1
pkill -x strata 2>/dev/null
echo "park test done"
