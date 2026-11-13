#!/usr/bin/env bash
# Verify --kv fp16 on the BATCH (rows_) path: the served engine takes two concurrent
# requests, each a different sequence in its own slot.  That is the only path that
# calls Window::init_slots, i.e. the only one that would hit the alloc(0) trap.
set -u
cd /Users/pc/src/Strata-fork
PORT=8199
LOG=/tmp/fp16_batch.log
CFG=/tmp/fp16_batch_cfg.json
pkill -f "server.py.*--port $PORT" 2>/dev/null
pkill -x strata 2>/dev/null
sleep 1

.venv/bin/python gap/tune/mk_cfg.py "$CFG" --port $PORT \
  --set --max-context 16384 --set --kv fp16 --set --expert-cache 1024 \
  --set --conversation-cache-mib 0 || { echo "cfg failed"; exit 1; }

.venv/bin/python serve/server.py --engine strata --config "$CFG" --port $PORT --parallel 2 > $LOG 2>&1 &
SPID=$!
echo "server pid $SPID; waiting..."
for i in $(seq 1 100); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" | grep -q model && break
  sleep 2
done
if ! curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" | grep -q model; then
  echo "FAIL: server never became ready"; tail -10 $LOG; kill $SPID 2>/dev/null; exit 1
fi
echo "ready. engine lines:"
grep -aiE "batch slot|fp16" $LOG | tail -4
echo "sending 2 concurrent requests"
PIDS=""
for i in 1 2; do
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"max_tokens\":160,\"messages\":[{\"role\":\"user\",\"content\":\"Write one short sentence about number $i.\"}]}" \
    -o /tmp/fp16_batch_$i.json &
  PIDS="$PIDS $!"
done
# wait only on the curls: a bare `wait` would also wait on the server started above, forever
wait $PIDS
echo "--- replies ---"
for i in 1 2; do
  .venv/bin/python -c "
import json
try:
    d = json.load(open('/tmp/fp16_batch_$i.json'))
    print('req $i OK:', repr(d['choices'][0]['message']['content'][:100]))
except Exception as e:
    print('req $i FAILED:', e, open('/tmp/fp16_batch_$i.json').read()[:200])
"
done
echo "--- engine errors ---"
grep -aiE "no function|Traceback|alloc .* failed|error" $LOG | tail -5
kill $SPID 2>/dev/null
sleep 1
pkill -x strata 2>/dev/null
echo "batch test done"
