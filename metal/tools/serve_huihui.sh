#!/bin/bash
# metal/tools/serve_huihui.sh - stop / run-tests / restart the 8180 Huihui (UD-IQ4_XS) server safely.
#
#   serve_huihui.sh status              who is serving, memory
#   serve_huihui.sh stop                graceful TERM, then KILL; waits for the port to close
#   serve_huihui.sh start               start exactly the config's server line; wait for loaded + one live answer
#   serve_huihui.sh restart             stop, then start: ONE command, so a down server is never left between steps
#   serve_huihui.sh setenv K=V [K=V..]  stop; edit the config's engine env (K= clears, K- removes); start.
#                                       The edit happens while stopped, so server.py cannot overwrite it on exit.
#   serve_huihui.sh run CMD...          stop; run CMD (any exit); ALWAYS start again; health-check
# Never runs two engines: it will not start while anything answers on the port.
# Designed to survive the assistant losing its own model mid-run: the restart always follows the test.
set -u
ROOT=/Users/pc/src/Strata-fork
CFG=${HUIHUI_CFG:-$ROOT/strata-ud-iq4_xs-huihui.json}
PORT=${HUIHUI_PORT:-8180}
PY=${HUIHUI_PY:-$ROOT/.venv/bin/python}   # the run scripts start serve/server.py with the venv's python; match them
[ -x "$PY" ] || PY=python3
MODEL=$(python3 -c "import json;print(json.load(open('$CFG'))['model_name'])" 2>/dev/null || echo qwen3.8-flash-next-ud-iq4_xs-huihui)
RLOG=/tmp/bw/restart.log
ELOG=$(python3 -c "import json;print(json.load(open('$CFG')).get('log',''))" 2>/dev/null)

say() { echo "[serve_huihui $(date +%H:%M:%S)] $*"; }
port_pid() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1; }
engine_pid() {
    local sp; sp=$(port_pid)
    [ -z "$sp" ] && return 0
    pgrep -P "$sp" 2>/dev/null | while read -r p; do ps -o comm= -p "$p" 2>/dev/null | grep -q strata && echo "$p"; done
}

status() {
    local p e
    p=$(port_pid)
    if [ -z "$p" ]; then say "port $PORT: nobody"; return 1; fi
    e=$(engine_pid)
    say "port $PORT: server.py pid $p, engine pid ${e:-?} ($(ps -o rss= -p "${e:-0}" 2>/dev/null | awk '{printf "%.1f GB", $1/1048576}') engine RSS)"
}

stop() {
    local p e
    p=$(port_pid)
    if [ -z "$p" ]; then say "already down"; return 0; fi
    say "stopping server pid $p (and children)"
    pkill -TERM -P "$p" 2>/dev/null
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 60); do
        p=$(port_pid)
        [ -z "$p" ] && break
        sleep 0.5
    done
    p=$(port_pid)
    if [ -n "$p" ]; then
        say "still listening after 30s: KILL"
        pkill -KILL -P "$p" 2>/dev/null
        kill -KILL "$p" 2>/dev/null
        sleep 2
    fi
    for e in $(pgrep -x strata 2>/dev/null); do kill -TERM "$e" 2>/dev/null; done
    sleep 3
    for e in $(pgrep -x strata 2>/dev/null); do kill -KILL "$e" 2>/dev/null; done
    if [ -n "$(port_pid)" ]; then say "ERROR: port $PORT still held"; return 1; fi
    say "down; free: $(vm_stat | awk '/page size of/{for(i=1;i<=NF;i++) if($i=="of") ps=$(i+1)} /Pages free/{f=$3} /Pages speculative/{s=$3} END{if(!ps) ps=4096; gsub(/\./,"",f); gsub(/\./,"",s); printf "%.1f GB", (f+s+0)*ps/1073741824}')"
}

wait_ready() {  # $1 = seconds
    local deadline=$(( $(date +%s) + ${1:-600} )) miss=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"loaded"'; then
            say "models loaded"
            return 0
        fi
        miss=$(( miss + 1 ))
        # an engine that died at start (a bad config, an allocation that failed) is never going to load:
        # stop after 60s without one instead of waiting out the whole window
        if [ "$miss" -ge 30 ] && [ -z "$(pgrep -x strata 2>/dev/null)" ] && [ -n "$(port_pid)" ]; then
            say "ERROR: the engine died at start (no strata process); last lines of ${ELOG:-the engine log}:"
            [ -n "$ELOG" ] && tail -4 "$ELOG" | sed 's/^/    /'
            return 1
        fi
        # server.py itself died before the port ever opened (a bad config, an import error): never wait it out
        if [ -n "${SPID:-}" ] && ! kill -0 "$SPID" 2>/dev/null && [ -z "$(port_pid)" ]; then
            say "ERROR: server.py (pid $SPID) exited before the port opened; last lines of $RLOG:"
            tail -6 "$RLOG" | sed 's/^/    /'
            return 1
        fi
        sleep 2
    done
    say "TIMEOUT waiting for /v1/models"
    return 1
}

probe_answer() {  # one real completion; content or reasoning counts as life, but a run of "!" (token 0) is corruption
    local deadline out
    deadline=$(( $(date +%s) + ${1:-120} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        out=$(curl -s --max-time 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":64,\"temperature\":0}")
        # The UD-IQ4_XS decode bug emits "!" (vocab id 0) forever; the old grep called that a live answer. A
        # completion whose content+reasoning, minus "!" and whitespace, is empty is the corruption, not life.
        if printf '%s' "$out" | "$PY" -c '
import sys, json
try:
    m = json.load(sys.stdin)["choices"][0]["message"]
except Exception:
    sys.exit(1)
text = (m.get("content") or "") + (m.get("reasoning_content") or "")
sys.exit(0 if text.strip("! \t\r\n") else 1)
' 2>/dev/null; then
            say "live answer ok"
            return 0
        fi
        if printf '%s' "$out" | grep -q '"choices"'; then
            say "PROBE: engine answered but emitted the !-corruption (argmax fell to token 0); not ready"
        fi
        sleep 3
    done
    say "PROBE TIMEOUT: no clean completion in window"
    return 1
}

start() {
    if [ -n "$(port_pid)" ]; then say "ERROR: something already answers on $PORT; refusing to start a second engine"; return 1; fi
    # A stale engine binary fails silently: the config's exe is whatever was last built, so a source edit that was
    # never compiled into it is simply absent from the run. Warn (never refuse) so it is never a mystery.
    sh "$ROOT/metal/tools/engine_freshness.sh" "$CFG" || true
    say "starting: serve/server.py --engine strata --config $CFG --port $PORT"
    cd "$ROOT" || return 1
    nohup "$PY" serve/server.py --engine strata --config "$CFG" --port "$PORT" >> "$RLOG" 2>&1 &
    SPID=$!
    wait_ready 600 || { tail -20 "$RLOG"; return 1; }
    # a start right after a boot can queue behind a big parked re-prefill (100k tokens is ~7 min): give the probe room
    probe_answer 600 || { tail -20 "$RLOG"; return 1; }
    say "serving again"
}

case "${1:-status}" in
    status) status ;;
    stop)   stop ;;
    start)  start ;;
    restart)
        stop || { say "stop failed; server left as is"; exit 2; }
        start || { say "START FAILED - retry manually: $0 start"; exit 3; }
        ;;
    setenv)
        shift
        if [ $# -lt 1 ]; then echo "usage: serve_huihui.sh setenv K=V [K=V...]   (K= clears, K- removes)"; exit 64; fi
        stop || { say "stop failed; config not touched, server left as is"; exit 2; }
        python3 - "$CFG" "$@" <<'PY' || { say "config edit failed; nothing started"; exit 4; }
import json, sys
p, toks = sys.argv[1], sys.argv[2:]
c = json.load(open(p))
env = c.setdefault("env", {})
for t in toks:
    if t.endswith("-") and "=" not in t:
        env.pop(t[:-1], None)
    elif "=" in t:
        k, v = t.split("=", 1)
        env[k] = v
    else:
        sys.exit(f"bad token: {t} (want K=V or K-)")
json.dump(c, open(p, "w"), indent=2)
print("[serve_huihui] env: " + ", ".join(f"{k}={v}" for k, v in sorted(env.items())))
PY
        start || { say "START FAILED with the new env - retry manually: $0 start"; exit 3; }
        ;;
    run)
        shift
        mkdir -p /tmp/bw
        stop || { say "stop failed; aborting test, server left as is"; exit 2; }
        ( "$@"; echo "[harness] test exit: $?" ) 2>&1 | tee /tmp/bw/test-output.log
        tesc=${PIPESTATUS[0]}
        start || { say "RESTART FAILED - retry manually: $0 start"; exit 3; }
        say "done (test exit $tesc)"
        ;;
    *) echo "usage: serve_huihui.sh {status|stop|start|restart|setenv K=V ...|run CMD...}"; exit 64 ;;
esac
