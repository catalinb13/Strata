#!/bin/bash
# metal/tools/switch_to_strata.sh - move this Mac from sushi (port 12345) to the Strata Huihui
# server (port 8180) in one step.
#
# Both engines are ~50 GB Metal loads and cannot be resident at once, so switching means
# fully stopping sushi (and waiting for its pages to return) before strata boots.
#
#   switch_to_strata.sh              stop sushi, start strata on 8180, health-check it
#   switch_to_strata.sh --fresh      also move the saved session aside first (do this after a "!" corruption:
#                                    the saved history is full of "!" and the fixed model would imitate it)
#   switch_to_strata.sh --to-sushi   the reverse: stop strata, restart sushi on 12345
#
# The strata start uses serve_huihui.sh, which waits for the model to load AND for one clean
# completion (a run of "!" is treated as the decode bug, not a live answer). If strata will not
# come up clean, sushi is NOT restarted - run --to-sushi to get the assistant back.
#
# Run this yourself in a terminal (it stops the model you may be talking through).
set -u
ROOT=/Users/pc/src/Strata-fork
HARNESS="$ROOT/metal/tools/serve_huihui.sh"
SUSHI_PORT=${SUSHI_PORT:-12345}
SUSHI_MODEL=${SUSHI_MODEL:-/Users/pc/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}
SUSHI_ARGS=${SUSHI_ARGS:-"--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 250000 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1"}

say() { echo "[switch $(date +%H:%M:%S)] $*"; }
port_pid() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }
free_gb() { vm_stat | awk '/page size of/{for(i=1;i<=NF;i++) if($i=="of") ps=$(i+1)} /Pages free/{f=$3} /Pages speculative/{s=$3} END{if(!ps) ps=4096; gsub(/\./,"",f); gsub(/\./,"",s); printf "%.1f", (f+s+0)*ps/1073741824}'; }

# Wait until a stopped engine's pages are actually back, so the next ~50 GB load does not start
# against the old process still draining. Returns once free memory stops climbing or crosses a floor.
wait_ram() {
    local floor=${1:-25} prev=0 now stable=0
    say "waiting for memory to free (need ${floor} GB)..."
    for _ in $(seq 90); do
        now=$(free_gb)
        if awk -v n="$now" -v f="$floor" 'BEGIN{exit !(n>=f)}'; then say "free: ${now} GB"; return 0; fi
        if awk -v n="$now" -v p="$prev" 'BEGIN{exit !(n>=p)}'; then stable=$((stable+1)); else stable=0; fi
        [ "$stable" -ge 6 ] && { say "free steady at ${now} GB (below floor; proceeding)"; return 0; }
        prev=$now; sleep 1
    done
    say "free: $(free_gb) GB after wait"
}

stop_sushi() {
    local p; p=$(port_pid "$SUSHI_PORT")
    if [ -z "$p" ]; then say "sushi already down"; return 0; fi
    if ! ps -o comm= -p "$p" 2>/dev/null | grep -q sushi; then
        say "ERROR: port $SUSHI_PORT held by '$(ps -o comm= -p "$p" 2>/dev/null)', not sushi; refusing to kill it"; return 1
    fi
    say "stopping sushi pid $p"
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 40); do [ -z "$(port_pid "$SUSHI_PORT")" ] && break; sleep 0.5; done
    p=$(port_pid "$SUSHI_PORT")
    if [ -n "$p" ]; then say "still up after 20s: KILL $p"; kill -KILL "$p" 2>/dev/null; sleep 2; fi
    say "sushi down; free: $(free_gb) GB"
}

clear_strata_session() {  # --fresh: move the auto-session aside (reversible) so the fixed engine does not
    # replay a history already full of "!" and imitate it back. Renames, never deletes.
    local d=/Users/pc/src/Strata-data/sessions f
    for f in "$d"/*uihui*/auto-session; do
        [ -e "$f" ] || continue
        mv "$f" "$f.broken-$(date +%Y%m%d-%H%M%S)" 2>/dev/null && say "moved poisoned session aside: $f -> $f.broken-*"
    done
}

start_sushi() {
    local p; p=$(port_pid "$SUSHI_PORT")
    if [ -n "$p" ]; then say "sushi already answering on $SUSHI_PORT (pid $p)"; return 0; fi
    command -v sushi >/dev/null || { say "ERROR: sushi not on PATH"; return 1; }
    say "starting sushi (model $SUSHI_MODEL)"
    ( setsid sushi serve --model "$SUSHI_MODEL" $SUSHI_ARGS >/tmp/bw/sushi.out 2>&1 & ) 2>/dev/null \
        || nohup sushi serve --model "$SUSHI_MODEL" $SUSHI_ARGS >/tmp/bw/sushi.out 2>&1 &
    for _ in $(seq 120); do [ -n "$(port_pid "$SUSHI_PORT")" ] && { say "sushi up on $SUSHI_PORT"; return 0; }; sleep 2; done
    say "WARN: sushi not listening after 240s; check /tmp/bw/sushi.out"
    return 1
}

mkdir -p /tmp/bw

TO_SUSHI=0; FRESH=0
for a in "$@"; do
    case "$a" in
        --to-sushi) TO_SUSHI=1 ;;
        --fresh)    FRESH=1 ;;
        *) say "ignoring unknown argument: $a" ;;
    esac
done

if [ "$TO_SUSHI" = 1 ]; then
    say "switching to sushi"
    "$HARNESS" stop || { say "strata stop failed"; exit 1; }
    wait_ram 25
    start_sushi || exit 1
    exit 0
fi

say "switching to strata on 8180"
stop_sushi || exit 1
[ "$FRESH" = 1 ] && clear_strata_session
wait_ram 40                     # strata wants its arena free before it wires
exec "$HARNESS" start
