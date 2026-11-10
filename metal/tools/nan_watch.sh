#!/bin/sh
# nan_watch.sh - unattended #879 incident capture for the live slot (fork-local, beside serve_huihui.sh).
#
# Tails the serve log; on any line naming #879 (guard fire, #879s checkpoint instrument, flush, retry
# error): appends it with a timestamp to the incident ledger, snapshots the live auto-session with an
# APFS clone the first time in a burst (the flush deletes the poisoned session file; the quarantine lesson
# of 2026-11-09: copy it before the flush destroys the evidence), and shows one macOS notification per
# burst.  The engine already self-heals (STRATA_NAN_CKPT_RESUME); this only notices and preserves.
#
# LOG, SESS_DIR, LEDGER override for testing.  Keep at most KEEP session snapshots.
LOG=${LOG:-/Users/pc/src/Strata-fork/strata-ud-iq4_xs-huihui.log}
SESS_DIR=${SESS_DIR:-/Users/pc/src/Strata-data/sessions/ud-iq4_xs-huihui}
LEDGER=${LEDGER:-/Users/pc/src/Strata-data/nan-incidents.log}
KEEP=${KEEP:-8}

say() { printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LEDGER"; }

snap() {
    f="$SESS_DIR/auto-session"
    [ -f "$f" ] || return 0
    tag=$(date +%Y%m%d-%H%M%S)
    if cp -c "$f" "$f.incident-$tag" 2>/dev/null || cp "$f" "$f.incident-$tag" 2>/dev/null; then
        sz=$(ls -lh "$f.incident-$tag" | awk '{print $5}')
        say "nan-watch: snapshot auto-session -> auto-session.incident-$tag ($sz)"
    fi
    ls -t "$SESS_DIR"/auto-session.incident-* 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do rm -f "$old"; done
}

notify() {
    osascript -e "display notification \"$(printf '%s' "$1" | sed 's/"/'"'"'/g; s/^[^{]*serve: //; s/^strata //')\" with title \"Strata #879\" sound name Glass" 2>/dev/null
}

echo "nan-watch: tailing $LOG (ledger $LEDGER)"
say "nan-watch: started, tailing $LOG"
burst=0
last=0
tail -n 0 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"#879"*)
            now=$(date +%s)
            [ $((now - last)) -gt 120 ] && burst=0
            last=$now
            say "$line"
            if [ "$burst" -eq 0 ]; then
                snap
                notify "$line"
            fi
            burst=1
            ;;
    esac
done
