#!/bin/sh
# Warn when the engine binary a launch script is about to start is older than the C++ sources it was built from.
# The run scripts do not rebuild: they exec whatever the config's "exe" points at, so after a source edit they
# start the PREVIOUS build silently and none of the changes are in it - "but the fix is in the tree" costs a
# whole debugging session when the binary is stale.  Kernel files (.metal) are deliberately NOT checked: the
# engine JIT-compiles them from metal/kernels/ at every start, so editing one needs no rebuild.
#
# Usage: engine_freshness.sh <config.json>
# Never fails a launch: prints to stderr and exits 0 in every case, including when the check cannot run.
CFG=${1:-}
[ -n "$CFG" ] && [ -f "$CFG" ] || exit 0
ROOT=$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd) || exit 0
PY=${STRATA_FRESH_PY:-python3}
EXE=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("exe",""))' "$CFG" 2>/dev/null) || exit 0
[ -n "$EXE" ] && [ -f "$EXE" ] || exit 0
newest=$(find "$ROOT/metal/src" "$ROOT/metal/include" "$ROOT/include" "$ROOT/src" \
              -newer "$EXE" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.mm' -o -name '*.c' \) \
              2>/dev/null | head -3)
[ -n "$newest" ] || exit 0
printf 'ENGINE STALE: %s is older than its sources, so this launch will not have them:\n' "$EXE" >&2
printf '%s\n' "$newest" | while IFS= read -r f; do printf '    %s\n' "$f" >&2; done
printf '  rebuild first:  ninja -C %s\n' "$(dirname "$EXE")" >&2
exit 0
