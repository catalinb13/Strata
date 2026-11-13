#!/usr/bin/env bash
# Final verification sequence for the fp16 KV work:
#   1. wait for the safe sweep's python driver to exit (no self-match: pattern requires "python")
#   2. fp16 batch mode (--parallel 2, two concurrent requests) -> the only init_slots path
#   3. fp16 conversation parking (two turns, one conversation)
#   4. restore the live 8180 server
#   5. print the final comparison table
set -u
cd /Users/pc/src/Strata-fork

echo "=== 1. waiting for the safe sweep driver ==="
while pgrep -f "python.*gap/tune/fp16_vs_int8[.]py safe" > /dev/null; do sleep 5; done
echo "safe sweep done:"
tail -8 /tmp/kvab_safe4.log

echo
echo "=== 2. fp16 BATCH mode (--parallel 2) ==="
bash gap/tune/fp16_batch_test.sh 2>&1 | tail -20

echo
echo "=== 3. fp16 conversation parking ==="
bash gap/tune/fp16_park_test.sh 2>&1 | tail -20

echo
echo "=== 4. restoring the live server ==="
pkill -x strata 2>/dev/null
sleep 2
HUIHUI_CFG=/Users/pc/src/Strata-fork/strata-ud-iq4_xs-huihui.json bash metal/tools/serve_huihui.sh restart 2>&1 | tail -6

echo
echo "=== 5. final comparison table ==="
.venv/bin/python gap/tune/kv_report.py
echo "ALL DONE"
