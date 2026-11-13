#!/usr/bin/env bash
# Runs everything left after the fp16 sweep, in order, unattended:
#   1. wait for the sweep driver to exit (pattern cannot match this shell)
#   2. fp16 batch mode  (--parallel 2)   -> exercises init_slots
#   3. fp16 parking     (two turns)      -> exercises window_pools
#   4. restart the live 8180 server
#   5. print the final comparison table
set -u
cd /Users/pc/src/Strata-fork

echo "=== 1. waiting for the sweep driver ==="
while pgrep -f "python.*fp16_vs_int8[.]py all" > /dev/null; do sleep 10; done
echo "sweep finished:"
tail -6 /tmp/kvab_real2.log

echo
echo "=== 2. fp16 BATCH mode (--parallel 2) ==="
bash gap/tune/fp16_batch_test.sh 2>&1 | tail -22

echo
echo "=== 3. fp16 conversation parking ==="
bash gap/tune/fp16_park_test.sh 2>&1 | tail -22

echo
echo "=== 4. restoring the live server ==="
pkill -x strata 2>/dev/null
sleep 2
HUIHUI_CFG=/Users/pc/src/Strata-fork/strata-ud-iq4_xs-huihui.json bash metal/tools/serve_huihui.sh restart 2>&1 | tail -6

echo
echo "=== 5. final comparison table ==="
.venv/bin/python gap/tune/kv_report.py
echo "ALL DONE"
