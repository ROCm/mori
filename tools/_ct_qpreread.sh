#!/usr/bin/env bash
# Read back whatever the last _ct_qpre*.sh got as far as, whether or not it was allowed to finish.
set -uo pipefail
echo "--- /tmp/qpre.log ---"
cat /tmp/qpre.log 2>/dev/null || echo "(no log)"
echo "--- running now ---"
pgrep -af 'bench_dispatch_combine|ep_test.sh' | head -5 || echo "(nothing)"
echo "QPREREAD_DONE"
