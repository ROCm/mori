#!/usr/bin/env bash
# Read a run log mid-flight. LOG names it; CTR names the container.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
LOG="${LOG:-/tmp/q_decide.log}"
docker exec "$CTR" bash -lc "
  grep -E '^START|^END|^###|^  [a-z]|^       ' $LOG | tail -70
  echo \"--- \$(wc -l < $LOG) lines, \$(pgrep -fc bench_dispatch_combine 2>/dev/null || echo 0) bench procs live\"
"
echo "QPOLL_DONE"
