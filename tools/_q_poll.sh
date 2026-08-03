#!/usr/bin/env bash
# Everything the step runner has recorded so far, from the log that lives in the container
# filesystem rather than /tmp, plus the ep_test breadcrumb -- a "run" line with no matching "done"
# names the spec that was live when a node stopped answering.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "
  echo '== results =='
  grep -E '^###|^  [a-z]|^       ' $SRC/.q_decide.log 2>/dev/null | tail -60
  echo '== breadcrumb (tail) =='
  tail -12 $SRC/.ep_test_last 2>/dev/null
  echo \"== live: \$(pgrep -fc bench_dispatch_combine 2>/dev/null || echo 0) bench procs ==\"
"
echo "QPOLL_DONE"
