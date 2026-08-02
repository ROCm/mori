#!/usr/bin/env bash
# Tail of a specific ep_test.sh run log, for the case where the row said rc=1 and the greps in
# ep_test.sh matched none of the error shapes it knows about.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
TAG="${TAG:-full_b256}"
docker exec "$CTR" bash -lc "tail -40 /tmp/ep_test_${TAG}.log 2>/dev/null || echo '(no log for ${TAG})'"
echo "QLOG_DONE"
