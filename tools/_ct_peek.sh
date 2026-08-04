#!/usr/bin/env bash
# Pull a pattern out of the last A/B sweep's FULL log (/tmp/aa1_full.log in the container).
# _ct_aa1.sh streams only a filtered view to the background log, so anything the filter does not
# name -- kernel printfs in particular -- exists only in the full file.
#   PAT   grep -E pattern            (default: CSPLIT)
#   NLINE how many matches to keep   (default: 40, taken from the END)
#   F     file, or a glob for several (default /tmp/aa1_full.log). ep_test.sh puts each spec's
#         python output in /tmp/ep_test_<spec>.log and only echoes the result row upward, so
#         kernel printfs are in THOSE, not in the sweep log.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
docker exec -e PAT="${PAT:-CSPLIT}" -e NLINE="${NLINE:-40}" -e F="${F:-/tmp/aa1_full.log}" \
  "$CTR" bash -lc '
  for f in $F; do
  [ -f "$f" ] || { echo "NO $f"; continue; }
  echo "== $f"
  echo "lines=$(wc -l < $f)  matches=$(grep -cE "$PAT" $f)"
  grep -E "$PAT" "$f" | tail -n "$NLINE"
  done
'
echo "PEEK_DONE"
