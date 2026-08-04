#!/usr/bin/env bash
# First run of debug-aa on hardware: does it compile, and does the known-good bf16 case still hold.
#
# bf16 ZC=0 64x8 with the check armed is the reference this branch must not have broken -- 236.7us /
# 896.9 GB/s on record. Nothing here touches fp8, so if this row is wrong the cause is the branch
# itself and not the scale prefetch.
#
# Every knob is overridable, so this doubles as the generic way to reach ep_test.sh from the laptop:
#   _send_ct.ps1 -Script tools/_ct_aa1.sh -Envp "BASE=... SPECS=..." -Tmo 1800
# Send it through _send_ct.ps1 (base64) rather than a bare ssh line: PowerShell drops the inner
# quotes of a remote command, which turns SPECS="a=1; b=2" into separate words the shell then tries
# to run. Values with no spaces in them survive either way -- put the shared gates in BASE and keep
# each spec's gate list to a single token if a one-liner is unavoidable.
set -uo pipefail
C=${C:-MORI-EPV2}
# ep_test.sh separates specs with ';', which a remote command line reaching this node through
# PowerShell -> ssh would read as a command separator long before bash ever sees the assignment.
# Accept ',' as the separator too and translate here (_ct_comb.sh:74-76 hit the same wall).
export SPECS="$(printf '%s' "${SPECS:-bf16_zc0!=}" | tr ',' ';')"
export BASE="${BASE:-}"
export ZC="${ZC:-0}" QT="${QT:-none}"
export CBN="${CBN:-64}" CWPB="${CWPB:-8}" DBN="${DBN:-64}" DWPB="${DWPB:-8}" WS="${WS:-4}"
# Same reason as SPECS, different separator: ep_test.sh splits CBNS on whitespace, and -Envp cannot
# carry a space. CBNS=64,128,256 runs every spec once per block count.
export CBNS="$(printf '%s' "${CBNS:-}" | tr ',' ' ')"
export TAIL="${TAIL:-40}"  # accepted and ignored; see the grep filter below
# REV=<branch or sha> pulls the node's checkout up before running. This is the only deployment
# channel that leaves a number attributable to a commit -- copying files in once compiled a source
# the copy had never reached and printed a log identical to the pre-fix one, which reads as "the fix
# does not work" rather than "the fix is not here". ep_test.sh prints the resulting HEAD either way.
#
# The branch is fetched BY NAME into FETCH_HEAD, not through origin/<branch>. This clone is shallow
# with a single-branch refspec (+refs/heads/tdm-dispatch:...), so `git fetch origin` succeeds,
# returns 0, and still leaves origin/<anything-else> undefined; the reset then fails with git's
# usage text, which reads like a quoting bug, and the run measures the OLD binary while printing a
# HEAD that looks plausible. Always check the HEAD line below against what you pushed.
REV="${REV:-}"
[ -n "$REV" ] && docker exec "$C" bash -lc \
  "cd /root/mori_tdm && git fetch --quiet origin '$REV' && git reset --hard FETCH_HEAD 2>&1 | tail -1"

# RMCORE=1 clears the GPU core dumps a crashed 4-rank run leaves in the repo root. Five of them
# once filled 115 GB and took dockerd down with the disk, so they are worth sweeping rather than
# letting accumulate; the run itself sets `ulimit -c 0` so no new ones appear.
[ "${RMCORE:-0}" = 1 ] && docker exec "$C" bash -lc \
  'cd /root/mori_tdm && du -ch gpucore.*.gpu 2>/dev/null | tail -1; rm -f gpucore.*.gpu; df -h / | tail -1'

docker exec -e SPECS -e BASE -e ZC -e QT -e CBN -e CBNS -e CWPB -e DBN -e DWPB -e WS -e TAIL \
  "$C" bash -lc '
  cd /root/mori_tdm || exit 1
  echo "HEAD=$(git rev-parse --short HEAD)"
  echo "=== comb ${CBN}x${CWPB} disp ${DBN}x${DWPB} WS=$WS ZC=$ZC QT=$QT BASE=[$BASE] ==="
  # tee + line-buffered grep, NOT tail. tail cannot emit anything until the pipe closes, so under
  # _ct_bg.sh the log stays at three lines for the entire run and a finished sweep is indis-
  # tinguishable from a hung one -- which cost a round of diagnosis chasing a run that had already
  # written "done ... rc=0" to .ep_test_last two minutes earlier. The filter keeps the result rows,
  # the section headers and any ABORT; everything else is in the file named below.
  # No tail on the end of this either, for the same reason -- it would re-introduce the buffering
  # one stage later. The filter is what keeps the output small; TAIL is now only a compatibility
  # no-op for callers that still set it.
  ./tools/ep_test.sh 2>&1 | tee /tmp/aa1_full.log \
    | grep --line-buffered -E "^  |^## |^ABORT|^EP_TEST_DONE|^\[CSPLIT\] rank=0"
  echo "  (full output: /tmp/aa1_full.log in $C)"
'
echo "AA1_DONE"
