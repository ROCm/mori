#!/usr/bin/env bash
# full - noq at one geometry, with the rows written to the NODE as they land.
#
# Two sweeps in a row now have come back empty: the ssh wrapper collects the whole stream into a
# variable and only prints it when the remote side exits, so a script the remote `timeout` kills
# loses every row it had already produced. Nothing about the runs was wrong -- 55 and 25 minutes of
# GPU time were spent and reported nothing. tee to /tmp/qpre.log fixes that independently of how
# the script ends; tools/_ct_qpreread.sh gets it back.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
LOG=/tmp/qpre.log
: > "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1" | tee -a "$LOG"
date -u +'START %H:%M:%S' | tee -a "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=256 CWPB=16 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       ' | tee -a "$LOG"
date -u +'MID %H:%M:%S' | tee -a "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=512 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       ' | tee -a "$LOG"
date -u +'END %H:%M:%S' | tee -a "$LOG"
echo "QPRE5_DONE" | tee -a "$LOG"
