#!/usr/bin/env bash
# Smoke: does the split-out quantise kernel build, load, and produce the right answer?
#
# One run only. The previous sweep put twelve in one script and hit the remote timeout with the
# whole grep buffer unflushed, so nothing at all came back from 55 minutes of GPU time -- hence
# --line-buffered below and one spec per script until the cost per run is known again.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
date -u +'START %H:%M:%S'
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  MORI_EP_TRACE_LAUNCH=1 SPECS='pre!=MORI_COMB_QPRE=1' ./tools/ep_test.sh" 2>&1 |
  grep --line-buffered -E '^  |^## |^       '
date -u +'END %H:%M:%S'
echo "QPRE1_DONE"
