#!/usr/bin/env bash
# Blockwise combine is 1411.8us where bf16 is 169.0 for twice the bytes, and the deletion probes
# say it is none of the obvious things: the peer scale reads price at 0 (QNOSC 1413.1), the local
# quantise pass prices at 0 (NOQUANT 1412.7), and the fold prices at 89us (NOREDUCE 1322.8). So
# ~1250us sits in the gather, which is ~106 MB of peer reads -- about 85 GB/s against the 1256.7
# the bf16 gather reaches on the same fabric. That is a fallback, not a tuning gap.
#
# This run says WHICH path it is on and whether the knobs that shape the gather reach it at all.
# MORI_EP_TRACE_LAUNCH prints the symbol and the LDS request, which is what the tile paths'
# runtime budget checks turn on.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
# bf16 reference first: same trace, so the two symbols and LDS figures sit side by side.
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  BASE='MORI_EP_TRACE_LAUNCH=1' SPECS='bf16ref!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry|^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  BASE='MORI_EP_TRACE_LAUNCH=1' \
  SPECS='q_full!=; q_quad4!=MORI_COMB_QUAD=4; q_quad4s4!=MORI_COMB_QUAD=4 MORI_COMB_QSPLIT=4; q_tdm1!=MORI_COMB_TDM=1; q_fast0!=MORI_COMB_FASTPATH=0' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPATH_DONE"
