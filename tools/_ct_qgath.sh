#!/usr/bin/env bash
# The blockwise gather, with the 778us local quantise pass held out by MORI_COMB_NOQUANT so the
# transport is not measured through it. Every row here is WRONG ON PURPOSE (NOQUANT folds whatever
# the previous launch left in combineInp); only the times mean anything.
#
# What is being asked: q_nq is 631.1us, of which the barrier is 15.8 and the fold ~113, so ~500us
# moves 106 MB = ~210 GB/s. bf16 moves twice those bytes at 1255 GB/s end to end. Half the payload
# cannot cost five times as much, so either the descriptor shape is wrong for a 1-byte element type
# or something other than the payload is in the loop.
#   g_chunk2   the shipping shape: MORI_COMB_TDM=2, 2 chunks x 4 sources per token
#   g_chunk1   whole-token chunks, half the descriptors, twice the row
#   g_quad4    QUAD depth 4, one descriptor per source per token, ring-buffered
#   g_quad4s2  QUAD depth 4 split 2, shorter reads but three in flight
#   g_pure     NOQUANT + QNOSC + NOREDUCE: reads issued and waited on, nothing folded
#   b_chunk2   bf16 on the SAME chunked shape, mori-owned buffer. The comparison that isolates the
#              element type: same descriptor count, same code, 2-byte elements instead of 1-byte.
#   b_quad     bf16 as shipped (QUAD), for the 169.1us reference
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  BASE='MORI_COMB_NOQUANT=1' \
  SPECS='g_chunk2=; g_chunk1=MORI_COMB_TDM=1; g_quad4=MORI_COMB_QUAD=4; g_quad4s2=MORI_COMB_QUAD=4 MORI_COMB_QSPLIT=2; g_pure=MORI_COMB_QNOSC=1 MORI_COMB_NOREDUCE=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='b_chunk2=MORI_COMB_QUAD=0; b_quad=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry'
echo "QGATH_DONE"
