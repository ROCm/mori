#!/usr/bin/env bash
# Split the 367.6us into its two kernels, without paying for a rebuild.
#
# MORI_COMB_QPRE=noq takes the split path and then never launches the pre-kernel, so the combine
# kernel is the gather and nothing else. full - noq is the quantise pass; noq is the gather. Both
# rows share the binary every row so far has used.
#
# What the answer has to beat: 169.2us / 1254.7 GB/s, bf16 zero-copy PULL at 64x8 on this tree.
# 367.6 is 2.2x that. Whichever of the two halves is the larger is where the next change goes, and
# guessing between them is exactly what the cache-key failure earlier in this work punished.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
date -u +'START %H:%M:%S'
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='256 512' CWPB=16 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       '
echo "QPRE4_HALF"
# Same two rows at 8 warps, where the PULL tiles are half the LDS and twice as many blocks fit per
# CU. Which of width and warp count the gather actually wants is not decided by the rows above.
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='512 1024' CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       '
date -u +'END %H:%M:%S'
echo "QPRE4_DONE"
