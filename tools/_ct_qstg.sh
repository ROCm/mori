#!/usr/bin/env bash
# A/B the blockwise quantise/stage pass over MORI_COMB_QSTGU, the number of scale blocks a subwarp
# keeps in flight. 1 is the old one-load-at-a-time chain, which prices at 778us of a 1409.5us
# combine. Every value computes the same bytes, so the check is armed on all of them: a row that
# is fast and rc=1 is not a result.
#
# 56 scale blocks, 2 subwarps per warp, so kStep = 2*QSTGU and 1/2/4/7/14 all divide 28 evenly --
# no value here is paying for a ragged tail.
#   s_nq   the same build with the pass deleted, to confirm the gather underneath did not move
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='s_u1!=MORI_COMB_QSTGU=1; s_u2!=MORI_COMB_QSTGU=2; s_u4!=MORI_COMB_QSTGU=4; s_u7!=MORI_COMB_QSTGU=7; s_u14!=MORI_COMB_QSTGU=14; s_nq=MORI_COMB_NOQUANT=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QSTG_DONE"
