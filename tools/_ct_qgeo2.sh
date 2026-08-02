#!/usr/bin/env bash
# Blockwise at ITS OWN geometry, with the read size raised to match what the fabric will actually
# deliver. Two things landed at once:
#
#   Width. 64x8 is the bf16 ZC=1 tuned point and blockwise had never been measured anywhere else.
#   full/nq at 64/128/256 blocks: 1044.2/636.0, 581.7/348.4, 385.9/222.1. Both halves scale with
#   the grid, so 64 blocks was starving both, and at 256 the gather is at 222.1 - 16 barrier - fold
#   which is right on its ceiling below.
#
#   Read size. epsim mode9 at grid 64 block 256 (VMM/uncached, 4 peers) prices the peer read purely
#   by how many bytes one descriptor asks for: 1792 B 327 GB/s, 3584 B 620 (655 at 8 in flight),
#   7168 B 927 at 2 in flight and 1087 at 4, 14336 B 1389. fp8's chunked read is 3584 B because
#   MORI_COMB_TDM=2 halves a token that is already half of bf16's -- so quantising walked the read
#   down two steps of that curve. MORI_COMB_TDM=1 puts it back to a whole 7168 B token.
#
# LDS is why this is not free at every width: TDM=1 needs warps * 4 * 7168 B of tiles, 229 KB at 8
# warps against the 320 KB budget, so it fits at 8 and not at 14. Rows that do not fit fall back to
# the lane gather and will read slow rather than wrong -- the check is armed on every full row.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='256 384 512' CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=; tdm1!=MORI_COMB_TDM=1; tdm1_nq=MORI_COMB_TDM=1 MORI_COMB_NOQUANT=1; tdm1_u14!=MORI_COMB_TDM=1 MORI_COMB_QSTGU=14' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QGEO2_HALF"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=256 CWPB=14 DBN=64 DWPB=8 WS=4 \
  SPECS='w14!=; w14_nq=MORI_COMB_NOQUANT=1' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QGEO2_DONE"
