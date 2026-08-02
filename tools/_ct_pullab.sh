#!/usr/bin/env bash
# "PULL + caller-owned input buffer": which route is correct, and what does it cost.
#
# Five specs at ZC=0, correctness check armed on every one (rc=0 IS the result being read):
#   push     PUSH baseline, the 318us / 667 GB/s default
#   host     stage on the host stream, then run the untouched zero-copy _p2p kernel
#   kern     kernel stages into combineInp itself -- the route that returned wrong results
#   kernf    same plus MORI_COMB_RELFENCE, the per-block release fence :693 left out
#   kernslow same as kern with the fast gather off, to say whether the bug is in the
#            staging or in the QUAD/TDM gather that only zero copy has ever exercised
# Then ZC=1 alone for the 169us / 1257 GB/s reference these are being compared against.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='push!=; host!=MORI_COMB_PULL=host; kern!=MORI_COMB_PULL=kernel; kernf!=MORI_COMB_PULL=kernel MORI_COMB_RELFENCE=1; kernslow!=MORI_COMB_PULL=kernel MORI_COMB_FASTPATH=0' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='zc1ref!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry|^       '
echo "PULLAB_DONE"
