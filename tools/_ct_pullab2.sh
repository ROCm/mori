#!/usr/bin/env bash
# "PULL + caller-owned input buffer", route A against route B, RE-RUN.
#
# The previous verdict on route B has to be withdrawn, not because B started working but because
# the experiment that rejected the release-fence explanation never ran: MORI_COMB_RELFENCE emitted
# a -D that the build cache key did not name, so the "with fence" build was the "without fence"
# build's .hsaco. It read rc=1 because it WAS the failing binary.
#
# Check armed on every row. A fast row with rc=1 is not a result.
#   p_push    MORI_COMB_PULL=off, the PUSH transport, for the number this has to beat
#   p_host    route A: d2d on the caller's stream, then the ordinary zero-copy _p2p kernel
#   p_kern    route B: the kernel stages into the registered buffer itself
#   p_kernf   route B with a per-block __threadfence_system() before the cross-device barrier.
#             The release side of that barrier fences on block 0's first worldSize threads only
#             (:738), so another block's stores can still be in flight when the peer flag goes up.
#   p_kernb   route B with 15 extra barriers after the staging. This is the discriminator: if B
#             fails on a visibility RACE, buying it tens of microseconds and fifteen more fences
#             should change the outcome; if B fails on a LAYOUT mismatch, it fails identically.
#   p_zc1     true zero copy, the correctness and speed reference both routes are chasing
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='p_push!=MORI_COMB_PULL=off; p_host!=MORI_COMB_PULL=host; p_kern!=MORI_COMB_PULL=kernel; p_kernf!=MORI_COMB_PULL=kernel MORI_COMB_RELFENCE=1; p_kernb!=MORI_COMB_PULL=kernel MORI_COMB_BARRIER2=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='p_zc1!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry'
echo "PULLAB2_DONE"
