#!/usr/bin/env bash
# Two questions in one session, because the ssh handshake to this node costs more than the runs.
#
# 1) Where do blockwise combine's 1410.9us go? Deletion method at 64x8 ZC=0, QPULL on by default.
#    Only q_full carries the correctness check; the rest are wrong on purpose, read as full-minus.
#      q_full    reference, rc=0
#      q_nosc    fold with scale 1: prices the UNCACHED PEER scale reads in the innermost fold
#      q_noquant skip the local quantise/staging pass: prices producing the fp8 at all
#      q_nored   fold one tile instead of all sources: prices the fold arithmetic
# 2) Why is MORI_COMB_PULL=kernel wrong when =host is right? "both" does the host copy AND leaves
#    the in-kernel staging on, so the staging loop is the only live difference. Pass means the
#    loop is innocent and the flag drags something else in with it; fail means the loop corrupts.
#    dflt re-reads the new default (host) now that it is the default rather than a gate.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='dflt!=; both!=MORI_COMB_PULL=both' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='q_full!=; q_nosc=MORI_COMB_QNOSC=1; q_noquant=MORI_COMB_NOQUANT=1; q_nored=MORI_COMB_NOREDUCE=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRICE_DONE"
