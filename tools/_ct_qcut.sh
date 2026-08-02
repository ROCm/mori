#!/usr/bin/env bash
# Three structurally different gathers land on the same number -- chunked (LDS 116 KB) 1411.5us,
# QUAD depth 4 (288 KB) 1413.4, whole-token chunks (231 KB) 1387.5 -- so the gather SHAPE is not
# what costs, and the earlier reading of NOREDUCE's remainder as "the peer reads" does not follow.
# Cut the kernel in half instead and see which half holds the time.
#   q_ponly     PUSHONLY: launch + local quantise/staging + barrier, no gather and no fold
#   q_ponly_nq  the same with the staging deleted too: launch + barrier alone
#   q_noroute   prices the per-token pointer/route setup
#   q_nq_chk!   NOQUANT with the CHECK ARMED. NOQUANT prices at 0us, which for a pass that reads
#               212 MB and writes 106 MB cannot be a cost -- it has to mean the pass is not
#               running. If deleting it still gives rc=0, that is proved, and it is also the
#               explanation for MORI_COMB_PULL=kernel being wrong: same staging arm.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='q_full!=; q_ponly=MORI_COMB_PUSHONLY=1; q_ponly_nq=MORI_COMB_PUSHONLY=1 MORI_COMB_NOQUANT=1; q_noroute=MORI_COMB_NOROUTE=1; q_nq_chk!=MORI_COMB_NOQUANT=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QCUT_DONE"
