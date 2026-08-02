#!/usr/bin/env bash
# Two questions blockwise still owes an answer to, both at zero code cost.
#
# 1. Is the local quantise pass still concurrency-starved? It is 410us at 64 blocks for 318 MB of
#    purely local traffic, while route A's host-side d2d moves 424 MB of the same kind in 67.7us
#    (~6.3 TB/s). 64 blocks x 8 warps is 512 warps on a machine with far more CUs than that, so
#    widening the grid is the cheapest test of whether depth or width is what is missing. full and
#    nq at each width, because the gather underneath moves with the grid too and only the
#    difference is this pass.
#
# 2. What is the peer-read ceiling as a function of READ SIZE at this exact geometry? epsim mode9
#    is the instrument the QUAD note already cites (2432 elems -> 801 GB/s, 4864 -> 1322, 7168 ->
#    1395), but every entry in it is a bf16 read. fp8's chunked read is 3584 B and its whole-token
#    read is 7168 B -- half and one quarter of those -- and nothing in the kernel A/B distinguishes
#    "the read is small" from "the loop is slow", because chunked, whole-token and QUAD all landed
#    within 4%. rdElems is in bf16 elems, so bytes are twice it: 896 -> 1792 B, 1792 -> 3584 B
#    (fp8 chunk), 3584 -> 7168 B (fp8 token, bf16 chunk), 7168 -> 14336 B (bf16 token).
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='64 128 256' CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=; nq=MORI_COMB_NOQUANT=1' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QGEO_HALF"
# _ct_epsim.sh drives docker itself, so it runs on the NODE and has to be staged there (-Aux).
CTR=$CTR NR=4 NT=4096 ITERS=50 MODES=9 GRIDS=64 BLOCK=256 ALLOC=vmm \
  RDSWEEP="896:4,1792:4,1792:8,3584:2,3584:4,7168:2" bash /tmp/_ct_epsim.sh 2>&1 | grep -E 'EPSIM|skip|COMPILE'
echo "QGEO_DONE"
