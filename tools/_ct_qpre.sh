#!/usr/bin/env bash
# Does splitting the blockwise quantise into its own kernel pay, and is it still right?
#
# The fused kernel has to launch the quantise at the gather's width. Deletion pricing says that is
# most of the cost: at 64x8 the pass is 408.2us and at 256 blocks 163.8, against 50us for the same
# 318 MB at the 6.3 TB/s a d2d copy gets. MORI_COMB_QPRE runs it as EpCombineQuantizeInputKernel at
# its own geometry and passes useExternalInpBuffer=0 to the combine launch so the inline arm goes.
#
# CORRECTNESS FIRST. Every '!' row runs with the bench check armed; rc=0 on the qpre rows is the
# precondition for reading any of the latencies, because the pass moved across a kernel boundary
# and the cross-card release moved with it (the pre-kernel ends on __threadfence_system).
#
# Rows: QPRE=0 is the inline reference on the same tree; nq holds the quantise out to price the
# gather alone; TDM=1 asks for whole-token 7168 B peer reads instead of the chunked 3584 B ones.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
# The 1250 GB/s bar, re-measured on this tree rather than quoted.
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='bf16ref!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRE_BAR"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='inline!=MORI_COMB_QPRE=0; pre!=MORI_COMB_QPRE=1; pre_nq=MORI_COMB_QPRE=1 MORI_COMB_NOQUANT=1; pre_t1!=MORI_COMB_QPRE=1 MORI_COMB_TDM=1; pre_t1_nq=MORI_COMB_QPRE=1 MORI_COMB_TDM=1 MORI_COMB_NOQUANT=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRE_HALF"
# Pre-pass width. Same -D set as 'pre' above, so these reuse its build and only the grid moves.
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='pre_g1=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=128 MORI_COMB_QPRE_WPB=4; pre_g2=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=512 MORI_COMB_QPRE_WPB=4; pre_g3=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256 MORI_COMB_QPRE_WPB=8; pre_g4=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=1024 MORI_COMB_QPRE_WPB=2' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRE_DONE"
