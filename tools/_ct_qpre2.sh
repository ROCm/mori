#!/usr/bin/env bash
# Where the split-out quantise stands, priced with knobs that do NOT change the build.
#
# A new -D costs a full JIT rebuild of ep_intranode.hip, measured at ~890s on this node, so a sweep
# that touches one is four runs an hour. MORI_COMB_QPRE and its geometry are read in Python and
# emit no -D at all, so every row here shares one binary and one cache entry -- the whole script
# costs about as much as a single recompile would.
#
# First data point: QPRE=1 at the default 1024x8 pre-grid reads 1158.5us rc=0 at 64x8 EP4, against
# 1044.2 recorded for the inline pass at the same width. Correct, but not yet a win, and the two
# candidate reasons are the pre-grid and the per-thread __threadfence_system the pre-kernel ends
# on. This script settles the first one; only the geometry moves between rows.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
date -u +'START %H:%M:%S'
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='inline!=MORI_COMB_QPRE=0; g1024x8=MORI_COMB_QPRE=1; g256x8=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; g512x8=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=512; g128x8=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=128; g512x4=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=512 MORI_COMB_QPRE_WPB=4; g256x16=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256 MORI_COMB_QPRE_WPB=16' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRE2_HALF"
# The bar, on this tree: bf16 zero-copy PULL at the same width is the 1250 GB/s the quantised path
# has to beat. Same build, only the bench arguments differ.
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='bf16ref!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
date -u +'END %H:%M:%S'
echo "QPRE2_DONE"
