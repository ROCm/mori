#!/usr/bin/env bash
# Now that the quantise has its own kernel, what does the combine kernel -- which is nothing but
# the gather -- want for a width?
#
# It could not be asked before. The fused kernel's width was one number serving two phases with
# opposite appetites, so every earlier CBN sweep moved both at once and neither reading was clean.
# MEASURED at 64x8 EP4 fp8_blockwise, one binary, only the pre-grid moving: inline 1011.3us,
# split at 1024x8 1143.5, 512x8 940.7, 128x8 876.0, 512x4 860.4, 256x8 856.4. The bar on the same
# tree is bf16 zero-copy PULL at 169.2us / 1254.7 GB/s.
#
# 856 minus a quantise pass that priced at 163.8us when it was inline at this width leaves ~690us
# of gather for 106 MB, i.e. 154 GB/s, against 1255 for twice the bytes in bf16. That is the whole
# remaining gap, and it is what this sweep is aimed at. Pre-grid pinned at 256x8 throughout so the
# only thing moving is the gather's width.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
date -u +'START %H:%M:%S'
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='64 128 256 384 512' CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='pre!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
echo "QPRE3_HALF"
# Warps per block at the two widths the sweep above is likely to like. 16 warps doubles the LDS the
# PULL tiles want, so a row that does not fit falls back to the lane gather and reads slow rather
# than wrong -- the [EPLAUNCH] line says which one actually launched.
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBNS='128 256' CWPB=16 DBN=64 DWPB=8 WS=4 \
  MORI_EP_TRACE_LAUNCH=1 SPECS='pre_w16=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       '
date -u +'END %H:%M:%S'
echo "QPRE3_DONE"
