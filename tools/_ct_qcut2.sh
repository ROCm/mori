#!/usr/bin/env bash
# Re-run of the blockwise deletion pricing, now that the build cache is keyed off the real -D list.
# The previous pass has to be thrown away wherever NOQUANT appears in it: PUSHONLY+NOQUANT read
# 795.4us against PUSHONLY's 796.2, which is not a small effect, it is the same binary twice.
#
# What is still trustworthy from it, because those gates were always in the key:
#   q_full 1412.2   q_ponly 796.2   q_noroute 1412.5
# so over half the blockwise kernel is spent before a single peer byte is read. This pass asks what
# that 796 is made of and re-prices the rest honestly.
#   q_nq        NOQUANT alone: the local quantise/stage pass, full minus this
#   q_ponly_nq  PUSHONLY+NOQUANT: launch + barrier with nothing in between
#   q_nored     NOREDUCE: the fp32 dequantise fold
#   q_nosc      QNOSC: folds with scale 1.0, i.e. prices the uncached peer scale reads
#   ref_bf16    the number blockwise has to beat, same geometry, mori-owned buffer
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='q_full!=; q_nq=MORI_COMB_NOQUANT=1; q_ponly_nq=MORI_COMB_PUSHONLY=1 MORI_COMB_NOQUANT=1; q_nored=MORI_COMB_NOREDUCE=1; q_nosc=MORI_COMB_QNOSC=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='ref_bf16!=' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry'
echo "QCUT2_DONE"
