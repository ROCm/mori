#!/usr/bin/env bash
# The two blockwise fixes, separately and together. Both are supposed to be exactly
# result-preserving, so the check is ARMED on every blockwise row: a fast row with rc=1 is not a
# result. Against the 1409.5us baseline and the 169.1us bf16 reference.
#   QSTGU  scale blocks in flight in the local quantise pass (1 = the old dependent chain)
#   QWIDE  describe the fp8 peer read in 4-byte elements instead of 1-byte (0 = off)
#   s_nq   the pair with the quantise pass deleted, to read the gather on its own
#   b_*    does bf16 want the widening too, or is it only the 1-byte descriptor that was broken
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='s_base!=MORI_COMB_QSTGU=1 MORI_COMB_QWIDE=0; s_stg!=MORI_COMB_QSTGU=4 MORI_COMB_QWIDE=0; s_wide!=MORI_COMB_QSTGU=1 MORI_COMB_QWIDE=1; s_both!=; s_u7!=MORI_COMB_QSTGU=7; s_nq=MORI_COMB_NOQUANT=1' \
  ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## |^       '
docker exec "$CTR" bash -lc "cd $SRC && QT=none ZC=1 CBN=64 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='b_ref!=; b_wide!=MORI_COMB_QWIDE=2' ./tools/ep_test.sh" 2>&1 | grep -E '^  |^## geometry'
echo "QFIX_DONE"
