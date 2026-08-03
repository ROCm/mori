#!/usr/bin/env bash
# What the quantised combine spends, phase by phase, against the 169.2us / 1254.7 GB/s bar.
#
# ORDERED BY BUILD KEY, not by interest. A new -D set costs a ~890s JIT rebuild on this node, so
# every phase below is one key and the phases are ordered so that the cheapest, most decisive ones
# finish first: this node has died mid-sweep four times, and a partial run has to still be worth
# having. Results are appended to $OUT on the node as they are produced.
#
# THE BUDGET THIS IS TESTING. Quantising moves 106 MB across the fabric instead of 212, but it adds
# a local pass that reads 212 MB of bf16 and writes 106 MB of fp8. At the 6.3 TB/s the d2d copy
# measures that pass is 50us, and 106 MB of peer reads at the 1254.7 GB/s bf16 gets is 85us -- so
# the whole idea is worth about 135us against 169.2, and only if BOTH halves run at their floor.
# It currently costs 367.6. Phase 1 says which half owns the 230us of overhead.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
OUT="${OUT:-/root/qbudget.txt}"
PH="${PH:-1 2 3}"
G="QT=fp8_blockwise ZC=0 WS=4 DBN=64 DWPB=8"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec "$CTR" bash -lc "date -u +'=== qbudget %F %H:%M:%S ===' >> $OUT"

# $1 = label, $2 = env prefix for ep_test.sh
row() {
  echo "-- $1"
  docker exec "$CTR" bash -lc "cd $SRC && $2 ./tools/ep_test.sh 2>&1 | grep -E '^  |^## geometry|^       ' | tee -a $OUT"
}

for p in $PH; do
case $p in

1)
# ---- PHASE 1: default build key. Correctness, the split, and the bar. No rebuild beyond the one
#      the default already needs, because QPRE=noq is read in Python and NOT a -D.
#      full  = quantise pre-kernel + gather.  noq = gather alone.  full - noq = the quantise pass.
echo "PHASE1"
row "p1 blockwise 256x16, correctness + split" \
  "$G CBN=256 CWPB=16 SPECS='full!=; noq=MORI_COMB_QPRE=noq'"
row "p1 blockwise 64x8, the geometry the complaint names" \
  "$G CBN=64 CWPB=8 SPECS='full!=; noq=MORI_COMB_QPRE=noq'"
row "p1 bf16 zero-copy PULL, the bar, same session" \
  "QT=none ZC=1 WS=4 DBN=64 DWPB=8 CBN=64 CWPB=8 SPECS='bar!='"
;;

2)
# ---- PHASE 2: MORI_COMB_SCPRE=0, one rebuild. Prices the scale-row prefetch honestly: same
#      source, same geometry, same session, one -D apart. Before this gate existed the only
#      available comparison was MORI_COMB_QNOSC, which deletes the scales and is wrong by
#      construction. Both rows have the check armed -- SCPRE=0 is the OLD code path, so if the
#      prefetch is wrong this is the row that still passes.
echo "PHASE2"
row "p2 prefetch off vs on, 256x16" \
  "$G CBN=256 CWPB=16 SPECS='scpre_off!=MORI_COMB_SCPRE=0; scpre_on!=MORI_COMB_SCPRE=1'"
row "p2 prefetch off vs on, gather alone, 256x16" \
  "$G CBN=256 CWPB=16 SPECS='goff=MORI_COMB_QPRE=noq MORI_COMB_SCPRE=0; gon=MORI_COMB_QPRE=noq MORI_COMB_SCPRE=1'"
;;

3)
# ---- PHASE 3: back on the default key, so free. Where each half wants to run. The two halves want
#      opposite widths, which is why they are separate kernels at all, and only the pre-kernel grid
#      has been swept (256x8 best of 1024/512/256/128). The GATHER's width at fp8 has not been:
#      every gather figure on record is at a width chosen for bf16, which moves twice the bytes per
#      descriptor and so has a different in-flight ceiling.
echo "PHASE3"
row "p3 gather width sweep, quantise deleted" \
  "$G CBNS='64 128 256 512' CWPB=8 SPECS='noq=MORI_COMB_QPRE=noq'"
row "p3 gather width sweep at 16 warps" \
  "$G CBNS='128 256 512' CWPB=16 SPECS='noq=MORI_COMB_QPRE=noq'"
;;

esac
done
echo "QBUDGET_DONE"
