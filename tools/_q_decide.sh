#!/usr/bin/env bash
# The run to fire the moment a gfx1250 node is back. It answers, in one sitting, the question the
# fp8 combine has been stuck on: why does a gather that moves HALF the bytes take 2.5x longer than
# the bf16 one, invariant to every read-shape knob?
#
# Order matters -- the first rows are the ones a short window must not lose.
#
# 1. pipe rows FIRST, with the check armed. MORI_COMB_PIPE=2 now admits blockwise (the fold it was
#    excluded from dequantises), and that code has never run. rc=0 is the gate on believing the
#    rest of the row; overlap is the only structural knob blockwise never had.
# 2. dcast: fp8 transport with NO scales and no blockwise fold. If this lands near half the bf16
#    time, the fp8 gather is fine and everything lost is blockwise-specific, which contradicts the
#    73us that deletion pricing puts on the scale reads and means the deletion pricing is what is
#    wrong. If it is slow too, the loss is in the 1-byte tile path itself.
# 3. the deletion decomposition, re-taken on this node so nothing is compared across machines.
#
# CBN/CWPB are swept at the two points that matter: 64x8 (the bf16 tuned point, and the geometry
# the ask names) and 256x16 (the best blockwise has managed, 367.6us).
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
LOG="${LOG:-/tmp/q_decide.log}"

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
docker exec -d "$CTR" bash -lc "
  cd $SRC && : > $LOG
  echo \"START \$(date -u +%H:%M:%S)\" >> $LOG
  for geo in '64 8' '256 16'; do
    set -- \$geo
    echo \"### comb \$1 x \$2\" >> $LOG
    # 1. the new path, correctness first
    QT=fp8_blockwise ZC=0 CBN=\$1 CWPB=\$2 DBN=64 DWPB=8 WS=4 \
      SPECS='bwq_pipe2!=MORI_COMB_PIPE=2; bwq_base!=' ./tools/ep_test.sh >> $LOG 2>&1
    # 2. fp8 with no scales at all
    QT=fp8_direct_cast ZC=0 CBN=\$1 CWPB=\$2 DBN=64 DWPB=8 WS=4 \
      SPECS='dcast!=' ./tools/ep_test.sh >> $LOG 2>&1
    # 3. where the time goes, on this node
    QT=fp8_blockwise ZC=0 CBN=\$1 CWPB=\$2 DBN=64 DWPB=8 WS=4 \
      SPECS='bwq_gather=MORI_COMB_QPRE=noq; bwq_nosc=MORI_COMB_QNOSC=1; bwq_nored=MORI_COMB_NOREDUCE=1; bwq_pure=MORI_COMB_QPRE=noq MORI_COMB_QNOSC=1 MORI_COMB_NOREDUCE=1' \
      ./tools/ep_test.sh >> $LOG 2>&1
    # 4. the bar, same node same session
    QT=none ZC=1 CBN=\$1 CWPB=\$2 DBN=64 DWPB=8 WS=4 SPECS='bf16_zc1!=' ./tools/ep_test.sh >> $LOG 2>&1
    QT=none ZC=0 CBN=\$1 CWPB=\$2 DBN=64 DWPB=8 WS=4 SPECS='bf16_zc0!=' ./tools/ep_test.sh >> $LOG 2>&1
  done
  echo \"END \$(date -u +%H:%M:%S)\" >> $LOG
"
echo "LAUNCHED -> $LOG (about 25 min; poll with _q_poll.sh)"
