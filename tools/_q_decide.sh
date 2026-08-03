#!/usr/bin/env bash
# ONE spec per invocation, appended to a log inside the container filesystem so it survives the
# node. STEP picks which.
#
# It is a step runner rather than a 25-minute sweep because f01-1 died in the middle of one and
# took the whole queue with it, leaving no way to say which spec was running -- and f01-2 died the
# same way earlier. Until something is known about what kills these boxes, no run may be longer
# than the evidence it produces.
#
# What the steps answer, in the order a short window should spend itself:
#   1 bf16_zc1   the bar, and a health check: it is the one row known to pass on a fresh node
#   2 bwq_pipe2  MORI_COMB_PIPE=2, which now admits blockwise. NEVER RUN. Check armed: rc is the
#                gate on believing the time.
#   3 bwq_base   blockwise as it ships, same geometry, same session
#   4 dcast      fp8 with NO scales and no blockwise fold. Near half the bf16 time here means the
#                fp8 transport is fine and the loss is blockwise-specific -- which contradicts the
#                73us that deletion pricing puts on the scale reads, and would mean the deletion
#                pricing is what is wrong.
#   5 bwq_gather the quantise pass deleted (QPRE=noq): the gather alone
#   6 bwq_pure   gather with the fold and the scale reads deleted too: transport alone. This is the
#                number the whole question turns on -- 106 MB against a fabric that does 1.4 TB/s.
#   7 bf16_zc0   the caller-owned-buffer path, which is the other half of the ask
#
# CBN/CWPB default to the 64x8 the ask names; re-run the same steps at 256 16 afterwards.
set -uo pipefail
CTR="${CTR:-MORI-F1}"
SRC="${SRC:-/root/mori_tdm}"
LOG="${LOG:-$SRC/.q_decide.log}"
STEP="${STEP:-1}"
CBN="${CBN:-64}"
CWPB="${CWPB:-8}"

case "$STEP" in
  1) QT=none;            ZC=1; SPECS='bf16_zc1!=' ;;
  2) QT=fp8_blockwise;   ZC=0; SPECS='bwq_pipe2!=MORI_COMB_PIPE=2' ;;
  3) QT=fp8_blockwise;   ZC=0; SPECS='bwq_base!=' ;;
  4) QT=fp8_direct_cast; ZC=0; SPECS='dcast!=' ;;
  5) QT=fp8_blockwise;   ZC=0; SPECS='bwq_gather=MORI_COMB_QPRE=noq' ;;
  6) QT=fp8_blockwise;   ZC=0; SPECS='bwq_pure=MORI_COMB_QPRE=noq MORI_COMB_QNOSC=1 MORI_COMB_NOREDUCE=1' ;;
  7) QT=none;            ZC=0; SPECS='bf16_zc0!=' ;;
  *) echo "unknown STEP=$STEP"; exit 1 ;;
esac

docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
echo "== step $STEP: QT=$QT ZC=$ZC $CBN x $CWPB  SPECS='$SPECS' =="
docker exec "$CTR" bash -lc "
  cd $SRC
  echo \"### step $STEP  $(date -u +%FT%TZ)  QT=$QT ZC=$ZC ${CBN}x${CWPB}\" >> $LOG
  QT=$QT ZC=$ZC CBN=$CBN CWPB=$CWPB DBN=64 DWPB=8 WS=4 SPECS='$SPECS' ./tools/ep_test.sh 2>&1 |
    tee -a $LOG | grep -E '^  |^## HEAD|^       '
  echo \"### step $STEP end \$(rocm-smi --showmeminfo vram 2>/dev/null | grep -c 'VRAM Total Used') cards report\" >> $LOG
"
echo "Q_STEP_${STEP}_DONE"
