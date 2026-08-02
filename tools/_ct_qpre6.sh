#!/usr/bin/env bash
# full - noq: how much of the split blockwise combine is the quantise kernel and how much the
# gather. Four runs, nothing else, because the last two attempts at a wide sweep ended with the
# card wedged.
#
# WHY IT WEDGED, since it cost two hours and the fix is a rule and not a knob: `timeout N bash -s`
# on the remote side kills the bash it started and nothing under it, so a sweep cut short leaves
# its ep_test.sh loop and its four bench processes running inside the container. The next sweep
# then puts a SECOND four-process job on the same four GPUs, both spin on their own cross-device
# barrier, neither can be scheduled to release the other, and every process lands in D state where
# even SIGKILL from the host will not touch it. It took docker stop plus rocm-smi --gpureset per
# device to get back to the 175 MB idle VRAM. Rule: one timing script in flight at a time, reap
# before starting, and keep every script comfortably inside its timeout.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
SRC="${SRC:-/root/mori_tdm}"
LOG=/tmp/qpre.log
pgrep -f 'bench_dispatch_combine|ep_test.sh' >/dev/null && { echo "BUSY: something is already running"; exit 1; }
: > "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1" | tee -a "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=256 CWPB=16 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       ' | tee -a "$LOG"
docker exec "$CTR" bash -lc "cd $SRC && QT=fp8_blockwise ZC=0 CBN=512 CWPB=8 DBN=64 DWPB=8 WS=4 \
  SPECS='full!=MORI_COMB_QPRE=1 MORI_COMB_QPRE_BN=256; noq=MORI_COMB_QPRE=noq' ./tools/ep_test.sh" 2>&1 |
  grep -E '^  |^## |^       ' | tee -a "$LOG"
echo "QPRE6_DONE" | tee -a "$LOG"
