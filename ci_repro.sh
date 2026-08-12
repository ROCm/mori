#!/bin/bash
# Mirror the nightly "MORI-EP (intranode)" step exactly: same timeout, same -v.
# On a hang the timeout fires and we keep the tail, so the -v log tells us which
# test id was in flight -- that is the pair the report reconstructed by hand.
REPO=/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori
CONTAINER=mori_dbg_714
N=${1:-20}
mkdir -p /tmp/ci_repro
# The trace prefix is resolved inside the container, so the directory has to exist
# there; utils.py opens it outside its try block and a missing path kills every
# worker, which looks exactly like the hang we are hunting.
podman exec $CONTAINER mkdir -p /tmp/ci_repro

# The container imports mori from site-packages, and the JIT hashes and compiles
# mori/_jit-sources -- not this repo. Editing the repo alone changes nothing that
# runs on the GPU, and the cache key does not move either, so the stale kernel is
# silently reused. Push the device sources over before every run.
JITSRC=/opt/venv/lib/python3.12/site-packages/mori/_jit-sources
for f in intranode.hpp intranode_ll.hpp intranode_1250x.hpp; do
  podman exec $CONTAINER cp "$REPO/src/ops/dispatch_combine/$f" \
    "$JITSRC/src/ops/dispatch_combine/$f"
done
for i in $(seq 1 "$N"); do
  LOG=/tmp/ci_repro/run_$i.log
  podman exec -e PYTHONPATH=$REPO -e MORI_WORKER_TRACE=/tmp/ci_repro/wt_${i} $CONTAINER bash -c \
    "cd $REPO && timeout 360 pytest tests/python/ops/test_dispatch_combine_intranode.py -v" \
    > "$LOG" 2>&1
  RC=$?
  SUM=$(grep -E "^[0-9]+ passed|passed," "$LOG" | tail -1)
  SLOT=$(grep -c SLOTCHK "$LOG")
  if [ "$RC" -ne 0 ]; then
    echo "!!! HANG/FAIL iter=$i rc=$RC slotchk=$SLOT"
    echo "--- last 5 test lines ---"
    grep -E "PASSED|FAILED|ERROR" "$LOG" | tail -5
    echo "--- slotchk ---"
    grep SLOTCHK "$LOG" | head -10
  else
    echo "ok iter=$i slotchk=$SLOT :: $SUM"
  fi
done
echo "CI LOOP DONE"
