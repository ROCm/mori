#!/bin/bash
# Reproduce the intranode hang and capture it while it is still live.
#
# The pytest timeout is deliberately long: the earlier 360s CI timeout killed the
# workers before they could be inspected. A host-side stall detector watches the
# -v log instead, and once no test has completed for STALL_S it dumps, for every
# worker, which GPU wave is still resident and what the dispatch epilogue's peer
# signal slots contain. That is the pair (waiting rank, missing peer) we need.
REPO=/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori
CONTAINER=mori_dbg_714
N=${1:-10}
STALL_S=${2:-90}
OUT=/tmp/hang_capture
mkdir -p $OUT
podman exec $CONTAINER mkdir -p $OUT

dump_waves() {
  local tag=$1
  podman exec $CONTAINER bash -c '
    for P in $(ps -eo pid,etimes,stat,comm --sort=etimes \
               | awk "\$2<3000 && \$3 ~ /R/ && \$4==\"python\" {print \$1}" | head -8); do
      echo "########## worker pid=$P ##########"
      timeout 90 rocgdb -p $P -batch \
        -ex "set pagination off" \
        -ex "info threads" 2>&1 \
        | grep -E "AMDGPU Wave" | head -5
      # The one resident wave is block 0 / wave 0 of the dispatch epilogue; report
      # which lanes are still spinning and what each lane observes.
      W=$(timeout 90 rocgdb -p $P -batch -ex "set pagination off" -ex "info threads" 2>&1 \
          | grep -E "AMDGPU Wave" | head -1 | awk "{print \$1}" | tr -d "*")
      if [ -n "$W" ]; then
        timeout 90 rocgdb -p $P -batch \
          -ex "set pagination off" \
          -ex "thread $W" \
          -ex "p/x \$exec" \
          -ex "p/x \$v6" -ex "p/x \$v7" -ex "p/d \$v3" \
          -ex "x/8xw \$v6" 2>&1 \
          | grep -vE "^\[New LWP|^\[Thread deb|^GNU gdb|^Copyright|^License|^This |^There |^Type |^For |^Find |^ +<http" | head -25
      fi
    done' > "$OUT/${tag}.dump" 2>&1
  echo "  dumped -> $OUT/${tag}.dump"
}

for i in $(seq 1 "$N"); do
  LOG=$OUT/run_$i.log
  podman exec -e PYTHONPATH=$REPO -e MORI_WORKER_TRACE=$OUT/wt_${i} $CONTAINER bash -c \
    "cd $REPO && timeout 1800 pytest tests/python/ops/test_dispatch_combine_intranode.py -v" \
    > "$LOG" 2>&1 &
  RUNPID=$!

  last_n=-1; same=0; hung=0
  while kill -0 $RUNPID 2>/dev/null; do
    sleep 10
    n=$(grep -cE "PASSED|FAILED|SKIPPED" "$LOG" 2>/dev/null)
    if [ "$n" = "$last_n" ]; then
      same=$((same + 10))
    else
      same=0; last_n=$n
    fi
    if [ "$same" -ge "$STALL_S" ]; then
      echo "!!! iter=$i STALLED after $n tests (${same}s no progress)"
      echo "    in flight: $(tail -1 "$LOG" | sed 's/.*intranode.py:://')"
      dump_waves "hang_$i"
      hung=1
      podman exec $CONTAINER bash -c "pkill -f 'pytest tests/python'" 2>/dev/null
      break
    fi
  done
  wait $RUNPID 2>/dev/null
  [ "$hung" = "1" ] && { echo "captured on iter $i"; break; }
  echo "ok iter=$i :: $(grep -E 'passed' "$LOG" | tail -1)"
done
echo "CAPTURE LOOP DONE"
