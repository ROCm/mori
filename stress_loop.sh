#!/bin/bash
# Repeatedly run the intranode stress loop across a few shapes, flagging hangs
# (timeout) and any SLOTCHK hit from the in-kernel slot-accounting probe.
REPO=/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori
N=${1:-40}
for i in $(seq 1 "$N"); do
  for KT in IntraNodeLL IntraNode; do
    for TOK in 1 8 128 4096; do
      OUT=$(podman exec -e PYTHONPATH=$REPO mori_dbg_714 bash -c \
        "cd $REPO && timeout 120 python tests/python/ops/bench_dispatch_combine.py \
         --cmd stress --kernel-type $KT --world-size 8 --max-tokens $TOK --hidden-dim 4096 2>&1" )
      RC=$?
      LAST=$(echo "$OUT" | grep -oE 'Round [0-9]+ begin' | tail -1)
      SLOT=$(echo "$OUT" | grep -c 'SLOTCHK')
      if [ "$RC" -ne 0 ] || [ "$SLOT" -ne 0 ]; then
        echo "!!! iter=$i kernel=$KT tokens=$TOK rc=$RC slotchk=$SLOT last=$LAST"
        echo "$OUT" | grep -E 'SLOTCHK|Error|error|Traceback' | head -20
      else
        echo "ok iter=$i kernel=$KT tokens=$TOK last=$LAST"
      fi
    done
  done
done
echo "LOOP DONE"
