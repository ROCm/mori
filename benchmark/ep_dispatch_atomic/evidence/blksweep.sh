#!/bin/bash
# If SlotReserve is contention on the single offTokOff word, its cost must scale
# with the NUMBER OF BLOCKS (each block issues npes atomics onto that word) and
# be nearly independent of anything else. Warps/block pinned at 8 throughout.
set -u
cd /app/mori/tests/python/ops/dispatch_combine_v2
export MORI_JIT_EXTRA_FLAGS=-DENABLE_PROFILER
export MORI_V2_EP_DISP_WARPS=8
for B in 8 16 32 64 128; do
  export MORI_V2_EP_DISP_BLOCKS=$B
  M=512 PAIRS=10 timeout 700 torchrun --standalone --nproc_per_node=4 /tmp/ept_prof.py \
      > /tmp/sweep_$B.log 2>&1
  echo "### blocks=$B  atomics_per_rank=$((B*4))"
  python3 - <<PY
import torch, statistics as st
MHZ=99.845; WPB=8; us=lambda t:t*1000.0/MHZ/1000.0
d=[]
for rk in range(4):
    b=torch.load(f"/tmp/eptrace_rank{rk}.pt").numpy().reshape(-1,2)
    per={}
    for ts,meta in b: per.setdefault(int(meta)>>16,[]).append((int(ts),(int(meta)>>2)&0x3FFF,int(meta)&3))
    for w,s in per.items():
        if w%WPB: continue
        launch,op=-1,{}
        for ts,slot,typ in s:
            if typ==0:
                if slot==0: launch+=1
                op[slot]=ts
            elif typ==1 and slot in op:
                t0=op.pop(slot)
                if slot==2 and launch>=1: d.append(us(ts-t0))
d.sort()
print(f"   SlotReserve  n={len(d):5d}  min {d[0]:6.2f}  p50 {d[len(d)//2]:6.2f}  "
      f"p95 {d[int(.95*len(d))]:6.2f}  max {d[-1]:6.2f} us")
PY
done
