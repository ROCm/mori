# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Complete per-warp picture, split by the role a warp played in the launch.

DrainSpin and InboundWait are opened by EVERY warp (their MORI_TRACE_NEXT sits
outside the isFirstArriver block), but only the first arriver actually waits --
for every other warp those spans are just "ticket drawn -> fell out of the
kernel". The first arriver is identifiable without guessing: it is the only warp
that emits FenceAgent, which IS inside the conditional.
"""
import torch
import statistics as st

MHZ = 99.845
S = {
    0: "Setup",
    1: "Routing",
    2: "SlotReserve",
    3: "MetaStage",
    4: "MetaTdm",
    5: "PayloadTdm",
    6: "GridTicket",
    7: "DrainSpin",
    8: "FenceAgent",
    9: "FenceSignal",
    10: "InboundWait",
}
BEGIN, END = 0, 1


def us(t):
    return t * 1000.0 / MHZ / 1000.0


rows = []  # (rank, launch, warp, slot, dur_us, t0, t1)
for rk in range(4):
    b = torch.load(f"/tmp/eptrace_rank{rk}.pt").numpy().reshape(-1, 2)
    per = {}
    for ts, meta in b:
        per.setdefault(int(meta) >> 16, []).append(
            (int(ts), (int(meta) >> 2) & 0x3FFF, int(meta) & 3)
        )
    for w, ev in per.items():
        launch, opened = -1, {}
        for ts, slot, typ in ev:
            if typ == BEGIN:
                if slot == 0:
                    launch += 1  # each launch restarts at Setup
                opened[slot] = ts
            elif typ == END and slot in opened:
                t0 = opened.pop(slot)
                rows.append((rk, launch, w, slot, us(ts - t0), t0, ts))

rows = [r for r in rows if r[1] >= 1]  # drop launch 0: it carries the host-barrier skew
launches = sorted({(r[0], r[1]) for r in rows})
# the first arriver = the only warp emitting FenceAgent(8) in that launch
fa = {(r[0], r[1]): r[2] for r in rows if r[3] == 8}
print(f"ranks=4  launches/rank={len(launches)//4}  warps/launch=512  spans={len(rows)}")
print(f"first-arriver warp id per launch (first 8): {list(fa.values())[:8]}\n")


def show(title, pred):
    print(f"{title}")
    print(f"{'slot':14} {'n':>6} {'mean us':>9} {'p50':>8} {'p95':>8} {'max':>8}")
    print("-" * 58)
    for sid in sorted(S):
        v = sorted(r[4] for r in rows if r[3] == sid and pred(r))
        if not v:
            continue
        print(
            f"{S[sid]:14} {len(v):6d} {st.mean(v):9.3f} {v[len(v)//2]:8.3f} "
            f"{v[int(.95*len(v))]:8.3f} {v[-1]:8.3f}"
        )
    print()


show(
    "ALL 512 WARPS  (the population -- what every warp actually costs)", lambda r: True
)
show(
    "FIRST-ARRIVER WARP ONLY  (the critical path: it owns the waits)",
    lambda r: fa.get((r[0], r[1])) == r[2],
)
show(
    "ORDINARY WARPS  (never enter the drain/inbound waits)",
    lambda r: fa.get((r[0], r[1])) != r[2],
)

# per-launch kernel span from the trace itself
sp = []
for rk, ln in launches:
    sel = [r for r in rows if r[0] == rk and r[1] == ln]
    sp.append(us(max(r[6] for r in sel) - min(r[5] for r in sel)))
sp.sort()
print(
    f"kernel span per launch (all warps): mean {st.mean(sp):.2f} us  p50 {sp[len(sp)//2]:.2f}  max {sp[-1]:.2f}"
)
