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
"""Stock (S) vs front rendezvous (F), same instrumented header, per slot.

FrontRdv's marker sits OUTSIDE the `#if MORI_EP_VA_FRONT` block, so both arms
emit it: in S it must be ~0 (nothing between it and MetaStage), in F it is the
rendezvous. That makes it the check that arm F really compiled the new path.
"""
import statistics as st
import torch

MHZ = 99.845
S = {
    0: "Setup",
    1: "Routing",
    2: "SlotReserve",
    11: "FrontRdv",
    3: "MetaStage",
    4: "MetaTdm",
    5: "PayloadTdm",
    6: "GridTicket",
    7: "DrainSpin",
    8: "FenceAgent",
    9: "FenceSignal",
    10: "InboundWait",
}
ORDER = [0, 1, 2, 11, 3, 4, 5, 6, 7, 8, 9, 10]


def us(t):
    return t / MHZ


def load(tag):
    rows, spans = [], []
    for rk in range(4):
        b = torch.load(f"/tmp/eptrace{tag}_rank{rk}.pt").numpy().reshape(-1, 2)
        per = {}
        for ts, meta in b:
            per.setdefault(int(meta) >> 16, []).append(
                (int(ts), (int(meta) >> 2) & 0x3FFF, int(meta) & 3)
            )
        lo, hi = {}, {}
        for w, ev in per.items():
            launch, opened = -1, {}
            for ts, slot, typ in ev:
                if typ == 0:
                    if slot == 0:
                        launch += 1
                    opened[slot] = ts
                    lo[launch] = min(lo.get(launch, ts), ts)
                elif typ == 1 and slot in opened:
                    t0 = opened.pop(slot)
                    rows.append((rk, launch, w, slot, us(ts - t0)))
                    hi[launch] = max(hi.get(launch, ts), ts)
        # launch 0 carries the host-barrier skew
        spans += [us(hi[ln] - lo[ln]) for ln in lo if ln >= 1 and ln in hi]
    rows = [r for r in rows if r[1] >= 1]
    fa = {(r[0], r[1]): r[2] for r in rows if r[3] == 8}
    return rows, spans, fa


def slot_stats(rows, pred):
    out = {}
    for sid in ORDER:
        v = sorted(r[4] for r in rows if r[3] == sid and pred(r))
        if v:
            out[sid] = (len(v), st.mean(v), v[len(v) // 2], v[int(0.95 * len(v))])
    return out


rS, spS, faS = load("S")
rF, spF, faF = load("F")
print(f"launches/rank  S={len(spS)//4}  F={len(spF)//4}   (launch 0 dropped)")
print("dispatch kernel span, first begin -> last end, per rank per launch:")
print(
    f"  stock  mean {st.mean(spS):6.2f}  p50 {sorted(spS)[len(spS)//2]:6.2f}  max {max(spS):6.2f} us"
)
print(
    f"  FRONT  mean {st.mean(spF):6.2f}  p50 {sorted(spF)[len(spF)//2]:6.2f}  max {max(spF):6.2f} us"
)
print()

for title, pS, pF in [
    ("ALL WARPS (mean us per warp)", lambda r: True, lambda r: True),
    (
        "FIRST-ARRIVER WARP (critical path, owns the waits)",
        lambda r: faS.get((r[0], r[1])) == r[2],
        lambda r: faF.get((r[0], r[1])) == r[2],
    ),
]:
    a, b = slot_stats(rS, pS), slot_stats(rF, pF)
    print(title)
    print(
        f"{'slot':12} {'stock mean':>11} {'p95':>7} | {'FRONT mean':>11} {'p95':>7} | {'delta':>7}"
    )
    print("-" * 66)
    for sid in ORDER:
        if sid not in a and sid not in b:
            continue
        sa = a.get(sid, (0, 0.0, 0.0, 0.0))
        sb = b.get(sid, (0, 0.0, 0.0, 0.0))
        print(
            f"{S[sid]:12} {sa[1]:11.3f} {sa[3]:7.3f} | {sb[1]:11.3f} {sb[3]:7.3f} | {sb[1]-sa[1]:+7.3f}"
        )
    print()
