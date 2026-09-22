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
"""Stock (S) vs variant A segmented output (A), same instrumented header (profilerF.hpp).

  python3 trace_compare_a.py S1,S2 A1,A2 [label]   # PROFTAG values of each arm's runs

[label] names the second arm in the output (default "seg"). Any arm that keeps the
stock tail compares the same way -- e.g. MORI_EP_TOKOFF_EXT=1, labelled "ext".

Arm A is MORI_EP_VARIANT_A=seg with -DMORI_EP_VA_NOPREFIX: the per-source segment
replaces the remote slot allocator and nothing compacts or publishes a prefix. Both
arms keep the stock tail, so the critical-path warp is the same in each: the first
arriver, the only warp that emits FenceAgent.

Reads /tmp/eptrace{TAG}_rank{0..3}.pt. Launch 0 of every run is dropped (it carries
the host-barrier skew). FrontRdv (11) sits outside the FRONT block in the header, so
both arms emit it and it must be ~0 in both -- the check that neither compiled FRONT.

Tail, per rank per launch, split at the moment this rank's LAST block finished its
payload (the latest PayloadTdm end):
  fan  = latest - earliest PayloadTdm end
  post = kernel end - latest PayloadTdm end (signalling + waiting on the peers)
min-over-ranks post is the rank that finished last, so it has no peer left to wait
on: its post is the signal itself. Cross-rank timestamps are NOT compared --
wall_clock64 is not synchronised across GPUs.
"""
import statistics as st
import sys

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
PAYLOAD, FENCE_AGENT = 5, 8


def us(t):
    return t / MHZ


def load(tags):
    """rows: (key, warp, slot, dur_us, t0, t1) with key = (tag, rank, launch)."""
    rows = []
    for tag in tags:
        for rk in range(4):
            b = torch.load(f"/tmp/eptrace{tag}_rank{rk}.pt").numpy().reshape(-1, 2)
            per = {}
            for ts, meta in b:
                per.setdefault(int(meta) >> 16, []).append(
                    (int(ts), (int(meta) >> 2) & 0x3FFF, int(meta) & 3)
                )
            for w, ev in per.items():
                launch, opened = -1, {}
                for ts, slot, typ in ev:
                    if typ == 0:
                        if slot == 0:
                            launch += 1
                        opened[slot] = ts
                    elif typ == 1 and slot in opened:
                        t0 = opened.pop(slot)
                        if launch >= 1:
                            rows.append(
                                ((tag, rk, launch), w, slot, us(ts - t0), t0, ts)
                            )
    return rows


def slot_stats(rows, pred):
    out = {}
    for sid in ORDER:
        v = sorted(r[3] for r in rows if r[2] == sid and pred(r))
        if v:
            out[sid] = (len(v), st.mean(v), v[int(0.95 * (len(v) - 1))])
    return out


def tail(rows):
    first, last, pay_lo, pay_hi = {}, {}, {}, {}
    for key, _, slot, _, t0, t1 in rows:
        first[key] = min(first.get(key, t0), t0)
        last[key] = max(last.get(key, t1), t1)
        if slot == PAYLOAD:
            pay_lo[key] = min(pay_lo.get(key, t1), t1)
            pay_hi[key] = max(pay_hi.get(key, t1), t1)
    keys = [k for k in first if k in pay_hi]
    span = [us(last[k] - first[k]) for k in keys]
    fan = [us(pay_hi[k] - pay_lo[k]) for k in keys]
    post = {k: us(last[k] - pay_hi[k]) for k in keys}
    # min over the 4 ranks of the same (tag, launch)
    grp = {}
    for (tag, rk, ln), v in post.items():
        grp.setdefault((tag, ln), []).append(v)
    minpost = [min(v) for v in grp.values() if len(v) == 4]
    return len(keys), span, fan, list(post.values()), minpost


def fmt(v):
    v = sorted(v)
    return f"mean {st.mean(v):6.2f}  p50 {v[len(v) // 2]:6.2f}  p95 {v[int(0.95 * (len(v) - 1))]:6.2f}"


tagsS, tagsA = sys.argv[1].split(","), sys.argv[2].split(",")
LBL = sys.argv[3] if len(sys.argv) > 3 else "seg"
rS, rA = load(tagsS), load(tagsA)
cS = {r[0]: r[1] for r in rS if r[2] == FENCE_AGENT}
cA = {r[0]: r[1] for r in rA if r[2] == FENCE_AGENT}
nS, spS, fanS, postS, mpS = tail(rS)
nA, spA, fanA, postA, mpA = tail(rA)
print(f"stock runs {tagsS}  {LBL} runs {tagsA}")
print(f"rank-launches  stock={nS}  {LBL}={nA}   (launch 0 of each run dropped)\n")

print("per rank per launch, us")
print(f"  kernel span  stock  {fmt(spS)}")
print(f"               {LBL:6} {fmt(spA)}")
print(f"  fan          stock  {fmt(fanS)}")
print(f"               {LBL:6} {fmt(fanA)}")
print(f"  post         stock  {fmt(postS)}")
print(f"               {LBL:6} {fmt(postA)}")
print(f"  min-rank post stock {fmt(mpS)}")
print(f"               {LBL:6} {fmt(mpA)}\n")

for title, pS, pA in [
    ("ALL WARPS (mean us per warp)", lambda r: True, lambda r: True),
    (
        "CRITICAL-PATH WARP (first arriver: the only FenceAgent warp, owns the waits)",
        lambda r: cS.get(r[0]) == r[1],
        lambda r: cA.get(r[0]) == r[1],
    ),
]:
    a, b = slot_stats(rS, pS), slot_stats(rA, pA)
    print(title)
    print(
        f"{'slot':12} {'stock mean':>11} {'p95':>7} | {LBL + ' mean':>9} {'p95':>7} | {'delta':>7}"
    )
    print("-" * 64)
    for sid in ORDER:
        if sid not in a and sid not in b:
            continue
        sa = a.get(sid, (0, 0.0, 0.0))
        sb = b.get(sid, (0, 0.0, 0.0))
        print(
            f"{S[sid]:12} {sa[1]:11.3f} {sa[2]:7.3f} | {sb[1]:9.3f} {sb[2]:7.3f} | {sb[1] - sa[1]:+7.3f}"
        )
    print()
