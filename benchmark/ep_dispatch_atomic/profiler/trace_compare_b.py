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
"""Stock (S) vs per-block done flags (B), same instrumented header (profilerB.hpp).

  python3 trace_compare_b.py S1,S2 B1,B2     # PROFTAG values of each arm's runs

Reads /tmp/eptrace{TAG}_rank{0..3}.pt. Launch 0 of every run is dropped (it carries
the host-barrier skew). Three views:

1. Per-slot means over ALL warps.
2. The critical-path warp. Stock: the first arriver, the only warp that emits
   FenceAgent. Flags: of block 0's warps (the only ones that emit FlagPoll), the one
   whose FlagPoll ends last -- the kernel cannot end before it.
3. The tail, per rank per launch, split at the moment this rank's LAST block
   finished its payload (the latest PayloadTdm end):
     fan  = latest - earliest PayloadTdm end: how spread the blocks' completions are
     post = kernel end - latest PayloadTdm end: signalling + waiting on the peers
   A cheaper signal shows up in `post`; a better-balanced grid in `fan`.
"""
import statistics as st
import sys

import torch

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
    12: "FlagSend",
    13: "FlagPoll",
    14: "FlagSum",
}
ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
PAYLOAD, FENCE_AGENT, FLAG_POLL = 5, 8, 13


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


def critical(rows, arm):
    """key -> warp id of the critical-path warp."""
    if arm == "S":
        return {r[0]: r[1] for r in rows if r[2] == FENCE_AGENT}
    best = {}
    for r in rows:
        if r[2] == FLAG_POLL and (r[0] not in best or r[5] > best[r[0]][1]):
            best[r[0]] = (r[1], r[5])
    return {k: v[0] for k, v in best.items()}


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
    post = [us(last[k] - pay_hi[k]) for k in keys]
    return len(keys), span, fan, post


def fmt(v):
    v = sorted(v)
    return f"mean {st.mean(v):6.2f}  p50 {v[len(v) // 2]:6.2f}  p95 {v[int(0.95 * (len(v) - 1))]:6.2f}"


tagsS, tagsB = sys.argv[1].split(","), sys.argv[2].split(",")
rS, rB = load(tagsS), load(tagsB)
cS, cB = critical(rS, "S"), critical(rB, "B")
nS, spS, fanS, postS = tail(rS)
nB, spB, fanB, postB = tail(rB)
emits = sorted({S[r[2]] for r in rB if r[2] in (6, 7, 8, 9, 10)})
print(f"stock runs {tagsS}  flags runs {tagsB}")
print(f"rank-launches  stock={nS}  flags={nB}   (launch 0 of each run dropped)")
print(f"flags arm emitted stock-tail slots: {emits or 'none'}  (must be none)")
print(f"flags arm emitted FlagSend spans: {sum(1 for r in rB if r[2] == 12)}\n")

print("per rank per launch, us")
print(f"  kernel span  stock  {fmt(spS)}")
print(f"               flags  {fmt(spB)}")
print(f"  fan          stock  {fmt(fanS)}")
print(f"               flags  {fmt(fanB)}")
print(f"  post         stock  {fmt(postS)}")
print(f"               flags  {fmt(postB)}\n")

for title, pS, pB in [
    ("ALL WARPS (mean us per warp)", lambda r: True, lambda r: True),
    (
        "CRITICAL-PATH WARP (stock: first arriver; flags: block-0 warp whose FlagPoll ends last)",
        lambda r: cS.get(r[0]) == r[1],
        lambda r: cB.get(r[0]) == r[1],
    ),
]:
    a, b = slot_stats(rS, pS), slot_stats(rB, pB)
    print(title)
    print(f"{'slot':12} {'stock mean':>11} {'p95':>7} | {'flags mean':>11} {'p95':>7}")
    print("-" * 56)
    for sid in ORDER:
        if sid not in a and sid not in b:
            continue
        sa = a.get(sid, (0, 0.0, 0.0))
        sb = b.get(sid, (0, 0.0, 0.0))
        print(f"{S[sid]:12} {sa[1]:11.3f} {sa[2]:7.3f} | {sb[1]:11.3f} {sb[2]:7.3f}")
    print()
