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
"""Serialisation queue, or barrier skew?

If the 4 peer atomics serialise on one remote word, the 64 blocks' SlotReserve
END timestamps should fan out into a ramp while their STARTs stay clustered, and
a block's duration should grow with how late it is served. If instead the blocks
START at scattered times and END together, the span is tracking arrival skew.
"""
import torch
import statistics as st

MHZ = 99.845
WPB = 8


def us(t):
    return t * 1000.0 / MHZ / 1000.0


ev = {}
for rk in range(4):
    b = torch.load(f"/tmp/eptrace_rank{rk}.pt").numpy().reshape(-1, 2)
    per = {}
    for ts, meta in b:
        per.setdefault(int(meta) >> 16, []).append(
            (int(ts), (int(meta) >> 2) & 0x3FFF, int(meta) & 3)
        )
    for w, s in per.items():
        launch, op = -1, {}
        for ts, slot, typ in s:
            if typ == 0:
                if slot == 0:
                    launch += 1
                op[slot] = ts
            elif typ == 1 and slot in op:
                t0 = op.pop(slot)
                if slot == 2 and w % WPB == 0:  # warp 0 = the one issuing the atomics
                    ev.setdefault((rk, launch), []).append((w // WPB, t0, ts))
for key in [k for k in ev if k[1] < 1]:
    del ev[key]

print(
    f"{'launch':>12} {'start spread':>13} {'end spread':>11} {'dur min':>9} {'dur max':>9} "
    f"{'corr(start,dur)':>16}"
)
print("-" * 78)
for key in sorted(ev)[:6]:
    v = ev[key]
    t0 = [x[1] for x in v]
    t1 = [x[2] for x in v]
    d = [x[2] - x[1] for x in v]
    m0, m1 = min(t0), min(t1)
    # correlation between how early a block starts and how long it waits
    mt, md = st.mean(t0), st.mean(d)
    num = sum((a - mt) * (b - md) for a, b in zip(t0, d))
    den = (sum((a - mt) ** 2 for a in t0) * sum((b - md) ** 2 for b in d)) ** 0.5
    print(
        f"{str(key):>12} {us(max(t0)-m0):12.2f}u {us(max(t1)-m1):10.2f}u "
        f"{us(min(d)):8.2f}u {us(max(d)):8.2f}u {num/den if den else 0:16.3f}"
    )

# are the END times evenly spaced (a draining queue) or clustered?
key = sorted(ev)[1]
v = sorted(ev[key], key=lambda x: x[2])
gaps = [us(v[i + 1][2] - v[i][2]) for i in range(len(v) - 1)]
print(f"\nlaunch {key}: 64 blocks sorted by END time")
print(
    f"  consecutive END gaps: mean {st.mean(gaps):.3f}us  p50 {sorted(gaps)[len(gaps)//2]:.3f}us  "
    f"max {max(gaps):.3f}us"
)
print(f"  first 12 gaps: {' '.join(f'{g:.2f}' for g in gaps[:12])}")
