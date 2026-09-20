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
"""At blocks=128 with 512 tokens, half the grid gets no tokens at all, so s_N[p]==0
and the `if (n > 0)` guard skips the atomic entirely. Those blocks therefore measure
the SlotReserve region with the atomic removed -- i.e. __syncthreads() on its own."""
import torch
import statistics as st

MHZ = 99.845
WPB = 8


def us(t):
    return t * 1000.0 / MHZ / 1000.0


sr, pay = {}, {}
for rk in range(4):
    b = torch.load(f"/tmp/eptrace_rank{rk}.pt").numpy().reshape(-1, 2)
    per = {}
    for ts, meta in b:
        per.setdefault(int(meta) >> 16, []).append(
            (int(ts), (int(meta) >> 2) & 0x3FFF, int(meta) & 3)
        )
    for w, s in per.items():
        if w % WPB:
            continue
        blk = w // WPB
        launch, op = -1, {}
        for ts, slot, typ in s:
            if typ == 0:
                if slot == 0:
                    launch += 1
                op[slot] = ts
            elif typ == 1 and slot in op:
                t0 = op.pop(slot)
                if launch < 1:
                    continue
                if slot == 2:
                    sr.setdefault(blk, []).append(us(ts - t0))
                if slot == 5:
                    pay.setdefault(blk, []).append(us(ts - t0))
blocks = sorted(sr)
print("blocks seen:", len(blocks), "range", blocks[0], "-", blocks[-1])
for name, grp in [
    ("blockIdx 0-63  (have tokens)", [b for b in blocks if b < 64]),
    ("blockIdx 64-127 (no tokens) ", [b for b in blocks if b >= 64]),
]:
    s = [x for b in grp for x in sr[b]]
    p = [x for b in grp for x in pay.get(b, [])]
    if not s:
        print(name, "n/a")
        continue
    s.sort()
    ps = f"{st.mean(p):6.2f}" if p else "  n/a "
    print(
        f"{name}: SlotReserve mean {st.mean(s):6.2f}  p50 {s[len(s)//2]:6.2f}  "
        f"max {s[-1]:6.2f} us | PayloadTdm mean {ps} us"
    )
