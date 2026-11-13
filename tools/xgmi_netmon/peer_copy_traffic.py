#!/usr/bin/env python3
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
"""Known XGMI traffic for checking xgmi-netmon: torch peer copies with exact byte counts.

Run inside a ROCm + torch container on an 8-GPU node while `xgmi-netmon record` (with
--push, or imported afterwards) runs on the host.
Phases (10 s each, 5 s apart): GPU0->GPU1, GPU1->GPU0, ring i->i+1 on all GPUs.
Prints one JSON line per phase with wall-clock start/end and the bytes copied per pair; compare
with sum_over_time(xgmi_bytes{...}[end-start]) at `end`.

Result on MI355X (2026-11-13): sender wr = receiver rd = 1.118x the copied bytes, and 0.569x in
the opposite direction on both ends.
"""
import json
import time

import torch

N = 1 << 30  # bytes per copy
ng = torch.cuda.device_count()
bufs = [
    (
        torch.empty(N, dtype=torch.uint8, device=f"cuda:{i}"),
        torch.empty(N, dtype=torch.uint8, device=f"cuda:{i}"),
    )
    for i in range(ng)
]


def sync():
    for i in range(ng):
        torch.cuda.synchronize(i)


def phase(name, pairs, dur):
    streams = [torch.cuda.Stream(device=f"cuda:{s}") for s, _ in pairs]
    sync()
    t0, k = time.time(), 0
    while time.time() - t0 < dur:
        for (s, d), st in zip(pairs, streams):
            with torch.cuda.stream(st):
                bufs[d][1].copy_(bufs[s][0], non_blocking=True)
        k += 1
        if k % 2 == 0:
            sync()
    sync()
    t1 = time.time()
    print(
        json.dumps(
            {
                "phase": name,
                "pairs": pairs,
                "start": t0,
                "end": t1,
                "bytes_per_pair": k * N,
                "GBps_per_pair": round(k * N / (t1 - t0) / 1e9, 2),
            }
        ),
        flush=True,
    )


print("gpus", ng, "peer 0->1", torch.cuda.can_device_access_peer(0, 1), flush=True)
phase("warmup", [(0, 1)], 1)
time.sleep(3)
phase("gpu0->gpu1", [(0, 1)], 10)
time.sleep(5)
phase("gpu1->gpu0", [(1, 0)], 10)
time.sleep(5)
phase("ring i->i+1", [(i, (i + 1) % ng) for i in range(ng)], 10)
