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
"""Compile-only check for the TDM dispatch kernel -- no GPU needed.

``COMPILE_ONLY=1`` makes FlyDSL trace, lower and codegen the kernel and then
return instead of launching, so this exercises every DSL construct in
``make_dispatch_tdm`` (and every compile-time shape assert) without touching a
device. It cannot say whether the kernel is *correct*, only whether it is
well-formed -- but it is seconds rather than the minutes a 4-rank run costs, so
it is the first thing to run after touching the kernel.

The cases cover both metadata paths (a 512-wide tile has no room to batch, so it
falls back to the scalar tail), both lane groupings (topk=8 packs 4 tokens per
wave, topk=6 only 1) and a single-rank grid.

    COMPILE_ONLY=1 python compile_dispatch_tdm.py
"""
import os
import sys
import time

os.environ.setdefault("COMPILE_ONLY", "1")
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
# The kernel's lane grouping is wave-size dependent and detect_gpu_arch() has no
# device to ask when this runs on a build box.
os.environ.setdefault("MORI_WAVE_SIZE", "32")

from mori.ops.dispatch_combine_v2.intranode_kernels_tdm import (  # noqa: E402
    make_dispatch_tdm,
    tdm_stage_capacity,
)

# (npes, topk, hidden, elem_size, block_num, warp_num)
CASES = [
    (4, 8, 7168, 2, 64, 8),
    (4, 8, 2048, 2, 64, 16),
    (1, 8, 2048, 2, 64, 8),
    (4, 6, 4096, 2, 64, 8),
    (8, 8, 512, 2, 64, 8),  # tile too small for a meta batch -> scalar metadata
]


def main():
    bad = 0
    for npes, topk, hidden, elem, block, warp in CASES:
        max_tok = 512
        cap, slots = tdm_stage_capacity(
            npes=npes,
            experts_per_token=topk,
            max_tok_per_rank=max_tok,
            block_num=block,
            warp_num_per_block=warp,
        )
        tag = f"npes={npes} topk={topk} hidden={hidden} elem={elem} {block}x{warp}"
        t0 = time.time()
        try:
            run = make_dispatch_tdm(
                rank=0,
                npes=npes,
                experts_per_rank=8,
                experts_per_token=topk,
                hidden_dim=hidden,
                hidden_elem_size=elem,
                max_tok_per_rank=max_tok,
                max_recv=npes * max_tok,
                block_num=block,
                warp_num_per_block=warp,
                off_tok_off=0,
                off_recv_num=256,
                off_tis=512,
                off_out_idx=4096,
                off_out_wts=8192,
                off_out_tok=16384,
            )
            run(*([0] * 11), 0, 8)
        except Exception as exc:  # noqa: BLE001 - report every case, not just the first
            bad += 1
            print(f"FAIL {tag}\n  {type(exc).__name__}: {exc}", flush=True)
            if os.environ.get("TB"):
                import traceback

                traceback.print_exc()
                return 1
        else:
            print(
                f"ok   {tag}  cap/blk={cap} slots={slots}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
    print(f"\n{len(CASES) - bad}/{len(CASES)} configs compiled")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
