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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The gfx1201 EP2/EP4 push transport policy for top-k 1..64.

BF16 and FP16 share token thresholds and launch defaults. Top-k 8 retains
its measured specialization; all other top-k values share runtime kernels.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rdna4EpPolicy:
    world_size: int
    hidden: int
    capacity: int
    dtype: str
    topk: int = 8

    def __post_init__(self):
        if not (
            self.world_size in (2, 4)
            and 2048 <= self.hidden <= 8192
            and self.hidden % 8 == 0
            and self.capacity > 0
            and self.dtype in ("bf16", "fp16")
            and 1 <= self.topk <= 64
        ):
            raise ValueError(
                "MORI_RDNA4_EP requires EP2/EP4, BF16/FP16, topk=1..64, positive capacity and H=2048..8192 aligned to 8"
            )

    def check_input(self, dtype, hidden, *, external=True):
        # EP4 places live weights and dispatch metadata in disjoint suffix
        # slices. Moving these slices across calls races a slower peer reader.
        if hidden != self.hidden or dtype != self.dtype or not external:
            raise ValueError(
                "MORI_RDNA4_EP requires external input with fixed config.hidden_dim "
                f"and dtype ({self.hidden}, {self.dtype}); create a new operator "
                "to change hidden size or dtype"
            )

    @staticmethod
    def small(tokens):
        return tokens is not None and 0 <= tokens <= 256

    def dispatch_kernel(self, tokens):
        dtype = self.dtype if self.topk == 8 else f"{self.dtype}_topk"
        if self.world_size == 4:
            return f"EpDispatchIntraNodeEp4HybridKernel_{dtype}"
        suffix = ""
        if self.small(tokens):
            # Capacity is symmetric. Large allocations must keep the count
            # protocol compatible with a peer running the large-token entry.
            suffix = "_small_v2" + ("_compat" if self.capacity > 256 else "")
        return f"EpDispatchIntraNodeEp2PushKernel_{dtype}{suffix}"

    def combine_kernel(self, tokens):
        dtype = self.dtype if self.topk == 8 else f"{self.dtype}_topk"
        tile = 512 if tokens * self.hidden <= 256 * 1024 else 1024
        if self.world_size == 4:
            suffix = f"_tile{tile}" if self.small(tokens) else ""
            return f"EpCombineIntraNodeEp4PushKernel_{dtype}{suffix}"
        suffix = f"_small_v2_{tile}" if self.small(tokens) else ""
        return f"EpCombineIntraNodeEp2Kernel_{dtype}_push{suffix}"

    def launch(self, tokens, *, dispatch):
        blocks = 32
        if dispatch and self.small(tokens):
            blocks = max(4 if self.world_size == 2 else 8, (tokens + 15) // 16)
        return blocks, 16
