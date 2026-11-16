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
"""The existing ``GemmAllReduceOp`` (BF16), behind the case's interface.

Verifier side: the implementer must not read this file (``hidden_paths.txt``); running
it as a baseline is fine. The op is constructed with its defaults (SDMA scatter, LSA
gather, BF16 on both legs, default schedule), which is what a caller gets.
"""

from mori.ops.gemm_ar import GemmAllReduceOp


class ExistingGemmAllReduce:
    #: SDMA queues per peer; 1 is the op's default.
    SDMA_QUEUES = 1

    @staticmethod
    def window_bytes_for(world_size, *, m_max, n, k):
        return GemmAllReduceOp.window_bytes_for(
            world_size, m_max=m_max, n=n, k=k, quant="bf16"
        )

    def __init__(self, comm, *, n, k, m_max):
        self.op = GemmAllReduceOp(
            comm, n=n, k=k, m_max=m_max, quant="bf16", sdma_queues=self.SDMA_QUEUES
        )
        self.weight = None
        # The op keeps no record of which path a call took.
        self.fused_path_ran = None

    def set_weight(self, b):
        self.weight = b.contiguous()

    def __call__(self, a):
        m = a.shape[0]
        padded = self.op.padded_m(m)
        if padded != m:
            a = self.op.pad_rows(a, padded)
        return self.op(a, self.weight, logical_m=m)[:m]

    def self_test(self):
        self.op.self_test()

    def close(self):
        self.op.close()


class ExistingGemmAllReduceQ8(ExistingGemmAllReduce):
    """The same op with 8 SDMA queues per peer, the existing benchmark's default."""

    SDMA_QUEUES = 8
