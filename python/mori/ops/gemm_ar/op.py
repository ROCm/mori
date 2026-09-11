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
"""Host op-layer for the fused GEMM + all-reduce. See ``README.md``."""

from __future__ import annotations

from typing import Optional

import torch

import flydsl.expr as fx
from mori.cco import CCODevCommRequirements, GDA_CONNECTION_NONE
from mori.tensor_utils import from_gpu_ptr

from .kernels_fused import compile_fused_gemm_scatter
from .kernels_sdma import build_sdma_phases
from .layout import ArConfig

# A destination's row band is BLOCK_M rows, so M has to divide into whole bands
# per destination. Both are also the tile the kernel is compiled for: 128x256 is
# what the block-scale path needs, because its per-K-block promotion doubles the
# accumulator VGPRs and 256x256 would spill.
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 256

# The block-scale group, on both operands: A is 1x128, B is 128x128.
SCALE_BLOCK_K = 128

# Chunks are how many separate pushes a destination receives, and so how early
# the first bytes leave. More is better until the pieces get small enough that
# the SDMA per-packet cost shows; 8 is the measured knee.
MAX_CHUNKS = 8


def padded_m(m: int, world_size: int, block_m: int = DEFAULT_BLOCK_M) -> int:
    """``m`` rounded up to a whole number of row bands per destination."""
    granule = world_size * block_m
    return (m + granule - 1) // granule * granule


def counter_chunks(m_pad: int, world_size: int, block_m: int = DEFAULT_BLOCK_M) -> int:
    """The largest chunk count that divides the row bands per destination.

    Chunks must divide evenly or the completion counter's modulo test never
    fires and the push is never issued. At small M there are fewer bands than
    ``MAX_CHUNKS``, so take the largest divisor rather than failing: a fixed 8
    has nothing to divide at ``m_pad = 4096`` (4 bands).

    Worth taking the largest: at M=15360 the divisor rule measures 1104.8us
    against 1434.0us for a chunk count of 1.
    """
    bands = m_pad // (world_size * block_m)
    return max(c for c in range(1, min(MAX_CHUNKS, bands) + 1) if bands % c == 0)


def supports(
    m: int,
    n: int,
    k: int,
    world_size: int,
    *,
    block_n: int = DEFAULT_BLOCK_N,
) -> bool:
    """Whether this shape is *expressible*, which is not whether it is faster.

    Profitability depends on how much GEMM there is to hide the transfer behind
    and is the caller's call -- see the measured curve in ``README.md``.
    """
    return (
        2 <= world_size <= 8
        and n % block_n == 0
        and n % SCALE_BLOCK_K == 0
        and k % SCALE_BLOCK_K == 0
        and m > 0
    )


class GemmAllReduceOp:
    """Fused fp8 block-scale GEMM + all-reduce over cco SDMA.

    Owns the symmetric window and a per-M kernel cache; one instance serves one
    ``(n, k)`` weight shape and any ``M`` up to ``m_max``.

    The window cannot grow once allocated, so size ``m_max`` for the largest M
    the deployment will see, not for the first one it does.
    """

    def __init__(
        self,
        comm,
        *,
        n: int,
        k: int,
        m_max: int,
        block_m: int = DEFAULT_BLOCK_M,
        block_n: int = DEFAULT_BLOCK_N,
        sdma_queues: int = 1,
    ):
        self.comm = comm
        self.rank = comm.rank
        self.world_size = comm.world_size
        self.n, self.k = n, k
        self.block_m, self.block_n = block_m, block_n
        self.sdma_queues = sdma_queues
        self.m_max = padded_m(m_max, self.world_size, block_m)

        if not supports(self.m_max, n, k, self.world_size, block_n=block_n):
            raise ValueError(
                f"unsupported shape: M<={self.m_max} N={n} K={k} "
                f"world_size={self.world_size}"
            )

        self.window_bytes = self._make_cfg(self.m_max).window_bytes
        self.mem = comm.alloc_mem(self.window_bytes)
        self.win = comm.register_window(self.mem.ptr, self.mem.size)
        # Counters live in this window and are read before they are first
        # written, so it has to start zeroed.
        from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        # One queue per (source, destination) pair is all the pipeline uses -- a
        # destination is one xGMI link, and splitting a transfer across queues
        # only costs bandwidth. Asking for world_size would create world_size
        # queues *per peer* (56 on an 8-rank node, to use 7); inside a process
        # that already holds SDMA engines that overruns the per-engine queue
        # slots and hsaKmtCreateQueueExt fails.
        reqs.sdma_queue_count = sdma_queues
        self.dev_comm = comm.create_dev_comm(reqs)

        self._cache: dict[int, tuple] = {}
        self._pad_in: Optional[torch.Tensor] = None

    def padded_m(self, m: int) -> int:
        """``m`` rounded up to what :meth:`__call__` accepts."""
        return padded_m(m, self.world_size, self.block_m)

    def _make_cfg(self, m: int) -> ArConfig:
        cfg = ArConfig(
            world_size=self.world_size,
            m=m,
            n=self.n,
            recv_slots=self.world_size,
            counter_chunks=counter_chunks(m, self.world_size, self.block_m),
        )
        cfg.validate()
        return cfg

    def _compiled(self, m: int):
        """Kernel plus SDMA phases for this M, compiled on first sight."""
        hit = self._cache.get(m)
        if hit is not None:
            return hit
        cfg = self._make_cfg(m)
        gemm = compile_fused_gemm_scatter(
            cfg,
            self.rank,
            K=self.k,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            b_preshuffled=True,
            fuse=True,
            transport="sdma",
            quant="blockscale",
            sdma_queues=self.sdma_queues,
            # The three C-store stages are off by default in
            # compile_fused_gemm_scatter, and blockscale requires swap_ab.
            swap_ab=True,
            permlane=True,
            lane_transpose=True,
        )
        parts = build_sdma_phases(cfg, self.rank, queues=self.sdma_queues)
        c = from_gpu_ptr(self.mem.ptr + cfg.input_off, (m, self.n), torch.bfloat16)
        out = from_gpu_ptr(self.mem.ptr + cfg.output_off, (m, self.n), torch.bfloat16)
        hit = (gemm, parts, c, out)
        self._cache[m] = hit
        return hit

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        A GEMM row and a reduce-scatter row both depend only on the same input
        row, so the added rows produce zeros that the caller slices off. Pad
        *before* quantising: that is what keeps the A scale's column-major
        ``[K/128, M]`` layout intact without touching it.
        """
        m, k = x.shape
        if self._pad_in is None or self._pad_in.shape[0] < m_pad:
            self._pad_in = torch.zeros((m_pad, k), dtype=x.dtype, device=x.device)
        buf = self._pad_in[:m_pad]
        buf[:m].copy_(x)
        buf[m:].zero_()
        return buf

    def __call__(
        self,
        a_fp8: torch.Tensor,
        b_preshuffled: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        """One fused GEMM + all-reduce; returns a view of the window's output.

        ``a_fp8`` is ``[M, K]`` with ``M`` a multiple of ``world_size *
        block_m`` (see :meth:`padded_m`), ``b_preshuffled`` is ``[N, K]``
        through :func:`preshuffle_b`.

        Both scale buffers are read linearly, in physical order: ``a_scale`` is
        logically ``[M, K/128]`` but column-major, i.e. physically
        ``[K/128, M]``; ``b_scale`` is ``[N/128, K/128]`` row-major.

        The returned tensor aliases the window and is overwritten by the next
        call. Clone it to keep it.
        """
        m = a_fp8.shape[0]
        if m > self.m_max:
            raise ValueError(
                f"M={m} exceeds the window's m_max={self.m_max}; the window "
                f"cannot grow after construction"
            )
        if m % (self.world_size * self.block_m):
            raise ValueError(
                f"M={m} must be a multiple of world_size * block_m = "
                f"{self.world_size * self.block_m}; use padded_m()/pad_rows()"
            )
        gemm, parts, c, out = self._compiled(m)
        stream = fx.Stream(torch.cuda.current_stream())
        gemm(
            a_fp8.contiguous().view(torch.int8).view(-1),
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            c.view(-1),
            a_scale.reshape(-1),
            b_scale.reshape(-1),
            m,
            self.n,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        parts["drain"](self.dev_comm.ptr, self.win.handle, stream=stream)
        parts["reduce"](self.dev_comm.ptr, self.win.handle, stream=stream)
        parts["gather"](self.dev_comm.ptr, self.win.handle, stream=stream)
        return out
