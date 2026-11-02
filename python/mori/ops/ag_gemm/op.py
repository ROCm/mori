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
"""``AgGemmOp``: the all-gather that feeds a ColumnParallelLinear, fused into
the fp8 GEMM.

Every rank passes its own tokens ``A_r [M, K]`` and its own weight shard
``B_r [N, K]``, and gets back ``concat_s(A_s) @ B_r.T``: ``[world*M, N]`` bf16,
source rank ``s`` at rows ``[s*M, (s+1)*M)``. That is the sequence-parallel
all-gather followed by the column-parallel projection, as one call.

Two modes, both over the copy engines:

``fused`` (default)
    One kernel: block 0 pushes this rank's rows to every peer band by band,
    local tiles run first, remote tiles wait only for the band they need.
``split``
    Push, barrier, the same GEMM. The baseline ``fused`` is measured against,
    and a fallback.

Every call ends in a cross-rank barrier. That is what lets a call refill the
peers' receive slots without a separate credit: no rank starts call ``i+1``
until every rank's GEMM ``i`` -- the only reader of those slots -- is done. It
is also why one instance can serve different M from call to call, although the
slots are laid out for the current M so the GEMM reads one contiguous operand.

The returned tensor is this op's own output buffer, not the window, and the
next call overwrites it: consume it first on the same stream, or clone it.
"""

from __future__ import annotations

from typing import NamedTuple

import flydsl.expr as fx
import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.tensor_utils import from_gpu_ptr

from ..gemm_ar.op import FP8_DTYPES, MIN_K, _PinnedLaunch
from .kernels import BLOCK_K, build_phases, compile_ag_gemm
from .layout import AgGemmConfig

MODES = ("fused", "split")
#: The tile. The lane-transpose store pairs two 128-column halves, and the
#: mainloop is gemm_ar's 256-row 8-wave pipeline.
BLOCK_M = 256
BLOCK_N = 256
#: ptpc only: its A scale is one fp32 per row and travels with the row band.
QUANTS = ("ptpc",)
#: Default cap on row bands per peer. Measured on Kimi-K3's TP8 projections: 4
#: wins from M=1024 rows per rank up and 2 at 512, the most the tiles allow;
#: one chunk barely beats the split path, because every remote tile then waits
#: for a peer's whole slab. Above 4 was not measured.
DEFAULT_CHUNKS = 4


class _Plan(NamedTuple):
    cfg: AgGemmConfig
    gemm: object
    phases: tuple
    rows: torch.Tensor
    scales: torch.Tensor


class AgGemmOp:
    """Fused all-gather + fp8 ptpc GEMM over cco SDMA.

    One instance serves one ``(n, k)`` weight shape and any M up to ``m_max``
    (padded, see :meth:`padded_m`). It owns its symmetric window, which cannot
    grow, so size ``m_max`` for the largest per-rank M the deployment will see.
    """

    def __init__(
        self,
        comm,
        *,
        n: int,
        k: int,
        m_max: int,
        quant: str = "ptpc",
        mode: str = "fused",
        chunks: int | None = None,
        sdma_queues: int = 1,
    ):
        self._closed = True  # a failed constructor leaves close() a no-op
        self.mem = self.win = self.dev_comm = None
        self._cache: dict[int, _Plan] = {}
        self._pad_in: torch.Tensor | None = None
        self._out: torch.Tensor | None = None

        if quant not in QUANTS:
            raise ValueError(
                f"quant={quant!r} is not one of {QUANTS}: blockscale and mxfp8 "
                f"keep the A scale K-block major, so a row band is strided"
            )
        if mode not in MODES:
            raise ValueError(f"mode={mode!r} is not one of {MODES}")
        if m_max < 1 or sdma_queues < 1:
            raise ValueError(
                f"m_max and sdma_queues must be >= 1; got m_max={m_max} "
                f"sdma_queues={sdma_queues}"
            )
        if chunks is not None and chunks < 1:
            raise ValueError(f"chunks must be >= 1, got {chunks}")
        why = shape_constraint(n, k)
        if why is not None:
            raise ValueError(f"unsupported shape: {why}")
        self.comm = comm
        self.rank = comm.rank
        self.world_size = comm.nranks
        self.n, self.k = n, k
        self.quant, self.mode = quant, mode
        self.chunks = chunks
        self.sdma_queues = sdma_queues
        self.m_max = self.padded_m(m_max)
        self.window_bytes = self.window_bytes_for(self.world_size, k=k, m_max=m_max)
        try:
            self.mem = comm.alloc_mem(self.window_bytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            # The epoch and the flags are read before they are first written.
            from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = sdma_queues
            self.dev_comm = comm.create_dev_comm(reqs)
            # A put on a queue that does not exist moves nothing, and the op
            # would return the previous call's rows without a word.
            if self.dev_comm.sdma_num_queue < sdma_queues:
                raise RuntimeError(
                    f"AgGemmOp needs {sdma_queues} SDMA queue(s) per peer but the "
                    f"communicator has {self.dev_comm.sdma_num_queue}: build mori "
                    f"with BUILD_CCO_SDMA=ON, set MORI_ENABLE_SDMA=1, and make sure "
                    f"the first dev-comm created on this communicator asked for at "
                    f"least that many"
                )
            if self.dev_comm.lsa_size != self.world_size:
                raise RuntimeError(
                    f"AgGemmOp needs every rank in one LSA domain (one node); "
                    f"lsa_size={self.dev_comm.lsa_size}, world_size={self.world_size}"
                )
            # Every rank's zeroing has to land before any peer's first push or
            # flag into this window.
            torch.cuda.synchronize()
            comm.barrier()
        except Exception:
            self._closed = False
            self.close()
            raise
        self._closed = False

    @staticmethod
    def window_bytes_for(world_size: int, *, k: int, m_max: int) -> int:
        """Window bytes the constructor will allocate, for sizing the
        communicator's VMM before the op exists."""
        m = (m_max + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        return AgGemmConfig(
            world_size=world_size, m_local=m, k=k, block_m=BLOCK_M
        ).window_bytes

    # --- shapes ------------------------------------------------------------

    def padded_m(self, m: int) -> int:
        """Rows to pass for a logical per-rank ``m``: a whole number of tiles."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        return (m + BLOCK_M - 1) // BLOCK_M * BLOCK_M

    def chunks_for(self, m: int) -> int:
        """Row bands each peer's rows are pushed in at this M (fused mode)."""
        tiles = m // BLOCK_M
        cap = DEFAULT_CHUNKS if self.chunks is None else self.chunks
        # The largest divisor of the row tiles up to the cap: a band is a whole
        # number of tiles, and a remote tile waits on exactly one band.
        return max(d for d in range(1, min(cap, tiles) + 1) if tiles % d == 0)

    def _check_m(self, m: int) -> None:
        if m % BLOCK_M:
            raise ValueError(
                f"M={m} must be a multiple of {BLOCK_M}; use padded_m() and pad_rows()"
            )
        if m > self.m_max:
            raise ValueError(f"M={m} exceeds the window's m_max={self.m_max}")

    def _compiled(self, m: int) -> _Plan:
        hit = self._cache.get(m)
        if hit is not None:
            return hit
        cfg = AgGemmConfig(
            world_size=self.world_size,
            m_local=m,
            k=self.k,
            block_m=BLOCK_M,
            chunks=self.chunks_for(m) if self.mode == "fused" else 1,
        )
        fused = self.mode == "fused"
        gemm = compile_ag_gemm(
            cfg,
            self.rank,
            N=self.n,
            BLOCK_N=BLOCK_N,
            fuse=fused,
            sdma_queues=self.sdma_queues,
        )
        phases = build_phases(cfg, self.rank, sdma_queues=self.sdma_queues)
        # Split: push, barrier0, GEMM, barrier1. Fused: GEMM, barrier1.
        before = () if fused else ("push", "barrier0")
        hit = _Plan(
            cfg=cfg,
            gemm=_PinnedLaunch(gemm),
            phases=(
                tuple(_PinnedLaunch(phases[p]) for p in before),
                _PinnedLaunch(phases["barrier1"]),
            ),
            rows=from_gpu_ptr(
                self.mem.ptr + cfg.recv_off, (cfg.m, self.k), torch.uint8
            ),
            scales=from_gpu_ptr(self.mem.ptr + cfg.scale_off, (cfg.m,), torch.float32),
        )
        self._cache[m] = hit
        return hit

    def local_rows(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's slot at ``m``: ``([m, K] fp8, [m] fp32)`` in the window.

        Quantise straight into these and pass them to :meth:`__call__` to skip
        the copy it otherwise makes. Valid until the next call at another M.
        """
        self._check_open()
        self._check_m(m)
        plan = self._compiled(m)
        r = slice(self.rank * m, (self.rank + 1) * m)
        return plan.rows[r].view(torch.float8_e4m3fn), plan.scales[r]

    # --- the call ----------------------------------------------------------

    def __call__(
        self,
        a_fp8: torch.Tensor,
        b_preshuffled: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One all-gather + GEMM; returns ``[world*M, N]`` bf16.

        ``a_fp8`` is this rank's ``[M, K]`` tokens, M from :meth:`padded_m`, the
        same M on every rank. ``b_preshuffled`` is this rank's ``[N, K]`` weight
        shard through :func:`~mori.ops.gemm_ar.preshuffle_b`. ``a_scale`` is
        ``[M]`` (or ``[M, 1]``) and ``b_scale`` ``[N]`` (or ``[1, N]``), fp32.

        Rows padded by :meth:`pad_rows` come back as zero rows inside every
        source's block; slice them off per source.

        ``out``, if given, must be a contiguous ``[world*M, N]`` bf16 tensor;
        otherwise the result is this op's buffer, overwritten by the next call.
        """
        self._check_open()
        m = a_fp8.shape[0]
        self._check_m(m)
        a_scale, b_scale = self._check_operands(a_fp8, b_preshuffled, a_scale, b_scale)
        plan = self._compiled(m)
        rows, scales = self.local_rows(m)
        # Skipped when the caller quantised into the window already.
        if a_fp8.data_ptr() != rows.data_ptr():
            rows.copy_(a_fp8)
        if a_scale.data_ptr() != scales.data_ptr():
            scales.copy_(a_scale)
        c = self._output(m, out)
        stream = fx.Stream(torch.cuda.current_stream())
        before, closing = plan.phases
        for phase in before:
            phase(self.dev_comm.ptr, self.win.handle, stream=stream)
        plan.gemm(
            plan.rows.view(torch.int8).view(-1),
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            c.view(-1),
            plan.scales,
            b_scale,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        closing(self.dev_comm.ptr, self.win.handle, stream=stream)
        return c

    def _output(self, m: int, out: torch.Tensor | None) -> torch.Tensor:
        shape = (self.world_size * m, self.n)
        if out is not None:
            if tuple(out.shape) != shape or out.dtype != torch.bfloat16:
                raise ValueError(
                    f"out must be a {shape} bfloat16 tensor, got "
                    f"{tuple(out.shape)} {out.dtype}"
                )
            if not out.is_contiguous() or out.device.type != "cuda":
                raise ValueError("out must be a contiguous GPU tensor")
            return out
        buf = self._out
        if buf is None or buf.numel() < shape[0] * shape[1]:
            buf = self._out = torch.empty(
                self.world_size * self.m_max * self.n,
                dtype=torch.bfloat16,
                device="cuda",
            )
        return buf[: shape[0] * shape[1]].view(shape)

    def _check_operands(self, a_fp8, b_preshuffled, a_scale, b_scale):
        """Reject what the launch would reinterpret as bytes; return flat scales."""
        m = a_fp8.shape[0]
        for name, t, shape in (
            ("a_fp8", a_fp8, (m, self.k)),
            ("b_preshuffled", b_preshuffled, (self.n, self.k)),
        ):
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.dtype not in FP8_DTYPES:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in FP8_DTYPES]}, got "
                    f"{t.dtype}; the launch reinterprets it as raw bytes"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        for name, t, numel in (("a_scale", a_scale, m), ("b_scale", b_scale, self.n)):
            if t.dtype != torch.float32:
                raise ValueError(f"{name} must be torch.float32, got {t.dtype}")
            if t.numel() != numel:
                raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        return a_scale.reshape(-1), b_scale.reshape(-1).contiguous()

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        Pad before quantising. The returned view is overwritten by the next
        ``pad_rows``.
        """
        if x.dim() != 2:
            raise ValueError(f"expected a 2-D [M, K] tensor, got {tuple(x.shape)}")
        m, k = x.shape
        if m_pad < m:
            raise ValueError(f"m_pad={m_pad} is smaller than the input's M={m}")
        buf = self._pad_in
        if (
            buf is None
            or buf.shape[0] < m_pad
            or buf.shape[1] != k
            or buf.dtype != x.dtype
            or buf.device != x.device
        ):
            buf = self._pad_in = torch.zeros((m_pad, k), dtype=x.dtype, device=x.device)
        buf = buf[:m_pad]
        buf[:m].copy_(x)
        buf[m:].zero_()
        return buf

    # --- checks and lifetime -----------------------------------------------

    def self_test(self) -> None:
        """Verify that the transport moves bytes; raise if it does not.

        A mori built with ``BUILD_CCO_SDMA=OFF``, or run without
        ``MORI_ENABLE_SDMA=1``, still launches every kernel and returns from
        every put; the puts move nothing and the GEMM reads whatever the
        receive slots held. So: fill this rank's slot with ``rank+1``, run the
        split push at ``m_max``, and check every peer's slot came back as its
        source's value. A collective: every rank must call it.
        """
        self._check_open()
        m = self.m_max
        plan = self._compiled(m)
        cfg = AgGemmConfig(
            world_size=self.world_size, m_local=m, k=self.k, block_m=BLOCK_M
        )
        phases = build_phases(cfg, self.rank, sdma_queues=self.sdma_queues)
        slots = plan.rows.view(self.world_size, m, self.k)
        slots.zero_()
        slots[self.rank].fill_(self.rank + 1)
        torch.cuda.synchronize()
        self.comm.barrier()
        stream = fx.Stream(torch.cuda.current_stream())
        for p in ("push", "barrier0", "barrier1"):
            phases[p](self.dev_comm.ptr, self.win.handle, stream=stream)
        torch.cuda.current_stream().synchronize()
        for src in range(self.world_size):
            if src != self.rank and not bool((slots[src] == src + 1).all()):
                raise RuntimeError(
                    f"ag_gemm self-test failed: the slot from rank {src} holds "
                    f"{int(slots[src].flatten()[0])}, expected {src + 1}. Check "
                    f"that mori was built with BUILD_CCO_SDMA=ON and that "
                    f"MORI_ENABLE_SDMA=1 is set: without either, every put "
                    f"silently does nothing."
                )

    def _check_open(self) -> None:
        if self.mem is None:
            raise RuntimeError("this AgGemmOp has been closed")

    def close(self) -> None:
        """Release the window and its memory. Idempotent; synchronise first if
        a call may still be in flight. The dev-comm is left to the communicator,
        as ``GemmAllReduceOp`` does."""
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._pad_in = self._out = None
        for handle in (self.win, self.mem):
            if handle is not None:
                handle.close()
        self.win = self.mem = None

    def __enter__(self) -> AgGemmOp:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def shape_constraint(n: int, k: int) -> str | None:
    """Why ``(n, k)`` cannot be served, or None."""
    if k % BLOCK_K:
        return f"k={k} must be a multiple of {BLOCK_K}"
    if k < MIN_K:
        return (
            f"k={k} is below the minimum {MIN_K}: the mainloop prefetches a "
            f"second K block and runs two tail steps"
        )
    if n < 1 or n % BLOCK_N:
        return f"n={n} must be a positive multiple of {BLOCK_N}"
    return None


def supports(n: int, k: int, world_size: int) -> bool:
    """Whether this shape is expressible, which is not whether it is faster."""
    return 2 <= world_size <= 8 and shape_constraint(n, k) is None


__all__ = ["MODES", "QUANTS", "AgGemmOp", "shape_constraint", "supports"]
