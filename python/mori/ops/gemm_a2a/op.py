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
"""``GemmA2AOp``: the fp8 GEMM with its column-sharded all-to-all fused in.

Every rank passes its own ``A [M, K]`` and the same ``B [N, K]``; every rank gets
back ``[world*M, N/world]`` bf16, source rank ``r`` at rows ``[r*M, (r+1)*M)`` --
the shape a QKV projection under Ulysses sequence parallelism needs.

Two modes, both over the copy engines:

``fused`` (default)
    The GEMM epilogue pushes each chunk to its destination as soon as the
    chunk's tiles are done, then a one-block kernel drains and barriers.
``split``
    The same GEMM writes the per-destination staging slabs, then one push per
    destination. The baseline ``fused`` is measured against, and a fallback.

The LSA paths in this package are not offered here: on this operator they lose
to SDMA, and the fused one writes into peers from every GEMM block, which the
recv-reuse credit (below) would have to check in every block.

**Recv reuse.** The returned tensor is this op's receive buffer, and the next
call writes it. A peer must not do that while this rank is still reading the
previous call's, and nothing about the end-of-call barrier prevents it: a peer
passes the barrier, starts its next call, and pushes. Each call therefore posts
a credit at its start and waits for a destination's credit before pushing to it
(``_credit``). That makes it safe **provided whatever reads the returned tensor
is ordered before this rank's next call** -- on the same stream, or behind an
event the next call's stream waits on. Clone the result to keep it longer.
"""

from __future__ import annotations

from typing import NamedTuple

import flydsl.expr as fx
import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.tensor_utils import from_gpu_ptr

from ..gemm_ar.op import MIN_K, FP8_DTYPES, _flatten_mxfp8_a_scale, _PinnedLaunch
from .kernels_fused import BLOCK_K, compile_fused_gemm_a2a
from .kernels_sdma import build_sdma_phases
from .layout import (
    DEFAULT_BLOCK_N,
    MAX_CHUNKS,
    MXFP8_BLOCK,
    MXFP8_BLOCK_M,
    TILE_M_GRANULE,
    a2a_config,
    default_block_m,
    default_chunks,
)

#: Quantisations the op takes. blockscale compiles but has not been measured on
#: this operator.
QUANTS = ("ptpc", "mxfp8")
MODES = ("fused", "split")


class _Plan(NamedTuple):
    gemm: object
    tail: object
    recv: torch.Tensor


class GemmA2AOp:
    """Fused fp8 GEMM + column-sharded all-to-all over cco SDMA.

    One instance serves one ``(n, k)`` weight and any ``M`` up to ``m_max``
    (padded, see :meth:`padded_m`). It owns its symmetric window, which cannot
    grow, so size ``m_max`` for the largest M the deployment will see.
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
        sdma_queues: int = 1,
        max_shapes: int = 8,
        chunks: int | None = None,
        war_credit: bool = True,
    ):
        """``war_credit=False`` drops the recv-reuse credit. It exists only so
        the race test can show the credit is what keeps it from failing."""
        self._closed = True  # a failed constructor leaves close() a no-op
        self.mem = self.win = self.dev_comm = None
        self._cache: dict[int, _Plan] = {}
        self._shape_slot: dict[int, int] = {}
        self._pad_in: torch.Tensor | None = None

        if quant not in QUANTS:
            raise ValueError(f"quant={quant!r} is not one of {QUANTS}")
        if mode not in MODES:
            raise ValueError(f"mode={mode!r} is not one of {MODES}")
        if m_max < 1 or sdma_queues < 1 or max_shapes < 1:
            raise ValueError(
                f"m_max, sdma_queues and max_shapes must be >= 1; got m_max={m_max} "
                f"sdma_queues={sdma_queues} max_shapes={max_shapes}"
            )
        if chunks is not None and not 1 <= chunks:
            raise ValueError(f"chunks must be >= 1, got {chunks}")
        if k % BLOCK_K:
            raise ValueError(f"k={k} must be a multiple of {BLOCK_K}")
        if k < MIN_K:
            raise ValueError(
                f"k={k} is below the minimum {MIN_K}: the mainloop prefetches a "
                f"second K block and runs two tail steps, so k/{BLOCK_K} must be >= 2"
            )
        if quant == "mxfp8" and n % MXFP8_BLOCK:
            raise ValueError(f"mxfp8 needs n % {MXFP8_BLOCK} == 0, got n={n}")
        self.comm = comm
        self.rank = comm.rank
        self.world_size = comm.nranks
        self.n, self.k = n, k
        self.quant, self.mode = quant, mode
        self.sdma_queues = sdma_queues
        self.max_shapes = max_shapes
        self.chunks = chunks
        self.war_credit = war_credit
        #: M rounds up to this: the tile's M granule, 256 under mxfp8.
        self.m_granule = MXFP8_BLOCK_M if quant == "mxfp8" else TILE_M_GRANULE
        self.m_max = self.padded_m(m_max)
        self._num_cus = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        # Validates n against world_size * block_n before anything is allocated.
        cap = self._cfg(self.m_max, slot=0)
        self.shard_n = cap.shard_n
        self.window_bytes = cap.window_bytes
        try:
            self.mem = comm.alloc_mem(self.window_bytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            # Counters, flags and credits are read before they are first
            # written, so the window has to start zeroed.
            from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = sdma_queues
            self.dev_comm = comm.create_dev_comm(reqs)
            # Both modes push over SDMA. cco fixes the queue count at the comm's
            # first dev-comm and only logs when a later one asks for more, or
            # when SDMA is off altogether; a put on a queue that does not exist
            # moves nothing, so the op would return stale bytes without a word.
            if self.dev_comm.sdma_num_queue < sdma_queues:
                raise RuntimeError(
                    f"GemmA2AOp needs {sdma_queues} SDMA queue(s) per peer but the "
                    f"communicator has {self.dev_comm.sdma_num_queue}: build mori "
                    f"with BUILD_CCO_SDMA=ON, set MORI_ENABLE_SDMA=1, and make sure "
                    f"the first dev-comm created on this communicator asked for at "
                    f"least that many"
                )
            # World ranks are used as LSA ranks throughout.
            if self.dev_comm.lsa_size != self.world_size:
                raise RuntimeError(
                    f"GemmA2AOp needs every rank in one LSA domain (one node); "
                    f"lsa_size={self.dev_comm.lsa_size}, world_size={self.world_size}"
                )
            # Every rank's zeroing has to land before any peer's first push or
            # credit into this window.
            torch.cuda.synchronize()
            comm.barrier()
        except Exception:
            self._closed = False
            self.close()
            raise
        self._closed = False

    @staticmethod
    def window_bytes_for(
        world_size: int,
        *,
        n: int,
        m_max: int,
        quant: str = "ptpc",
        max_shapes: int = 8,
        chunks: int | None = None,
    ) -> int:
        """Window bytes the constructor will allocate, for sizing the
        communicator's VMM before the op exists. The tile does not enter it:
        payload is laid out for the capacity M and the counters for MAX_CHUNKS."""
        g = MXFP8_BLOCK_M if quant == "mxfp8" else TILE_M_GRANULE
        cap = (m_max + g - 1) // g * g
        return a2a_config(
            world_size=world_size,
            m=cap,
            n=n,
            block_m=g,
            staged=True,
            counter_chunks=1,
            counter_capacity=max(MAX_CHUNKS, chunks or 0),
            counter_shape_slots=max_shapes,
            capacity_m=cap,
        ).window_bytes

    # --- shapes ------------------------------------------------------------

    def padded_m(self, m: int) -> int:
        """Rows to pass for logical ``m``: rounded up to the tile's M granule."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        g = self.m_granule
        return (m + g - 1) // g * g

    def block_m_for(self, m: int) -> int:
        return default_block_m(m, self.n, self.quant, num_cus=self._num_cus)

    def chunks_for(self, m: int) -> int:
        bm = self.block_m_for(m)
        if self.chunks is None:
            return default_chunks(m, self.n // self.world_size, bm)
        # Round a request down to a divisor of the row tiles, as the bench does.
        for c in range(min(self.chunks, m // bm), 0, -1):
            if (m // bm) % c == 0:
                return c
        return 1

    def _slot(self, m: int) -> int:
        """This M's counter set, assigned on first sight. Every call leaves its
        election counters back at zero; a set per M is still kept, so a shape's
        counters never depend on another shape's calls having completed."""
        slot = self._shape_slot.get(m)
        if slot is None:
            if len(self._shape_slot) >= self.max_shapes:
                raise ValueError(
                    f"this op already serves {self.max_shapes} distinct M values "
                    f"{sorted(self._shape_slot)}; M={m} needs another counter set. "
                    f"Raise max_shapes or pad to fewer distinct M."
                )
            slot = self._shape_slot[m] = len(self._shape_slot)
        return slot

    def _cfg(self, m: int, *, slot: int):
        return a2a_config(
            world_size=self.world_size,
            m=m,
            n=self.n,
            block_m=self.block_m_for(m),
            block_n=DEFAULT_BLOCK_N,
            staged=True,
            counter_chunks=self.chunks_for(m),
            # Pinned to the instance, not the shape: every M has to see the
            # control region and payload at the same offsets.
            counter_capacity=max(MAX_CHUNKS, self.chunks or 0),
            counter_shape_slots=self.max_shapes,
            counter_shape_index=slot,
            capacity_m=self.m_max,
        )

    def _check_m(self, m: int) -> None:
        if m % self.m_granule:
            raise ValueError(
                f"M={m} must be a multiple of {self.m_granule}; use padded_m() and "
                f"pad_rows()"
            )
        if m > self.m_max:
            raise ValueError(f"M={m} exceeds the window's m_max={self.m_max}")

    def _compiled(self, m: int) -> _Plan:
        hit = self._cache.get(m)
        if hit is not None:
            return hit
        cfg = self._cfg(m, slot=self._slot(m))
        fused = self.mode == "fused"
        gemm = compile_fused_gemm_a2a(
            cfg,
            self.rank,
            K=self.k,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            quant=self.quant,
            transport="sdma",
            fuse=fused,
            chunks=cfg.counter_chunks,
            sdma_queues=self.sdma_queues,
            war_credit=self.war_credit,
        )
        phases = build_sdma_phases(
            cfg, self.rank, queues=self.sdma_queues, war_credit=self.war_credit
        )
        hit = _Plan(
            gemm=_PinnedLaunch(gemm),
            tail=_PinnedLaunch(phases["drain" if fused else "scatter"]),
            recv=from_gpu_ptr(
                self.mem.ptr + cfg.recv_off,
                (self.world_size * m, cfg.shard_n),
                torch.bfloat16,
            ),
        )
        self._cache[m] = hit
        return hit

    # --- the call ----------------------------------------------------------

    def __call__(
        self,
        a_fp8: torch.Tensor,
        b_preshuffled: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        """One GEMM + all-to-all; returns ``[world*M, N/world]`` bf16.

        ``a_fp8`` is ``[M, K]`` with M from :meth:`padded_m`; ``b_preshuffled``
        is ``[N, K]`` through :func:`~mori.ops.gemm_ar.preshuffle_b`, identical
        on every rank.

        ``ptpc``: ``a_scale`` is ``[M]`` (or ``[M, 1]``) and ``b_scale`` ``[N]``
        (or ``[1, N]``), fp32 -- per token and per channel.

        ``mxfp8``: ``a_scale`` is the flat int32 buffer
        :func:`~mori.ops.gemm_ar.preshuffle_a_scale` returns, and ``b_scale``
        the ``[N/32, K/32]`` ue8m0 bytes K-block major and widened, i.e.
        ``e.t().contiguous().to(torch.int32)``.

        The result aliases this op's window and the next call overwrites it.
        Whatever reads it must be ordered before this rank's next call on the
        device (same stream, or an event), which is what lets the next call's
        peers know it has been read; clone it to keep it longer.
        """
        if self.mem is None:
            raise RuntimeError("this GemmA2AOp has been closed")
        m = a_fp8.shape[0]
        self._check_m(m)
        a_scale, b_scale = self._check_operands(a_fp8, b_preshuffled, a_scale, b_scale)
        plan = self._compiled(m)
        stream = fx.Stream(torch.cuda.current_stream())
        plan.gemm(
            a_fp8.contiguous().view(torch.int8).view(-1),
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            # The C argument only sizes a descriptor the staging store does
            # not use; the bytes go to the window.
            plan.recv.view(-1),
            a_scale,
            b_scale,
            m,
            self.n,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        plan.tail(self.dev_comm.ptr, self.win.handle, stream=stream)
        return plan.recv

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
        if self.quant == "mxfp8":
            a_scale = _flatten_mxfp8_a_scale(a_scale, m, self.k)
            want = (
                ("a_scale", a_scale, torch.int32, a_scale.numel()),
                (
                    "b_scale",
                    b_scale,
                    torch.int32,
                    (self.n // MXFP8_BLOCK) * (self.k // MXFP8_BLOCK),
                ),
            )
        else:
            want = (
                ("a_scale", a_scale, torch.float32, m),
                ("b_scale", b_scale, torch.float32, self.n),
            )
        for name, t, dtype, numel in want:
            if t.dtype != dtype:
                raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
            if t.numel() != numel:
                raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        return a_scale.reshape(-1), b_scale.reshape(-1).contiguous()

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        Pad before quantising. The added rows produce zero rows in every
        source's block of the result; slice them off by source. The returned
        view is overwritten by the next ``pad_rows``.
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
        every put -- the puts move nothing, and the result is this rank's own
        block plus whatever the window held. The constructor refuses a
        communicator that reports fewer SDMA queues than asked for; this checks
        that the bytes actually move. So: write ``src+1`` into every
        staging slab, run the split push at ``m_max``, and check every remote
        block of recv came back as its source's value.
        """
        if self.mem is None:
            raise RuntimeError("this GemmA2AOp has been closed")
        m = self.m_max
        cfg = self._cfg(m, slot=self._slot(m))
        phases = build_sdma_phases(cfg, self.rank, queues=self.sdma_queues)
        ws, s = self.world_size, cfg.shard_n
        staging = from_gpu_ptr(
            self.mem.ptr + cfg.staging_off, (ws, cfg.cap_m, s), torch.bfloat16
        )
        recv = from_gpu_ptr(
            self.mem.ptr + cfg.recv_off, (ws, cfg.cap_m, s), torch.bfloat16
        )
        staging.fill_(float(self.rank + 1))
        recv.zero_()
        phases["scatter"](
            self.dev_comm.ptr,
            self.win.handle,
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        torch.cuda.current_stream().synchronize()
        for src in range(ws):
            if src == self.rank:
                continue
            got = recv[src, :m].float()
            if not bool((got == float(src + 1)).all()):
                raise RuntimeError(
                    f"gemm_a2a self-test failed: recv block from rank {src} holds "
                    f"{got.flatten()[0].item():g}, expected {src + 1}. Check that "
                    f"mori was built with BUILD_CCO_SDMA=ON and that "
                    f"MORI_ENABLE_SDMA=1 is set: without either, every put "
                    f"silently does nothing."
                )

    def close(self) -> None:
        """Release the window and its memory. Idempotent; synchronise first if
        a call may still be in flight. The dev-comm is left to the communicator,
        as ``GemmAllReduceOp`` does."""
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._pad_in = None
        for handle in (self.win, self.mem):
            if handle is not None:
                handle.close()
        self.win = self.mem = None

    def __enter__(self) -> GemmA2AOp:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["MODES", "QUANTS", "GemmA2AOp"]
