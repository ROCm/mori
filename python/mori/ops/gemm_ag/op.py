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
"""``GemmAgOp``: a GEMM whose output every rank needs, with the all-gather fused.

Every rank passes its own ``A [M, K]`` and the same ``B [N, K]``, and gets back
``[world*M, N]``, source rank ``r`` at rows ``[r*M, (r+1)*M)``: a projection
computed on a context- or sequence-parallel token shard whose output every rank
consumes -- DeepSeek-V4's compressor ``wkv_gate`` is the one it was built for.

Two precisions, one interface:

``in_dtype="fp8"``
    gemm_ar's 8-wave fp8 GEMM, ``quant`` ptpc / blockscale / mxfp8, bf16 out.
``in_dtype="bf16"``
    the quad-subtile bf16 GEMM, bf16 or fp32 out (``wkv_gate`` is fp32), with
    the tile picked per M from the measured table (``pick_tile``).

Two modes, both over the copy engines:

``fused`` (default)
    The GEMM writes C into this rank's receive slot and its epilogue pushes each
    finished chunk of rows to every peer; a one-block kernel drains and barriers.
``split``
    The same GEMM into the slot, then one push per peer. The baseline, and a
    fallback.

**Recv reuse.** The returned tensor is this op's receive buffer, and the next
call refills it. A peer must not push into it while this rank still reads the
previous call's, and the end-of-call barrier does not prevent that; each call
therefore posts a credit at its start and waits for a destination's before
pushing to it (``_credit``). So it is safe **provided whatever reads the result
is ordered before this rank's next call** -- the same stream, or an event the
next call's stream waits on. Clone the result to keep it longer.
"""

from __future__ import annotations

from typing import NamedTuple

import flydsl.expr as fx
import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.tensor_utils import from_gpu_ptr

from ..gemm_ar.op import (
    FP8_DTYPES,
    MIN_K,
    _flatten_a_scale,
    _flatten_mxfp8_a_scale,
    _PinnedLaunch,
)
from ._gemm_a16w16_8wave import BLOCK_K as BF16_BLOCK_K
from ._gemm_a16w16_8wave import compile_bf16_gemm_ag, pick_chunks, pick_tile
from .kernels_fused import compile_fused_gemm_ag, compile_gemm_local
from .kernels_sdma import build_sdma_phases
from .layout import (
    DEFAULT_BLOCK_N,
    MXFP8_BLOCK,
    MXFP8_BLOCK_M,
    ag_config,
    counter_chunks,
)

IN_DTYPES = ("fp8", "bf16")
QUANTS = ("ptpc", "blockscale", "mxfp8")
OUT_DTYPES = ("bf16", "fp32")
MODES = ("fused", "split")
#: Rows are padded to this, so any tile either GEMM may pick divides them.
M_GRANULE = 256
#: N for every tile either GEMM may pick.
N_GRANULE = DEFAULT_BLOCK_N
#: The fp8 GEMM's row tile; mxfp8 needs MXFP8_BLOCK_M for its packed A scale.
FP8_BLOCK_M = 128
#: blockscale's group along K, on both operands.
SCALE_BLOCK_K = 128
#: Ceiling on chunks per call, hence on the SDMA queues the fused path needs:
#: the lane-parallel epilogue keeps two chunks of one destination apart by
#: queue. pick_chunks measured 4 within 2% of 8 where it would pick 8, and a
#: queue per chunk per peer is what runs out of engine slots first.
DEFAULT_MAX_CHUNKS = 4


class _Plan(NamedTuple):
    gemm: object
    tail: object
    recv: torch.Tensor
    slot: torch.Tensor


class GemmAgOp:
    """Fused GEMM + all-gather over cco SDMA.

    One instance serves one ``(n, k)`` weight and precision, and any M up to
    ``m_max`` (padded, see :meth:`padded_m`). It owns its symmetric window,
    which cannot grow, so size ``m_max`` for the largest M the deployment sees.
    """

    def __init__(
        self,
        comm,
        *,
        n: int,
        k: int,
        m_max: int,
        in_dtype: str = "fp8",
        quant: str = "ptpc",
        out_dtype: str = "bf16",
        mode: str = "fused",
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_shapes: int = 8,
        chunks: int | None = None,
        tile: tuple[int, int] | None = None,
        war_credit: bool = True,
    ):
        """``chunks`` pins the fused chunk count instead of picking it per M
        (rounded down to a divisor of the row tiles, and capped by
        ``max_chunks``). ``tile`` pins ``(BLOCK_M, BLOCK_N)`` for every M instead of picking
        it per M; an illegal one is rejected here. ``war_credit=False`` drops
        the recv-reuse credit, and exists only so the race test can show the
        credit is what keeps it from failing."""
        self._closed = True  # a failed constructor leaves close() a no-op
        self.mem = self.win = self.dev_comm = None
        self._cache: dict[int, _Plan] = {}
        self._shape_slot: dict[int, int] = {}
        self._pad_in: torch.Tensor | None = None

        why = shape_constraint(
            n, k, in_dtype=in_dtype, quant=quant, out_dtype=out_dtype
        )
        if why is not None:
            raise ValueError(f"unsupported configuration: {why}")
        if mode not in MODES:
            raise ValueError(f"mode={mode!r} is not one of {MODES}")
        if chunks is not None and chunks < 1:
            raise ValueError(f"chunks must be >= 1, got {chunks}")
        if m_max < 1 or max_chunks < 1 or max_shapes < 1:
            raise ValueError(
                f"m_max, max_chunks and max_shapes must be >= 1; got m_max={m_max} "
                f"max_chunks={max_chunks} max_shapes={max_shapes}"
            )
        self.comm = comm
        self.rank = comm.rank
        self.world_size = comm.nranks
        self.n, self.k = n, k
        self.in_dtype, self.quant, self.out_dtype = in_dtype, quant, out_dtype
        self.mode = mode
        self.max_chunks = max_chunks
        self.max_shapes = max_shapes
        self.war_credit = war_credit
        self.tile = tuple(tile) if tile is not None else None
        self.chunks = chunks
        self.elem_bytes = 4 if out_dtype == "fp32" else 2
        self.out_torch = torch.float32 if out_dtype == "fp32" else torch.bfloat16
        #: One queue per chunk per peer on the fused path; one on the split one.
        self.sdma_queues = max_chunks if mode == "fused" else 1
        self.m_max = self.padded_m(m_max)
        if self.tile is not None:
            bm, bn = self.tile
            if self.in_dtype == "fp8" and (
                bn != N_GRANULE
                or bm != (MXFP8_BLOCK_M if quant == "mxfp8" else bm)
                or bm not in (128, 256)
            ):
                raise ValueError(
                    f"tile={self.tile}: the fp8 GEMM takes BLOCK_N=256 and "
                    f"BLOCK_M 128 or 256 (256 under mxfp8)"
                )
        # Validates the tile against the layout before anything is allocated.
        self.window_bytes = self._cfg(self.m_max, slot=0).window_bytes
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
            reqs.sdma_queue_count = self.sdma_queues
            self.dev_comm = comm.create_dev_comm(reqs)
            # A put on a queue that does not exist moves nothing, and the op
            # would return the previous call's rows without a word.
            if self.dev_comm.sdma_num_queue < self.sdma_queues:
                raise RuntimeError(
                    f"GemmAgOp needs {self.sdma_queues} SDMA queue(s) per peer but "
                    f"the communicator has {self.dev_comm.sdma_num_queue}: build "
                    f"mori with BUILD_CCO_SDMA=ON, set MORI_ENABLE_SDMA=1, make sure "
                    f"the first dev-comm created on this communicator asked for at "
                    f"least that many, or lower max_chunks"
                )
            if self.dev_comm.lsa_size != self.world_size:
                raise RuntimeError(
                    f"GemmAgOp needs every rank in one LSA domain (one node); "
                    f"lsa_size={self.dev_comm.lsa_size}, world_size={self.world_size}"
                )
            # Every rank's zeroing has to land before any peer's first push,
            # flag or credit into this window.
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
        out_dtype: str = "bf16",
        max_shapes: int = 8,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> int:
        """Window bytes the constructor will allocate, for sizing the
        communicator's VMM before the op exists."""
        m = (m_max + M_GRANULE - 1) // M_GRANULE * M_GRANULE
        return ag_config(
            world_size=world_size,
            m=m,
            n=n,
            elem_bytes=4 if out_dtype == "fp32" else 2,
            block_m=M_GRANULE,
            block_n=N_GRANULE,
            counter_capacity=max(8, max_chunks),
            counter_shape_slots=max_shapes,
        ).window_bytes

    # --- shapes ------------------------------------------------------------

    def padded_m(self, m: int) -> int:
        """Rows to pass for logical ``m``: a multiple of 256, which every tile
        either GEMM may pick divides."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        return (m + M_GRANULE - 1) // M_GRANULE * M_GRANULE

    def tile_for(self, m: int) -> tuple[int, int]:
        """``(BLOCK_M, BLOCK_N)`` at this M."""
        if self.tile is not None:
            return self.tile
        if self.in_dtype == "bf16":
            return pick_tile(m, self.n, self.out_dtype, fused=self.mode == "fused")
        return (MXFP8_BLOCK_M if self.quant == "mxfp8" else FP8_BLOCK_M), N_GRANULE

    def chunks_for(self, m: int) -> int:
        """Chunks the fused epilogue pushes a call in at this M: one per
        workgroup round of the tile (pick_chunks), capped by ``max_chunks``."""
        if self.mode != "fused":
            return 1
        tile = self.tile_for(m)
        if self.chunks is not None:
            return counter_chunks(m // tile[0], min(self.chunks, self.max_chunks))
        return pick_chunks(m, self.n, tile, max_chunks=self.max_chunks)

    def _slot(self, m: int) -> int:
        """This M's counter set, assigned on first sight. The election counters
        are never reset, so two shapes sharing a set would elect on each
        other's residue."""
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
        bm, bn = self.tile_for(m)
        return ag_config(
            world_size=self.world_size,
            m=m,
            n=self.n,
            elem_bytes=self.elem_bytes,
            block_m=bm,
            block_n=bn,
            counter_chunks=self.chunks_for(m),
            # Pinned to the instance, not the shape, so every M sees the control
            # region and the start of recv at the same offsets. recv itself is
            # laid out for the current M -- contiguous [world*M, N] -- which is
            # safe across shapes for the same reason consecutive calls are: the
            # credit holds every push until its owner has finished reading.
            counter_capacity=max(8, self.max_chunks),
            counter_shape_slots=self.max_shapes,
            counter_shape_index=slot,
        )

    def _check_m(self, m: int) -> None:
        if m % M_GRANULE:
            raise ValueError(
                f"M={m} must be a multiple of {M_GRANULE}; use padded_m() and "
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
        bm, bn = cfg.block_m, cfg.block_n
        common = dict(BLOCK_M=bm, BLOCK_N=bn)
        if self.in_dtype == "bf16":
            gemm = compile_bf16_gemm_ag(
                cfg,
                self.rank,
                K=self.k,
                out_dtype=self.out_dtype,
                fuse=fused,
                chunks=cfg.counter_chunks,
                sdma_queues=self.sdma_queues,
                war_credit=self.war_credit,
                **common,
            )
        elif fused:
            gemm = compile_fused_gemm_ag(
                cfg,
                self.rank,
                K=self.k,
                quant=self.quant,
                fuse=True,
                chunks=cfg.counter_chunks,
                sdma_queues=self.sdma_queues,
                war_credit=self.war_credit,
                **common,
            )
        else:
            # gemm-only's kernel with C pointed at this rank's slot.
            gemm = compile_gemm_local(
                cfg, self.rank, K=self.k, quant=self.quant, **common
            )
        phases = build_sdma_phases(
            cfg, self.rank, queues=self.sdma_queues, war_credit=self.war_credit
        )
        recv = from_gpu_ptr(
            self.mem.ptr + cfg.recv_off, (self.world_size * m, self.n), self.out_torch
        )
        hit = _Plan(
            gemm=_PinnedLaunch(gemm),
            tail=_PinnedLaunch(phases["drain" if fused else "gather"]),
            recv=recv,
            slot=recv[self.rank * m : (self.rank + 1) * m],
        )
        self._cache[m] = hit
        return hit

    # --- the call ----------------------------------------------------------

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor | None = None,
        b_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One GEMM + all-gather; returns ``[world*M, N]`` in ``out_dtype``.

        ``a`` is this rank's ``[M, K]`` with M from :meth:`padded_m`, the same M
        on every rank; ``b`` is ``[N, K]``, identical on every rank.

        ``fp8``: ``a``/``b`` are float8_e4m3fn, ``b`` through
        :func:`~mori.ops.gemm_ar.preshuffle_b`. Scales follow ``quant``, as in
        :class:`~mori.ops.gemm_ar.GemmAllReduceOp`: ptpc ``a_scale [M]``,
        ``b_scale [N]`` fp32; blockscale ``a_scale [M, K/128]`` (column-major,
        or ``[K/128, M]``, or flat) and ``b_scale [N/128, K/128]`` fp32; mxfp8
        ``a_scale`` from :func:`~mori.ops.gemm_ar.preshuffle_a_scale` and
        ``b_scale`` the ``[N/32, K/32]`` ue8m0 bytes K-block major as int32.

        ``bf16``: ``a``/``b`` are bfloat16, ``b`` plain row-major; no scales.

        The result aliases this op's window and the next call overwrites it.
        Whatever reads it must be ordered before this rank's next call on the
        device (same stream, or an event); clone it to keep it longer.
        """
        self._check_open()
        m = a.shape[0] if a.dim() == 2 else -1
        self._check_m(m)
        args = self._operands(a, b, a_scale, b_scale, m)
        plan = self._compiled(m)
        stream = fx.Stream(torch.cuda.current_stream())
        plan.gemm(
            *args[:2],
            plan.slot.reshape(-1),
            *args[2:],
            m,
            self.n,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        plan.tail(self.dev_comm.ptr, self.win.handle, stream=stream)
        return plan.recv

    def _operands(self, a, b, a_scale, b_scale, m):
        """Check what the launch would reinterpret as bytes; return its args."""
        bf16 = self.in_dtype == "bf16"
        want = (torch.bfloat16,) if bf16 else FP8_DTYPES
        for name, t, shape in (("a", a, (m, self.k)), ("b", b, (self.n, self.k))):
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.dtype not in want:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in want]} for "
                    f"in_dtype={self.in_dtype!r}, got {t.dtype}; the launch "
                    f"reinterprets it as raw bytes"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        if bf16:
            if a_scale is not None or b_scale is not None:
                raise ValueError("in_dtype='bf16' takes no scales")
            # int16 views: the bf16 buffer tensor reuses the incoming layout,
            # and a byte-counted one would describe twice the elements.
            return (
                a.contiguous().view(torch.int16).view(-1),
                b.contiguous().view(torch.int16).view(-1),
            )
        if a_scale is None or b_scale is None:
            raise ValueError(f"quant={self.quant!r} needs a_scale and b_scale")
        if self.quant == "mxfp8":
            sa = _flatten_mxfp8_a_scale(a_scale, m, self.k)
            sb_numel = (self.n // MXFP8_BLOCK) * (self.k // MXFP8_BLOCK)
            sdtype = torch.int32
        elif self.quant == "blockscale":
            sa = _flatten_a_scale(a_scale, m, self.k // SCALE_BLOCK_K)
            sb_numel = (self.n // SCALE_BLOCK_K) * (self.k // SCALE_BLOCK_K)
            sdtype = torch.float32
        else:
            sa = a_scale.reshape(-1)
            sb_numel = self.n
            sdtype = torch.float32
            if sa.numel() != m:
                raise ValueError(f"a_scale must have {m} elements, got {sa.numel()}")
        for name, t in (("a_scale", sa), ("b_scale", b_scale)):
            if t.dtype != sdtype:
                raise ValueError(f"{name} must be {sdtype}, got {t.dtype}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        if b_scale.numel() != sb_numel:
            raise ValueError(
                f"b_scale must have {sb_numel} elements, got {b_scale.numel()}"
            )
        return (
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            sa.contiguous(),
            b_scale.reshape(-1).contiguous(),
        )

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        Pad before quantising. The added rows come back as zero rows in every
        source's block of the result; slice them off per source. The returned
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
        every put; the puts move nothing and the result holds whatever the
        window did. So: fill this rank's slot with ``rank+1``, run the split
        push at ``m_max``, and check every peer's slot came back as its source's
        value. A collective: every rank must call it.
        """
        self._check_open()
        m = self.m_max
        cfg = self._cfg(m, slot=self._slot(m))
        phases = build_sdma_phases(
            cfg, self.rank, queues=self.sdma_queues, war_credit=self.war_credit
        )
        recv = from_gpu_ptr(
            self.mem.ptr + cfg.recv_off, (self.world_size, m, self.n), self.out_torch
        )
        recv.zero_()
        recv[self.rank].fill_(float(self.rank + 1))
        torch.cuda.synchronize()
        self.comm.barrier()
        phases["gather"](
            self.dev_comm.ptr,
            self.win.handle,
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        torch.cuda.current_stream().synchronize()
        for src in range(self.world_size):
            if src != self.rank and not bool((recv[src] == float(src + 1)).all()):
                raise RuntimeError(
                    f"gemm_ag self-test failed: the slot from rank {src} holds "
                    f"{recv[src].flatten()[0].item():g}, expected {src + 1}. Check "
                    f"that mori was built with BUILD_CCO_SDMA=ON and that "
                    f"MORI_ENABLE_SDMA=1 is set: without either, every put "
                    f"silently does nothing."
                )

    def _check_open(self) -> None:
        if self.mem is None:
            raise RuntimeError("this GemmAgOp has been closed")

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

    def __enter__(self) -> GemmAgOp:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def shape_constraint(
    n: int,
    k: int,
    *,
    in_dtype: str = "fp8",
    quant: str = "ptpc",
    out_dtype: str = "bf16",
) -> str | None:
    """Why this configuration cannot be served, or None."""
    if in_dtype not in IN_DTYPES:
        return f"in_dtype={in_dtype!r} is not one of {IN_DTYPES}"
    if out_dtype not in OUT_DTYPES:
        return f"out_dtype={out_dtype!r} is not one of {OUT_DTYPES}"
    if in_dtype == "fp8":
        if quant not in QUANTS:
            return f"quant={quant!r} is not one of {QUANTS}"
        if out_dtype != "bf16":
            return "the fp8 GEMM's epilogue only emits bf16"
        step, min_k = SCALE_BLOCK_K, MIN_K
    else:
        if quant != "ptpc":
            return "quant does not apply to in_dtype='bf16'; leave it at its default"
        step, min_k = BF16_BLOCK_K, 2 * BF16_BLOCK_K
    if k % step or k < min_k:
        return f"K={k} must be a multiple of {step} and at least {min_k}"
    if n < 1 or n % N_GRANULE:
        return f"N={n} must be a positive multiple of {N_GRANULE}"
    return None


def supports(n: int, k: int, world_size: int, **kwargs) -> bool:
    """Whether this shape is expressible, which is not whether it is faster."""
    return 2 <= world_size <= 8 and shape_constraint(n, k, **kwargs) is None


__all__ = [
    "IN_DTYPES",
    "MODES",
    "OUT_DTYPES",
    "QUANTS",
    "GemmAgOp",
    "shape_constraint",
    "supports",
]
