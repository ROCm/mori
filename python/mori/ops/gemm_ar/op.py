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

from typing import NamedTuple, Optional

import torch

import flydsl.expr as fx
from mori.cco import CCODevCommRequirements, GDA_CONNECTION_NONE
from mori.tensor_utils import from_gpu_ptr

from .kernels_fused import BLOCK_K, compile_fused_gemm_scatter
from .kernels_sdma import build_sdma_phases
from .layout import GATHER_TRANSPORTS, WIRE_DTYPES, ArConfig

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

# The mainloop prefetches K block 1 and then runs tail steps K_ITERS-2 and
# K_ITERS-1, so a single-K-block pipeline does not exist: K_ITERS must be >= 2.
# K=128 otherwise passes the divisibility check and returns finite garbage
# (relL2 1.41 against 2.4e-3 at K=256).
MIN_K = 2 * BLOCK_K

# Distinct M values one instance can serve. Each needs its own counter set --
# see ArConfig.counter_shape_slots -- and each also costs a FlyDSL compile, so
# this is a small number by nature rather than a tuning knob.
DEFAULT_MAX_SHAPES = 8


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


def _gemm_constraints(n: int, k: int, block_n: int) -> Optional[str]:
    """Why the GEMM cannot take this shape, or None. See :func:`supports`."""
    if k % SCALE_BLOCK_K:
        return f"K={k} must be a multiple of {SCALE_BLOCK_K} (the block-scale group)"
    if k < MIN_K:
        return (
            f"K={k} is below the minimum {MIN_K}: the mainloop prefetches a "
            f"second K block and runs two tail steps, so K/{BLOCK_K} must be >= 2"
        )
    if n % block_n:
        return f"N={n} must be a multiple of block_n={block_n}"
    if n % SCALE_BLOCK_K:
        return f"N={n} must be a multiple of {SCALE_BLOCK_K} (the B scale group)"
    return None


def _build_cfg(
    *,
    world_size: int,
    m_pad: int,
    n: int,
    block_m: int,
    gather_dtype: str,
    gather_transport: str,
    counter_capacity: int,
    shape_slots: int,
    shape_index: int,
) -> ArConfig:
    """The one place an ``ArConfig`` is built, so every caller agrees.

    ``supports``, ``window_bytes_for`` and the op itself all come through here,
    which is what keeps the support predicate from drifting away from what
    execution actually accepts.
    """
    cfg = ArConfig(
        world_size=world_size,
        m=m_pad,
        n=n,
        recv_slots=world_size,
        counter_chunks=counter_chunks(m_pad, world_size, block_m),
        counter_capacity=counter_capacity,
        counter_shape_slots=shape_slots,
        counter_shape_index=shape_index,
        gather_dtype=gather_dtype,
        gather_transport=gather_transport,
    )
    cfg.validate()
    return cfg


def supports(
    m: int,
    n: int,
    k: int,
    world_size: int,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    gather_dtype: str = "bf16",
    gather_transport: str = "lsa",
) -> bool:
    """Whether this shape is *expressible*, which is not whether it is faster.

    Profitability depends on how much GEMM there is to hide the transfer behind
    and is the caller's call -- see the measured curve in ``README.md``.

    The collective half is answered by building the config and validating it,
    rather than by a second copy of its rules: a predicate that says yes where
    construction raises, or vice versa, is worse than no predicate.
    """
    if m <= 0 or _gemm_constraints(n, k, block_n) is not None:
        return False
    try:
        _build_cfg(
            world_size=world_size,
            m_pad=padded_m(m, world_size, block_m),
            n=n,
            block_m=block_m,
            gather_dtype=gather_dtype,
            gather_transport=gather_transport,
            counter_capacity=MAX_CHUNKS,
            shape_slots=DEFAULT_MAX_SHAPES,
            shape_index=0,
        )
    except ValueError:
        return False
    return True


#: fp8 encodings the kernel's byte reinterpretation is valid for. Both are 1
#: byte and the GEMM only ever sees the bytes, but forwarding a bf16 tensor here
#: reinterprets two rows as one and produces finite nonsense.
FP8_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


def _flatten_a_scale(a_scale: torch.Tensor, m: int, kb: int) -> torch.Tensor:
    """A's block scales in the physical order the kernel indexes.

    The kernel reads element ``(row, kb)`` at ``kb * M + row`` -- the layout
    ``gemm_a8w8_blockscale_bpreshuffle`` consumes, and what sglang's
    ``materialize_bpreshuffle_fp8_scale`` produces. Three spellings are accepted
    and they mean different things:

    * ``[M, K/128]`` -- the logical shape. Transposed here, which is free for the
      column-major tensor the quantiser returns (``.t()`` is then contiguous) and
      a copy for a row-major one. **Passing this through ``reshape(-1)`` instead
      is the bug this function exists to prevent**: for the column-major case
      that walks the logical rows, pairing every scale with the wrong K block,
      which still validates and still runs (relL2 0.26 against 2.4e-3).
    * ``[K/128, M]`` -- the physical shape, taken as-is.
    * 1-D -- already flattened in physical order, taken as-is.

    ``m == kb`` would be ambiguous; the logical reading wins, and K would have to
    be ``128 * M`` for it to arise.
    """
    if a_scale.dim() == 1:
        if a_scale.numel() != m * kb:
            raise ValueError(
                f"a_scale has {a_scale.numel()} elements, expected M*K/128 = {m * kb}"
            )
        return a_scale
    if a_scale.dim() != 2:
        raise ValueError(
            f"a_scale must be [M, K/128], [K/128, M] or flat, got shape "
            f"{tuple(a_scale.shape)}"
        )
    if tuple(a_scale.shape) == (m, kb):
        return a_scale.t().reshape(-1)
    if tuple(a_scale.shape) == (kb, m):
        return a_scale.reshape(-1)
    raise ValueError(
        f"a_scale shape {tuple(a_scale.shape)} matches neither [M, K/128] = "
        f"{(m, kb)} nor its transpose"
    )


class _Plan(NamedTuple):
    """One M's compiled pipeline.

    ``tail`` is already resolved from ``parts["fused_order"]`` into the
    launchers to call, in order -- the hot path should not be looking phase
    names up in a dict that also holds the order under a string key.
    """

    gemm: object
    tail: tuple
    input: torch.Tensor
    output: torch.Tensor


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
        gather_dtype: str = "bf16",
        gather_transport: str = "lsa",
        max_shapes: int = DEFAULT_MAX_SHAPES,
    ):
        self._closed = True  # so a failed constructor leaves close() a no-op
        if gather_dtype not in WIRE_DTYPES:
            raise ValueError(
                f"gather_dtype={gather_dtype!r} is not one of {sorted(WIRE_DTYPES)}"
            )
        if gather_transport not in GATHER_TRANSPORTS:
            raise ValueError(
                f"gather_transport={gather_transport!r} is not one of "
                f"{sorted(GATHER_TRANSPORTS)}"
            )
        if block_n != DEFAULT_BLOCK_N:
            raise ValueError(
                f"block_n must be {DEFAULT_BLOCK_N}: the op always compiles with "
                f"permlane, whose lane transpose is written for that width"
            )
        if block_m < 1 or m_max < 1 or sdma_queues < 1 or max_shapes < 1:
            raise ValueError(
                f"block_m, m_max, sdma_queues and max_shapes must all be >= 1; got "
                f"block_m={block_m} m_max={m_max} sdma_queues={sdma_queues} "
                f"max_shapes={max_shapes}"
            )
        why = _gemm_constraints(n, k, block_n)
        if why is not None:
            raise ValueError(f"unsupported shape: {why}")
        self.comm = comm
        self.rank = comm.rank
        # Communicator calls it nranks; DevCommHandle calls the same thing
        # world_size. Normalise here so the op's own surface has one name.
        self.world_size = comm.nranks
        self.n, self.k = n, k
        self.block_m, self.block_n = block_m, block_n
        self.sdma_queues = sdma_queues
        #: fp8 halves the all-gather's bytes, which is ~40% of a fused layer.
        #: It costs relL2 ~2.1e-2 against a bf16 wire that is exact -- e4m3's
        #: mantissa, not a tuning knob -- so it is off unless asked for.
        self.gather_dtype = gather_dtype
        #: Only meaningful with fp8. "lsa" pulls each peer's slice over xGMI and
        #: widens it on the way in -- one kernel instead of an SDMA push plus a
        #: dequantise, and it does not read the landed fp8 back out of HBM.
        #: Measured 957us against SDMA's 1019 on the fused layer.
        self.gather_transport = gather_transport
        self.max_shapes = max_shapes
        self.m_max = padded_m(m_max, self.world_size, block_m)
        #: Which counter set each M uses, assigned in first-seen order. The sets
        #: are disjoint, so two M values never elect on each other's residue.
        self._shape_slot: dict[int, int] = {}

        self.mem = self.win = self.dev_comm = None
        self.window_bytes = self._make_cfg(self.m_max).window_bytes
        try:
            self.mem = comm.alloc_mem(self.window_bytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            # Counters live in this window and are read before they are first
            # written, so it has to start zeroed.
            from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()

            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            # One queue per (source, destination) pair is all the pipeline uses
            # -- a destination is one xGMI link, and splitting a transfer across
            # queues only costs bandwidth. Asking for world_size would create
            # world_size queues *per peer* (56 on an 8-rank node, to use 7);
            # inside a process that already holds SDMA engines that overruns the
            # per-engine queue slots and hsaKmtCreateQueueExt fails.
            reqs.sdma_queue_count = sdma_queues
            self.dev_comm = comm.create_dev_comm(reqs)
        except Exception:
            # Roll back whatever was acquired: the communicator holds a strong
            # reference to each handle, so a half-built op leaks the window for
            # as long as the communicator lives.
            self._closed = False
            self.close()
            raise
        self._closed = False

        self._cache: dict[int, _Plan] = {}
        self._pad_in: Optional[torch.Tensor] = None
        #: M values that have actually run, for :meth:`_check_growth`.
        self._m_seen: set[int] = set()

    @staticmethod
    def window_bytes_for(
        world_size: int,
        *,
        m_max: int,
        n: int,
        block_m: int = DEFAULT_BLOCK_M,
        gather_dtype: str = "bf16",
        gather_transport: str = "lsa",
        max_shapes: int = DEFAULT_MAX_SHAPES,
    ) -> int:
        """Symmetric-window bytes an op for this shape will allocate.

        Needed before the op exists: the window comes out of the communicator's
        VMM reservation, so ``Communicator.init(per_rank_vmm=...)`` has to be
        sized first. Same arithmetic the constructor uses.
        """
        return _build_cfg(
            world_size=world_size,
            m_pad=padded_m(m_max, world_size, block_m),
            n=n,
            block_m=block_m,
            gather_dtype=gather_dtype,
            gather_transport=gather_transport,
            counter_capacity=MAX_CHUNKS,
            shape_slots=max_shapes,
            shape_index=0,
        ).window_bytes

    def padded_m(self, m: int) -> int:
        """``m`` rounded up to what :meth:`__call__` accepts."""
        return padded_m(m, self.world_size, self.block_m)

    def _check_growth(self, m: int) -> None:
        """Refuse the first call at an M larger than one already run.

        Known-bad and not yet root-caused: after a call at some M, the first call
        at a *larger* M comes back wrong about half the time, always with the
        same value, while the repeat is correct. Measured on 2 ranks, 8 runs
        each: 512 then 1024 fails 4/8; 1024 then 512 fails 0/8; 2048 then 4096
        (same chunk count) also fails, so it is the growth and not the chunk
        count. Zeroing the window and synchronising between the calls avoids it,
        which points at a read of a region the smaller shape left dirty rather
        than at the counters -- those are already per-shape.

        Until that is found, this is an error rather than a wrong answer. Serve
        the largest M first (padding to it is usually what a server does anyway),
        or give each M its own op.
        """
        if self._m_seen and m > max(self._m_seen):
            raise ValueError(
                f"M={m} is larger than every M this instance has run "
                f"{sorted(self._m_seen)}. Growing M within one op is a known "
                f"defect -- the first call at the larger M is intermittently "
                f"wrong. Call the largest M first, or build a second op."
            )

    def _slot_for(self, m: int) -> int:
        """This M's counter set, assigned on first sight and then fixed."""
        slot = self._shape_slot.get(m)
        if slot is None:
            if len(self._shape_slot) >= self.max_shapes:
                raise ValueError(
                    f"this op already serves {self.max_shapes} distinct M values "
                    f"{sorted(self._shape_slot)} and M={m} would need another "
                    f"counter set; raise max_shapes, or pad to fewer distinct M"
                )
            slot = len(self._shape_slot)
            self._shape_slot[m] = slot
        return slot

    def _make_cfg(self, m: int) -> ArConfig:
        return _build_cfg(
            world_size=self.world_size,
            m_pad=m,
            n=self.n,
            block_m=self.block_m,
            gather_dtype=self.gather_dtype,
            gather_transport=self.gather_transport,
            # Pinned to the capacity, not to this M's chunk count: the control
            # region has to sit at the same offsets for every M the instance
            # serves, or one shape reads the previous shape's payload as locks.
            counter_capacity=MAX_CHUNKS,
            shape_slots=self.max_shapes,
            shape_index=self._slot_for(m),
        )

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
        # fused_order rather than a literal list: the fp8 wire inserts a
        # quantise and a dequantise around the gather, and a hardcoded
        # drain/reduce/gather would skip them and reduce into zeros.
        tail = tuple(parts[name] for name in parts["fused_order"])
        hit = _Plan(
            gemm=gemm,
            tail=tail,
            input=from_gpu_ptr(
                self.mem.ptr + cfg.input_off, (m, self.n), torch.bfloat16
            ),
            output=from_gpu_ptr(
                self.mem.ptr + cfg.output_off, (m, self.n), torch.bfloat16
            ),
        )
        self._cache[m] = hit
        return hit

    def close(self) -> None:
        """Release this op's window and memory. Idempotent.

        The communicator holds a strong reference to each handle, so dropping
        the op is not enough -- without this the window (about 700 MiB at the
        model shape) lives until the communicator does. Ownership of ``comm``
        stays with the caller.

        Synchronise first if anything may still be in flight: the phases are
        launched on the current stream and this frees memory they address.
        """
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._pad_in = None
        # Window first, then the memory behind it: deregister before free.
        #
        # The dev-comm is deliberately left to the communicator, which is also
        # what EpDispatchCombineOp does. Destroying it here and then letting the
        # communicator tear down aborts the process (exit 255, after this op's
        # own work has finished) -- its SDMA queues are comm-scoped, and the
        # queue count is fixed by the first ccoDevCommCreate on a comm anyway.
        # The window is what this exists to release: ~700 MiB at the model shape,
        # against a handful of queues.
        for handle in (self.win, self.mem):
            if handle is not None:
                handle.close()
        self.win = self.mem = None

    def __enter__(self) -> "GemmAllReduceOp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        A GEMM row and a reduce-scatter row both depend only on the same input
        row, so the added rows produce zeros that the caller slices off. Pad
        *before* quantising: that is what keeps the A scale's column-major
        ``[K/128, M]`` layout intact without touching it.

        The returned tensor is a view of a buffer this op reuses, so a later
        ``pad_rows`` call overwrites it. Clone it to keep it.
        """
        if x.dim() != 2:
            raise ValueError(
                f"expected a 2-D [M, K] tensor, got shape {tuple(x.shape)}"
            )
        m, k = x.shape
        if m_pad < m:
            raise ValueError(f"m_pad={m_pad} is smaller than the input's M={m}")
        # Key on dtype and device as well as capacity: reusing a bf16 buffer for
        # an fp32 input would silently round it, and this promises zero-extension
        # and nothing else.
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

        ``a_scale`` may be ``[M, K/128]`` (the logical shape, either memory
        order), ``[K/128, M]``, or already flat in physical order -- see
        :func:`_flatten_a_scale`, which is where the distinction is made rather
        than assumed. ``b_scale`` is ``[N/128, K/128]`` row-major.

        The returned tensor aliases the window and is overwritten by the next
        call. Clone it to keep it. One instance is not usable concurrently: the
        window, the kernel cache and the padding buffer are all shared, and the
        phases run on the current stream.
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
        self._check_operands(a_fp8, b_preshuffled, a_scale, b_scale, m)
        self._check_growth(m)
        plan = self._compiled(m)
        self._m_seen.add(m)
        stream = fx.Stream(torch.cuda.current_stream())
        plan.gemm(
            a_fp8.contiguous().view(torch.int8).view(-1),
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            plan.input.view(-1),
            _flatten_a_scale(a_scale, m, self.k // SCALE_BLOCK_K),
            b_scale.reshape(-1),
            m,
            self.n,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        for phase in plan.tail:
            phase(self.dev_comm.ptr, self.win.handle, stream=stream)
        return plan.output

    def _check_operands(self, a_fp8, b_preshuffled, a_scale, b_scale, m: int) -> None:
        """Reject what the launch boundary would otherwise reinterpret as bytes.

        ``a_fp8``/``b_preshuffled`` go in as ``int8``, so a bf16 tensor of the
        right shape reaches the kernel and comes back finite and wrong. These
        are all metadata checks -- no device work, nothing that shows in a
        profile.
        """
        if self.mem is None:
            raise RuntimeError("this GemmAllReduceOp has been closed")
        kb = self.k // SCALE_BLOCK_K
        for name, t, shape in (
            ("a_fp8", a_fp8, (m, self.k)),
            ("b_preshuffled", b_preshuffled, (self.n, self.k)),
        ):
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.dtype not in FP8_DTYPES:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in FP8_DTYPES]}, got "
                    f"{t.dtype}; the launch reinterprets it as raw bytes, so a "
                    f"wider dtype runs and returns nonsense rather than failing"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        for name, t, numel in (
            ("a_scale", a_scale, m * kb),
            ("b_scale", b_scale, (self.n // SCALE_BLOCK_K) * kb),
        ):
            if t.dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {t.dtype}")
            if t.numel() != numel:
                raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
