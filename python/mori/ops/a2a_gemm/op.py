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
"""``A2aGemmOp``: the all-to-all fused into the fp8 GEMM that consumes it.

After Ulysses attention every rank holds every token for its own head shard,
``[world*M, K/world]``; the o-projection needs this rank's ``M`` tokens over all
of ``K``. Every rank passes its shard and the same ``B [N, K]`` and gets back
``C [M, N]`` bf16 for its own tokens.

Two modes, both over the copy engines:

``fused`` (default)
    Block 0 of the GEMM posts every band to every peer, and each tile waits
    only for the bands its rows need; a barrier-only kernel closes the call.
``split``
    Push, barrier, the GEMM, barrier. The baseline ``fused`` is measured
    against, and a fallback.

**Inputs in the window.** The op's ``send`` region is where the exchange reads
from. :meth:`send_buffer` returns it for ``M``; a caller that quantizes the
attention output straight into it saves the copy ``__call__`` otherwise makes.

**Reuse.** The returned ``C`` is this op's buffer for that ``M`` and the next
call with the same ``M`` writes it; clone it to keep it. ``send`` and ``recv``
are safe to refill on the stream after a call returns: the closing barrier
holds every rank until every rank's GEMM has finished reading them.
"""

from __future__ import annotations

from typing import NamedTuple

import flydsl.expr as fx
import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.tensor_utils import from_gpu_ptr

from ..gemm_ar.op import FP8_DTYPES, _PinnedLaunch
from .kernels import ACQUIRE_MODES, WAIT_MODES, build_phases, compile_a2a_gemm
from .layout import MAX_CHUNKS, A2aGemmConfig

MODES = ("fused", "split")
#: The tile the op compiles. ``BLOCK_M=128`` is expressible and slower (README).
BLOCK_M = 256
BLOCK_N = 256


class _Plan(NamedTuple):
    cfg: A2aGemmConfig
    gemm: object
    phases: dict
    out: torch.Tensor


def default_chunks(m: int, world_size: int) -> int:
    """Bands per call, fitted on the README's two shapes.

    P=4 (MiniMax-M3): the best count rises with M -- 1, 2, 4 at 2048, 4096,
    8192 tokens per rank, one band per 2048 rows. P=8 (Llama-70B): one band at
    2048, then one per 1024 rows (c4 at 4096 is level with the best c2; c8 at
    8192 is best by 4%). Rounded down to a divisor of the row tiles.
    """
    if world_size >= 8:
        want = 1 if m <= 2048 else m // 1024
    else:
        want = m // 2048
    tiles = m // BLOCK_M
    for c in range(max(1, min(want, MAX_CHUNKS, tiles)), 0, -1):
        if tiles % c == 0:
            return c
    return 1


def default_wait(m: int, world_size: int) -> str:
    """``segment`` is best or within 0.5% everywhere except P=8 at M >= 4096,
    where ``tile`` is ahead by 0.4-3% (README)."""
    return "tile" if world_size >= 8 and m >= 4096 else "segment"


class A2aGemmOp:
    """Fused all-to-all + fp8 ptpc GEMM over cco SDMA, for Ulysses' o-projection.

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
        mode: str = "fused",
        chunks: int | None = None,
        wait: str | None = None,
        acquire: str = "inv",
    ):
        """``chunks`` / ``wait`` override the per-M defaults (``default_chunks``,
        ``default_wait``). ``acquire="none"`` is what the README's tables use;
        it relies on cco's default uncached window, so the default is ``inv``."""
        self._closed = True  # a failed constructor leaves close() a no-op
        self.mem = self.win = self.dev_comm = None
        self._cache: dict[int, _Plan] = {}

        if mode not in MODES:
            raise ValueError(f"mode={mode!r} is not one of {MODES}")
        if wait is not None and wait not in WAIT_MODES:
            raise ValueError(f"wait={wait!r} is not one of {WAIT_MODES}")
        if acquire not in ACQUIRE_MODES:
            raise ValueError(f"acquire={acquire!r} is not one of {ACQUIRE_MODES}")
        if chunks is not None and not 1 <= chunks <= MAX_CHUNKS:
            raise ValueError(f"chunks must be in [1, {MAX_CHUNKS}], got {chunks}")
        if m_max < 1:
            raise ValueError(f"m_max must be >= 1, got {m_max}")
        if n % BLOCK_N:
            raise ValueError(f"n={n} must be a multiple of {BLOCK_N}")
        self.comm = comm
        self.rank = comm.rank
        self.world_size = comm.nranks
        self.n, self.k = n, k
        self.mode, self.chunks, self.wait, self.acquire = mode, chunks, wait, acquire
        self.m_max = self.padded_m(m_max)
        # Validates world_size and K (a whole number of K steps per source, and
        # at least two of them) before anything is allocated.
        cap = A2aGemmConfig(self.world_size, self.m_max, k, BLOCK_M, 1)
        cap.validate()
        self.k_shard = cap.k_shard
        self.window_bytes = cap.window_bytes
        try:
            self.mem = comm.alloc_mem(self.window_bytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            # The epoch, the barriers and the counter snapshots are read before
            # they are first written, so the window has to start zeroed.
            from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = 1
            self.dev_comm = comm.create_dev_comm(reqs)
            # A put on a queue that does not exist moves nothing, and a band
            # whose count never arrives hangs the GEMM: refuse up front.
            if self.dev_comm.sdma_num_queue < 1:
                raise RuntimeError(
                    "A2aGemmOp needs an SDMA queue per peer but the communicator "
                    "has none: build mori with BUILD_CCO_SDMA=ON and set "
                    "MORI_ENABLE_SDMA=1"
                )
            if self.dev_comm.lsa_size != self.world_size:
                raise RuntimeError(
                    f"A2aGemmOp needs every rank in one LSA domain (one node); "
                    f"lsa_size={self.dev_comm.lsa_size}, world_size={self.world_size}"
                )
            torch.cuda.synchronize()
            comm.barrier()
            # The copy engines' signal counters run on across ops; count this
            # op's bands from where they stand, with nothing in flight anywhere.
            build_phases(cap, self.rank)["arm"](
                self.dev_comm.ptr,
                self.win.handle,
                stream=fx.Stream(torch.cuda.current_stream()),
            )
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
        cap = (m_max + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        return A2aGemmConfig(world_size, cap, k, BLOCK_M, 1).window_bytes

    # --- shapes ------------------------------------------------------------

    def padded_m(self, m: int) -> int:
        """Tokens per rank to pass for logical ``m``: rounded up to the tile."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        return (m + BLOCK_M - 1) // BLOCK_M * BLOCK_M

    def chunks_for(self, m: int) -> int:
        if self.chunks is None:
            return default_chunks(m, self.world_size)
        tiles = m // BLOCK_M
        for c in range(min(self.chunks, tiles), 0, -1):
            if tiles % c == 0:
                return c
        return 1

    def wait_for(self, m: int) -> str:
        return self.wait or default_wait(m, self.world_size)

    def _check_m(self, m: int) -> None:
        if m % BLOCK_M:
            raise ValueError(f"M={m} must be a multiple of {BLOCK_M}; use padded_m()")
        if m > self.m_max:
            raise ValueError(f"M={m} exceeds the window's m_max={self.m_max}")

    def _compiled(self, m: int) -> _Plan:
        hit = self._cache.get(m)
        if hit is not None:
            return hit
        cfg = A2aGemmConfig(self.world_size, m, self.k, BLOCK_M, self.chunks_for(m))
        cfg.validate()
        gemm = compile_a2a_gemm(
            cfg,
            self.rank,
            N=self.n,
            BLOCK_N=BLOCK_N,
            fuse=self.mode == "fused",
            self_from_send=True,
            wait=self.wait_for(m),
            acquire=self.acquire,
        )
        phases = {
            name: _PinnedLaunch(fn) for name, fn in build_phases(cfg, self.rank).items()
        }
        out = torch.empty((m, self.n), dtype=torch.bfloat16, device="cuda")
        hit = self._cache[m] = _Plan(cfg, _PinnedLaunch(gemm), phases, out)
        return hit

    def send_buffer(self, m: int) -> torch.Tensor:
        """The ``[world*m, k_shard]`` fp8 slab ``__call__`` exchanges from.

        Rows ``[d*m, (d+1)*m)`` go to rank ``d``. Writing the attention output
        here directly, then passing this tensor as ``a_fp8``, skips a copy.
        """
        if self.mem is None:
            raise RuntimeError("this A2aGemmOp has been closed")
        self._check_m(m)
        cfg = A2aGemmConfig(self.world_size, m, self.k, BLOCK_M, 1)
        return from_gpu_ptr(
            self.mem.ptr + cfg.send_off,
            (self.world_size * m, self.k_shard),
            torch.float8_e4m3fn,
        )

    # --- the call ----------------------------------------------------------

    def __call__(
        self,
        a_fp8: torch.Tensor,
        b_preshuffled: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        """One all-to-all + GEMM; returns ``C [M, N]`` bf16 for this rank's tokens.

        ``a_fp8`` is ``[world*M, K/world]``: every token, this rank's head
        columns, M from :meth:`padded_m`. ``b_preshuffled`` is ``[N, K]``
        through :func:`~mori.ops.gemm_ar.preshuffle_b`, identical on every rank.
        ``a_scale`` is ``[M]`` fp32, one per token over all of K -- computed
        before quantization from the row maxima gathered across ranks, so it is
        local and never travels; ``b_scale`` is ``[N]`` fp32, per channel.

        The result is overwritten by the next call with the same M.
        """
        if self.mem is None:
            raise RuntimeError("this A2aGemmOp has been closed")
        ws = self.world_size
        if a_fp8.dim() != 2 or a_fp8.shape[0] % ws or a_fp8.shape[1] != self.k_shard:
            raise ValueError(
                f"a_fp8 must be [world*M, {self.k_shard}], got {tuple(a_fp8.shape)}"
            )
        m = a_fp8.shape[0] // ws
        self._check_m(m)
        self._check_operands(a_fp8, b_preshuffled, a_scale, b_scale, m)
        plan = self._compiled(m)
        cfg = plan.cfg
        send = from_gpu_ptr(
            self.mem.ptr + cfg.send_off, (ws * m, self.k_shard), torch.uint8
        )
        if a_fp8.data_ptr() != send.data_ptr():
            send.copy_(a_fp8.contiguous().view(torch.uint8))
        operand = from_gpu_ptr(
            self.mem.ptr + cfg.send_off, (cfg.operand_bytes,), torch.int8
        )
        stream = fx.Stream(torch.cuda.current_stream())
        dc, win = self.dev_comm.ptr, self.win.handle
        if self.mode == "split":
            plan.phases["push"](dc, win, stream=stream)
            plan.phases["barrier0"](dc, win, stream=stream)
        plan.gemm(
            operand,
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            plan.out.view(-1),
            a_scale.reshape(-1).contiguous(),
            b_scale.reshape(-1).contiguous(),
            dc,
            win,
            stream=stream,
        )
        plan.phases["barrier1"](dc, win, stream=stream)
        return plan.out

    def _check_operands(self, a_fp8, b_preshuffled, a_scale, b_scale, m):
        """Reject what the launch would reinterpret as bytes."""
        if tuple(b_preshuffled.shape) != (self.n, self.k):
            raise ValueError(
                f"b_preshuffled must be {(self.n, self.k)}, "
                f"got {tuple(b_preshuffled.shape)}"
            )
        for name, t in (("a_fp8", a_fp8), ("b_preshuffled", b_preshuffled)):
            if t.dtype not in FP8_DTYPES:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in FP8_DTYPES]}, got "
                    f"{t.dtype}; the launch reinterprets it as raw bytes"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        for name, t, numel in (("a_scale", a_scale, m), ("b_scale", b_scale, self.n)):
            if t.dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {t.dtype}")
            if t.numel() != numel:
                raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")

    # --- checks and lifetime -----------------------------------------------

    def self_test(self) -> None:
        """Verify that the transport moves bytes; raise if it does not.

        A mori built with ``BUILD_CCO_SDMA=OFF``, or run without
        ``MORI_ENABLE_SDMA=1``, still launches every kernel and returns from
        every put -- the puts move nothing. Fill every ``send`` slab with
        ``rank + 1``, run the split exchange at ``m_max``, and check every
        remote ``recv`` slab came back as its source's value.
        """
        if self.mem is None:
            raise RuntimeError("this A2aGemmOp has been closed")
        # Only the exchange is under test: the phases, not a GEMM compile.
        cfg = A2aGemmConfig(self.world_size, self.m_max, self.k, BLOCK_M, 1)
        phases = build_phases(cfg, self.rank)
        ws = self.world_size
        send = from_gpu_ptr(
            self.mem.ptr + cfg.send_off, (ws, cfg.slab_bytes), torch.uint8
        )
        recv = from_gpu_ptr(
            self.mem.ptr + cfg.recv_off, (ws, cfg.slab_bytes), torch.uint8
        )
        send.fill_(self.rank + 1)
        recv.zero_()
        torch.cuda.synchronize()
        self.comm.barrier()
        stream = fx.Stream(torch.cuda.current_stream())
        dc, win = self.dev_comm.ptr, self.win.handle
        phases["push"](dc, win, stream=stream)
        phases["barrier0"](dc, win, stream=stream)
        # barrier1 re-takes the counter snapshot the next call counts from.
        phases["barrier1"](dc, win, stream=stream)
        torch.cuda.current_stream().synchronize()
        for src in range(ws):
            if src == self.rank:
                continue
            if not bool((recv[src] == src + 1).all()):
                raise RuntimeError(
                    f"a2a_gemm self-test failed: recv slab from rank {src} holds "
                    f"{int(recv[src][0])}, expected {src + 1}. Check that mori was "
                    f"built with BUILD_CCO_SDMA=ON and that MORI_ENABLE_SDMA=1 is "
                    f"set: without either, every put silently does nothing."
                )

    def close(self) -> None:
        """Release the window and its memory. Idempotent; synchronise first if a
        call may still be in flight. The dev-comm is left to the communicator,
        as ``GemmA2AOp`` does."""
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        for handle in (self.win, self.mem):
            if handle is not None:
                handle.close()
        self.win = self.mem = None

    def __enter__(self) -> A2aGemmOp:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["MODES", "A2aGemmOp", "default_chunks", "default_wait"]
