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
"""``GemmArV2``: BF16 GEMM + all-reduce over cco SDMA, the gemm_ar_bf16 case's
interface (``benchmark/cco/harness/cases/gemm_ar_bf16/interface.py``).

Per call (``mode="stream"``, the default):

1. the GEMM in ``chunks`` launches on the caller's stream, chunk ``c`` = row
   tile ``c`` of every destination's band, into this rank's partial ``Y``;
2. on a side stream, after each chunk's event, one put per peer of that chunk
   into the peer's ``R[rank]``; then the scatter drain + barrier;
3. back on the caller's stream: the reduce of band ``rank``, then the gather
   (push band ``rank`` to every peer, drain, barrier).

``mode="split"``: one GEMM launch, then one put per peer (the in-module split
baseline; reports ``fused_path_ran = False``). ``mode="fused"``: one GEMM launch
whose epilogue posts each (destination, chunk) from the block that completes it.

Switches (constructor keywords, or ``MORI_GARV2_<NAME>`` environment variables):
``mode``, ``chunks`` (0 = per-M rule), ``ctr_order``, ``emit_put``, ``wait``,
``stop_after`` (``gemm`` / ``scatter`` / ``reduce``; diagnosis only, the result is
then wrong), ``swap_store``.
"""

from __future__ import annotations

import os

import flydsl.expr as fx
import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.ops.gemm_ar_v2 import kernels as KR
from mori.ops.gemm_ar_v2 import layout as L
from mori.tensor_utils import from_gpu_ptr

_DEFAULTS = dict(
    mode="stream",
    chunks=0,
    ctr_order="acq_rel",
    emit_put=True,
    wait=True,
    stop_after="",
    swap_store=True,
)
_BOOL = {"emit_put", "wait", "swap_store"}


def _resolve(opts):
    out = dict(_DEFAULTS)
    for key in _DEFAULTS:
        env = os.environ.get(f"MORI_GARV2_{key.upper()}")
        if env is not None:
            out[key] = (env not in ("0", "false", "False", "")) if key in _BOOL else (
                int(env) if key == "chunks" else env
            )
    out.update({k: v for k, v in opts.items() if v is not None})
    if out["mode"] not in ("stream", "split", "fused"):
        raise ValueError(f"mode={out['mode']!r}")
    if out["stop_after"] not in ("", "gemm", "scatter", "reduce"):
        raise ValueError(f"stop_after={out['stop_after']!r}")
    return out


class GemmArV2:
    """Fused BF16 GEMM + all-reduce. See the module docstring."""

    @staticmethod
    def window_bytes_for(world_size, *, m_max, n, k):
        return L.Layout(world_size, n, L.padded_m(m_max, world_size)).window_bytes

    def __init__(self, comm, *, n, k, m_max, **opts):
        self._closed = True
        self.mem = self.win = self.dev_comm = None
        self.o = _resolve(opts)
        self.comm = comm
        self.rank, self.world = comm.rank, comm.nranks
        if n % L.BLOCK_N or k % 64 or k < 128:
            raise ValueError(f"unsupported n={n} k={k}")
        self.n, self.k = n, k
        self.m_cap = L.padded_m(m_max, self.world)
        self.lay = L.Layout(self.world, n, self.m_cap)
        self.queues = L.MAX_CHUNKS if self.o["mode"] == "fused" else 1
        self.fused_path_ran = None
        self._plans = {}
        self._b = None
        self._apad = None
        try:
            nbytes = self.lay.window_bytes
            self.mem = comm.alloc_mem(nbytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            from_gpu_ptr(self.mem.ptr, (nbytes,), torch.uint8).zero_()
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = self.queues
            self.dev_comm = comm.create_dev_comm(reqs)
            if self.dev_comm.sdma_num_queue < self.queues:
                raise RuntimeError(
                    f"needs {self.queues} SDMA queue(s) per peer, the communicator has "
                    f"{self.dev_comm.sdma_num_queue}: build with BUILD_CCO_SDMA=ON and "
                    f"set MORI_ENABLE_SDMA=1"
                )
            if self.dev_comm.lsa_size != self.world:
                raise RuntimeError("all ranks must be in one LSA domain (one node)")
            torch.cuda.synchronize()
            comm.barrier()
        except Exception:
            self._closed = False
            self.close()
            raise
        self._closed = False
        self._y = from_gpu_ptr(
            self.mem.ptr + self.lay.y_off, (self.m_cap * n,), torch.bfloat16
        )
        self._ctrl = from_gpu_ptr(self.mem.ptr, (L.CTRL_BYTES // 4,), torch.int32)
        self._side = torch.cuda.Stream()
        self._events = [torch.cuda.Event() for _ in range(L.MAX_CHUNKS)]

    # --- configuration -----------------------------------------------------

    def shape_for(self, mp):
        band_tiles = mp // self.world // L.BLOCK_M
        if self.o["mode"] == "split":
            chunks = 1
        elif self.o["chunks"]:
            chunks = L.default_chunks(band_tiles, min(int(self.o["chunks"]), L.MAX_CHUNKS))
        else:
            chunks = L.default_chunks(band_tiles)
        return L.Shape(self.world, self.n, mp, chunks)

    def _plan(self, mp):
        hit = self._plans.get(mp)
        if hit is not None:
            return hit
        o, r = self.o, self.rank
        sh = self.shape_for(mp)
        mode = o["mode"]
        plan = dict(
            shape=sh,
            gemm=KR.compile_gemm(
                sh,
                self.lay,
                r,
                K=self.k,
                grid="chunk" if mode == "stream" else "all",
                fused=mode == "fused",
                queues=self.queues,
                ctr_order=o["ctr_order"],
                emit_put=o["emit_put"],
                swap_store=o["swap_store"],
            ),
            put=KR.compile_put_chunk(sh, self.lay, r),
            scatter=KR.compile_barrier(
                sh, self.lay, r, kind="scatter", quiet_all=mode == "fused", wait=o["wait"]
            ),
            reduce=KR.compile_reduce(sh, self.lay, r),
            gather=KR.compile_barrier(sh, self.lay, r, kind="gather", wait=o["wait"]),
        )
        self._plans[mp] = plan
        return plan

    # --- the interface -----------------------------------------------------

    def set_weight(self, b):
        if tuple(b.shape) != (self.n, self.k) or b.dtype != torch.bfloat16:
            raise ValueError(f"b must be bf16 [{self.n}, {self.k}], got {b.dtype} {tuple(b.shape)}")
        self._bw = b.contiguous()
        self._b = self._bw.view(torch.int16).view(-1)

    def _a_arg(self, a, m, mp):
        if a.dtype != torch.bfloat16 or a.dim() != 2 or a.shape[1] != self.k:
            raise ValueError(f"a must be bf16 [M, {self.k}]")
        if m == mp:
            return a.contiguous().view(torch.int16).view(-1)
        if self._apad is None:
            self._apad = torch.zeros(
                (self.m_cap, self.k), dtype=torch.bfloat16, device=a.device
            )
        buf = self._apad[:mp]
        buf[:m].copy_(a)
        buf[m:].zero_()
        return buf.view(torch.int16).view(-1)

    def __call__(self, a):
        if self.mem is None:
            raise RuntimeError("closed")
        m = a.shape[0]
        if not 1 <= m <= self.m_cap:
            raise ValueError(f"M={m} outside [1, {self.m_cap}]")
        mp = L.padded_m(m, self.world)
        p = self._plan(mp)
        sh = p["shape"]
        A = self._a_arg(a, m, mp)
        o = self.o
        stop = o["stop_after"]
        cur = torch.cuda.current_stream()
        s = fx.Stream(cur)
        dc, wh = self.dev_comm.ptr, self.win.handle
        if o["mode"] == "stream":
            side = self._side
            side.wait_stream(cur)
            ss = fx.Stream(side)
            for c in range(sh.chunks):
                p["gemm"](A, self._b, c, dc, wh, stream=s)
                if stop != "gemm":
                    ev = self._events[c]
                    ev.record(cur)
                    side.wait_event(ev)
                    p["put"](dc, wh, c, stream=ss)
            if stop != "gemm":
                p["scatter"](dc, wh, stream=ss)
            cur.wait_stream(side)
        else:
            p["gemm"](A, self._b, 0, dc, wh, stream=s)
            if stop != "gemm":
                if o["mode"] == "split":
                    p["put"](dc, wh, 0, stream=s)
                p["scatter"](dc, wh, stream=s)
        if stop in ("", "reduce"):
            p["reduce"](wh, stream=s)
        if stop == "":
            p["gather"](dc, wh, stream=s)
        self.fused_path_ran = o["mode"] != "split"
        return self._y[: m * self.n].view(m, self.n)

    # --- checks ------------------------------------------------------------

    def self_test(self):
        """Collective: fill band ``rank`` with ``rank+1``, gather, check every band.
        Without BUILD_CCO_SDMA / MORI_ENABLE_SDMA a put silently moves nothing
        (G950-ENV-01)."""
        mp = L.padded_m(1, self.world)
        p = self._plan(mp)
        ms = mp // self.world
        y = self._y[: mp * self.n].view(self.world, ms, self.n)
        y.zero_()
        y[self.rank].fill_(float(self.rank + 1))
        torch.cuda.synchronize()
        self.comm.barrier()
        p["gather"](self.dev_comm.ptr, self.win.handle, stream=fx.Stream(torch.cuda.current_stream()))
        torch.cuda.synchronize()
        for src in range(self.world):
            if not bool((y[src] == float(src + 1)).all()):
                raise RuntimeError(
                    f"gemm_ar_v2 self-test: band {src} holds {y[src].flatten()[0].item():g}, "
                    f"expected {src + 1}; is mori built with BUILD_CCO_SDMA=ON and "
                    f"MORI_ENABLE_SDMA=1 set?"
                )
        self.comm.barrier()

    def debug_preset_counters(self, value):
        """Set every running epoch, arrival slot and counter (and the counter base)
        to ``value`` on this rank; collective (ends with a barrier)."""
        v = int(value) & 0xFFFFFFFF
        v = v - (1 << 32) if v >= (1 << 31) else v
        torch.cuda.synchronize()
        for off in L.EPOCH_WORDS + L.COUNTER_WORDS:
            self._ctrl[off // 4] = v
        torch.cuda.synchronize()
        self.comm.barrier()

    def debug_dump(self):
        """The control words, for a hang report."""
        c = self._ctrl.cpu().tolist()
        u = lambda off: c[off // 4] & 0xFFFFFFFF  # noqa: E731
        return {
            "FS": u(L.FS_OFF),
            "FG": u(L.FG_OFF),
            "S": [u(L.S_OFF + 4 * i) for i in range(self.world)],
            "G": [u(L.G_OFF + 4 * i) for i in range(self.world)],
            "Bctr": u(L.BCTR_OFF),
            "C": [u(L.CTR_OFF + 4 * i) for i in range(self.world * L.MAX_CHUNKS)],
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._plans.clear()
        for h in (self.win, self.mem):
            if h is not None:
                h.close()
        self.win = self.mem = None


def _variant(name, **kw):
    def __init__(self, comm, *, n, k, m_max, **opts):
        GemmArV2.__init__(self, comm, n=n, k=k, m_max=m_max, **{**kw, **opts})

    return type(name, (GemmArV2,), {"__init__": __init__})


# Arms for op_harness (--impl python/mori/ops/gemm_ar_v2/op.py:<Class>).
GemmArV2Split = _variant("GemmArV2Split", mode="split")
GemmArV2Fused = _variant("GemmArV2Fused", mode="fused")
GemmArV2StopGemm = _variant("GemmArV2StopGemm", stop_after="gemm")
GemmArV2StopScatter = _variant("GemmArV2StopScatter", stop_after="scatter")
GemmArV2StopReduce = _variant("GemmArV2StopReduce", stop_after="reduce")
GemmArV2SplitStopGemm = _variant("GemmArV2SplitStopGemm", mode="split", stop_after="gemm")
GemmArV2FusedStopScatter = _variant("GemmArV2FusedStopScatter", mode="fused", stop_after="scatter")
GemmArV2FusedNoPut = _variant(
    "GemmArV2FusedNoPut", mode="fused", emit_put=False, stop_after="scatter"
)
