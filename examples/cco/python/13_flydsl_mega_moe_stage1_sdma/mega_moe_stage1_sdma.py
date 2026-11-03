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
"""MegaMoEV2 Stage1 with the dispatch on the SDMA copy engines: the host op ``MegaMoEStage1Sdma``.

aiter's fused Stage1 kernel moves every routed row to its destination with P2P stores
from the compute units (producer workgroups inside the GEMM1 kernel). Here the copy
engines move them, and GEMM1 starts on an expert as soon as its rows have landed:

    pack   (``pack.build_pack``, one block per dispatch segment) this rank's rows, exactly
           the production producer's rows, into a local send slab grouped by (destination,
           segment); this rank's own rows straight to the landing buffer; the route headers
           (srcmap word, weight) to the destination over P2P, as aiter does
    fused  (``gemm1_8wave_sdma``) block 0 puts the slab to the peers' landing buffers at the
           production payload row numbers, one signalled put pair per (peer, segment) in
           expert order; optional shuffler blocks put the landed A scales into the
           contiguous GEMM's layout; the other blocks run the 8-wave GEMM1 + SwiGLU, an
           expert's M-tiles waiting for its rows

aiter's compact prepare (routing plan, ``TILE_INPUT_BASE``), Stage2 and combine are used
as they are; only the Stage1 launch is replaced. ``install()`` swaps ``moe._s1_mega`` and
``moe._s1_mega_bundle`` for this op (``uninstall()`` restores them, or use it as a context
manager), so ``moe(x, wts, ids)`` then runs the SDMA Stage1. The output is byte-identical
to aiter's.

All the copy engine touches is in one cco window, same layout on every rank::

    [ slab x | slab e8m0 | landing x | landing e8m0 | spare | shuffled e8m0 | flags ]

The slab holds what this rank sends (``max_tok * topk`` rows), the landing buffer what it
receives (twice that: the route-major payload's row space); see ``gemm1_8wave_sdma``.

Signals: each peer's signal count from this rank grows by ``epr + 1`` per call (one per
segment, empty ones included), never reset; the host keeps the running target in
``sig_base`` (advanced on the stream before the launch, so CUDA Graph replays advance it
too).

Scale modes (``scale_mode``), see ``gemm1_8wave_sdma``: "shuffler" (``shufflers`` blocks
put the A scales into the contiguous layout once per M-tile) or "ldsw" (every work item
waits on the peer signals and stages its scales into LDS). "auto" picks per call from the
active Stage1 ``sort_block_m``: shuffler at 128 (aiter's 4096 and 8192 buckets), ldsw
otherwise (the 2048 bucket uses 64). Measured on 8x MI355X, v4_pro, Stage1 including
prepare, CUDA Graph, against aiter: at 2048 the shuffler is +0.07..+0.09 ms, ldsw +0.008
(the first M-tile waits for expert 0's shuffle, a large share at small batches); at 4096
the shuffler is -0.04..-0.05 ms; at 8192 -0.17..-0.23 ms.

Needs aiter's compact Stage1 with the route-major payload (``max_tok_per_rank`` below
32768), the a8w4 SwiGLU ("silu") configuration, and mori built with SDMA.
"""

import torch
import torch.distributed as dist

import flydsl.expr as fx

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from gemm1_8wave_sdma import SCALE_MODES, compile_gemm1_8wave_sdma
from pack import build_pack


def _view(ptr, nbytes, device):
    """uint8 tensor over ``nbytes`` of device memory at ``ptr`` (no ownership)."""

    class V:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(V(), device=device)


def _align(n, a=4096):
    return (n + a - 1) // a * a


class MegaMoEStage1Sdma:
    """SDMA Stage1 for one ``MegaMoEV2`` instance (see the module docstring).

    ``scale_mode``: "auto", "shuffler" or "ldsw"; ``shufflers``: shuffler blocks in the
    shuffler mode. At 8192 the fused kernel alone took 2.04 / 1.84 / 1.83 ms with 1 / 2 / 4
    of them; 2 and 4 are within run-to-run noise of each other end to end.
    """

    def __init__(self, moe, scale_mode="auto", shufflers=4):
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

        assert scale_mode == "auto" or scale_mode in SCALE_MODES, scale_mode
        assert moe._a_dtype == "fp8" and moe.act == "silu", "a8w4 SwiGLU only"
        self.moe = moe
        self.scale_mode = scale_mode
        self.shufflers = int(shufflers)
        dev = moe.dev
        K = moe.model_dim
        self.K, self.SC = K, K // 32
        self.s_rows = moe.mtpr * moe.topk  # slab: every route of this rank's tokens
        self.r_rows = 2 * self.s_rows  # landing: the route-major payload's row space
        self.off_slab_x = 0
        self.off_slab_sc = _align(self.off_slab_x + self.s_rows * K)
        self.off_land_x = _align(self.off_slab_sc + self.s_rows * self.SC)
        self.off_land_sc = _align(self.off_land_x + self.r_rows * K)
        self.off_spare = _align(self.off_land_sc + self.r_rows * self.SC)
        # Shuffler mode: the A scales in the shuffle_scale layout at the sorted rows (padded
        # to 256, plus 256 rows a short last M-tile may read past), and the per-block flags.
        self.sh_rows = (int(moe._s1_nvm) + 255) // 256 * 256 + 256
        self.off_shsc = _align(self.off_spare + 4096)
        self.off_flag = _align(self.off_shsc + self.sh_rows * self.SC)
        total = self.off_flag + 4096
        uid = [Communicator.get_unique_id() if moe.rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        self.comm = Communicator.init(
            moe.world_size,
            moe.rank,
            uid[0],
            per_rank_vmm=max(Communicator.DEFAULT_PER_RANK_VMM, total + (512 << 20)),
        )
        self.mem = self.comm.alloc_mem(total)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        _view(self.mem.ptr, total, dev).zero_()
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dev_comm = self.comm.create_dev_comm(reqs)
        assert (
            getattr(self.dev_comm, "sdma_num_queue", 0) >= 1
        ), "mori was built without SDMA"
        self.land_x = _view(self.mem.ptr + self.off_land_x, self.r_rows * K, dev)
        self.shsc = _view(self.mem.ptr + self.off_shsc, self.sh_rows * self.SC, dev)
        self.sig_base = torch.zeros(1, dtype=torch.int64, device=dev)
        self.cursor = torch.zeros(
            4, dtype=torch.int32, device=dev
        )  # [0]: GEMM work items
        self.dummy = torch.zeros(
            64, dtype=torch.uint8, device=dev
        )  # A_scale in the ldsw mode
        self.pack = build_pack(
            K=K,
            npes=moe.world_size,
            epr=moe.epr,
            topk=moe.topk,
            mtpr=moe.mtpr,
            rank=moe.rank,
        )
        self.kernels = {}
        self._prod = None
        torch.cuda.synchronize()
        self.comm.barrier()

    def scale_mode_for(self, sort_block_m):
        if self.scale_mode != "auto":
            return self.scale_mode
        return "shuffler" if sort_block_m == 128 else "ldsw"

    def kernel_for(self, sbm, num_cu, scale_mode):
        key = (sbm, num_cu, scale_mode)
        if key not in self.kernels:
            moe = self.moe
            self.kernels[key] = compile_gemm1_8wave_sdma(
                K=self.K,
                N=2 * moe.inter_dim,
                experts=moe.epr,
                SBM=sbm,
                grid_ctas=num_cu,
                out_rows=int(moe._s1_nvm),
                scale_bytes=moe._s1_osd.numel(),
                rank=moe.rank,
                npes=moe.world_size,
                off_slab_x=self.off_slab_x,
                off_slab_sc=self.off_slab_sc,
                off_land_x=self.off_land_x,
                off_land_sc=self.off_land_sc,
                off_spare=self.off_spare,
                scale_mode=scale_mode,
                shufflers=self.shufflers,
                swiglu_limit=moe.swiglu_limit,
            )
        return self.kernels[key]

    def __call__(
        self,
        out,
        x,
        w,
        scale_x,
        scale_w,
        sorted_token_ids,
        expert_ids,
        num_valid_ids,
        out_scale,
        tokens,
        addr_disp,
        i32_cur_tok,
        addr_in_tok,
        addr_in_idx,
        addr_in_wts,
        addr_in_sc,
        addr_parity,
        addr_expected,
        stream,
        *,
        sort_block_m,
        num_cu,
        **_ignored,
    ):
        """Drop-in for aiter's ``run_mega_moe_stage1`` (same positional arguments)."""
        moe, ws = self.moe, self.moe._s1_dispatch_workspace
        assert (
            not moe._s1_fixed_slot and int(moe.mtpr) < 32768
        ), "compact, route-major payload only"
        mode = self.scale_mode_for(sort_block_m)
        shuffler = mode == "shuffler"
        launch = self.kernel_for(sort_block_m, num_cu, mode)
        self.sig_base.add_(moe.epr + 1)
        self.cursor.zero_()
        base = self.mem.ptr
        _run_compiled(
            self.pack,
            addr_disp,
            addr_in_tok,
            addr_in_sc,
            addr_in_wts,
            addr_in_idx,
            addr_parity,
            fx.Int64(base + self.off_slab_x),
            fx.Int64(base + self.off_slab_sc),
            fx.Int64(base + self.off_land_x),
            fx.Int64(base + self.off_land_sc),
            stream,
        )
        _run_compiled(
            launch,
            self.land_x,  # A: the landing rows (production payload row numbers)
            moe._s1_w1.view(-1),  # B_T: w1
            self.shsc if shuffler else self.dummy,  # A_scale: the shufflers' output
            moe._s1_w1_scale.view(-1),  # B_scale
            self.sh_rows if shuffler else 0,  # c_m: A_scale rows
            fx.Int64(ws["expert_tile_end"].data_ptr()),  # tile_end
            fx.Int64(base + self.off_land_sc),  # land_sc: landing e8m0 rows
            fx.Int64(moe._s1_tile_input_base.data_ptr()),  # tib: TILE_INPUT_BASE
            addr_parity,  # parity_p
            fx.Int64(moe._s1_out_kernel.data_ptr()),  # s1_out: Stage1 output rows
            fx.Int64(ws["plan_ready"].data_ptr()),  # plan_ready
            addr_expected,  # expected_p: expected value per parity
            fx.Int64(base + self.off_flag),  # shuf_flags
            fx.Int64(moe._s1_osd.data_ptr()),  # s1_out_scale: Stage1 output scales
            fx.Int64(self.cursor.data_ptr()),  # cursor: work-item counter
            fx.Int64(
                base + self.off_shsc
            ),  # shuf_scale: the shufflers' output (= A_scale)
            addr_disp,  # addr_disp: the dispatch table
            fx.Int64(self.win.handle),  # win: the cco window
            fx.Int64(self.dev_comm.ptr),  # dev_comm: the cco DevComm
            fx.Int64(
                self.sig_base.data_ptr()
            ),  # sig_base: this call's signal target base
            stream,
        )

    def bundle(self, *args, **kw):
        """Drop-in for aiter's ``run_mega_moe_stage1_bundle``: 18 common arguments, the
        variant id, the stream."""
        common, stream = args[:18], args[19]
        cfg = self.moe._active_config.stage1
        self(*common, stream, sort_block_m=cfg.sort_block_m, num_cu=kw["num_cu"])

    def install(self):
        """Route ``moe``'s Stage1 launches through this op."""
        if self._prod is None:
            self._prod = (self.moe._s1_mega, self.moe._s1_mega_bundle)
            self.moe._s1_mega, self.moe._s1_mega_bundle = self, self.bundle
        return self

    def uninstall(self):
        """Back to aiter's Stage1."""
        if self._prod is not None:
            self.moe._s1_mega, self.moe._s1_mega_bundle = self._prod
            self._prod = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()

    def close(self):
        self.uninstall()
        torch.cuda.synchronize()
        self.comm.barrier()
        self.comm.close_all_resources()
