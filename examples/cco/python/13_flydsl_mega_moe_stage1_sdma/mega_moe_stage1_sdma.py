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

``fuse_quant=True`` folds the copy into the quantisation instead (``quant_pack``): aiter's
prepare then only plans (its quant workgroups are not launched), ``route`` records where each
token's rows go and writes the route headers, and ``qpack`` quantises each bf16 row once,
straight into its slab / landing rows -- the same bytes as aiter's quantisation, without
pack's extra read and write of every routed row.

``gemm1="7001"`` (``--gemm1 7001``) runs aiter#7001's fused Stage1 GEMM1 tiles instead of the
8-wave GEMM1, in one launch that also packs the rows (``gemm1_7001_sdma``): producer
workgroups pack each (destination, segment) in expert order and the copy puts of a segment
start as soon as it is packed, so there is no pack / qpack launch before it. It needs aiter
with aiter#7001's ``mega_moe/gemm1.py`` and the prepare's quantisation (no ``fuse_quant``).

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

import dataclasses

import torch
import torch.distributed as dist

import flydsl.expr as fx

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from gemm1_8wave_sdma import SCALE_MODES, compile_gemm1_8wave_sdma
from pack import build_pack
from quant_pack import build_quant_pack, build_route

#: GEMM1 of the fused kernel: example 13's 8-wave GEMM1, or aiter#7001's Stage1 tiles.
GEMM1_MODES = ("8wave", "7001")


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
    ``fuse_quant``: quantise straight into the send rows (``quant_pack``) instead of pack;
    ``moe`` must then be called with bf16 input (``forward``), the prepare's quant input.
    ``gemm1``: "8wave" (``gemm1_8wave_sdma``) or "7001" (aiter#7001's Stage1 tiles, the rows
    packed by ``producers`` workgroups of the same launch; ``gemm1_7001_sdma``).
    """

    def __init__(
        self,
        moe,
        scale_mode="auto",
        shufflers=4,
        fuse_quant=False,
        gemm1="8wave",
        producers=32,
    ):
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

        assert scale_mode == "auto" or scale_mode in SCALE_MODES, scale_mode
        assert moe._a_dtype == "fp8" and moe.act == "silu", "a8w4 SwiGLU only"
        assert gemm1 in GEMM1_MODES, gemm1
        assert not (
            gemm1 == "7001" and fuse_quant
        ), "--gemm1 7001 packs the prepare's fp8 rows itself (no --fuse-quant)"
        self.gemm1 = gemm1
        self.producers = int(producers)
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
        # --gemm1 7001: per (destination, segment), the call whose rows it holds packed.
        self.seg_done = torch.zeros(
            (moe.epr + 1) * moe.world_size, dtype=torch.int32, device=dev
        )
        self._prod = None
        self.fuse_quant = bool(fuse_quant)
        self._x_bf16 = (
            None  # fuse_quant: this call's bf16 input, taken from the prepare
        )
        if self.fuse_quant:
            self.inv = torch.full(
                (moe.mtpr * moe.topk,), -1, dtype=torch.int32, device=dev
            )
            self.route = build_route(
                npes=moe.world_size,
                epr=moe.epr,
                topk=moe.topk,
                mtpr=moe.mtpr,
                rank=moe.rank,
            )
            self.qpacks = {}  # per token count: one block per token
        torch.cuda.synchronize()
        self.comm.barrier()

    def scale_mode_for(self, sort_block_m):
        if self.scale_mode != "auto":
            return self.scale_mode
        return "shuffler" if sort_block_m == 128 else "ldsw"

    def kernel_7001(self, sbm, num_cu, cfg):
        """``--gemm1 7001``: the launcher for one Stage1 config (``cfg`` has the fields of
        aiter's Stage1Config)."""
        from gemm1_7001_sdma import compile_gemm1_7001_sdma

        key = ("7001", sbm, num_cu, cfg["tile_n"], cfg["num_waves"], cfg["work_shards"])
        if key not in self.kernels:
            moe = self.moe
            assert cfg.get("grid_mult", 1) == 1, "compact Stage1: one workgroup per CU"
            self.kernels[key] = compile_gemm1_7001_sdma(
                model_dim=self.K,
                inter_dim=moe.inter_dim,
                rank=moe.rank,
                experts_per_rank=moe.epr,
                fuse_npes=moe.world_size,
                fuse_mtpr=moe.mtpr,
                sort_block_m=sbm,
                tile_n=cfg["tile_n"],
                tile_k=cfg["tile_k"],
                num_waves=cfg["num_waves"],
                pipe_weights=cfg["pipe_weights"],
                mfma_amajor=cfg["mfma_amajor"],
                swizzle_a=cfg["swizzle_a"],
                async_a_copy=cfg["async_a_copy"],
                use_tile_resource=cfg["use_tile_resource"],
                waves_per_eu_hint=cfg["waves_per_eu_hint"],
                num_cu=num_cu,
                b_nt=cfg["b_nt"],
                work_shards=cfg["work_shards"],
                swiglu_limit=moe.swiglu_limit,
                off_slab_x=self.off_slab_x,
                off_slab_sc=self.off_slab_sc,
                off_land_x=self.off_land_x,
                off_land_sc=self.off_land_sc,
                off_spare=self.off_spare,
                topk=moe.topk,
                producers=self.producers,
            )
        return self.kernels[key]

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
        if self.gemm1 == "7001":
            self.sig_base.add_(moe.epr + 1)
            self._call_7001(
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
                addr_in_tok,
                addr_in_idx,
                addr_in_wts,
                addr_in_sc,
                addr_parity,
                addr_expected,
                stream,
                sort_block_m,
                num_cu,
                _ignored,
            )
            return
        launch = self.kernel_for(sort_block_m, num_cu, mode)
        self.sig_base.add_(moe.epr + 1)
        self.cursor.zero_()
        base = self.mem.ptr
        if self.fuse_quant:
            assert (
                self._x_bf16 is not None
            ), "fuse_quant: no bf16 input from the prepare"
            xb = self._x_bf16
            tokens_now = int(xb.shape[0])
            if tokens_now not in self.qpacks:
                self.qpacks[tokens_now] = build_quant_pack(
                    K=self.K, topk=moe.topk, rank=moe.rank, tokens=tokens_now
                )
            _run_compiled(
                self.route,
                addr_disp,
                addr_in_wts,
                addr_in_idx,
                addr_parity,
                fx.Int64(self.inv.data_ptr()),
                stream,
            )
            _run_compiled(
                self.qpacks[tokens_now],
                fx.Int64(xb.data_ptr()),
                fx.Int64(self.inv.data_ptr()),
                fx.Int64(base + self.off_slab_x),
                fx.Int64(base + self.off_slab_sc),
                fx.Int64(base + self.off_land_x),
                fx.Int64(base + self.off_land_sc),
                stream,
            )
        else:
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

    def _call_7001(
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
        addr_in_tok,
        addr_in_idx,
        addr_in_wts,
        addr_in_sc,
        addr_parity,
        addr_expected,
        stream,
        sort_block_m,
        num_cu,
        kw,
    ):
        """``--gemm1 7001``: one launch packs, puts and runs #7001's GEMM1 tiles on the
        landing rows (``gemm1_7001_sdma``); no pack / qpack before it."""
        moe, K = self.moe, self.K
        # The Stage1 config: the launch's own keywords, or (bundle launches) the active one.
        cfg = kw if "tile_n" in kw else dataclasses.asdict(moe._active_config.stage1)
        land_x = _view(self.mem.ptr + self.off_land_x, self.r_rows * K, moe.dev)
        land_sc = _view(self.mem.ptr + self.off_land_sc, self.r_rows * self.SC, moe.dev)
        _run_compiled(
            self.kernel_7001(sort_block_m, num_cu, cfg),
            out,
            land_x.view(x.dtype).view(self.r_rows, K),  # A: the landing rows
            w,
            land_sc.view(scale_x.dtype).view(self.r_rows, -1),  # their raw e8m0
            scale_w,
            sorted_token_ids,
            expert_ids,
            num_valid_ids,
            out_scale,
            tokens,
            addr_disp,
            addr_parity,
            addr_expected,
            fx.Int64(self.win.handle),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.sig_base.data_ptr()),
            addr_in_tok,  # the prepare's fp8 rows
            addr_in_sc,  # and their e8m0
            addr_in_wts,
            addr_in_idx,
            fx.Int64(self.seg_done.data_ptr()),
            fx.Int64(self.mem.ptr),  # this rank's window base
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
            if self.fuse_quant:
                prepare = self.moe._run_compact_prepare

                def run_prepare(*args, quant_input=None, **kw):
                    # The prepare only plans; qpack quantises the bf16 input.
                    self._x_bf16 = quant_input
                    return prepare(*args, **kw)

                self.moe._run_compact_prepare = run_prepare
        return self

    def uninstall(self):
        """Back to aiter's Stage1."""
        if self._prod is not None:
            self.moe._s1_mega, self.moe._s1_mega_bundle = self._prod
            self._prod = None
            self.moe.__dict__.pop("_run_compact_prepare", None)

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()

    def close(self):
        self.uninstall()
        torch.cuda.synchronize()
        self.comm.barrier()
        self.comm.close_all_resources()
