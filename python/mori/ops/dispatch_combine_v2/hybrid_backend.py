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
"""Hybrid kernel backend: HIP/JIT dispatch + FlyDSL combine.

HIP dispatch benefits from the mature gfx1250 TDM path and avoids the FlyDSL
compiler on the copy-bound scatter leg.  FlyDSL combine brings the full gather
path with per-block cross-device barrier and reset_total_recv.

Arena layout is the same as both pure backends (they share region names and
sizes), so one SymmArena serves both kernel families.
"""

from __future__ import annotations

import torch

import flydsl.expr as fx
from mori.tensor_utils import from_gpu_ptr

from . import ep_plans as cb
from .intranode_kernels import make_combine, xdb_flag_slots
from .dispatch_combine_op import (
    EpDispatchCombineOp,
    KernelSet,
)
from .symm_arena import SymmArena

_HIP_DISPATCH_DTYPES = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fn: 1,
    torch.float8_e4m3fnuz: 1,
    torch.float4_e2m1fn_x2: 1,
}

# HIP region-name mapping (C++ offset stem -> arena region name).
_REGIONS = {
    "tokOff": "tok_off",
    "recvNum": "recv_num",
    "recvToSrc": "recv_to_src_token",
    "outIdx": "out_idx",
    "outWts": "out_wts",
    "dispOut": "disp_out",
    "outTok": "out_tok",
    "xdb": "cross_device_barrier",
}


class EpDispatchCombineOpHybrid(EpDispatchCombineOp, backend="hybrid"):
    """HIP dispatch + FlyDSL gather-combine.  No scatter, no quant, no replay,
    no per-token scales, no StdMoE — dispatch-side constraints from HIP still
    apply; combine-side gets FlyDSL's full gather path."""

    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self._recv_cap = cfg.effective_max_recv
        self._closed = False
        # No gfx1250 arch probe here: the HIP dispatch plan selects its own
        # portable-vs-TDM body at render time, and the gfx1250 combine's fan
        # buffer is unused because combine is FlyDSL, not the TDM kernel.

        self._gate(
            KernelSet(dispatch={}, combine={}, unsupported=self._unsupported(cfg))
        )

        self.arena = SymmArena(comm, self._regions(cfg))
        self.arena.zero()

        self._dispatch_specs, self._combine_specs = self._specs_from(cfg)
        self._kernels = self._build_kernels(cfg, self.arena)

        topk = cfg.num_experts_per_token
        max_tok = cfg.max_num_inp_token_per_rank
        i32 = dict(dtype=torch.int32, device=dev)
        self.token_dest_map = torch.zeros(max_tok * topk, **i32)
        self.dest_pe_counter = torch.zeros(cfg.world_size, **i32)
        self.total_recv = torch.zeros(1, **i32)
        self.dispatch_barrier = torch.zeros(1, dtype=torch.uint32, device=dev)
        self.combine_barrier = torch.zeros(1, dtype=torch.int32, device=dev)
        # FlyDSL combine uses per-block xdb flag counters.
        self.cross_device_flag = torch.ones(
            xdb_flag_slots, dtype=torch.int64, device=dev
        )
        self.combine_out = torch.zeros(
            max_tok * cfg.hidden_dim, dtype=cfg.combine_dtype, device=dev
        )
        self.combine_out_weights = torch.zeros(
            max_tok * topk, dtype=torch.float32, device=dev
        )

    # -- backend hooks --------------------------------------------------------

    def _regions(self, cfg):
        cap = cfg.effective_max_recv
        topk = cfg.num_experts_per_token
        return [
            ("tok_off", 4),
            ("recv_num", cfg.world_size * 4),
            ("recv_to_src_token", cap * 4),
            ("out_idx", cap * topk * 4),
            ("out_wts", cap * topk * 4),
            ("disp_out", cap * cfg.token_nbytes),
            ("out_tok", cap * cfg.combine_token_nbytes),
            ("cross_device_barrier", cfg.world_size * 8),
        ]

    _COMBINE_DTYPES = {torch.bfloat16, torch.float32}

    def _unsupported(self, cfg) -> tuple[str, ...]:
        bad = []
        if cfg.dispatch_dtype not in _HIP_DISPATCH_DTYPES:
            bad.append(
                f"dispatch dtype {cfg.dispatch_dtype} (have bf16, fp32, fp8, fp4)"
            )
        if cfg.combine_dtype not in self._COMBINE_DTYPES:
            bad.append(
                f"combine dtype {cfg.combine_dtype} (have bf16, fp32)"
            )
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (gather only)")
        if cfg.quant_type != "none":
            bad.append(f"quant_type={cfg.quant_type!r}")
        if cfg.enable_std_moe:
            bad.append("enable_std_moe")
        if cfg.scale_dim and cfg.scale_type_size:
            bad.append("per-token scales forwarding")
        worst = cfg.world_size * cfg.max_num_inp_token_per_rank
        if cfg.effective_max_recv < worst:
            bad.append(
                f"max_total_recv_tokens below the worst case "
                f"({cfg.effective_max_recv} < {worst}); token dropping is not implemented"
            )
        return tuple(bad)

    def _build_kernels(self, cfg, arena) -> KernelSet:
        bad = self._unsupported(cfg)
        if bad:
            return KernelSet(dispatch={}, combine={}, unsupported=bad)

        # -- HIP dispatch plans -----------------------------------------------
        common = dict(
            world_size=cfg.world_size,
            max_tok_per_rank=cfg.max_num_inp_token_per_rank,
            num_expert_per_rank=cfg.num_experts_per_rank,
            num_expert_per_token=cfg.num_experts_per_token,
            max_recv=cfg.effective_max_recv,
            use_weights=True,
            arena=arena,
            region_names=_REGIONS,
        )
        disp_cfg = dict(
            hidden_dim=cfg.hidden_dim // 2 if cfg.is_fp4 else cfg.hidden_dim,
            dtype=cfg.dispatch_dtype,
        )
        dispatch = {}
        self._plans = []
        for b, w in self._dispatch_specs:
            plan = cb.EpDispatchPlan(
                **common, **disp_cfg, block_num=b, warp_per_block=w
            )
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            dispatch[(b, w)] = self._wrap_dispatch(plan)

        # -- FlyDSL combine kernels -------------------------------------------
        topk = cfg.num_experts_per_token
        hidden_dim = cfg.hidden_dim
        max_tok_per_rank = cfg.max_num_inp_token_per_rank
        recv_cap = cfg.effective_max_recv

        self._combine_variants = {
            (b, w): make_combine(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=topk,
                hidden_dim=hidden_dim,
                hidden_elem_size=cfg.combine_elem_size,
                max_tok_per_rank=max_tok_per_rank,
                max_recv=recv_cap,
                block_num=b,
                warp_num_per_block=w,
                off_out_tok=arena.offset("out_tok"),
                off_xdb_mem=arena.offset("cross_device_barrier"),
                off_out_wts=arena.offset("out_wts"),
                reset_total_recv=True,
                fp4=(cfg.combine_dtype == torch.float4_e2m1fn_x2),
            )
            for (b, w) in self._combine_specs
        }
        combine = {k: self._wrap_combine(k) for k in self._combine_variants}

        return KernelSet(
            dispatch=dispatch,
            combine=combine,
            dispatch_replay=None,
            # FlyDSL combine: host stages tokens, kernel resets its own counters.
            combine_stages_in_kernel=False,
            combine_resets_counters=True,
            # HIP dispatch: op must zero local scratch.
            dispatch_resets_counters=False,
            capabilities=frozenset({"gather"}),
        )

    def _close_backend(self):
        for plan in getattr(self, "_plans", ()):
            plan.close()

    # -- views ----------------------------------------------------------------

    def recv_tokens(self):
        cols = self.cfg.hidden_dim // 2 if self.cfg.is_fp4 else self.cfg.hidden_dim
        return from_gpu_ptr(
            self.arena.local_ptr("disp_out"),
            (self._recv_cap, cols),
            self.cfg.dispatch_dtype,
        )

    def combine_in_view(self):
        return from_gpu_ptr(
            self.arena.local_ptr("out_tok"),
            (self._recv_cap, self.cfg.hidden_dim),
            self.cfg.combine_dtype,
        )

    def recv_weights(self):
        return from_gpu_ptr(
            self.arena.local_ptr("out_wts"),
            (self._recv_cap, self.cfg.num_experts_per_token),
            torch.float32,
        )

    def recv_indices(self):
        return from_gpu_ptr(
            self.arena.local_ptr("out_idx"),
            (self._recv_cap, self.cfg.num_experts_per_token),
            torch.int32,
        )

    def recv_scales(self):
        return None

    def local_expert_count(self):
        raise NotImplementedError(
            "local_expert_count is flydsl-only; use backend='flydsl'"
        )

    def convert_dispatch_output(self):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    def convert_combine_input(self, routing):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    # -- kernel adapters ------------------------------------------------------

    def _wrap_dispatch(self, plan):
        def run(*, input, indices, weights, scales, dest_map, num_tokens):
            plan.launch(
                stream=torch.cuda.current_stream().cuda_stream,
                token_indices=indices,
                inp_token_buf=input,
                weights_buf=weights,
                disp_dest_tok_id_map=dest_map,
                dest_pe_token_counter=self.dest_pe_counter,
                total_recv_token_num=self.total_recv,
                grid_barrier=self.dispatch_barrier,
                num_tokens=num_tokens,
            )

        return run

    def _wrap_combine(self, spec):
        def run(*, input, dest_map, total_recv, num_tokens, want_weights=False):
            self._combine_variants[spec](
                self.arena.handle,
                dest_map.data_ptr(),
                self.combine_barrier.data_ptr(),
                self.cross_device_flag.data_ptr(),
                total_recv.data_ptr(),
                self.combine_out.data_ptr(),
                self.combine_out_weights.data_ptr(),
                self.cfg.rank,
                num_tokens,
                fx.Stream(torch.cuda.current_stream()),
            )

        return run
