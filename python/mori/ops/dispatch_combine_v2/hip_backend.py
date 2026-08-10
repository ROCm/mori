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
"""HIP/JIT kernel backend for the v2 EP op.

Same surface as the FlyDSL backend (``EpDispatchCombineOpFlyDSL``): same
constructor, same ``dispatch``/``combine`` signatures and return shapes, same
routing handle. What differs is where the kernels come from, and which configs
can be served -- this backend implements the bf16/fp32 gather path and rejects
everything else at CONSTRUCTION rather than at launch.

Imports ``ep_plans`` (the C++/JIT plans) but never flydsl, so it works where
FlyDSL is not installed.
"""

from __future__ import annotations

import torch

from mori.tensor_utils import from_gpu_ptr

from . import ep_plans as cb
from .dispatch_combine_op import EpDispatchCombineOp, KernelSet
from .symm_arena import SymmArena

# C++ offset-argument stem -> arena region name. The C++ side names the offsets
# after its own EpArgs fields (offTokOff -> "tokOff"); the region names match the
# FlyDSL op's, so one arena can serve either backend.
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

# Only what EpDType enumerates. fp16 is deliberately absent: the wire code for an
# enum field comes from mori.jit.v2.plan_api.DTYPES, which has no fp16 entry, so
# advertising it here would either raise or -- worse -- alias onto another code.
_DTYPE_BYTES = {torch.bfloat16: 2, torch.float32: 4}


class EpDispatchCombineOpHip(EpDispatchCombineOp, backend="hip"):
    """C++/JIT-kernel EP op: bf16/fp32, gather combine, no quant, no replay."""

    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self._recv_cap = cfg.effective_max_recv
        self._closed = False

        # Gate FIRST: rejecting a config after taking a symmetric window would
        # leak it (the arena is registered with the communicator), and the whole
        # point of the gate is that an unsupported config never gets that far.
        self._gate(KernelSet(dispatch={}, combine={}, unsupported=self._unsupported(cfg)))

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
        self.combine_barrier = torch.zeros(1, dtype=torch.uint32, device=dev)
        # Monotone epoch. Starts at 1 so the zeroed barrier slots cannot alias
        # the first launch's flag value.
        self.cross_device_flag = torch.ones(1, dtype=torch.int64, device=dev)
        self.combine_out = torch.zeros(
            max_tok * cfg.hidden_dim, dtype=cfg.combine_dtype, device=dev
        )
        self.combine_out_weights = torch.zeros(
            max_tok * topk, dtype=torch.float32, device=dev
        )

    # -- backend hooks -----------------------------------------------------

    def _regions(self, cfg):
        elem = _DTYPE_BYTES.get(cfg.dispatch_dtype, 2)
        celem = _DTYPE_BYTES.get(cfg.combine_dtype, 2)
        cap = cfg.effective_max_recv
        topk = cfg.num_experts_per_token
        return [
            ("tok_off", 4),
            ("recv_num", cfg.world_size * 4),
            ("recv_to_src_token", cap * 4),
            ("out_idx", cap * topk * 4),
            ("out_wts", cap * topk * 4),
            ("disp_out", cap * cfg.hidden_dim * elem),
            ("out_tok", cap * cfg.hidden_dim * celem),
            ("cross_device_barrier", cfg.world_size * 8),
        ]

    def _unsupported(self, cfg) -> tuple[str, ...]:
        """Everything this backend cannot do, checked before anything is built."""
        bad = []
        if cfg.dispatch_dtype not in _DTYPE_BYTES:
            bad.append(f"dispatch dtype {cfg.dispatch_dtype} (have bf16, fp32)")
        if cfg.combine_dtype not in _DTYPE_BYTES:
            bad.append(f"combine dtype {cfg.combine_dtype} (have bf16, fp32)")
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (gather only)")
        if cfg.quant_type != "none":
            bad.append(f"quant_type={cfg.quant_type!r}")
        if cfg.enable_std_moe:
            bad.append("enable_std_moe")
        if cfg.scale_dim and cfg.scale_type_size:
            bad.append("per-token scales forwarding")
        if cfg.is_asymmetric_dtype:
            bad.append("asymmetric dispatch/combine dtype")
        # The C++ validator rejects a shrunk cap: the recv capacity is also the
        # flat-index stride, so an overflow re-encodes to the next peer instead
        # of merely overrunning the region.
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
            # Build nothing when the config is out of range: constructing a Plan
            # compiles, and compiling a kernel we are about to reject is both
            # slow and misleading.
            return KernelSet(dispatch={}, combine={}, unsupported=bad)

        common = dict(
            world_size=cfg.world_size,
            hidden_dim=cfg.hidden_dim,
            max_tok_per_rank=cfg.max_num_inp_token_per_rank,
            num_expert_per_rank=cfg.num_experts_per_rank,
            num_expert_per_token=cfg.num_experts_per_token,
            max_recv=cfg.effective_max_recv,
            dtype=cfg.dispatch_dtype,
            use_weights=True,
            arena=arena,
            region_names=_REGIONS,
        )
        # One plan per (block, warp) the schedule can select. Compilation happens
        # here and only here, so _pick never touches the compiler.
        dispatch, combine = {}, {}
        self._plans = []
        for b, w in self._dispatch_specs:
            plan = cb.EpDispatchPlan(**common, block_num=b, warp_per_block=w)
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            dispatch[(b, w)] = self._wrap_dispatch(plan)
        for b, w in self._combine_specs:
            plan = cb.EpCombinePlan(**common, block_num=b, warp_per_block=w)
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            combine[(b, w)] = self._wrap_combine(plan)

        return KernelSet(
            dispatch=dispatch,
            combine=combine,
            dispatch_replay=None,  # no replay path in this backend
            # The combine kernel stages into out_tok itself (and skips the copy
            # when the caller already wrote there), so the op must not do it.
            stages_in_kernel=True,
            # These are plain local buffers, not symmetric regions: the kernels
            # do not reset them, the op must.
            self_resets_counters=False,
            capabilities=frozenset({"gather"}),
        )

    def _close_backend(self):
        for plan in getattr(self, "_plans", ()):
            plan.close()

    # -- views (same contract as the FlyDSL backend) -----------------------

    def recv_tokens(self):
        return from_gpu_ptr(
            self.arena.local_ptr("disp_out"),
            (self._recv_cap, self.cfg.hidden_dim),
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
        """Always None: this backend forwards no scales. Not an error -- it is
        the same answer FlyDSL gives for a config built without them, and a
        config that actually asks for scales is rejected by the gate."""
        return None

    def local_expert_count(self):
        raise NotImplementedError("local_expert_count is flydsl-only; use backend='flydsl'")

    def convert_dispatch_output(self):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    def convert_combine_input(self, routing):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    # -- ops ---------------------------------------------------------------

    # -- kernel adapters: the ctypes plan -> the base's named convention --

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

    def _wrap_combine(self, plan):
        def run(*, input, dest_map, total_recv, num_tokens):
            plan.launch(
                stream=torch.cuda.current_stream().cuda_stream,
                inp_token_buf=input,
                out_token_buf=self.combine_out,
                out_weights_buf=self.combine_out_weights,
                disp_dest_tok_id_map=dest_map,
                total_recv_token_num=total_recv,
                grid_barrier=self.combine_barrier,
                xdb_flag=self.cross_device_flag,
                num_tokens=num_tokens,
            )

        return run
