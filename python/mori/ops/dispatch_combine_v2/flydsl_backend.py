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
"""mori-parity host op-layer for the cco-LSA intranode dispatch/combine kernels.

One SymmArena window holds the symmetric staging; per-rank metadata are plain
device tensors surfaced to the caller via from_gpu_ptr.
"""

import torch

import flydsl.expr as fx
from mori.tensor_utils import from_gpu_ptr

from .intranode_kernels import (
    xdb_flag_slots,
    make_dispatch,
    make_combine,
    make_combine_scatter,
    make_convert_dispatch_output,
    make_convert_combine_input,
    make_local_expert_count,
)
from .dispatch_combine_op import (  # noqa: F401
    _DT,
    _FP8_DTYPES,
    _QUANT_TYPES,
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
    EpDispatchRoutingHandle,
    KernelSet,
)
from .symm_arena import SymmArena  # noqa: F401


class EpDispatchCombineOpFlyDSL(EpDispatchCombineOp, backend="flydsl"):
    """FlyDSL-kernel EP op. The full v2 feature set: gather + scatter combine,
    fp8/fp4, quant, StdMoE, per-token scales, routing replay."""

    def __init__(self, cfg: EpDispatchCombineConfig, comm):
        self.cfg = cfg
        self.comm = comm
        device = torch.device("cuda", torch.cuda.current_device())
        self.dev = device
        elem_size = cfg.elem_size
        is_fp4 = cfg.is_fp4
        is_fp8 = cfg.is_fp8
        token_nbytes = cfg.token_nbytes  # per-token transport bytes (fp4 = hidden/2)
        if (is_fp4 or is_fp8) and cfg.is_scatter:
            raise ValueError(
                "plain fp4/fp8 token dtype is gather-only "
                "(fp8 quant uses quant_type=fp8_direct_cast, not data_type)"
            )
        topk = cfg.num_experts_per_token
        hidden_dim = cfg.hidden_dim
        max_tok_per_rank = cfg.max_num_inp_token_per_rank
        recv_cap = cfg.effective_max_recv  # recv-slot cap (== ws*M unless capped)
        self._recv_cap = recv_cap

        self._scale_bytes = cfg.scale_dim * cfg.scale_type_size
        self._scale_num_i32 = (self._scale_bytes + 3) // 4
        self._enable_scales = self._scale_bytes > 0

        regions = [
            ("tok_off", 4),
            ("recv_num", cfg.world_size * 4),
            ("recv_to_src_token", recv_cap * 4),
            ("out_idx", recv_cap * topk * 4),
            ("out_wts", recv_cap * topk * 4),
            # disp_out: dispatch scatter dest / expert-GEMM input (recv_x). Kept
            # separate from out_tok so combine's copy-in never clobbers the
            # dispatched tokens the expert still reads — callers can skip .clone().
            ("disp_out", recv_cap * token_nbytes),
            # out_tok: combine staging (post-expert results that peers gather).
            ("out_tok", recv_cap * cfg.combine_token_nbytes),
            ("cross_device_barrier", cfg.world_size * 8),
        ]
        if self._enable_scales:
            regions.append(("out_scales", recv_cap * self._scale_num_i32 * 4))
        # scatter combine needs its own staging regions
        if cfg.is_scatter:
            wire_elem_size = cfg.wire_elem_size
            regions.append(
                (
                    "comb_inp",
                    cfg.world_size * max_tok_per_rank * hidden_dim * wire_elem_size,
                )
            )
            regions.append(("comb_wts", cfg.world_size * max_tok_per_rank * topk * 4))
            if cfg.fp8_blockwise:
                regions.append(
                    (
                        "comb_scales",
                        cfg.world_size * max_tok_per_rank * cfg.combine_scale_dim * 4,
                    )
                )
        self.arena = SymmArena(comm, regions)
        self.arena.zero()

        self.token_dest_map = torch.full(
            (max_tok_per_rank * topk,), -1, dtype=torch.int32, device=device
        )
        self.dest_pe_counter = torch.zeros(
            cfg.world_size, dtype=torch.int32, device=device
        )
        self.dispatch_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=device)
        self.combine_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        # Per-block xdb flag counters for the gather combine entry barrier: one
        # i64 per block, fixed at xdb_flag_slots (== CU count, the max combine
        # block_num). Every block owns a private counter and block 0 fills the
        # unused tail so all stay in lockstep across calls with different block_num.
        self.cross_device_flag = torch.ones(
            xdb_flag_slots, dtype=torch.int64, device=device
        )
        c_dt = cfg.combine_dtype  # combine output dtype
        c_elem = cfg.combine_elem_size
        if c_dt == torch.float4_e2m1fn_x2:  # fp4 combine outputs fp4 (hidden/2 B/token)
            self.combine_out = torch.zeros(
                max_tok_per_rank * (hidden_dim // 2), dtype=torch.int8, device=device
            )
        elif c_dt in _FP8_DTYPES:  # fp8 combine outputs fp8 (1 byte/elem)
            self.combine_out = torch.zeros(
                max_tok_per_rank * hidden_dim, dtype=torch.int8, device=device
            )
        else:
            self.combine_out = torch.zeros(
                max_tok_per_rank * hidden_dim,
                dtype=torch.int16 if c_elem == 2 else torch.int32,
                device=device,
            )
        self.combine_out_weights = torch.zeros(
            max_tok_per_rank * topk, dtype=torch.float32, device=device
        )

        arena = self.arena
        # Distinct (block, warp) variants to precompile. With a per-token schedule
        # the op picks the best (block, warp) at runtime from cur_rank_num_token;
        # otherwise it is single-shot. Scatter combine is not schedule-tuned.
        dispatch_specs, combine_specs = self._specs_from(cfg)
        self._dispatch_specs = dispatch_specs
        self._combine_specs = combine_specs

        self._dispatch_kwargs = dict(
            rank=cfg.rank,
            npes=cfg.world_size,
            experts_per_rank=cfg.num_experts_per_rank,
            experts_per_token=topk,
            hidden_dim=hidden_dim,
            hidden_elem_size=elem_size,
            max_tok_per_rank=max_tok_per_rank,
            max_recv=recv_cap,
            off_tok_off=arena.offset("tok_off"),
            off_recv_num=arena.offset("recv_num"),
            off_tis=arena.offset("recv_to_src_token"),
            off_out_idx=arena.offset("out_idx"),
            off_out_wts=arena.offset("out_wts"),
            off_out_tok=arena.offset("disp_out"),
            off_out_scales=arena.offset("out_scales") if self._enable_scales else 0,
            scale_dim=cfg.scale_dim,
            scale_type_size=cfg.scale_type_size,
            fp4=is_fp4,
        )
        # (block, warp) -> compiled dispatch / combine kernel.
        self._dispatch_variants = {
            (b, w): make_dispatch(
                block_num=b, warp_num_per_block=w, **self._dispatch_kwargs
            )
            for (b, w) in dispatch_specs
        }
        self._dispatch_replay_variants_raw = {}  # lazily compiled per (block, warp)
        if cfg.is_scatter:
            self._combine_variants = {
                (b, w): make_combine_scatter(
                    rank=cfg.rank,
                    npes=cfg.world_size,
                    experts_per_token=topk,
                    hidden_dim=hidden_dim,
                    hidden_elem_size=elem_size,
                    max_tok_per_rank=max_tok_per_rank,
                    max_recv=recv_cap,
                    block_num=b,
                    warp_num_per_block=w,
                    off_out_tok=arena.offset("out_tok"),
                    off_comb_inp=arena.offset("comb_inp"),
                    off_tis=arena.offset("recv_to_src_token"),
                    off_xdb_mem=arena.offset("cross_device_barrier"),
                    off_out_wts=arena.offset("out_wts"),
                    off_comb_wts=arena.offset("comb_wts"),
                    off_comb_scales=(
                        arena.offset("comb_scales") if cfg.fp8_blockwise else 0
                    ),
                    fp8_direct_cast=cfg.fp8_direct_cast,
                    fp8_blockwise=cfg.fp8_blockwise,
                    scale_dim=cfg.combine_scale_dim,
                    reset_total_recv=False,
                )
                for (b, w) in combine_specs
            }
        else:
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
                for (b, w) in combine_specs
            }

        self._local_expert_count_buf = torch.zeros(
            cfg.num_experts_per_rank, dtype=torch.int32, device=device
        )
        self._local_expert_count = make_local_expert_count(
            rank=cfg.rank,
            experts_per_rank=cfg.num_experts_per_rank,
            experts_per_token=topk,
            block_num=cfg.dispatch_block_num,
            warp_num_per_block=cfg.warp_num_per_block,
        )

        if cfg.enable_std_moe:
            assert elem_size == 2, "StdMoE convert path is bf16-only"
            experts_per_rank = cfg.num_experts_per_rank
            max_tok_per_expert = cfg.world_size * max_tok_per_rank
            self._max_tok_per_expert = max_tok_per_expert
            self.packed_x = torch.zeros(
                experts_per_rank * max_tok_per_expert * hidden_dim,
                dtype=torch.int16,
                device=device,
            )
            self.packed_count = torch.zeros(
                experts_per_rank, dtype=torch.int32, device=device
            )
            self.packed_src = torch.zeros(
                experts_per_rank * max_tok_per_expert, dtype=torch.int32, device=device
            )
            self.slot_map = torch.full(
                (recv_cap * topk,), -1, dtype=torch.int64, device=device
            )
            self._convert_dispatch = make_convert_dispatch_output(
                rank=cfg.rank,
                experts_per_rank=experts_per_rank,
                experts_per_token=topk,
                hidden_dim=hidden_dim,
                hidden_elem_size=elem_size,
                max_tok_per_expert=max_tok_per_expert,
                block_num=cfg.dispatch_block_num,
                warp_num_per_block=cfg.warp_num_per_block,
            )
            self._convert_combine = make_convert_combine_input(
                rank=cfg.rank,
                experts_per_rank=experts_per_rank,
                experts_per_token=topk,
                hidden_dim=hidden_dim,
                hidden_elem_size=elem_size,
                max_tok_per_expert=max_tok_per_expert,
                block_num=cfg.combine_block_num,
                warp_num_per_block=cfg.combine_warp_num_per_block,
            )
        self._kernels = KernelSet(
            dispatch={
                k: self._wrap_dispatch(k, replay=False) for k in self._dispatch_variants
            },
            combine={k: self._wrap_combine(k) for k in self._combine_variants},
            # Replay variants stay lazily compiled -- building all of them
            # eagerly would cost every op a compile it usually never uses. The
            # wrapper keeps that laziness inside the backend instead of on the
            # base's hot path.
            dispatch_replay={
                k: self._wrap_dispatch(k, replay=True) for k in self._dispatch_variants
            },
            # FlyDSL stages combine's input on the host (see combine()), and its
            # kernels reset their own counters.
            stages_in_kernel=False,
            self_resets_counters=True,
            capabilities=frozenset(
                {
                    "gather",
                    "scatter",
                    "quant",
                    "std_moe",
                    "scales",
                    "replay",
                    "local_expert_count",
                    "asymmetric_dtype",
                    "recv_cap",
                }
            ),
        )
        self._gate(self._kernels)
        self._closed = False

    def recv_tokens(self):
        """Arena disp_out [max_recv, hidden] (dispatch dest / expert-GEMM input).
        Separate from out_tok, so combine's copy-in never overwrites it — the
        expert can read this in place without a defensive .clone().
        fp4 packs 2 e2m1 per float4_e2m1fn_x2 element -> last dim is hidden/2."""
        cols = self.cfg.hidden_dim // 2 if self.cfg.is_fp4 else self.cfg.hidden_dim
        return from_gpu_ptr(
            self.arena.local_ptr("disp_out"),
            (self._recv_cap, cols),
            self.cfg.dispatch_dtype,
        )

    def combine_in_view(self):
        """Symmetric buffer [max_recv, hidden] that gather-mode combine reads
        from via P2P.  Scatter mode uses a different staging layout (comb_inp)
        and cannot use this buffer.

        To skip the d2d copy inside combine(), write expert output directly
        into this view (e.g. point the GEMM output pointer here), then pass
        it as the ``input`` argument to combine().  combine() detects the
        matching data_ptr and elides the copy."""
        cdt = self.cfg.combine_dtype
        cols = (
            self.cfg.hidden_dim // 2
            if cdt == torch.float4_e2m1fn_x2
            else self.cfg.hidden_dim
        )
        return from_gpu_ptr(
            self.arena.local_ptr("out_tok"), (self._recv_cap, cols), cdt
        )

    def convert_dispatch_output(self):
        """mori ConvertDispatchOutput: repack recv tokens into per-local-expert
        buckets. Returns (packed_x, packed_count, packed_src); GEMM overwrites
        packed_x in place."""
        assert self.cfg.enable_std_moe, "op built without enable_std_moe"
        self.packed_count.zero_()
        self.slot_map.fill_(-1)
        arena = self.arena
        stream = fx.Stream(torch.cuda.current_stream())
        self._convert_dispatch(
            arena.local_ptr("disp_out"),
            arena.local_ptr("out_idx"),
            arena.local_ptr("recv_to_src_token"),
            self.total_recv.data_ptr(),
            self.packed_x.data_ptr(),
            self.packed_count.data_ptr(),
            self.packed_src.data_ptr(),
            self.slot_map.data_ptr(),
            stream,
        )
        experts_per_rank = self.cfg.num_experts_per_rank
        max_tok_per_expert = self._max_tok_per_expert
        hidden_dim = self.cfg.hidden_dim
        packed_x_view = from_gpu_ptr(
            self.packed_x.data_ptr(),
            (experts_per_rank, max_tok_per_expert, hidden_dim),
            self.cfg.dispatch_dtype,
        )
        return packed_x_view, self.packed_count, self.packed_src

    def convert_combine_input(self, routing):
        """mori ConvertCombineInput: weighted-reduce each recv token's local-expert
        outputs from packed_x back into out_tok. Run after GEMM, before combine."""
        assert self.cfg.enable_std_moe, "op built without enable_std_moe"
        arena = self.arena
        stream = fx.Stream(torch.cuda.current_stream())
        self._convert_combine(
            arena.local_ptr("out_tok"),
            arena.local_ptr("out_wts"),
            routing.total_recv_token_num.data_ptr(),
            self.packed_x.data_ptr(),
            self.slot_map.data_ptr(),
            stream,
        )

    def recv_weights(self):
        """Arena out_wts as [max_recv, topk] f32 (forwarded per-token weights)."""
        return from_gpu_ptr(
            self.arena.local_ptr("out_wts"),
            (self._recv_cap, self.cfg.num_experts_per_token),
            torch.float32,
        )

    def recv_indices(self):
        """Arena out_idx as [max_recv, topk] i32 (forwarded expert indices)."""
        return from_gpu_ptr(
            self.arena.local_ptr("out_idx"),
            (self._recv_cap, self.cfg.num_experts_per_token),
            torch.int32,
        )

    def recv_scales(self):
        """Forwarded per-token scales as opaque i32 dwords [max_recv, scale_num_i32],
        or None if built without scales."""
        if not self._enable_scales:
            return None
        return from_gpu_ptr(
            self.arena.local_ptr("out_scales"),
            (self._recv_cap, self._scale_num_i32),
            torch.int32,
        )

    # -- kernel adapters: FlyDSL's positional convention -> the base's named one --

    def _wrap_dispatch(self, spec, *, replay):
        """One dispatch variant, callable the way ep_backend expects.

        Everything the base does not know about -- the window handle, the
        counters, the barrier, rank, the stream -- is captured here.
        """

        def run(*, input, indices, weights, scales, dest_map, num_tokens):
            if replay:
                kern = self._dispatch_replay_variants_raw.get(spec)
                if kern is None:
                    kern = self._dispatch_replay_variants_raw[spec] = make_dispatch(
                        replay=True,
                        block_num=spec[0],
                        warp_num_per_block=spec[1],
                        **self._dispatch_kwargs,
                    )
            else:
                kern = self._dispatch_variants[spec]
            kern(
                self.arena.handle,
                input.data_ptr(),
                indices.data_ptr(),
                weights.data_ptr() if weights is not None else 0,
                dest_map.data_ptr(),
                self.dest_pe_counter.data_ptr(),
                self.dispatch_barrier.data_ptr(),
                self.total_recv.data_ptr(),
                (
                    scales.data_ptr()
                    if (scales is not None and self._enable_scales)
                    else 0
                ),
                self.cfg.rank,
                num_tokens,
                fx.Stream(torch.cuda.current_stream()),
            )

        return run

    def _wrap_combine(self, spec):
        # want_weights is accepted and ignored: the FlyDSL combine kernels take the
        # out_weights pointer unconditionally and have no null gate for it, so the
        # fold always runs here. The base still returns None when it was not asked
        # for, so the two backends agree on what combine() hands back.
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

    def local_expert_count(self):
        """[num_experts_per_rank] i32: recv tokens per local expert. Call after
        dispatch, before combine (gather resets total_recv)."""
        self._local_expert_count_buf.zero_()
        stream = fx.Stream(torch.cuda.current_stream())
        self._local_expert_count(
            self.arena.local_ptr("out_idx"),
            self.total_recv.data_ptr(),
            self._local_expert_count_buf.data_ptr(),
            stream,
        )
        return self._local_expert_count_buf
