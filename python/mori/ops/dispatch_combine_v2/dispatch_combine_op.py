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
"""Backend-agnostic base + entry point for the v2 EP op.

Two backends live beside this module -- FlyDSL (``flydsl_backend``) and HIP/JIT
(``hip_backend``) -- behind one class. ``EpDispatchCombineOp(cfg, comm)`` returns
whichever ``cfg.kernel_backend`` (or ``MORI_V2_KERNEL_BACKEND``) names, defaulting
to flydsl; ``EpDispatchCombineOpHip(cfg, comm)`` is the explicit form, and
``isinstance(op, EpDispatchCombineOp)`` holds either way.

Neither backend is imported here. Selecting one imports only that one, so the HIP
backend works on a machine with no flydsl.

A subclass supplies four hooks -- two with no default, two with one::

    _regions(cfg)              -> [(name, nbytes)]   arena layout it needs
    _build_kernels(cfg, arena) -> KernelSet          bound, ready-to-launch kernels
    _unsupported(cfg)          -> (reason, ...)      why it cannot serve this cfg;
                                                     empty tuple = it can serve it
    _region(name)              -> name               arena region a shared view reads

That is the HIP backend's route. The FlyDSL one overrides ``__init__`` and builds
its own arena and KernelSet there, so ``_unsupported`` is the only hook it defines.

Everything else -- arena, scratch buffers, variant selection, lifecycle -- is here.
Behavioural differences between the backends are *data* on the KernelSet, not
methods to override, after aiter's ``MOEMetadata``.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from dataclasses import dataclass
from typing import Callable

import torch

from mori.tensor_utils import from_gpu_ptr
from mori.jit.config import detect_wave_size

WAVE = detect_wave_size()

# Where each backend lives. Imported lazily, on selection only.
_BACKEND_MODULES = {"flydsl": "flydsl_backend", "hip": "hip_backend"}


def _default_backend() -> str:
    from mori.ops.dispatch_combine import _is_gfx125x

    return "hip" if _is_gfx125x() else "flydsl"


_FP4_QUANT_TYPES = ("fp4_blockwise", "fp4_blockwise_fp32")
# The quant types the hip backend runs as its gfx125x push combine: the sender
# quantizes into the destination's landing rows and the destination reduces them.
_PUSH_QUANT_TYPES = _FP4_QUANT_TYPES + ("fp8_blockwise",)
_QUANT_TYPES = ("none", "fp8_direct_cast", "fp8_blockwise") + _FP4_QUANT_TYPES

_INTERNODE_KERNELS = ("auto", "v2", "v2_ll")

_DT = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fnuz: 1,
    torch.float8_e4m3fn: 1,
}
_FP8_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


@dataclass
class EpDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    # Base token dtype; dispatch_data_type / combine_data_type override it per-op
    # (None => data_type). Asymmetric (fp8 dispatch -> bf16 combine) fits an expert
    # op that converts dtype between the two. gather mode only.
    data_type: torch.dtype = torch.bfloat16
    dispatch_data_type: torch.dtype = None
    combine_data_type: torch.dtype = None
    # Per-token quant scales forwarded verbatim to dest out_scales (0 disables).
    scale_dim: int = 0
    scale_type_size: int = 0
    # "gather" (UseP2PRead) or "scatter" (mori _nop2p, fp8 compression home).
    combine_mode: str = "gather"
    quant_type: str = "none"  # none | fp8_direct_cast | fp8_blockwise
    # Geometry: None => the tuned schedule for this device/shape/dtype; pin any of
    # these to opt out. Combine keeps its own warp count -- its K-deep per-lane MLP
    # saturates sooner than dispatch's copy.
    #
    # Pinning is per FIELD, not all-or-nothing: on the internode path a pinned
    # field overrides the tuning table for every token bucket while the fields
    # left None stay tuned, so you can fix one knob without hand-writing the
    # other five. MORI_EP_DISP_GEOM / MORI_EP_COMB_GEOM override the whole triple
    # for a leg and bypass the table entirely.
    dispatch_block_num: int = None
    combine_block_num: int = None
    warp_num_per_block: int = None
    combine_warp_num_per_block: int = None
    # Optional per-token plan: tuple of (max_tok_inclusive | None, disp_block,
    # disp_warp, comb_block, comb_warp) buckets. When set, the op precompiles the
    # distinct (block, warp) variants and picks one at runtime from
    # cur_rank_num_token. None => auto (from tuning_configs) or single-shot fallback.
    schedule: tuple = None
    enable_std_moe: bool = False
    max_total_recv_tokens: int = 0  # mori maxTotalRecvTokens; 0 = worst-case ws*M
    # --- internode -----------------------------------------------------------
    # GPUs per physical node. None => one node (== world_size), which is what
    # every intranode config is and keeps their behaviour unchanged. Setting it
    # smaller is what selects the internode path: it is EP's own idea of a node
    # and must agree with the communicator's LSA team, which the op checks at
    # construction rather than assuming.
    gpu_per_node: int = None
    # QPs per peer on the RDMA leg. Only read on the internode path, where 1
    # starves it (~1.5x); the intranode path never reaches RDMA and ignores it,
    # so the default is the value the only consumer wants.
    num_qp_per_pe: int = 2
    # Widest transported element, which is what sizes the staging buffers. None
    # => max over the two legs, which is what the buffers actually have to hold.
    max_token_type_size: int = None
    # RDMA-block split of the grid, per phase. Runtime, never compiled in:
    # dispatch and combine are deliberately tuned to DIFFERENT values (see
    # internode_tuning_configs), so one compiled-in value cannot serve both.
    # Pinnable like the block/warp fields above, and clamped to < block_num.
    dispatch_rdma_block_num: int = None
    combine_rdma_block_num: int = None
    # Which of the two internode kernel families runs. "v2" and "v2_ll" are
    # SEPARATE kernels -- separate JIT modules, entry symbols and cache keys, not
    # two branches of one body -- so naming one here compiles only that one.
    #   "v2"     the general path: chunked, deduplicating, sized for wide tokens.
    #   "v2_ll"  low latency: no dedup across expert slots, one entry per node
    #            per token, which wins while the wire is not the bottleneck.
    #   "auto"   compile BOTH and pick per launch from the token count, at
    #            internode_auto_ll_max_tokens. This is the only mode that can switch
    #            at runtime, and it is the only one that pays for both compiles.
    internode_kernel: str = "auto"
    # The "auto" crossover. Compared against THIS CALL's token count -- dispatch
    # uses input.shape[0], combine routing.cur_rank_num_token -- so one op
    # alternates between the families as the batch changes; it is not a property
    # of the shape. Unrelated to max_num_inp_token_per_rank, which is the
    # capacity. Read only when internode_kernel == "auto".
    internode_auto_ll_max_tokens: int = 512
    # Which kernel backend serves this op: "flydsl" (default, full intranode
    # feature set) or "hip" (HIP/JIT: gather only, no quant, no StdMoE, no
    # routing replay; dispatch transports bf16/fp32/fp8/fp4 and combine reduces
    # in bf16/fp32). "hip" is the only backend with an internode path -- flydsl's
    # _unsupported rejects any config whose gpu_per_node < world_size -- so a
    # multi-node config must name it. None = MORI_V2_KERNEL_BACKEND, else the
    # default. Only consulted when constructing the BASE class; naming a
    # subclass directly wins.
    kernel_backend: str = None
    # Opt in to the AsyncLL kernel family (hip backend only). It is not a member
    # of internode_kernel because it is not an internode-only choice: AsyncLL runs
    # on one node too, which is where it is validated, and hanging it off a field
    # named "internode" would say otherwise.
    #
    # What it buys is the send/recv split -- dispatch_send/dispatch_recv and
    # combine_send/combine_recv become separate launches, so the caller can run
    # its own work in the seam while the transfers drain. Every other family does
    # send and recv in one kernel and has no such seam.
    #
    # Intranode it moves tokens with ccoSdma, the copy engine, rather than with
    # peer stores: that is the v1 AsyncLL protocol transcribed onto cco, and it is
    # what the cross-node leg will reuse. It is therefore NOT the fastest way to
    # dispatch on a single node -- the ordinary v2 intranode kernels are, by about
    # 2x on MI355X -- and it is off by default for that reason.
    async_ll: bool = False
    # How AsyncLL's transfer passes move bytes between same-node peers.
    #
    #   "sdma" (default) -- ccoSdma, the copy engine. The transfer occupies no
    #       CUs, so it can drain under whatever the caller runs in the send/recv
    #       seam, and it is the shape the cross-node (ccoGda) leg will reuse.
    #   "lsa" -- peer stores over the flat symmetric VA. Faster only at small
    #       batch; it spends the grid on the copy, so there is nothing left to
    #       overlap and it contends with its own HBM traffic.
    #
    # Measured on an idle reserved MI355X node, EP8, hidden 7168, bf16,
    # dispatch+combine as a graph, two sweeps agreeing within 0.3pp:
    #
    #   tok/rank      64     128     256     512    1024    2048   (us)
    #   sdma       102.0   136.8   194.3   292.1   523.1   982.0
    #   lsa         97.7   138.9   201.5   320.3   581.4  1094.2
    #
    # So "lsa" wins by ~4% at 64 tokens and loses by up to ~11% from 128 up,
    # which is why "sdma" is the default. Neither closes the gap to the ordinary
    # v2 intranode kernels (62.5 / 856.9 us at the ends): AsyncLL stages each
    # payload through an extra buffer so an RDMA leg can reuse the protocol, and
    # that cost is the same for both transports.
    #
    # Pick "lsa" for a small-batch, empty-seam call; keep "sdma" otherwise, when
    # the point is the overlap, or when this op will later run across nodes.
    async_ll_transport: str = "sdma"

    def __post_init__(self):
        # Node grouping first: is_internode gates checks further down, and the
        # derived defaults below are cheaper to reason about once it is fixed.
        if self.gpu_per_node is None:
            self.gpu_per_node = self.world_size
        if self.gpu_per_node <= 0 or self.world_size % self.gpu_per_node:
            raise ValueError(
                f"world_size ({self.world_size}) must be a positive multiple of "
                f"gpu_per_node ({self.gpu_per_node}); EP addresses a peer as "
                "node * gpu_per_node + local rank and a partial node has no such "
                "encoding"
            )
        if self.num_qp_per_pe < 1:
            raise ValueError(f"num_qp_per_pe must be >= 1, got {self.num_qp_per_pe}")

        # all-or-none: setting only one silently defaults the other to data_type.
        if (self.dispatch_data_type is None) != (self.combine_data_type is None):
            raise ValueError(
                "dispatch_data_type / combine_data_type must be set together "
                "(all-or-none): got dispatch_data_type="
                f"{self.dispatch_data_type}, combine_data_type={self.combine_data_type}. "
                "Set data_type alone for a symmetric op, or set both explicitly for "
                "asymmetric dispatch/combine dtypes."
            )
        if self.quant_type not in _QUANT_TYPES:
            raise ValueError(
                f"quant_type must be one of {_QUANT_TYPES}, got {self.quant_type!r}"
            )
        if self.is_internode and self.max_num_inp_token_per_rank % WAVE:
            # The internode send buffer is addressed as
            # pe * max_num_inp_token_per_rank + slot, but slots are handed out in
            # whole wavefronts: the largest index a full last chunk produces is
            # ceil(capacity / WAVE) * WAVE - 1, which exceeds the capacity
            # whenever the capacity is not a multiple of WAVE. Rounding the
            # capacity up makes the two agree by construction, without changing
            # the layout the kernel and the region table both assume. (Inherited
            # from v1, which computes the same stride and the same chunk count.)
            self.max_num_inp_token_per_rank = (
                (self.max_num_inp_token_per_rank + WAVE - 1) // WAVE
            ) * WAVE
        if self.internode_kernel not in _INTERNODE_KERNELS:
            raise ValueError(
                f"internode_kernel must be one of {_INTERNODE_KERNELS}, got "
                f"{self.internode_kernel!r}"
            )
        if self.internode_auto_ll_max_tokens < 0:
            raise ValueError(
                "internode_auto_ll_max_tokens must be >= 0, got "
                f"{self.internode_auto_ll_max_tokens}"
            )
        if self.combine_mode not in ("gather", "scatter"):
            raise ValueError(
                f"combine_mode must be gather|scatter, got {self.combine_mode!r}"
            )
        # A quantised combine scatters: each destination writes its own partial
        # back. That is intranode-only, and it is what kept quant off the
        # internode path -- the backend would accept the quant type and then
        # reject the scatter this rule had just set.
        #
        # The internode kernels do carry an fp8 staging path
        # (EpCombineAllInternalFp8 reduces nNodes fp8 slots into one bf16
        # output), but it does not produce correct tokens: measured on 2x8
        # MI308X, weights are exact while ~51% of hidden elements are outside a
        # 30% tolerance and some are exactly zero. Rejecting here rather than
        # relying on the scatter rule to reject by accident, because an accepted
        # config that returns corrupt data is the worst of the three outcomes.
        if self.quant_type != "none":
            if self.is_internode:
                raise ValueError(
                    f"quant_type={self.quant_type!r} is not supported on the "
                    "internode path (world_size > gpu_per_node): the fp8 combine "
                    "staging path there is incomplete and returns wrong tokens. "
                    "Use quant_type='none', or an intranode config."
                )
            self.combine_mode = "scatter"
        # Token copy moves whole 16 B (vec4) chunks; a non-16 B-aligned per-token
        # size would over-read/write a few dwords past the token.
        if self.token_nbytes % 16 != 0:
            raise ValueError(
                f"per-token transport bytes must be 16 B aligned (vec4 copy); "
                f"hidden_dim={self.hidden_dim}, dispatch_dtype={self.dispatch_dtype} -> "
                f"token_nbytes={self.token_nbytes}"
            )
        if self.is_asymmetric_dtype:
            # dispatch output (disp_out, dispatch dtype) and combine staging
            # (out_tok, combine dtype) are separate buffers. gather/non-quant/
            # non-StdMoE only (the asymmetric path is implemented for gather).
            push = self.quant_type in _PUSH_QUANT_TYPES
            if (
                (self.combine_mode != "gather" and not push)
                or (self.quant_type != "none" and not push)
                or self.enable_std_moe
            ):
                raise ValueError(
                    "combine_data_type (asymmetric dtype) requires combine_mode=gather, "
                    f"quant_type=none (or one of {_PUSH_QUANT_TYPES}), "
                    "enable_std_moe=False"
                )
            # fp4 dispatch + bf16 combine (the SGLang/aiter fp4-asym path) is
            # supported; fp4 on the combine side is not.
            if self.combine_dtype == torch.float4_e2m1fn_x2:
                raise ValueError(
                    "asymmetric combine dtype does not support fp4 (fp4 dispatch "
                    "+ bf16 combine is supported; fp4 combine is not)"
                )
            if self.combine_token_nbytes % 16 != 0:
                raise ValueError(
                    f"combine per-token bytes must be 16 B aligned; combine_data_type="
                    f"{self.combine_data_type} -> {self.combine_token_nbytes}"
                )
        # The staging buffers hold whichever leg is wider, so that is the default.
        # Left settable because v1 configs pass it explicitly and the two must
        # agree for the arena sizes to match.
        widest = max(self.elem_size, self.combine_elem_size)
        if self.max_token_type_size is None:
            self.max_token_type_size = widest
        elif self.max_token_type_size < widest:
            # It sizes every staging region the kernel writes and is compiled into
            # the kernel as the transport stride, so a value below the widest leg
            # does not fail -- it silently writes past the region into whatever
            # the arena put next.
            raise ValueError(
                f"max_token_type_size={self.max_token_type_size} is smaller than "
                f"the widest transported element ({widest} bytes: dispatch "
                f"{self.elem_size}, combine {self.combine_elem_size}); it sizes "
                "the staging regions, so a smaller value corrupts them"
            )

        # What the CALLER pinned, captured before _resolve_geometry() fills the
        # rest: afterwards every field holds a number and the two are
        # indistinguishable. The internode backend overlays these on top of its
        # tuning table, so a pinned field wins and an unpinned one is tuned.
        self._pinned_geometry = frozenset(
            field_name
            for field_name in (
                "dispatch_block_num",
                "combine_block_num",
                "warp_num_per_block",
                "combine_warp_num_per_block",
                "dispatch_rdma_block_num",
                "combine_rdma_block_num",
            )
            if getattr(self, field_name) is not None
        )

        self._resolve_geometry()

    def _resolve_geometry(self):
        """Fill block/warp/schedule. Tuned-by-default: when the caller pinned
        neither a schedule nor any block/warp, pull the tuned geometry for this
        device/shape/dtype from the SELECTED BACKEND's tuning table (so the plain
        constructor is tuned automatically — EpDispatchCombineConfig.tuned() is now
        just an explicit alias). If any field is pinned, honor it and fill the rest
        with the single-shot fallback (no schedule).

        Each backend tunes its OWN kernel: the hip and flydsl kernels are different
        implementations with different optima, so hip reads hip_tuning_configs and
        flydsl reads tuning_configs -- they never share a table."""
        pinned = self.schedule is not None or any(
            g is not None
            for g in (
                self.dispatch_block_num,
                self.combine_block_num,
                self.warp_num_per_block,
                self.combine_warp_num_per_block,
            )
        )
        if self.is_internode:
            # The intranode tables are keyed on a single node's geometry and
            # carry no rdma_block_num, so they have nothing to say about this
            # config. Internode geometry comes from internode_tuning_configs,
            # which the backend applies per phase and per token count. A
            # `schedule` bucket already carries per-phase block and warp counts,
            # but it has no rdma_block_num field at all, so it cannot carry the
            # DIFFERENT rdma values dispatch and combine are tuned to.
            self.schedule = None
            if self.dispatch_block_num is None:
                self.dispatch_block_num = 96
            if self.combine_block_num is None:
                self.combine_block_num = 96
            if self.warp_num_per_block is None:
                self.warp_num_per_block = 8
            if self.combine_warp_num_per_block is None:
                self.combine_warp_num_per_block = 8
            if self.dispatch_rdma_block_num is None:
                self.dispatch_rdma_block_num = 64
            if self.combine_rdma_block_num is None:
                self.combine_rdma_block_num = 64
            return
        if not pinned:
            backend = (
                self.kernel_backend
                or os.environ.get("MORI_V2_KERNEL_BACKEND")
                or _default_backend()
            )
            if backend == "hip":
                from .hip_tuning_configs import lookup

                # hip keys on the expert count too: it sizes dispatch's per-block
                # expert counters, so it can move the geometry. FlyDSL's table has no
                # such axis, hence the split call rather than a shared kwarg.
                t = lookup(
                    self.world_size,
                    self.hidden_dim,
                    self.num_experts_per_token,
                    dtype=self.dispatch_dtype_str,
                    experts_per_rank=self.num_experts_per_rank,
                    quant_type=self.quant_type,
                )
            else:
                from .tuning_configs import lookup

                t = lookup(
                    self.world_size,
                    self.hidden_dim,
                    self.num_experts_per_token,
                    dtype=self.dtype_str,
                )
            self.dispatch_block_num = t["dispatch_block_num"]
            self.combine_block_num = t["combine_block_num"]
            self.warp_num_per_block = t["warp_num_per_block"]
            self.combine_warp_num_per_block = t["combine_warp_num_per_block"]
            self.schedule = t["schedule"]
        else:
            # explicit geometry: fill any unset field with the single-shot default
            if self.dispatch_block_num is None:
                self.dispatch_block_num = 64
            if self.combine_block_num is None:
                self.combine_block_num = 80
            if self.warp_num_per_block is None:
                self.warp_num_per_block = 16
            if self.combine_warp_num_per_block is None:
                self.combine_warp_num_per_block = 4

        # Precise world_size check against the resolved geometry.  The combine
        # xdb barrier polls with `tid < npes`, so every schedule bucket must
        # have blockDim (= comb_warp * WAVE) >= world_size.
        if self.schedule:
            min_comb_warp = min(bucket[4] for bucket in self.schedule)
        else:
            min_comb_warp = self.combine_warp_num_per_block
        max_peers = min_comb_warp * WAVE
        if self.world_size > max_peers:
            raise ValueError(
                f"world_size ({self.world_size}) exceeds the smallest combine "
                f"blockDim in the schedule ({min_comb_warp} warps × {WAVE}-wide "
                f"wave = {max_peers} threads); the `tid < npes` barrier requires "
                f"world_size <= blockDim"
            )

    @property
    def nodes(self):
        """Physical nodes this op spans. 1 for every intranode config."""
        return self.world_size // self.gpu_per_node

    @property
    def is_internode(self):
        """True when some peer is not reachable over the flat LSA VA.

        This is the whole selector: there is no kernel_type enum on the v2 side.
        A config whose world fits in one node takes the intranode kernels even if
        gpu_per_node was passed explicitly.
        """
        return self.nodes > 1

    @property
    def is_scatter(self):
        return self.combine_mode == "scatter"

    @property
    def fp8_direct_cast(self):
        return self.quant_type == "fp8_direct_cast"

    @property
    def fp8_blockwise(self):
        return self.quant_type == "fp8_blockwise"

    @property
    def combine_scale_dim(self):
        """Per-token block count for fp8_blockwise combine (block_elems=128)."""
        return self.hidden_dim // 128 if self.fp8_blockwise else 0

    @property
    def wire_elem_size(self):
        """comb_inp transport element size: 1 byte for fp8 paths, else elem_size."""
        return (
            1
            if self.quant_type in ("fp8_direct_cast", "fp8_blockwise")
            else self.elem_size
        )

    @classmethod
    def tuned(cls, **kwargs):
        """Build a config with block/warp geometry pulled from tuning_configs
        (unless explicitly overridden in kwargs). Kept for back-compat and to
        force per-field tuning even when some geometry is overridden; the plain
        constructor is now also tuned-by-default (see _resolve_geometry)."""
        from .tuning_configs import lookup

        dt = kwargs.get("data_type", torch.bfloat16)
        dtype = (
            "fp4"
            if dt == torch.float4_e2m1fn_x2
            else ("fp8" if dt in _FP8_DTYPES else "bf16")
        )
        # The intranode tables say nothing about an internode config -- they are
        # keyed on a single node's geometry and carry no rdma_block_num -- and
        # pre-filling here would be indistinguishable from a caller pin, which
        # the internode backend honours over its own table. Leave it alone and
        # let _resolve_geometry / internode_tuning_configs do the work.
        gpu_per_node = kwargs.get("gpu_per_node") or kwargs["world_size"]
        if kwargs["world_size"] // gpu_per_node > 1:
            return cls(**kwargs)

        t = lookup(
            kwargs["world_size"],
            kwargs["hidden_dim"],
            kwargs["num_experts_per_token"],
            dtype=dtype,
        )
        for k, v in t.items():
            kwargs.setdefault(k, v)
        return cls(**kwargs)

    @property
    def dispatch_dtype(self):
        """Dispatch transport dtype (== data_type unless dispatch_data_type set)."""
        return (
            self.dispatch_data_type
            if self.dispatch_data_type is not None
            else self.data_type
        )

    @property
    def is_fp4(self):
        return self.dispatch_dtype == torch.float4_e2m1fn_x2

    @property
    def is_fp8(self):
        return self.dispatch_dtype in _FP8_DTYPES

    @property
    def elem_size(self):
        # fp4 is 0.5 B/elem; return a nominal 1 (kernels use the fp4 flag +
        # token_nbytes for actual sizing, never elem_size for fp4 token buffers).
        return 1 if self.is_fp4 else _DT[self.dispatch_dtype]

    @property
    def token_nbytes(self):
        """Per-token transport bytes (fp4 packs 2 e2m1/byte -> hidden/2)."""
        return self.hidden_dim // 2 if self.is_fp4 else self.hidden_dim * self.elem_size

    @property
    def combine_dtype(self):
        """Combine transport dtype (== data_type unless combine_data_type set)."""
        return (
            self.combine_data_type
            if self.combine_data_type is not None
            else self.data_type
        )

    @property
    def combine_elem_size(self):
        cdt = self.combine_dtype
        return 1 if cdt == torch.float4_e2m1fn_x2 else _DT[cdt]

    @property
    def combine_token_nbytes(self):
        cdt = self.combine_dtype
        return (
            self.hidden_dim // 2
            if cdt == torch.float4_e2m1fn_x2
            else self.hidden_dim * self.combine_elem_size
        )

    @property
    def is_asymmetric_dtype(self):
        return self.dispatch_dtype != self.combine_dtype

    @property
    def dispatch_dtype_str(self):
        """What dispatch transports, "fp4" / "fp8" / "bf16": hip_tuning_configs' key.

        HIP tunes dispatch and combine apart, and its combine half is picked by
        quant_type, so the combine dtype has no place in this key.
        """
        if self.is_fp4:
            return "fp4"
        if self.is_fp8:
            return "fp8"
        return "bf16"

    @property
    def dtype_str(self):
        """FlyDSL's tuning key (tuning_configs.lookup), which names both halves."""
        if self.is_fp4:
            # fp4 dispatch + non-fp4 combine (asymmetric) moves 2 B/elem on the
            # combine side, so it needs the bf16 combine geometry, not the
            # fp4-combine one -> its own "fp4_disp_bf16_comb" tuning key.
            if self.combine_dtype != torch.float4_e2m1fn_x2:
                return "fp4_disp_bf16_comb"
            return "fp4"
        if self.is_fp8:
            return "fp8"
        return "bf16"

    @property
    def max_recv(self):
        """Sentinel / sender-side atomic-add allocation bound (always ws*M)."""
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def effective_max_recv_per_rank(self):
        if self.max_total_recv_tokens <= 0:
            return self.max_num_inp_token_per_rank
        per = (self.max_total_recv_tokens + self.world_size - 1) // self.world_size
        return min(per, self.max_num_inp_token_per_rank)

    @property
    def effective_max_recv(self):
        """Recv-slot cap passed to the kernels as max_recv (mori MaxNumTokensToRecv)."""
        return self.world_size * self.effective_max_recv_per_rank


@dataclass
class KernelSet:
    """What a backend hands back: the kernels, plus how it wants to be driven.

    The flags exist so the shared op body can branch on *data* instead of the
    base calling overridable hooks for every small difference. Adding a backend
    with a new quirk adds a flag here, not a method to every subclass.
    """

    # (block_num, warp_num) -> callable. Keys must cover every spec the op's
    # schedule can select; the op clamps to what is present.
    dispatch: dict[tuple[int, int], Callable]
    combine: dict[tuple[int, int], Callable]
    # Replay-routing dispatch. None = this backend has no replay.
    dispatch_replay: dict[tuple[int, int], Callable] | None = None

    # The AsyncLL split. Each is the same (block_num, warp_num) -> callable shape
    # as `dispatch`, but launching only half the pass sequence, so the caller can
    # put its own work between the two. None = this backend runs send and recv in
    # one launch and has no seam to expose.
    dispatch_send: dict[tuple[int, int], Callable] | None = None
    dispatch_recv: dict[tuple[int, int], Callable] | None = None
    combine_send: dict[tuple[int, int], Callable] | None = None
    combine_recv: dict[tuple[int, int], Callable] | None = None

    # True  -> combine's kernel stages the caller's tokens into out_tok itself.
    # False -> the op must copy them in on the host first (one extra torch kernel).
    stages_in_kernel: bool = False

    # Advertised feature names, for callers that want to probe before asking.
    capabilities: frozenset[str] = frozenset()
    # Non-empty => this backend cannot serve the config it was built for. The op
    # raises with these at construction, so a caller never holds an object that
    # will fail later.
    unsupported: tuple[str, ...] = ()


class EpDispatchRoutingHandle:
    """Per-call routing (mori EpDispatchRoutingHandle parity).

    Lifetime: by default the handle ALIASES op-owned buffers -- its
    disp_dest_tok_id_map is the op's routing_dest_map and its total_recv_token_num
    is the op's total_recv -- so it is valid only until the next
    dispatch(return_routing=True) on the same op, which overwrites both. That keeps
    the serving path (dispatch -> experts -> combine) free of allocations. A caller
    that holds several handles at once (one per MoE layer, then the backward in
    reverse) passes dispatch(..., snapshot=True): the handle then owns its dest map
    and count, and combine / replay with it stay correct after later dispatches. On a
    gfx1250 LL128 op it also owns copies of its reverse map and forwarded ids (recv_maps),
    taken after the dispatch: that combine sends each received row back from them, not
    from the dest map.

    Either way only the routing is the handle's. What dispatch delivered into the
    arena (recv tokens, weights, indices, scales, and the reverse map below until
    its first access clones it) is overwritten by the next dispatch. Combine
    re-stages the tokens from its `input`, but the weight fold
    (combine(..., weights=...)) reads the forwarded weights straight from the
    arena, so the caller must make sure the arena holds this handle's: a weighted
    combine is valid with the handle of the latest dispatch, or after a replay
    dispatch(..., routing=handle), which rewrites them at the handle's slots.

    disp_dest_tok_id_map: forward (src_tok,k)->dest flat slot (v2 tok_map).
    disp_tok_id_to_src_tok_id_local: reverse recv-slot->src token (v2 tis).
    inter_node_*: empty placeholders on every path, internode included. v2's
    combine takes its routing from disp_dest_tok_id_map alone; the internode
    kernels keep their own send/dest maps in the arena and never hand one back,
    so there is nothing to put here. Kept for 5-tensor shape parity with v1 so
    downstream unpacking works.

    The reverse map (disp_tok_id_to_src_tok_id_local) is materialized LAZILY on
    first access. recv_to_src_token is written into this rank's arena by peers via
    P2P during dispatch and, per the mori contract, is only visible after the
    caller's post-dispatch comm.barrier(). Cloning it eagerly inside dispatch()
    (before that barrier) races those P2P writes and captures stale entries on
    high-CU parts (seen flaky on MI355X at high occupancy). Deferring the clone to
    first access lets it run after the barrier; it also skips the copy entirely for
    the common combine path, which never reads the reverse map.
    """

    def __init__(
        self,
        disp_dest_tok_id_map,
        inter_node_disp_dest_tok_id_map,
        inter_node_disp_send_map,
        total_recv_token_num,
        disp_tok_id_to_src_tok_id_local=None,
        cur_rank_num_token=0,
        *,
        reverse_src_view=None,
        reverse_encode=None,
        recv_maps=None,
    ):
        self.disp_dest_tok_id_map = disp_dest_tok_id_map
        self.recv_maps = recv_maps
        self.inter_node_disp_dest_tok_id_map = inter_node_disp_dest_tok_id_map
        self.inter_node_disp_send_map = inter_node_disp_send_map
        self.total_recv_token_num = total_recv_token_num
        self.cur_rank_num_token = cur_rank_num_token
        # Either an already-materialized reverse map (from_tensors round-trip) or
        # a live arena view to clone on first access (dispatch()).
        self._reverse_cache = disp_tok_id_to_src_tok_id_local
        self._reverse_src_view = reverse_src_view
        self._reverse_encode = reverse_encode

    @property
    def disp_tok_id_to_src_tok_id_local(self):
        if self._reverse_cache is None:
            # First access (post-barrier): clone off the arena so it survives the
            # next dispatch overwriting the region.
            src = self._reverse_src_view
            encode = self._reverse_encode
            self._reverse_cache = src.clone() if encode is None else encode(src)
        return self._reverse_cache

    def tensors(self):
        return (
            self.disp_dest_tok_id_map,
            self.inter_node_disp_dest_tok_id_map,
            self.inter_node_disp_send_map,
            self.total_recv_token_num,
            self.disp_tok_id_to_src_tok_id_local,
        )

    @classmethod
    def from_tensors(cls, tensors, cur_rank_num_token=0):
        return cls(*tensors, cur_rank_num_token=cur_rank_num_token)


class EpDispatchCombineOp:
    """Base + backend selector. Instantiating this picks a subclass."""

    _BACKENDS: dict[str, type] = {}
    backend_name: str | None = None

    # What the AsyncLL recv halves need from their send halves. Class-level so
    # every backend has them whether or not it implements the split, and so a
    # recv without a send raises rather than hitting a missing attribute.
    _async_pending_dispatch_tokens = None
    _async_pending_dispatch_input = None
    _async_pending_dispatch_map = None
    _async_pending_combine = None

    # -- registration / selection -----------------------------------------

    def __init_subclass__(cls, backend: str | None = None, **kw):
        super().__init_subclass__(**kw)
        if backend:
            cls.backend_name = backend
            EpDispatchCombineOp._BACKENDS[backend] = cls

    @staticmethod
    def _resolve_backend(name: str) -> type:
        if name not in EpDispatchCombineOp._BACKENDS:
            mod = _BACKEND_MODULES.get(name)
            if mod is None:
                raise ValueError(
                    f"unknown kernel backend {name!r}; "
                    f"known: {sorted(_BACKEND_MODULES)}"
                )
            try:
                importlib.import_module("." + mod, __package__)
            except ImportError as e:
                raise ImportError(
                    f"kernel backend {name!r} is not available: importing {mod!r} "
                    f"failed with {e}. Pick another backend with "
                    f"MORI_V2_KERNEL_BACKEND or cfg.kernel_backend."
                ) from e
            if name not in EpDispatchCombineOp._BACKENDS:
                raise ImportError(
                    f"module {mod!r} imported but registered no {name!r} backend"
                )
        return EpDispatchCombineOp._BACKENDS[name]

    def __new__(cls, cfg=None, comm=None, *a, **kw):
        # Only the base dispatches; instantiating a subclass directly is honoured.
        if cls is EpDispatchCombineOp:
            name = (
                getattr(cfg, "kernel_backend", None)
                or os.environ.get("MORI_V2_KERNEL_BACKEND")
                or _default_backend()
            )
            cls = EpDispatchCombineOp._resolve_backend(name)
        return super().__new__(cls)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Backends that actually import on this machine."""
        out = []
        for name in _BACKEND_MODULES:
            try:
                cls._resolve_backend(name)
                out.append(name)
            except (ImportError, ValueError):
                pass
        return out

    # -- hooks a subclass must supply --------------------------------------
    #
    # Exactly four, and no more: the arena layout, the kernels, and (optionally)
    # what the backend cannot do and which arena region a shared view reads.
    # Everything a caller touches -- dispatch(), combine(), the views, _pick,
    # close -- is implemented once, here.

    def _regions(self, cfg) -> list[tuple[str, int]]:
        """[(region_name, nbytes)] this backend needs carved out of the arena.

        The subclass maps region names to its own offset parameters itself; the
        base never passes offsets around. That is deliberate -- FlyDSL's
        `off_out_tok` means "disp_out" to make_dispatch and "out_tok" to
        make_combine, so any shared offset-passing path silently wires one of
        them to the wrong buffer.
        """
        raise NotImplementedError

    def _build_kernels(self, cfg, arena) -> KernelSet:
        """Compile/bind every variant and describe how they want to be driven.

        The callables in the returned KernelSet MUST accept these keyword
        arguments and nothing else -- the base calls them uniformly, so each
        backend adapts its own convention here rather than the base learning
        both:

            dispatch(input, indices, weights, scales, dest_map, num_tokens)
            combine (input, dest_map, total_recv, num_tokens)

        Everything else a kernel needs (the window handle, counters, barriers,
        rank, stream, output buffers) is backend-internal state the closure
        captures. Compilation happens HERE and only here: `_pick` must never be
        able to trigger one, or a capture would fork a compiler mid-graph.
        """
        raise NotImplementedError

    def _unsupported(self, cfg) -> tuple[str, ...]:
        """Reasons this backend cannot serve `cfg`. Empty = it can."""
        return ()

    def _region(self, name):
        """Arena region a shared view should read.

        Identity by default -- one layout, one set of names. A backend that
        serves more than one layout (the HIP one does: intranode and internode
        arenas hold the same things under different names) overrides this so the
        views stay a single contract instead of branching at every call site.
        """
        return name

    # -- shared: capability gate -------------------------------------------

    def _gate(self, kernels: KernelSet) -> None:
        if kernels.unsupported:
            raise ValueError(
                f"{self.backend_name} backend cannot serve this config:\n  "
                + "\n  ".join(kernels.unsupported)
            )

    @property
    def capabilities(self) -> frozenset[str]:
        return self._kernels.capabilities

    def scale_stride_bytes(self) -> int:
        """Bytes between consecutive out_scales rows, 0 when off.

        A backend may lay them down wider than the row it was handed (the HIP one
        pads to 128 B). On the base so a consumer never branches on the backend;
        the default is the row unchanged, which is right for one that does not pad.
        """
        return self._scale_row_bytes()

    def _scale_row_bytes(self) -> int:
        """The caller's row in bytes, dword-rounded. 0 when the transport is off."""
        n = self.cfg.scale_dim * self.cfg.scale_type_size
        return ((n + 3) // 4) * 4 if n else 0

    # -- shared: variant selection -----------------------------------------

    def _pick(self, num_tokens, snapshot=False):
        """((disp_block, disp_warp), (comb_block, comb_warp)) for a runtime token
        count via the per-token schedule; falls back to the single-shot specs
        otherwise. Clamped to variants that were actually built, so the hot path
        can never trigger a compile. `snapshot`: the call's handle owns its routing
        (dispatch(snapshot=True)), which a backend may pick other plans for."""
        schedule = self.cfg.schedule
        disp_spec = comb_spec = None
        if schedule:
            for bucket in schedule:
                max_tok = bucket[0]
                if max_tok is None or num_tokens <= max_tok:
                    disp_spec, comb_spec = (bucket[1], bucket[2]), (
                        bucket[3],
                        bucket[4],
                    )
                    break
            if disp_spec is None:
                last = schedule[-1]
                disp_spec, comb_spec = (last[1], last[2]), (last[3], last[4])
        else:
            disp_spec, comb_spec = self._dispatch_specs[0], self._combine_specs[0]
        if disp_spec not in self._kernels.dispatch:
            disp_spec = self._dispatch_specs[-1]
        if comb_spec not in self._kernels.combine:
            comb_spec = self._combine_specs[-1]
        return disp_spec, comb_spec

    @staticmethod
    def _specs_from(cfg):
        """The (block, warp) variants to build, de-duplicated from the schedule."""
        if cfg.schedule and not cfg.is_scatter:
            disp = sorted({(db, dw) for (_, db, dw, _, _) in cfg.schedule})
            comb = sorted({(cb, cw) for (_, _, _, cb, cw) in cfg.schedule})
        elif cfg.schedule:
            disp = sorted({(db, dw) for (_, db, dw, _, _) in cfg.schedule})
            comb = [(cfg.combine_block_num, cfg.combine_warp_num_per_block)]
        else:
            disp = [(cfg.dispatch_block_num, cfg.warp_num_per_block)]
            comb = [(cfg.combine_block_num, cfg.combine_warp_num_per_block)]
        return disp, comb

    # -- shared: lifecycle --------------------------------------------------

    def close(self):
        """Free this op's symmetric arena window. Call (or use as a context
        manager) when the op is discarded but its Communicator lives on."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._close_backend()
        self.arena.close()

    def _close_backend(self):
        """Release backend-owned handles BEFORE the arena goes away. The C++
        plans hold the cco window and their kernels dereference it, so the order
        is not cosmetic -- reversed, a late launch faults on freed device memory."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- shared: scratch the base's own methods use -------------------------

    @property
    def _empty_i32(self):
        """Inter-node placeholders in the routing handle. Owned here because the
        base's dispatch() is what puts them in the handle."""
        cache = getattr(self, "_empty_i32_cache", None)
        if cache is None:
            cache = self._empty_i32_cache = torch.empty(
                0, dtype=torch.int32, device=self.dev
            )
        return cache

    def _snapshot_recv_views(self, disp_spec):
        """What a snapshot handle must copy out of the arena because this backend's combine
        reads it back, as (recv_to_src, recv_idx) views, copied once the dispatch has written
        them; None when the combine routes from the dest map and the count alone."""
        return None

    def _reverse_source(self, disp_spec):
        """The arena view a dispatch through disp_spec leaves its reverse map in, and the
        function that re-encodes it as src_rank * max_num_inp_token_per_rank + src_tok, or
        None when it already is."""
        # A live arena view: the reverse map is cloned lazily on first access,
        # which must happen after the caller's post-dispatch barrier (see
        # EpDispatchRoutingHandle). Built once -- the pointer and the shape are
        # fixed for the op's lifetime, and rebuilding it per dispatch is pure
        # host cost on a path where the host already paces the GPU.
        reverse = getattr(self, "_reverse_view", None)
        if reverse is None:
            reverse = from_gpu_ptr(
                self.arena.local_ptr(self._region("recv_to_src_token")),
                (self._recv_cap,),
                torch.int32,
            )
            self._reverse_view = reverse
        return reverse, None

    # -- shared: the ops ---------------------------------------------------

    def dispatch(
        self,
        input,
        weights,
        scales,
        indices,
        *,
        routing=None,
        return_routing=False,
        snapshot=False,
    ):
        """mori-parity dispatch. input [n_tok,hidden], weights [n_tok,topk] f32,
        scales [n_tok,scale_dim] (or None), indices [n_tok,topk] i32.

        routing=: replay a prior handle (reuse the cached dest-slot layout, skip
        atomic routing). return_routing=: also return the handle. Mutually
        exclusive. Returns (out, out_weights, out_scales, out_indices,
        total_recv[, routing]); out == arena disp_out, safe to read without
        .clone() because combine stages into a separate out_tok buffer.

        snapshot=: with return_routing=True, give the handle its own dest map
        (the kernel writes it directly, no copy) and its own copy of total_recv --
        on an LL128 op also its own reverse map and forwarded ids, written by the
        kernel next to the arena's -- so it outlives later return_routing
        dispatches on this op. Without it the
        handle is valid only until the next one; see EpDispatchRoutingHandle.
        Intranode gather combine without StdMoE only: the internode and scatter
        combines also read routing state the op keeps in the arena, and StdMoE's
        convert reads the op's slot map and forwarded weights.

        out_scales is [max_recv, scale_dim_i32] on both backends but is not always
        CONTIGUOUS -- a backend may lay the rows down wider and return a strided
        view. Read by pointer, stride by scale_stride_bytes(), not by this shape.

        total_recv is a DEVICE tensor. Reading it on the host is a full sync that
        costs more than the kernel and makes the op uncapturable, so neither
        backend does it.
        """
        with self._asyncll_call("dispatch"):
            return self._dispatch_impl(
                input,
                weights,
                scales,
                indices,
                routing=routing,
                return_routing=return_routing,
                kernels=self._kernels.dispatch,
                snapshot=snapshot,
            )

    def _dispatch_impl(
        self,
        input,
        weights,
        scales,
        indices,
        *,
        routing,
        return_routing,
        kernels,
        snapshot=False,
    ):
        """The body of dispatch(), parameterised by which kernel table runs.

        dispatch() passes the whole-phase table; dispatch_send() passes the send
        half. Everything else -- the capacity check, the replay choice, the views
        returned -- is identical, which is why this is one function and not two
        that drift.
        """
        if routing is not None and return_routing:
            raise ValueError(
                "pass either routing= (replay) or return_routing=True, not both"
            )
        if snapshot and not return_routing:
            raise ValueError(
                "snapshot=True applies to the handle return_routing=True returns"
            )
        if snapshot and (
            self.cfg.is_internode or self.cfg.is_scatter or self.cfg.enable_std_moe
        ):
            raise ValueError(
                "snapshot=True is intranode gather without StdMoE only: the internode "
                "and scatter combines and StdMoE's convert also read state the op "
                "keeps in the arena"
            )
        # AsyncLL's combine reads dispReceiverIdxMap, which every dispatch's
        # RecvCopy rewrites, so a handle cannot outlive the next dispatch however
        # its dest map is held. The pattern snapshot exists for -- dispatch A,
        # dispatch B, then combine A -- is also exactly what AsyncLL's call-order
        # check refuses, because its signal slots pair each dispatch with the
        # combine that follows it.
        if snapshot and self.cfg.async_ll:
            raise ValueError(
                "snapshot=True does not apply to async_ll: AsyncLL pairs every "
                "dispatch with the next combine (its recv counts and slot map are "
                "per round), so a routing handle is only ever used by that combine"
            )
        n = input.shape[0]
        cap = self.cfg.max_num_inp_token_per_rank
        if n > cap:
            raise ValueError(f"{n} tokens exceeds max_num_inp_token_per_rank={cap}")
        disp_spec, _ = self._pick(n, snapshot)

        recv_views = None
        if routing is not None:
            table = self._kernels.dispatch_replay
            if table is None:
                raise NotImplementedError(
                    f"the {self.backend_name} backend has no routing replay"
                )
            kern, dest_map = table[disp_spec], routing.disp_dest_tok_id_map
        else:
            kern = kernels[disp_spec]
            if snapshot:
                # No fill: dispatch writes every (tok, k) entry below n, and
                # nothing reads past n.
                dest_map = torch.empty_like(self.routing_dest_map)
                recv_views = self._snapshot_recv_views(disp_spec)
            elif return_routing:
                dest_map = self.routing_dest_map
            else:
                dest_map = self.token_dest_map

        kern(
            input=input,
            indices=indices,
            weights=weights,
            scales=scales,
            dest_map=dest_map,
            num_tokens=n,
        )
        recv_maps = None if recv_views is None else tuple(v.clone() for v in recv_views)
        # What dispatch_recv() needs to launch the other half -- recorded only
        # when this WAS the send half. Recording it for a plain dispatch() was not
        # harmless bookkeeping: it kept the previous round's input activations
        # alive until the next call, for every kernel family. dispatch_recv()
        # drops the references again once it has launched.
        if kernels is not self._kernels.dispatch:
            self._async_pending_dispatch_tokens = n
            self._async_pending_dispatch_input = input
            self._async_pending_dispatch_map = dest_map

        base = (
            self.recv_tokens(),
            self.recv_weights(),
            self.recv_scales(),
            self.recv_indices(),
            self.total_recv,
        )
        if not return_routing:
            return base

        reverse, encode = self._reverse_source(disp_spec)
        own = recv_maps is not None and encode is None
        handle = EpDispatchRoutingHandle(
            disp_dest_tok_id_map=dest_map,
            inter_node_disp_dest_tok_id_map=self._empty_i32,
            inter_node_disp_send_map=self._empty_i32,
            # A copy rather than a buffer handed to the kernel: total_recv is
            # also read through the op (local_expert_count, StdMoE convert), so
            # the kernel keeps writing the op's.
            total_recv_token_num=(
                self.total_recv.clone() if snapshot else self.total_recv
            ),
            disp_tok_id_to_src_tok_id_local=recv_maps[0] if own else None,
            cur_rank_num_token=n,
            reverse_src_view=reverse if recv_maps is None else recv_maps[0],
            reverse_encode=encode,
            recv_maps=recv_maps,
        )
        return base + (handle,)

    def combine(self, input, weights=None, indices=None, *, routing):
        """mori-parity combine. input [<=max_recv,hidden] post-expert tokens.
        indices are accepted for API parity but unused; routing carries the mapping.

        `weights` is a REQUEST, not an input: passing it asks for the weight fold,
        whose values come from the forwarded out_wts, not from this argument, so
        they must be this routing's (see EpDispatchRoutingHandle). The fold is for
        training backward (mori a668e25e) and costs more than the whole token
        payload on gfx1250, so inference passes None -- as v1 does, gating on the
        same null pointer.

        Returns (out [ct,hidden], out_weights [ct,topk] or None)."""
        self._asyncll_check_weight_fold(weights)
        with self._asyncll_call("combine"):
            return self._combine_impl(
                input, weights, indices, routing=routing, kernels=self._kernels.combine
            )

    def _combine_impl(self, input, weights, indices, *, routing, kernels):
        """The body of combine(), parameterised by which kernel table runs. See
        _dispatch_impl."""
        ct = routing.cur_rank_num_token
        _, comb_spec = self._pick(
            ct, routing.total_recv_token_num is not self.total_recv
        )

        if not self._kernels.stages_in_kernel and not self.cfg.enable_std_moe:
            # StdMoE has already written the weighted-reduced tokens into out_tok;
            # copying `input` over them would clobber that result.
            out_tok_ptr = self.arena.local_ptr(self._region("out_tok"))
            if input.data_ptr() != out_tok_ptr:
                dst = self.combine_in_view().view(-1)[: input.numel()]
                dst.copy_(input.reshape(-1))
        # No combine_barrier.zero_(): the cross-device barrier clears it itself.
        # No combine_out.zero_(): the kernel reduce-then-stores every token in
        # [0, ct), so the prior contents of the returned slice never leak.

        want_weights = weights is not None
        maps = getattr(routing, "recv_maps", None)
        own = {} if maps is None else dict(recv_to_src=maps[0], recv_idx=maps[1])
        kernels[comb_spec](
            input=input,
            dest_map=routing.disp_dest_tok_id_map,
            total_recv=routing.total_recv_token_num,
            num_tokens=ct,
            want_weights=want_weights,
            **own,
        )
        # Same rule as dispatch: only the send half leaves anything for later.
        if kernels is not self._kernels.combine:
            self._async_pending_combine = (
                input,
                routing.disp_dest_tok_id_map,
                ct,
                want_weights,
            )

        cdt = self.cfg.combine_dtype
        hidden = self.cfg.hidden_dim
        topk = self.cfg.num_experts_per_token
        cols = (
            hidden // 2 if cdt == torch.float4_e2m1fn_x2 else hidden
        )  # fp4 packs 2/elem
        out = self.combine_out[: ct * cols].view(cdt).view(ct, cols)
        # None rather than a stale buffer when the fold was not asked for: the
        # kernel leaves combine_out_weights untouched, so returning it would hand
        # back the previous call's values.
        outw = (
            self.combine_out_weights[: ct * topk].view(ct, topk)
            if want_weights
            else None
        )
        return out, outw

    # -- the AsyncLL seam ------------------------------------------------
    #
    # dispatch() and combine() above run a whole phase in one call. On the
    # AsyncLL family each splits in two, and the point of the split is what the
    # caller does in between:
    #
    #     out, ..., routing = op.dispatch_send(x, w, s, idx, return_routing=True)
    #     ...                       # anything; the SDMA copies drain under it
    #     op.dispatch_recv()        # now the tokens are in `out`
    #
    # The send half returns the same tuple dispatch() does, but the output buffers
    # are NOT populated until the matching recv call has run. Reading them in the
    # seam reads the previous round.

    # The AsyncLL protocol's call order, as a state machine. What each call is
    # allowed to see, and what it leaves behind:
    #
    #   idle --dispatch_send--> sent --dispatch_recv--> dispatched
    #   dispatched --combine_send--> combining --combine_recv--> idle
    #
    # with dispatch() and combine() taking both steps of their half at once.
    #
    # This is not API tidiness; the protocol's signal slots carry the coupling
    # and an unpaired call leaves state the NEXT call misreads:
    #
    #   * dispatch twice, no combine between: the recv counts are cleared only by
    #     combine's send leg, so the second RecvTransfer finds the previous
    #     round's non-zero count and its "0 means not arrived" wait passes at
    #     once -- it then reads a payload that may still be in flight. Observed
    #     on 8xMI308X and 8xMI355X, both transports, as wrong received counts
    #     (560 against 548) and wrong combine output, in 1-2 of 6 attempts.
    #   * dispatch_send twice: the slot counter keeps climbing past
    #     MaxNumTokensToSendPerRank unchecked and writes into the next PE's slice.
    #   * combine twice: the count is already 0, so the biased read underflows.
    #     The kernels now skip instead of faulting, but the round is still wrong.
    #   * cross-node combine twice: the epoch is an add of 1 per combine while
    #     crossDeviceBarrierFlag moves once per dispatch, so one unpaired call
    #     shifts them permanently and the wait never matches again -- a hang that
    #     only reset() clears.
    #
    # Hence an error at the call rather than a diagnosis three kernels later. It
    # is host-side bookkeeping on a path that already launches nine kernels, and
    # it is only armed for async_ll: no other family couples its calls this way.
    _ASYNCLL_ORDER = {
        "dispatch": ("idle", "dispatched"),
        "dispatch_send": ("idle", "sent"),
        "dispatch_recv": ("sent", "dispatched"),
        "combine": ("dispatched", "idle"),
        "combine_send": ("dispatched", "combining"),
        "combine_recv": ("combining", "idle"),
    }
    _asyncll_state = "idle"

    def _asyncll_check_weight_fold(self, weights):
        """Refuse an AsyncLL combine weight fold at the CALL, not three layers in.

        The backend raises this too, but from inside the kernel wrapper -- after
        _combine_impl has already staged `input` into the out_tok region, and
        after the call-order check below has advanced the state. Both are wrong
        places to find out: a caller that catches the error is left with a
        half-written staging buffer and a state machine that has moved on.
        """
        if weights is not None and getattr(self.cfg, "async_ll", False):
            raise NotImplementedError(
                "AsyncLL combine does not produce output weights: its combine "
                "payload carries no per-token metadata (v1's AsyncLL is the "
                "same). Call combine() with weights=None, or use a non-AsyncLL "
                "kernel family if you need the fold."
            )

    @contextlib.contextmanager
    def _asyncll_call(self, call):
        """Refuse `call` if it is out of order; advance only once it has run.

        Advancing at the top -- before the body -- meant a call that then failed
        its own argument checks (too many tokens, snapshot=True on async_ll,
        routing= together with return_routing=) left the state moved on for a
        launch that never happened, and the caller's next, correct call was
        refused as out of order. So the check is up front and the advance is on
        the way out: an exception from the body leaves the state where it was.
        No-op off the AsyncLL family.
        """
        if not getattr(self.cfg, "async_ll", False):
            yield
            return
        want, nxt = self._ASYNCLL_ORDER[call]
        if self._asyncll_state != want:
            raise RuntimeError(
                f"{call}() is not valid here: AsyncLL is {self._asyncll_state!r} "
                f"and this call needs {want!r}. The order is dispatch_send -> "
                "dispatch_recv -> combine_send -> combine_recv (dispatch() and "
                "combine() take both steps of their half). An unpaired call "
                "leaves the recv counts and the combine epoch out of step, which "
                "shows up as wrong data or a hang in a LATER round, not here. "
                "Call reset() if you need to start over."
            )
        yield
        self._asyncll_state = nxt

    def _async_half(self, which):
        table = getattr(self._kernels, which)
        if table is None:
            raise NotImplementedError(
                f"the {self.backend_name} backend has no {which}: it runs send and "
                "recv in one launch. Build the op with async_ll=True (hip backend) "
                "to get the split."
            )
        return table

    def dispatch_send(
        self, input, weights, scales, indices, *, routing=None, return_routing=False
    ):
        """Enqueue the send half of a dispatch and return immediately.

        Same arguments and same return as :meth:`dispatch`. The returned views are
        only valid after :meth:`dispatch_recv`.
        """
        with self._asyncll_call("dispatch_send"):
            return self._dispatch_impl(
                input,
                weights,
                scales,
                indices,
                routing=routing,
                return_routing=return_routing,
                kernels=self._async_half("dispatch_send"),
            )

    def dispatch_recv(self):
        """Run the recv half of the dispatch the last :meth:`dispatch_send` began."""
        with self._asyncll_call("dispatch_recv"):
            n = self._async_pending_dispatch_tokens
            if n is None:
                raise RuntimeError("dispatch_recv without a preceding dispatch_send")
            self._async_pending_dispatch_tokens = None
            spec, _ = self._pick(n)
            self._async_half("dispatch_recv")[spec](
                input=self._async_pending_dispatch_input,
                dest_map=self._async_pending_dispatch_map,
                num_tokens=n,
            )
            # Launched, so nothing reads these again; holding them would keep the
            # caller's input alive until the next round. combine_recv already drops
            # its whole tuple the same way.
            self._async_pending_dispatch_input = None
            self._async_pending_dispatch_map = None

    def combine_send(self, input, weights=None, indices=None, *, routing):
        """Enqueue the send half of a combine and return immediately."""
        self._asyncll_check_weight_fold(weights)
        with self._asyncll_call("combine_send"):
            return self._combine_impl(
                input,
                weights,
                indices,
                routing=routing,
                kernels=self._async_half("combine_send"),
            )

    def combine_recv(self):
        """Run the recv half of the combine the last :meth:`combine_send` began."""
        with self._asyncll_call("combine_recv"):
            state = self._async_pending_combine
            if state is None:
                raise RuntimeError("combine_recv without a preceding combine_send")
            self._async_pending_combine = None
            inp, dest_map, ct, want_weights = state
            _, spec = self._pick(ct)
            self._async_half("combine_recv")[spec](
                input=inp,
                dest_map=dest_map,
                total_recv=None,
                num_tokens=ct,
                want_weights=want_weights,
            )

    def reset(self):
        """Zero the arena staging + per-rank counters (mori LaunchReset). The
        kernels self-reset their counters already; this forces a clean slate.

        Collective, like every other call here: cross_device_flag is the local
        half of the cross-device barrier and arena.zero() has just wiped the peer
        half, so one rank resetting alone would leave the two disagreeing."""
        # This is the one way out of a half-finished exchange, and it is what the
        # call-order error tells a caller to use: the counts and the epoch it
        # zeroes below are exactly the state that order was protecting.
        self._asyncll_state = "idle"
        self.arena.zero()
        self.token_dest_map.fill_(-1)
        self.routing_dest_map.fill_(self._null_flat)
        self.dest_pe_counter.zero_()
        self.dispatch_barrier.zero_()
        self.combine_barrier.zero_()
        getattr(self, "_recv_ctr", self.total_recv).zero_()
        # The two backends seed this differently -- intranode starts the epoch at
        # 1, internode at 0 (see the two constructors in hip_backend) -- so a
        # single literal here silently desynchronises one of them from its peers.
        self.cross_device_flag.fill_(0 if self.cfg.is_internode else 1)
        # The docstring above says the kernels self-reset their counters. That is
        # true of dest_pe_counter and the two grid barriers, and NOT of the
        # internode chunk-slot allocator: only EpCombineAll clears
        # blockFlagCounter, so a reset taken between a dispatch and its combine
        # would leave it advanced and the next dispatch would address slots past
        # its node's slice of the send buffer.
        for name in ("block_flag_counter", "inter_blocks_barrier"):
            counter = getattr(self, name, None)
            if counter is not None:
                counter.zero_()
        # A peer's next call writes into the state zeroed above, so, as when the op
        # is built, every rank finishes its zeroes before any rank goes on.
        torch.cuda.synchronize()
        comm = getattr(self, "comm", None)
        if comm is not None:
            comm.barrier()

    def __repr__(self):
        return (
            f"{type(self).__name__}(backend={self.backend_name}, "
            f"rank={self.cfg.rank}/{self.cfg.world_size}, "
            f"hidden={self.cfg.hidden_dim}, topk={self.cfg.num_experts_per_token})"
        )
