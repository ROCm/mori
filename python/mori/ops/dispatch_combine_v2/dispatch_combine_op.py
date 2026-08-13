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
from dataclasses import dataclass
import os

import torch

import flydsl.expr as fx
from mori.tensor_utils import from_gpu_ptr

from .intranode_kernels import (
    xdb_flag_slots,
    make_dispatch,
    make_dispatch_token_centric,
    make_combine,
    make_combine_scatter,
    make_convert_dispatch_output,
    make_convert_combine_input,
    make_local_expert_count,
)

_QUANT_TYPES = ("none", "fp8_direct_cast", "fp8_blockwise")

_DT = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fnuz: 1,
    torch.float8_e4m3fn: 1,
}
_FP8_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


def _align_up(x, a):
    return (x + a - 1) // a * a


class SymmArena:
    """One cco symmetric window carved into named, aligned sub-regions. A kernel
    reaches peer pe's copy of region R via cco.Window(handle).lsa_ptr(pe, off_R)."""

    _ALIGN = 256

    def __init__(self, comm, regions):
        self._comm = comm
        self._offsets = {}
        self._sizes = {}
        off = 0
        for name, nbytes in regions:
            off = _align_up(off, self._ALIGN)
            self._offsets[name] = off
            self._sizes[name] = nbytes
            off += nbytes
        self._total = max(_align_up(off, self._ALIGN), self._ALIGN)
        self._mem = comm.alloc_mem(self._total)
        self._win = comm.register_window(self._mem.ptr, self._total)

    @property
    def handle(self):
        return self._win.handle

    @property
    def total_bytes(self):
        return self._total

    def offset(self, name):
        return self._offsets[name]

    def local_ptr(self, name):
        return self._win.local_ptr + self._offsets[name]

    def zero(self, name=None):
        """Zero the whole window, or just region `name` if given. Wraps the raw
        pointer as a zero-copy int8 torch view (borrowed via
        __cuda_array_interface__ — no ownership taken) and memsets it."""
        if name is None:
            ptr, nbytes = self._win.local_ptr, self._total
        else:
            ptr, nbytes = self.local_ptr(name), self._sizes[name]
        from_gpu_ptr(ptr, (nbytes,), torch.int8).zero_()

    def close(self):
        """Free the symmetric window (deregister before freeing the backing mem)."""
        self._win.close()
        self._mem.close()


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
    # Experimental BF16-input -> FP8-wire blockwise quantization in dispatch.
    dispatch_fp8_blockwise: bool = False
    # Geometry: None => auto (pull the tuned schedule for this device/shape/dtype
    # from tuning_configs in __post_init__). Pin any of these to opt out and use a
    # fixed geometry instead. Combine wants few warps (K-deep per-lane MLP already
    # saturates), kept separate from dispatch's warp count.
    dispatch_block_num: int = None
    combine_block_num: int = None
    warp_num_per_block: int = None
    combine_warp_num_per_block: int = None
    # Optional per-token plan: tuple of (max_tok_inclusive | None, disp_block,
    # disp_warp, comb_block, comb_warp) buckets. When set, the op precompiles the
    # distinct (block, warp) variants and picks one at runtime from
    # cur_rank_num_token. None => auto (from tuning_configs) or single-shot fallback.
    schedule: tuple = None
    geometry_mode: str = "hybrid"  # table | analytical | hybrid
    enable_std_moe: bool = False
    max_total_recv_tokens: int = 0  # mori maxTotalRecvTokens; 0 = worst-case ws*M
    # Dispatch stall A/B switches. Prefetch route payload before the slot
    # allocation atomic; defer the local destination counter atomic until after
    # route metadata/scales are published (but before the token copy).
    prefetch_route_payload: bool = False
    defer_dest_ctr_atomic: bool = False
    # Normal-dispatch A/B: use the destination's slot-allocation counter as the
    # final receive count, eliminating the redundant per-source counter atomics.
    # Replay does not allocate slots and therefore keeps the legacy count path.
    use_tok_off_total_recv: bool = False
    # Remote LSA store cache-policy A/B switches (SC0|SC1 on FlyDSL 0.2.x).
    uncached_token_store: bool = False
    uncached_metadata_store: bool = False
    # Optional tuned runtime thresholds. The op lazily compiles cache-policy
    # variants and selects them by actual token count; 0 disables auto selection.
    uncached_token_store_max_tokens: int = 0
    uncached_metadata_store_max_tokens: int = 0
    # Replay-only A/B: decode destination PE/slot from the cached token map and
    # skip route loads plus duplicate detection.
    replay_fast_path: bool = False
    # Workgroup-per-token load-once/store-many dispatch A/B.
    token_centric_dispatch: bool = False
    token_centric_rotate_peer_order: bool = False
    token_centric_min_tokens: int = 0
    token_centric_max_tokens: int = 0
    token_centric_schedule: tuple = None
    # Peer-order A/B switches. Dispatch permutes each token's slot-to-warp
    # assignment; gather combine phase-shifts expert reads across warps.
    rotate_dispatch_slot_order: bool = False
    rotate_combine_peer_order: bool = False

    def __post_init__(self):
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
        if self.combine_mode not in ("gather", "scatter"):
            raise ValueError(
                f"combine_mode must be gather|scatter, got {self.combine_mode!r}"
            )
        if self.dispatch_fp8_blockwise:
            if (
                self.data_type != torch.bfloat16
                or self.dispatch_data_type is not None
                or self.combine_data_type is not None
                or self.combine_mode != "gather"
                or self.quant_type != "none"
                or self.enable_std_moe
            ):
                raise ValueError(
                    "dispatch_fp8_blockwise requires symmetric BF16 input/output, "
                    "combine_mode='gather', quant_type='none', and StdMoE disabled"
                )
            self.token_centric_dispatch = True
        if self.quant_type != "none":
            self.combine_mode = "scatter"
        if self.rotate_combine_peer_order and self.combine_mode != "gather":
            raise ValueError("rotate_combine_peer_order requires combine_mode='gather'")
        # The dispatch grid barrier iterates peers as `range(lane, npes, 64)` and
        # resets the barrier inside the loop, which is only correct when each lane
        # runs it at most once (npes <= wavefront). Intranode is single-node so
        # this always holds, but make the assumption explicit.
        if self.world_size > 64:
            raise ValueError(
                f"intranode op supports world_size <= 64, got {self.world_size}"
            )
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
            if (
                self.combine_mode != "gather"
                or self.quant_type != "none"
                or self.enable_std_moe
            ):
                raise ValueError(
                    "combine_data_type (asymmetric dtype) requires combine_mode=gather, "
                    "quant_type=none, enable_std_moe=False"
                )
            # fp4 DISPATCH + bf16 COMBINE (the SGLang/aiter fp4-asym path) is
            # supported: dispatch is a plain byte-mover (hidden/2 B) with e8m0
            # scales forwarded, combine stays bf16. fp4 on the combine side is
            # not implemented for the asymmetric path.
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
        self._resolve_geometry()

    def _resolve_geometry(self):
        """Fill block/warp/schedule. Tuned-by-default: when the caller pinned
        neither a schedule nor any block/warp, pull the tuned geometry for this
        device/shape/dtype from tuning_configs.lookup (so the plain constructor is
        tuned automatically — EpDispatchCombineConfig.tuned() is now just an
        explicit alias). If any field is pinned, honor it and fill the rest with
        the single-shot fallback (no schedule)."""
        pinned = self.schedule is not None or any(
            g is not None
            for g in (
                self.dispatch_block_num,
                self.combine_block_num,
                self.warp_num_per_block,
                self.combine_warp_num_per_block,
            )
        )
        if not pinned:
            from .tuning_configs import (
                has_tuned_entry,
                lookup,
                lookup_analytical,
            )

            mode = os.environ.get("MORI_EPV2_GEOMETRY", self.geometry_mode)
            if mode not in ("table", "analytical", "hybrid"):
                raise ValueError(
                    f"MORI_EPV2_GEOMETRY must be table|analytical|hybrid, got {mode!r}"
                )
            use_analytical = mode == "analytical" or (
                mode == "hybrid"
                and not has_tuned_entry(
                    self.world_size,
                    self.hidden_dim,
                    self.num_experts_per_token,
                    dtype=self.dtype_str,
                )
            )
            resolver = lookup_analytical if use_analytical else lookup
            t = resolver(
                self.world_size,
                self.hidden_dim,
                self.num_experts_per_token,
                dtype=self.dtype_str,
            )
            if use_analytical:
                # Isolate geometry in A/B: retain any orthogonal tuned kernel
                # policies (tok_off/cache/token-centric) for known shapes.
                table_cfg = lookup(
                    self.world_size,
                    self.hidden_dim,
                    self.num_experts_per_token,
                    dtype=self.dtype_str,
                )
                geometry_keys = {
                    "dispatch_block_num",
                    "combine_block_num",
                    "warp_num_per_block",
                    "combine_warp_num_per_block",
                    "schedule",
                }
                t.update(
                    {
                        k: v
                        for k, v in table_cfg.items()
                        if k not in geometry_keys
                    }
                )
            self.geometry_mode = mode
            self.dispatch_block_num = t["dispatch_block_num"]
            self.combine_block_num = t["combine_block_num"]
            self.warp_num_per_block = t["warp_num_per_block"]
            self.combine_warp_num_per_block = t["combine_warp_num_per_block"]
            self.schedule = t["schedule"]
            self.use_tok_off_total_recv = self.use_tok_off_total_recv or bool(
                t.get("use_tok_off_total_recv", False)
            )
            self.replay_fast_path = self.replay_fast_path or bool(
                t.get("replay_fast_path", False)
            )
            self.uncached_token_store_max_tokens = max(
                self.uncached_token_store_max_tokens,
                int(t.get("uncached_token_store_max_tokens", 0)),
            )
            self.uncached_metadata_store_max_tokens = max(
                self.uncached_metadata_store_max_tokens,
                int(t.get("uncached_metadata_store_max_tokens", 0)),
            )
            self.token_centric_min_tokens = max(
                self.token_centric_min_tokens,
                int(t.get("token_centric_min_tokens", 0)),
            )
            self.token_centric_max_tokens = max(
                self.token_centric_max_tokens,
                int(t.get("token_centric_max_tokens", 0)),
            )
            if self.token_centric_schedule is None:
                self.token_centric_schedule = t.get("token_centric_schedule")
            self.token_centric_rotate_peer_order = (
                self.token_centric_rotate_peer_order
                or bool(t.get("token_centric_rotate_peer_order", False))
            )
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
        if self.dispatch_fp8_blockwise:
            return self.hidden_dim
        return self.hidden_dim // 2 if self.is_fp4 else self.hidden_dim * self.elem_size

    @property
    def dispatch_wire_dtype(self):
        if self.dispatch_fp8_blockwise:
            from .tuning_configs import _topology

            return (
                torch.float8_e4m3fn
                if _topology()[1] in (90500, 120500)
                else torch.float8_e4m3fnuz
            )
        return self.dispatch_dtype

    @property
    def dispatch_quant_scale_dim(self):
        return self.hidden_dim // 128 if self.dispatch_fp8_blockwise else 0

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
    def dtype_str(self):
        """Token/dispatch dtype key for tuning_configs.lookup (fp4/fp8/default)."""
        if self.dispatch_fp8_blockwise:
            return "bf16_disp_fp8bw"
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


class EpDispatchRoutingHandle:
    """Per-call routing snapshot (mori EpDispatchRoutingHandle parity).

    disp_dest_tok_id_map: forward (src_tok,k)->dest flat slot (v2 tok_map).
    disp_tok_id_to_src_tok_id_local: reverse recv-slot->src token (v2 tis).
    inter_node_*: empty placeholders (v2 is intranode-only; kept for 5-tensor
    shape parity so downstream unpacking works).

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
    ):
        self.disp_dest_tok_id_map = disp_dest_tok_id_map
        self.inter_node_disp_dest_tok_id_map = inter_node_disp_dest_tok_id_map
        self.inter_node_disp_send_map = inter_node_disp_send_map
        self.total_recv_token_num = total_recv_token_num
        self.cur_rank_num_token = cur_rank_num_token
        # Either an already-materialized reverse map (from_tensors round-trip) or
        # a live arena view to clone on first access (dispatch()).
        self._reverse_cache = disp_tok_id_to_src_tok_id_local
        self._reverse_src_view = reverse_src_view

    @property
    def disp_tok_id_to_src_tok_id_local(self):
        if self._reverse_cache is None:
            # First access (post-barrier): clone off the arena so it survives the
            # next dispatch overwriting the region.
            self._reverse_cache = self._reverse_src_view.clone()
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
        if cfg.dispatch_fp8_blockwise:
            regions.append(
                (
                    "disp_quant_scales",
                    recv_cap * cfg.dispatch_quant_scale_dim * 4,
                )
            )
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
        self._empty_i32 = torch.empty(
            0, dtype=torch.int32, device=device
        )  # inter-node placeholders
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
        schedule = cfg.schedule
        if schedule:
            dispatch_specs = sorted({(db, dw) for (_, db, dw, _, _) in schedule})
            combine_specs = sorted({(cb, cw) for (_, _, _, cb, cw) in schedule})
        else:
            dispatch_specs = [(cfg.dispatch_block_num, cfg.warp_num_per_block)]
            combine_specs = [(cfg.combine_block_num, cfg.combine_warp_num_per_block)]
        if cfg.is_scatter:
            combine_specs = [(cfg.combine_block_num, cfg.combine_warp_num_per_block)]
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
            prefetch_route_payload=cfg.prefetch_route_payload,
            defer_dest_ctr_atomic=cfg.defer_dest_ctr_atomic,
            rotate_dispatch_slot_order=cfg.rotate_dispatch_slot_order,
            use_tok_off_total_recv=cfg.use_tok_off_total_recv,
            uncached_token_store=cfg.uncached_token_store,
            uncached_metadata_store=cfg.uncached_metadata_store,
            replay_fast_path=cfg.replay_fast_path,
            token_centric_rotate_peer_order=cfg.token_centric_rotate_peer_order,
            dispatch_fp8_blockwise=cfg.dispatch_fp8_blockwise,
            off_disp_quant_scales=(
                arena.offset("disp_quant_scales")
                if cfg.dispatch_fp8_blockwise
                else 0
            ),
            dispatch_quant_scale_dim=cfg.dispatch_quant_scale_dim,
        )
        # (block, warp) -> compiled dispatch / combine kernel.
        self._dispatch_variants = {
            (b, w): make_dispatch(
                block_num=b, warp_num_per_block=w, **self._dispatch_kwargs
            )
            for (b, w) in dispatch_specs
        }
        # Cache-policy variants are compiled lazily because only small-token
        # buckets use uncached peer stores.
        self._dispatch_policy_variants = {}
        self._dispatch_replay_variants = {}
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
                    rotate_combine_peer_order=cfg.rotate_combine_peer_order,
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
        self._closed = False

    def close(self):
        """Free this op's symmetric arena window. Call (or use as a context
        manager) when the op is discarded but its Communicator lives on —
        otherwise the arena stays in comm._resources until the comm is destroyed."""
        if self._closed:
            return
        self._closed = True
        self.arena.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def recv_tokens(self):
        """Arena disp_out [max_recv, hidden] (dispatch dest / expert-GEMM input).
        Separate from out_tok, so combine's copy-in never overwrites it — the
        expert can read this in place without a defensive .clone().
        fp4 packs 2 e2m1 per float4_e2m1fn_x2 element -> last dim is hidden/2."""
        cols = self.cfg.hidden_dim // 2 if self.cfg.is_fp4 else self.cfg.hidden_dim
        return from_gpu_ptr(
            self.arena.local_ptr("disp_out"),
            (self._recv_cap, cols),
            self.cfg.dispatch_wire_dtype,
        )

    def recv_quant_scales(self):
        """FP32 blockwise dispatch scales [max_recv, hidden/128], or None."""
        if not self.cfg.dispatch_fp8_blockwise:
            return None
        return from_gpu_ptr(
            self.arena.local_ptr("disp_quant_scales"),
            (self._recv_cap, self.cfg.dispatch_quant_scale_dim),
            torch.float32,
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

    def expert_output_buffer(self, total_recv=None):
        """Gather-combine destination for direct expert GEMM output.

        Writing expert results into this view and passing the same view to
        combine() elides the otherwise-required D2D staging copy.
        """
        if self.cfg.combine_mode != "gather":
            raise ValueError("expert_output_buffer is gather-mode only")
        view = self.combine_in_view()
        return view if total_recv is None else view[:total_recv]

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

    def _use_token_centric(self, num_tokens):
        if self.cfg.token_centric_dispatch:
            return True
        lo = self.cfg.token_centric_min_tokens
        hi = self.cfg.token_centric_max_tokens
        return lo > 0 and num_tokens >= lo and (hi <= 0 or num_tokens <= hi)

    def _token_centric_spec(self, num_tokens, fallback):
        schedule = self.cfg.token_centric_schedule
        if not schedule:
            return fallback
        for max_tok, block, warp in schedule:
            if max_tok is None or num_tokens <= max_tok:
                return block, warp
        return schedule[-1][1], schedule[-1][2]

    def _pick(self, num_tokens):
        """((disp_block, disp_warp), (comb_block, comb_warp)) for a runtime token
        count via the per-token schedule; falls back to the single-shot specs
        otherwise. Returned specs are clamped to the precompiled variants."""
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
        use_token_centric = self._use_token_centric(num_tokens)
        if use_token_centric:
            disp_spec = self._token_centric_spec(num_tokens, disp_spec)
        elif disp_spec not in self._dispatch_variants:
            disp_spec = self._dispatch_specs[-1]
        if comb_spec not in self._combine_variants:
            comb_spec = self._combine_specs[-1]
        return disp_spec, comb_spec

    def _dispatch_cache_policy(self, num_tokens):
        token_uncached = self.cfg.uncached_token_store or (
            self.cfg.uncached_token_store_max_tokens > 0
            and num_tokens <= self.cfg.uncached_token_store_max_tokens
        )
        metadata_uncached = self.cfg.uncached_metadata_store or (
            self.cfg.uncached_metadata_store_max_tokens > 0
            and num_tokens <= self.cfg.uncached_metadata_store_max_tokens
        )
        return bool(token_uncached), bool(metadata_uncached)

    def _get_dispatch_variant(self, spec, num_tokens, *, replay=False):
        """Select/compile the cache-policy specialization for this token bucket."""
        token_uncached, metadata_uncached = self._dispatch_cache_policy(num_tokens)
        token_centric = self._use_token_centric(num_tokens)
        if (
            not replay
            and not token_centric
            and token_uncached == self.cfg.uncached_token_store
            and metadata_uncached == self.cfg.uncached_metadata_store
        ):
            return self._dispatch_variants[spec]

        variants = (
            self._dispatch_replay_variants
            if replay
            else self._dispatch_policy_variants
        )
        key = (spec, token_uncached, metadata_uncached, token_centric)
        kern = variants.get(key)
        if kern is None:
            kwargs = dict(self._dispatch_kwargs)
            kwargs.update(
                uncached_token_store=token_uncached,
                uncached_metadata_store=metadata_uncached,
            )
            factory = (
                make_dispatch_token_centric if token_centric else make_dispatch
            )
            kern = variants[key] = factory(
                replay=replay,
                block_num=spec[0],
                warp_num_per_block=spec[1],
                **kwargs,
            )
        return kern

    def dispatch(
        self, input, weights, scales, indices, *, routing=None, return_routing=False
    ):
        """mori-parity dispatch. input [n_tok,hidden], weights [n_tok,topk] f32,
        scales [n_tok,scale_dim] (or None), indices [n_tok,topk] i32.

        routing=: replay a prior handle (reuse cached dest-slot layout, skip
        atomic routing). return_routing=: also return the handle. Mutually
        exclusive. Returns (out, out_weights, out_scales, out_indices,
        total_recv[, routing]); out == arena disp_out (safe to read without
        .clone() — combine stages into a separate out_tok buffer).
        """
        if routing is not None and return_routing:
            raise ValueError(
                "pass either routing= (replay) or return_routing=True, not both"
            )
        num_input_tokens = input.shape[0]
        disp_spec, _ = self._pick(num_input_tokens)
        # total_recv is self-reset inside the dispatch kernel (warp 0, Phase 2).
        scale_ptr = (
            scales.data_ptr() if (scales is not None and self._enable_scales) else 0
        )
        stream = fx.Stream(torch.cuda.current_stream())
        weight_ptr = weights.data_ptr() if weights is not None else 0
        if routing is not None:
            kern = self._get_dispatch_variant(
                disp_spec, num_input_tokens, replay=True
            )
            dest_map_ptr = routing.disp_dest_tok_id_map.data_ptr()
            kern(
                self.arena.handle,
                input.data_ptr(),
                indices.data_ptr(),
                weight_ptr,
                dest_map_ptr,
                self.dest_pe_counter.data_ptr(),
                self.dispatch_barrier.data_ptr(),
                self.total_recv.data_ptr(),
                scale_ptr,
                self.cfg.rank,
                num_input_tokens,
                stream,
            )
        else:
            dest_map = (
                torch.full_like(self.token_dest_map, -1)
                if return_routing
                else self.token_dest_map
            )
            self._get_dispatch_variant(disp_spec, num_input_tokens)(
                self.arena.handle,
                input.data_ptr(),
                indices.data_ptr(),
                weight_ptr,
                dest_map.data_ptr(),
                self.dest_pe_counter.data_ptr(),
                self.dispatch_barrier.data_ptr(),
                self.total_recv.data_ptr(),
                scale_ptr,
                self.cfg.rank,
                num_input_tokens,
                stream,
            )

        out = self.recv_tokens()
        out_weights = self.recv_weights()
        out_scales = self.recv_scales()
        out_indices = self.recv_indices()
        base = (out, out_weights, out_scales, out_indices, self.total_recv)
        if not return_routing:
            return base

        recv_to_src_view = from_gpu_ptr(
            self.arena.local_ptr("recv_to_src_token"), (self._recv_cap,), torch.int32
        )
        routing = EpDispatchRoutingHandle(
            disp_dest_tok_id_map=dest_map,
            inter_node_disp_dest_tok_id_map=self._empty_i32,
            inter_node_disp_send_map=self._empty_i32,
            total_recv_token_num=self.total_recv,
            cur_rank_num_token=num_input_tokens,
            reverse_src_view=recv_to_src_view,
        )
        return base + (routing,)

    def combine(self, input, weights=None, indices=None, *, routing):
        """mori-parity combine. input [<=max_recv,hidden] post-expert tokens
        (copied into arena out_tok if not already there). weights/indices are
        accepted for API parity but unused (weights come from forwarded out_wts,
        routing carries the mapping). Returns (out [ct,hidden], out_weights [ct,topk]).
        """
        out_tok_ptr = self.arena.local_ptr("out_tok")
        # StdMoE: convert_combine_input() has already staged the weighted-reduced
        # tokens into out_tok, so `input` is unused here — copying it in would
        # clobber that result. (Non-StdMoE: `input` holds the post-expert tokens
        # to combine; since the disp_out/out_tok split it no longer aliases out_tok,
        # so the copy is required.)
        if not self.cfg.enable_std_moe and input.data_ptr() != out_tok_ptr:
            # copy in the combine-dtype layout (not recv_tokens()'s dispatch view)
            dst = self.combine_in_view().view(-1)[: input.numel()]
            dst.copy_(input.reshape(-1))
        stream = fx.Stream(torch.cuda.current_stream())
        _, comb_spec = self._pick(routing.cur_rank_num_token)
        self._combine_variants[comb_spec](
            self.arena.handle,
            routing.disp_dest_tok_id_map.data_ptr(),
            self.combine_barrier.data_ptr(),
            self.cross_device_flag.data_ptr(),
            routing.total_recv_token_num.data_ptr(),
            self.combine_out.data_ptr(),
            self.combine_out_weights.data_ptr(),
            self.cfg.rank,
            routing.cur_rank_num_token,
            stream,
        )
        count = routing.cur_rank_num_token
        hidden_dim = self.cfg.hidden_dim
        topk = self.cfg.num_experts_per_token
        cdt = self.cfg.combine_dtype
        cols = (
            hidden_dim // 2 if cdt == torch.float4_e2m1fn_x2 else hidden_dim
        )  # fp4 out is hidden/2 float4 elems
        out = self.combine_out[: count * cols].view(cdt).view(count, cols)
        return out, self.combine_out_weights[: count * topk].view(count, topk)

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

    def reset(self):
        """Zero arena staging + per-rank counters/barriers (mori LaunchReset).
        Kernels self-reset counters already; this forces a clean slate."""
        self.arena.zero()
        self.token_dest_map.fill_(-1)
        self.dest_pe_counter.zero_()
        self.dispatch_barrier.zero_()
        self.total_recv.zero_()
        self.combine_barrier.zero_()
        self.cross_device_flag.fill_(1)
