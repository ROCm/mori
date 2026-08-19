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

A subclass supplies exactly two things::

    _regions(cfg)              -> [(name, nbytes)]   arena layout it needs
    _build_kernels(cfg, arena) -> KernelSet          bound, ready-to-launch kernels

Everything else -- arena, scratch buffers, variant selection, lifecycle -- is here.
Behavioural differences between the backends are *data* on the KernelSet, not
methods to override, after aiter's ``MOEMetadata``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable

import torch

from mori.tensor_utils import from_gpu_ptr

# Where each backend lives. Imported lazily, on selection only.
_BACKEND_MODULES = {"flydsl": "flydsl_backend", "hip": "hip_backend"}

DEFAULT_BACKEND = "flydsl"


_QUANT_TYPES = ("none", "fp8_direct_cast", "fp8_blockwise")

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
    # Which kernel backend serves this op: "flydsl" (default, full feature set)
    # or "hip" (HIP/JIT, bf16/fp32 gather only). None = MORI_V2_KERNEL_BACKEND,
    # else the default. Only consulted when constructing the BASE class; naming a
    # subclass directly wins.
    kernel_backend: str = None

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
        if self.quant_type != "none":
            self.combine_mode = "scatter"
        # The dispatch grid barrier resets inside a `range(lane, npes, 64)` loop,
        # correct only while each lane runs it once (npes <= wavefront).
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
        if not pinned:
            backend = (
                self.kernel_backend
                or os.environ.get("MORI_V2_KERNEL_BACKEND")
                or DEFAULT_BACKEND
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
                    dtype=self.dtype_str,
                    experts_per_rank=self.num_experts_per_rank,
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
    def dtype_str(self):
        """Token/dispatch dtype key for tuning_configs.lookup (fp4/fp8/default)."""
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

    # True  -> combine's kernel stages the caller's tokens into out_tok itself.
    # False -> the op must copy them in on the host first (one extra torch kernel).
    stages_in_kernel: bool = False
    # True  -> the kernels reset their own counters/barriers; the op must not
    #          memset them, and on symmetric regions it MUST NOT (a peer may have
    #          already delivered a signal, and wiping it hangs both ranks).
    self_resets_counters: bool = True

    # Advertised feature names, for callers that want to probe before asking.
    capabilities: frozenset[str] = frozenset()
    # Non-empty => this backend cannot serve the config it was built for. The op
    # raises with these at construction, so a caller never holds an object that
    # will fail later.
    unsupported: tuple[str, ...] = ()


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
    """Base + backend selector. Instantiating this picks a subclass."""

    _BACKENDS: dict[str, type] = {}
    backend_name: str | None = None

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
                or DEFAULT_BACKEND
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
    # Exactly three, and no more: the arena layout, the kernels, and (optionally)
    # what the backend cannot do. Everything a caller touches -- dispatch(),
    # combine(), the views, _pick, close -- is implemented once, here.

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

    # -- shared: variant selection -----------------------------------------

    def _pick(self, num_tokens):
        """((disp_block, disp_warp), (comb_block, comb_warp)) for a runtime token
        count via the per-token schedule; falls back to the single-shot specs
        otherwise. Clamped to variants that were actually built, so the hot path
        can never trigger a compile."""
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

    # -- shared: the ops ---------------------------------------------------

    def dispatch(
        self, input, weights, scales, indices, *, routing=None, return_routing=False
    ):
        """mori-parity dispatch. input [n_tok,hidden], weights [n_tok,topk] f32,
        scales [n_tok,scale_dim] (or None), indices [n_tok,topk] i32.

        routing=: replay a prior handle (reuse the cached dest-slot layout, skip
        atomic routing). return_routing=: also return the handle. Mutually
        exclusive. Returns (out, out_weights, out_scales, out_indices,
        total_recv[, routing]); out == arena disp_out, safe to read without
        .clone() because combine stages into a separate out_tok buffer.

        total_recv is a DEVICE tensor. Reading it on the host is a full sync that
        costs more than the kernel and makes the op uncapturable, so neither
        backend does it.
        """
        if routing is not None and return_routing:
            raise ValueError(
                "pass either routing= (replay) or return_routing=True, not both"
            )
        n = input.shape[0]
        cap = self.cfg.max_num_inp_token_per_rank
        if n > cap:
            raise ValueError(f"{n} tokens exceeds max_num_inp_token_per_rank={cap}")
        disp_spec, _ = self._pick(n)

        if not self._kernels.self_resets_counters:
            # Only this one: the kernels self-clear dest_pe_counter and the grid
            # barrier, but total_recv is cleared in COMBINE, not in dispatch.
            self.total_recv.zero_()

        if routing is not None:
            table = self._kernels.dispatch_replay
            if table is None:
                raise NotImplementedError(
                    f"the {self.backend_name} backend has no routing replay"
                )
            kern, dest_map = table[disp_spec], routing.disp_dest_tok_id_map
        else:
            kern = self._kernels.dispatch[disp_spec]
            if return_routing:
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

        base = (
            self.recv_tokens(),
            self.recv_weights(),
            self.recv_scales(),
            self.recv_indices(),
            self.total_recv,
        )
        if not return_routing:
            return base

        # A live arena view: the reverse map is cloned lazily on first access,
        # which must happen after the caller's post-dispatch barrier (see
        # EpDispatchRoutingHandle).
        reverse = from_gpu_ptr(
            self.arena.local_ptr("recv_to_src_token"), (self._recv_cap,), torch.int32
        )
        handle = EpDispatchRoutingHandle(
            disp_dest_tok_id_map=dest_map,
            inter_node_disp_dest_tok_id_map=self._empty_i32,
            inter_node_disp_send_map=self._empty_i32,
            total_recv_token_num=self.total_recv,
            cur_rank_num_token=n,
            reverse_src_view=reverse,
        )
        return base + (handle,)

    def combine(self, input, weights=None, indices=None, *, routing):
        """mori-parity combine. input [<=max_recv,hidden] post-expert tokens.
        indices are accepted for API parity but unused; routing carries the mapping.

        `weights` is a REQUEST, not an input: passing it asks for the weight fold,
        whose values come from the forwarded out_wts, not from this argument. The
        fold is for training backward (mori a668e25e) and costs more than the whole
        token payload on gfx1250, so inference passes None -- as v1 does, gating on
        the same null pointer.

        Returns (out [ct,hidden], out_weights [ct,topk] or None)."""
        ct = routing.cur_rank_num_token
        _, comb_spec = self._pick(ct)

        if not self._kernels.stages_in_kernel and not self.cfg.enable_std_moe:
            # StdMoE has already written the weighted-reduced tokens into out_tok;
            # copying `input` over them would clobber that result.
            out_tok_ptr = self.arena.local_ptr("out_tok")
            if input.data_ptr() != out_tok_ptr:
                dst = self.combine_in_view().view(-1)[: input.numel()]
                dst.copy_(input.reshape(-1))
        # No combine_barrier.zero_(): the cross-device barrier clears it itself.
        # No combine_out.zero_(): the kernel reduce-then-stores every token in
        # [0, ct), so the prior contents of the returned slice never leak.

        want_weights = weights is not None
        self._kernels.combine[comb_spec](
            input=input,
            dest_map=routing.disp_dest_tok_id_map,
            total_recv=routing.total_recv_token_num,
            num_tokens=ct,
            want_weights=want_weights,
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

    def reset(self):
        """Zero the arena staging + per-rank counters (mori LaunchReset). The
        kernels self-reset their counters already; this forces a clean slate.

        Collective, like every other call here: cross_device_flag is the local
        half of the cross-device barrier and arena.zero() has just wiped the peer
        half, so one rank resetting alone would leave the two disagreeing."""
        self.arena.zero()
        self.token_dest_map.fill_(-1)
        self.dest_pe_counter.zero_()
        self.dispatch_barrier.zero_()
        self.combine_barrier.zero_()
        self.total_recv.zero_()
        self.cross_device_flag.fill_(1)

    def __repr__(self):
        return (
            f"{type(self).__name__}(backend={self.backend_name}, "
            f"rank={self.cfg.rank}/{self.cfg.world_size}, "
            f"hidden={self.cfg.hidden_dim}, topk={self.cfg.num_experts_per_token})"
        )
