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
can be served -- this backend implements both combine transports (gather and
scatter) with a bf16/fp32 combine and a bf16/fp32/fp8/fp4 dispatch, and rejects
everything else at CONSTRUCTION rather than at launch.

Imports ``ep_plans`` (the C++/JIT plans) but never flydsl, so it works where
FlyDSL is not installed.
"""

from __future__ import annotations

import os
import re

import torch

from mori.tensor_utils import from_gpu_ptr

from . import ep_plans as cb
from .dispatch_combine_op import EpDispatchCombineOp, KernelSet
from .symm_arena import SymmArena

# C++ offset-argument stem -> arena region name. The C++ side names the offsets
# after its own EpArgs fields (offTokOff -> "tokOff"); the region names match the
# FlyDSL op's, and so do the sizes EXCEPT out_scales -- this backend pads the row,
# FlyDSL does not. A plan binds offsets, never sizes, so an arena sized for the
# wrong one is overrun silently: size it from scale_stride_bytes().
_REGIONS = {
    "tokOff": "tok_off",
    "recvNum": "recv_num",
    "recvToSrc": "recv_to_src_token",
    "outIdx": "out_idx",
    "outWts": "out_wts",
    "dispOut": "disp_out",
    "outTok": "out_tok",
    "xdb": "cross_device_barrier",
    "outScales": "out_scales",  # only laid out when scales are on; binds to 0 otherwise
    "combPush": "comb_push",  # only laid out for scatter combine; binds to 0 otherwise
    "xdbBlk": "comb_block_barrier",
}

# Must match EpCombinePushSlotAlign in include/mori/ops/dispatch_combine_v2/ep_cfg.hpp.
_PUSH_SLOT_ALIGN = 128

# op-layer quant_type -> EpCfg::combineQuant. Only the modes this backend serves
# appear here; _unsupported rejects the rest before a plan is ever built, so a
# KeyError here would mean the gate and this table disagree.
_QUANT_TO_INT = {"none": 0, "fp8_direct_cast": 1, "fp4": 3}


def push_wire_nbytes(cfg) -> int:
    """Per-token bytes a scatter combine puts ON THE WIRE: EpCombineWireBytes.

    Equal to the token's own size unless quant compresses it, in which case one
    byte per element. This is the slot's payload width, so it sizes the region;
    the LDS tile is NOT this (see push_lds_need).
    """
    if cfg.quant_type == "fp4":
        # e2m1 payload (2/byte) + one e8m0 scale byte per 8-element (pk8) group, the
        # scale row padded to 16 B. Must equal EpCombineWireBytes in ep_cfg.hpp.
        group = 8
        raw_scale = (cfg.hidden_dim + group - 1) // group
        scale = (raw_scale + 15) // 16 * 16
        return cfg.hidden_dim // 2 + scale
    if cfg.quant_type != "none":
        return cfg.hidden_dim
    return cfg.combine_token_nbytes


def push_slot_bytes(wire_nbytes: int, topk: int) -> int:
    """A scatter-combine slot's stride: EpCombinePushSlotBytes in ep_cfg.hpp.

    The weights ride behind the payload (the reduce walks slots by source rank,
    which is not how out_wts is indexed), and the whole thing is padded so every
    slot starts on a 128 B row -- a TDM store's destination has to.

    Mirrored from the C++ because the arena is sized before any kernel exists.
    """
    packed = wire_nbytes + topk * 4
    return (packed + _PUSH_SLOT_ALIGN - 1) // _PUSH_SLOT_ALIGN * _PUSH_SLOT_ALIGN


# Must match Ep1250xLdsBytes / EpCombinePushRRTile in ep_cfg.hpp.
_COMB_LDS_BUDGET = 327680
_PUSH_RR_TILE = 512


def push_lds_need(
    warp_per_block: int,
    topk: int,
    world_size: int,
    combine_token_nbytes: int,
    use_weights: bool = True,
    wire_nbytes: int = 0,
    qpipe: int = 4,
) -> int:
    """LDS the scatter send phase needs: EpCombinePushLdsNeed in ep_cfg.hpp.

    One whole token per warp, after the pointer arrays and the round-robin
    bookkeeping. gfx1250 combine gets the whole budget as dynamic shared memory,
    so the bookkeeping is in there too rather than in a static __shared__.

    wire_nbytes identifies the second tile the quantized send phase converts
    into; that tile also carries the appended weights and slot padding. It is 0
    without quant. Pass it whenever quant is on even though the
    kernel can be built to convert in place instead: this is the gate deciding
    whether the TDM send path exists, so it must not come out under what the
    kernel might ask for.
    """
    ptr = ((1 + int(bool(use_weights))) * warp_per_block * topk * 8 + 127) // 128 * 128
    # Two _PUSH_RR_TILE arrays: slot ids, plus the recvToSrc entry the grouping
    # pass read for each. Counted unconditionally for the same reason wire_nbytes
    # is -- this sizes the arena before the device build exists.
    rr = (2 * _PUSH_RR_TILE + 4 * world_size) * 4
    wire_tile = push_slot_bytes(wire_nbytes, topk) if wire_nbytes else 0
    # The narrow pipeline alternates two chunks. Non-quantized and unchunked
    # sends still stage one whole token.
    load_tile = (
        2 * combine_token_nbytes // qpipe
        if wire_nbytes and qpipe >= 2
        else combine_token_nbytes
    )
    return (ptr + rr + 127) // 128 * 128 + warp_per_block * (
        load_tile + wire_tile
    )

# Only what EpDType enumerates -- fp16 is absent because plan_api.DTYPES has no code
# for it, and advertising it here would alias onto another one. Dispatch only copies,
# so any fixed-width type transports; combine sums, so it needs an arithmetic one.
_DISPATCH_DTYPES = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fn: 1,
    torch.float8_e4m3fnuz: 1,
    torch.float4_e2m1fn_x2: 1,  # nominal: cfg.token_nbytes is what sizes buffers
}
_COMBINE_DTYPES = {torch.bfloat16: 2, torch.float32: 4}
# Must match EpScaleAlign in include/mori/ops/dispatch_combine_v2/ep_cfg.hpp.
_SCALE_ALIGN = 128


def scale_stride_bytes(scale_bytes: int) -> int:
    """What a DESTINATION scale row is laid down at: EpScaleStride in ep_cfg.hpp.

    Anything addressing the region itself strides by this; recv_scales() hides it.
    Mirrored from the C++ because sizing happens before any kernel exists.
    """
    if scale_bytes <= 0:
        return 0
    return (scale_bytes + _SCALE_ALIGN - 1) // _SCALE_ALIGN * _SCALE_ALIGN


# Must match EpXdbFlagSlots in include/mori/ops/dispatch_combine_v2/ep_cfg.hpp.
_XDB_FLAG_SLOTS = 256


class EpDispatchCombineOpHip(EpDispatchCombineOp, backend="hip"):
    """C++/JIT-kernel EP op: gather or scatter combine, no replay.

    Dispatch transports bf16/fp32/fp8/fp4, combine reduces in bf16/fp32, and the
    two need not match -- an fp8-in/bf16-out op is just two plans with different
    dtypes. On the dispatch leg mori does no quantizing: fp8/fp4 payloads arrive
    already packed and move as opaque bytes.

    The combine leg is where mori does quantize, and only there: quant_type=
    fp8_direct_cast casts each element into the peer's slot on write and widens
    it back in the reduce, which halves the bytes on the wire. It needs the
    scatter transport (the sender has to hold a whole token) and gfx125x (the
    cast happens inside the TDM tile).
    """

    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self._recv_cap = cfg.effective_max_recv
        self._closed = False
        # gfx125x routes to the TDM kernel, which needs a superset arena (plan A).
        _arch = getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "") or ""
        self._is1250 = _arch.split(":")[0].startswith("gfx125")

        # Gate FIRST: rejecting a config after taking a symmetric window would
        # leak it (the arena is registered with the communicator), and the whole
        # point of the gate is that an unsupported config never gets that far.
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
        self._null_flat = cfg.world_size * cfg.effective_max_recv
        self.routing_dest_map = torch.full_like(self.token_dest_map, self._null_flat)
        self.dest_pe_counter = torch.zeros(cfg.world_size, **i32)
        self.total_recv = torch.zeros(1, **i32)
        self.dispatch_barrier = torch.zeros(1, dtype=torch.uint32, device=dev)
        self.combine_barrier = torch.zeros(1, dtype=torch.uint32, device=dev)
        # Monotone epoch. Starts at 1 so the zeroed barrier slots cannot alias
        # the first launch's flag value. The gfx1250 combine barrier gives every
        # block a private slot (EpXdbFlagSlots in ep_cfg.hpp), so it is the only
        # writer of its own epoch; the portable path only ever uses slot 0.
        self.cross_device_flag = torch.ones(
            _XDB_FLAG_SLOTS if self._is1250 else 1, dtype=torch.int64, device=dev
        )
        self.combine_out = torch.zeros(
            max_tok * cfg.hidden_dim, dtype=cfg.combine_dtype, device=dev
        )
        self.combine_out_weights = torch.zeros(
            max_tok * topk, dtype=torch.float32, device=dev
        )
        # gfx1250 combine's intra-grid barrier fan-out (local scratch, 16 lines/block);
        # size to the largest combine block_num any variant launches. Portable path
        # never touches it -> left None (binds as 0).
        self.combine_barrier_fan = None
        if self._is1250:
            max_comb_blocks = max(b for b, _ in self._combine_specs)
            if max_comb_blocks > _XDB_FLAG_SLOTS:
                raise ValueError(
                    f"combine block_num {max_comb_blocks} exceeds the {_XDB_FLAG_SLOTS} "
                    "per-block xdb epoch slots the entry barrier owns"
                )
            self.combine_barrier_fan = torch.zeros(max_comb_blocks * 16, **i32)

    # -- backend hooks -----------------------------------------------------

    @staticmethod
    def _scale_i32(cfg) -> int:
        """Dwords in the SOURCE scale row. _unsupported rejects a row that is not
        already dword-sized, so the rounding here never actually rounds."""
        return (cfg.scale_dim * cfg.scale_type_size + 3) // 4

    def scale_stride_bytes(self) -> int:
        """Padded to 128 B here; the base returns the row unchanged. Use the
        module-level function when there is no op yet (sizing an arena)."""
        return scale_stride_bytes(self._scale_row_bytes())

    @classmethod
    def _scale_stride_i32(cls, cfg) -> int:
        """Dwords per DESTINATION scale row, 0 when the transport is off."""
        return scale_stride_bytes(cls._scale_i32(cfg) * 4) // 4

    def _regions(self, cfg):
        # token_nbytes / combine_token_nbytes rather than elem*hidden: they are the
        # only forms that are right for fp4, where 2 values share a byte.
        cap = cfg.effective_max_recv
        topk = cfg.num_experts_per_token
        regions = [
            ("tok_off", 4),
            ("recv_num", cfg.world_size * 4),
            ("recv_to_src_token", cap * 4),
            ("out_idx", cap * topk * 4),
            ("out_wts", cap * topk * 4),
            ("disp_out", cap * cfg.token_nbytes),
            ("out_tok", cap * cfg.combine_token_nbytes),
            ("cross_device_barrier", cfg.world_size * 8),
        ]
        if self._scale_i32(cfg):
            # Sized by the DESTINATION stride, which is the caller's row padded to
            # 128 B: the kernel lays the rows down at that pitch, so an arena sized
            # for the unpadded row would be overrun by the last tokens.
            # Padded rows only land aligned if the region does; that is another
            # file's constant, and lowering it reads as a perf regression.
            assert SymmArena._ALIGN % _SCALE_ALIGN == 0, (
                f"SymmArena._ALIGN={SymmArena._ALIGN} does not keep scale regions "
                f"{_SCALE_ALIGN} B-aligned; the padding in EpScaleStride buys nothing"
            )
            regions.append(("out_scales", cap * self._scale_stride_i32(cfg) * 4))
        if cfg.is_scatter:
            # Scatter's landing zone, one slot per (source rank, destination-local
            # token). Sized by world_size * max_tok_per_rank, NOT by the recv cap:
            # a slot is addressed by the DESTINATION's token id, so the recv cap
            # (which can be smaller when max_total_recv_tokens is set) would leave
            # the high slots off the end of the region.
            assert SymmArena._ALIGN % _PUSH_SLOT_ALIGN == 0, (
                f"SymmArena._ALIGN={SymmArena._ALIGN} does not keep push slots "
                f"{_PUSH_SLOT_ALIGN} B-aligned; the TDM store in the send phase needs that"
            )
            slot = push_slot_bytes(
                push_wire_nbytes(cfg), cfg.num_experts_per_token
            )
            regions.append(
                (
                    "comb_push",
                    cfg.world_size * cfg.max_num_inp_token_per_rank * slot,
                )
            )
        # One arrival slot per (source rank, source block) for the gfx1250 combine
        # entry barrier's MORI_COMB_BARDIRECT path. Appended LAST on purpose: every
        # region before it keeps the offset it had, so a run with the gate off is
        # measuring the same arena layout it measured before this existed.
        # Strided by the compile-time slot count rather than by the largest
        # block_num in the tuning table, because that constant is what the kernel
        # indexes with -- sizing it from anything else makes the two disagree the
        # first time the table changes.
        regions.append(("comb_block_barrier", cfg.world_size * _XDB_FLAG_SLOTS * 8))
        return regions

    def _unsupported(self, cfg) -> tuple[str, ...]:
        """Everything this backend cannot do, checked before anything is built."""
        bad = []
        if cfg.dispatch_dtype not in _DISPATCH_DTYPES:
            bad.append(
                f"dispatch dtype {cfg.dispatch_dtype} (have bf16, fp32, fp8, fp4)"
            )
        if cfg.combine_dtype not in _COMBINE_DTYPES:
            bad.append(f"combine dtype {cfg.combine_dtype} (have bf16, fp32)")
        # fp8_direct_cast is served, on gfx125x only and on the scatter transport
        # only. Both restrictions are the kernel's: the narrow happens inside the
        # TDM tile between the load and the store, so it needs the gfx125x body
        # AND a transport whose sender holds a whole token. (The op layer already
        # forces scatter for any quant_type, so the second one is belt and
        # braces rather than a case a caller can reach.)
        if cfg.quant_type in ("fp8_direct_cast", "fp4"):
            if not self._is1250:
                bad.append(f"quant_type={cfg.quant_type!r} off gfx125x")
            elif not cfg.is_scatter:
                bad.append(f"quant_type={cfg.quant_type!r} with combine_mode=gather")
            elif cfg.combine_dtype is not torch.bfloat16:
                bad.append(
                    f"quant_type={cfg.quant_type!r} with combine dtype "
                    f"{cfg.combine_dtype} (the cast is written for bf16)"
                )
            elif cfg.quant_type == "fp8_direct_cast" and cfg.hidden_dim % 4:
                bad.append(
                    f"quant_type={cfg.quant_type!r} at hidden_dim={cfg.hidden_dim}; "
                    "the in-place narrow moves 4 elements at a time"
                )
            elif cfg.quant_type == "fp4" and cfg.hidden_dim % 8:
                bad.append(
                    f"quant_type={cfg.quant_type!r} at hidden_dim={cfg.hidden_dim}; "
                    "fp4 packs 8 e2m1 elements (one pk8 scale group) at a time"
                )
        elif cfg.quant_type != "none":
            bad.append(f"quant_type={cfg.quant_type!r}")
        if cfg.enable_std_moe:
            bad.append("enable_std_moe")
        # The kernel walks the source scale rows with the PADDED dword stride, so
        # a caller row that is not itself a whole number of dwords would be read
        # at the wrong pitch. _scale_i32 rounds up, which hides that from the Cfg
        # validator -- check the caller's own width here instead.
        raw_scale_bytes = cfg.scale_dim * cfg.scale_type_size
        if raw_scale_bytes % 4:
            bad.append(
                f"per-token scale row of {raw_scale_bytes} B "
                f"(scale_dim={cfg.scale_dim} x {cfg.scale_type_size}); "
                "the row must be a whole number of dwords"
            )
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
            max_tok_per_rank=cfg.max_num_inp_token_per_rank,
            num_expert_per_rank=cfg.num_experts_per_rank,
            num_expert_per_token=cfg.num_experts_per_token,
            max_recv=cfg.effective_max_recv,
            use_weights=True,
            arena=arena,
            region_names=_REGIONS,
        )
        # The two legs are separate Plans, so each carries its own dtype and its own
        # element count -- which is what makes an asymmetric config (fp8/fp4 in,
        # bf16 out) just two ordinary kernels. hiddenDim is "elements of THIS leg's
        # dtype", so fp4 halves it: 2 e2m1 live in one transported byte.
        disp_cfg = dict(
            hidden_dim=cfg.hidden_dim // 2 if cfg.is_fp4 else cfg.hidden_dim,
            dtype=cfg.dispatch_dtype,
            # Dword-padded: the kernel copies the row as dwords, and EpCfgIsValid
            # rejects a Cfg whose row is not a whole number of them.
            scale_bytes=self._scale_i32(cfg) * 4,
        )
        comb_cfg = dict(
            hidden_dim=cfg.hidden_dim,
            dtype=cfg.combine_dtype,
            # Scatter: the rank holding the expert result writes it into the
            # owner's slot. Dispatch has no such choice, so this is combine-only.
            combine_push=cfg.is_scatter,
            combine_quant=_QUANT_TO_INT[cfg.quant_type],
        )
        # One plan per (block, warp) the schedule can select. Compilation happens
        # here and only here, so _pick never touches the compiler.
        dispatch, combine = {}, {}
        self._plans = []
        for b, w in self._dispatch_specs:
            plan = cb.EpDispatchPlan(
                **common, **disp_cfg, block_num=b, warp_per_block=w
            )
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            dispatch[(b, w)] = self._wrap_dispatch(plan)
        for b, w in self._combine_specs:
            if cfg.is_scatter and self._is1250:
                # The send phase stages one token per warp through TDM. Without
                # the tile it still works, by lane copy -- which is the slow path
                # TDM is here to avoid, and nothing downstream would report the
                # difference. Refuse instead, and name the way out.
                flags = os.environ.get("MORI_JIT_EXTRA_FLAGS", "")
                qpipe_match = re.search(
                    r"(?:^|\s)-DMORI_COMB_QPIPE=(\d+)(?:\s|$)", flags
                )
                qpipe = int(qpipe_match.group(1)) if qpipe_match else 4
                if qpipe < 1 or (
                    cfg.quant_type != "none" and cfg.hidden_dim % qpipe != 0
                ):
                    raise ValueError(
                        f"MORI_COMB_QPIPE={qpipe} must be positive and divide "
                        f"hidden_dim={cfg.hidden_dim}"
                    )
                need = push_lds_need(
                    w,
                    cfg.num_experts_per_token,
                    cfg.world_size,
                    cfg.combine_token_nbytes,
                    wire_nbytes=(
                        push_wire_nbytes(cfg) if cfg.quant_type != "none" else 0
                    ),
                    qpipe=qpipe,
                )
                if need > _COMB_LDS_BUDGET:
                    raise ValueError(
                        f"scatter combine at {w} warps/block needs {need} B of LDS for its "
                        f"send tiles but only {_COMB_LDS_BUDGET} B exist "
                        f"(hidden_dim={cfg.hidden_dim}, {cfg.combine_token_nbytes} B/token); "
                        f"pin a smaller warp_per_block, at most "
                        f"{(_COMB_LDS_BUDGET - (need - w * cfg.combine_token_nbytes)) // cfg.combine_token_nbytes}"
                    )
            plan = cb.EpCombinePlan(**common, **comb_cfg, block_num=b, warp_per_block=w)
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
            capabilities=frozenset(
                {"gather", "scatter", "scales", "fp8_direct_cast"}
            ),
        )

    def _close_backend(self):
        for plan in getattr(self, "_plans", ()):
            plan.close()

    # -- views (same contract as the FlyDSL backend) -----------------------

    def recv_tokens(self):
        # fp4 packs 2 e2m1 per element of the torch dtype -> last dim is hidden/2.
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
        """The forwarded scale rows, or None when the transport is off -- the same
        answer FlyDSL gives, and the same (recv_cap, dwords) int32 view, so a caller
        cannot tell the backends apart.

        Strided, not packed: the rows sit scale_stride_bytes() apart. Anything
        reading the region by pointer needs that pitch, not this shape.
        """
        n_i32 = self._scale_i32(self.cfg)
        if not n_i32:
            return None
        stride_i32 = self._scale_stride_i32(self.cfg)
        rows = from_gpu_ptr(
            self.arena.local_ptr("out_scales"),
            (self._recv_cap, stride_i32),
            torch.int32,
        )
        return rows[:, :n_i32]

    def local_expert_count(self):
        raise NotImplementedError(
            "local_expert_count is flydsl-only; use backend='flydsl'"
        )

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
                scales_buf=scales,
                disp_dest_tok_id_map=dest_map,
                dest_pe_token_counter=self.dest_pe_counter,
                total_recv_token_num=self.total_recv,
                grid_barrier=self.dispatch_barrier,
                num_tokens=num_tokens,
            )

        return run

    def _wrap_combine(self, plan):
        def run(*, input, dest_map, total_recv, num_tokens, want_weights=False):
            plan.launch(
                stream=torch.cuda.current_stream().cuda_stream,
                inp_token_buf=input,
                out_token_buf=self.combine_out,
                # Null == "skip the weight fold" (the kernel's only gate on it).
                out_weights_buf=self.combine_out_weights if want_weights else None,
                disp_dest_tok_id_map=dest_map,
                total_recv_token_num=total_recv,
                grid_barrier=self.combine_barrier,
                xdb_flag=self.cross_device_flag,
                combine_barrier_fan=self.combine_barrier_fan,  # None on non-gfx1250 -> 0
                num_tokens=num_tokens,
            )

        return run
