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
can be served -- everything out of range is rejected at CONSTRUCTION rather than
at launch.

Two paths live here, selected by ``cfg.is_internode`` (world_size larger than
gpu_per_node; there is no kernel_type enum on the v2 side):

* intranode -- one dispatch kernel and one combine kernel, gather only, a
  bf16/fp32/fp8/fp4 dispatch and a bf16/fp32 combine, one plan compiled per
  (block, warp) the tuning schedule can select.
* internode -- a sequence of separately compiled passes over the
  ``internode_regions`` arena and a device communicator: copystaging +
  dispatch for a round of dispatch, combinesync + combinesyncbarrier + combine
  + combineall for a round of combine. Gather only, bf16/fp32/fp8 on either
  leg, plus packed fp4 dispatch with bf16/fp32 combine. FP4 combine is not
  supported. "v2" and "v2_ll" are two distinct kernel families rather than
  a runtime branch: ``cfg.internode_kernel`` (auto | v2 | v2_ll) says which are
  compiled, and only "auto" compiles both and chooses per launch, at
  ``cfg.internode_auto_ll_max_tokens``. ``hip_tuning_configs.resolve_schedule``
  resolves both paths to the same kernel configurations before plans are built.
  Internode geometry changes reuse the compiled module; each plan retains its
  own grid, block and RDMA split.
  No other backend implements internode.

Imports ``ep_plans`` (the C++/JIT plans) but never flydsl, so it works where
FlyDSL is not installed.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import logging
import os

import torch

from mori.ops import utils as _gpu
from mori.tensor_utils import from_gpu_ptr

from . import ep_plans as cb
from . import hip_tuning_configs as tuning
from .dispatch_combine_op import _PUSH_QUANT_TYPES, EpDispatchCombineOp, KernelSet
from .internode_regions import (
    asyncll_regions,
    asyncll_seg_layout,
    asyncll_seg_words,
    internode_regions,
)
from .symm_arena import SymmArena

logger = logging.getLogger(__name__)

# segment_error_flags() bits.
SEGMENT_ERR_OVERFLOW = 1
SEGMENT_ERR_MISMATCH = 2
# EpInterNodeArgs.segStride bit 63: the op has the segmented layout, so compact
# rounds on it still exchange the mode word (asyncll::kSegCapableBit).
SEG_CAPABLE_BIT = 1 << 63

# C++ offset-argument stem -> arena region name. The C++ side names the offsets
# after its own EpArgs fields (offTokOff -> "tokOff"); the region names match the
# FlyDSL op's, and so do the sizes EXCEPT out_scales -- this backend pads the row,
# FlyDSL does not. The sizes come from the plans (region.<stem> in their info,
# EpRegion* in ep_cfg.hpp): size an arena with ep_plans.arena_regions().
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
}

_LL_REGIONS = dict(
    _REGIONS,
    recvNum="ll_recv_num",
    recvToSrc="ll_recv_to_src_token",
    outIdx="ll_out_idx",
    outWts="ll_out_wts",
    dispOut="ll_disp_out",
    outTok="ll_out_tok",
    outScales="ll_out_scales",
)

_LL_SPEC = ("ll128",)

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

# Which tuning-table dtype column a dispatch dtype reads.
_FP8_TUNING_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


# SDMA queues per GPU pair that a PART is known to fit, when AsyncLL would
# otherwise ask for one per peer. Only parts that CANNOT hold the full eight
# belong here; everything else keeps the peer count.
#
# Keyed on the model (mori.ops.utils.detect_model, i.e. the PCI device id), not
# the arch: MI300X, MI325X and MI308X are all gfx942, and only MI308X is short of
# engines. Its xGMI engines fold to the even ids {2,4,6} and ROCr's own blit
# queues take part of that, so three is what a pair fits -- asking for eight on
# 8xMI308X returns HSAKMT_STATUS_NO_MEMORY from hsaKmtCreateQueueExt and anvil
# exit()s the process; with three the AsyncLL suite passes there. anvil singles
# MI308X out the same way (AnvilLib::isMi308x, DID 0x74A2). Capping all of gfx942
# would have held MI300X and MI325X, which have the full set, to three as well.
_ASYNCLL_SDMA_QUEUE_CAP = {"mi308x": 3}

# Per device: how many live AsyncLL ops use each SDMA queue offset.
_ASYNCLL_QUEUE_OFFSET_USERS = collections.defaultdict(collections.Counter)


def _take_sdma_queue_offset(device_index: int, num_queues: int) -> int:
    """Least-used offset among live ops on the device, lowest on a tie.

    close() gives it back. More live ops than queues share one: slower, still
    correct.
    """
    users = _ASYNCLL_QUEUE_OFFSET_USERS[device_index]
    offset = min(range(num_queues), key=lambda o: (users[o], o))
    users[offset] += 1
    return offset


def _give_back_sdma_queue_offset(device_index: int, offset: int) -> None:
    users = _ASYNCLL_QUEUE_OFFSET_USERS[device_index]
    users[offset] -= 1
    if users[offset] <= 0:
        del users[offset]


def _hsa_cu_mask_cus(device_index: int, mp_count: int):
    """CUs HSA_CU_MASK leaves this process on `device_index`, or None if unrestricted.

    ROCr's syntax is `<gpus>:<cus>[;<gpus>:<cus>...]`, each side a comma list of
    decimal indices or `a-b` ranges, e.g. `0-7:0-63`. GPU indices are ROCr's
    enumeration order, which is the HIP/torch order unless HIP_VISIBLE_DEVICES
    reorders devices. A setting this cannot read returns None, so the ceiling
    falls back to the whole device rather than failing the op.
    """
    raw = os.environ.get("HSA_CU_MASK", "").strip()
    if not raw:
        return None

    def indices(text):
        out = set()
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            lo, sep, hi = part.partition("-")
            lo, hi = int(lo), int(hi) if sep else int(lo)
            if lo < 0 or hi < lo:
                raise ValueError(part)
            out.update(range(lo, hi + 1))
        return out

    try:
        for entry in raw.split(";"):
            if not entry.strip():
                continue
            gpus, _, cus = entry.partition(":")
            if device_index in indices(gpus):
                found = indices(cus)
                if mp_count > 0:
                    found = {c for c in found if c < mp_count}
                return len(found)
    except ValueError:
        return None
    return None


def _asyncll_sdma_queues(model, gpu_per_node: int) -> int:
    """How many SDMA queues per pair to ask cco for, 0 to leave the env default.

    `model` is detect_model()'s answer, None when the part is not one it knows.

    An explicit MORI_SDMA_NUM_CHANNELS is the operator saying what this machine
    can take, so it wins: returning 0 lets cco fall through to it. Without that
    deference this request silently beat the env knob, and the only way out of
    the MI308X abort was to edit the source.
    """
    if os.environ.get("MORI_SDMA_NUM_CHANNELS"):
        return 0
    want = min(8, max(1, gpu_per_node))
    cap = _ASYNCLL_SDMA_QUEUE_CAP.get(model)
    return min(want, cap) if cap else want


def _raw_stream(device_index: int) -> int:
    """The current stream on `device_index`, as a raw pointer.

    `torch.cuda.current_stream().cuda_stream` builds a Python Stream wrapper on
    every call -- ~14us a launch in the host profile, on a path where the host
    already paces the GPU. The private entry returns the pointer directly; it is
    what torch.compile's generated code uses. Fall back if it is ever renamed.

    The device index is passed in rather than queried: `torch.cuda.current_device()`
    is a lazy-init check plus a C call (~1.5us a launch in the same profile) for a
    value the op fixes at construction. The STREAM still has to be read every call
    -- a caller may run us under a different one.
    """
    try:
        return torch._C._cuda_getCurrentRawStream(device_index)
    except AttributeError:
        return torch.cuda.current_stream().cuda_stream


def scale_stride_bytes(scale_bytes: int) -> int:
    """What a DESTINATION scale row is laid down at (EpScaleStride in ep_cfg.hpp).

    Anything addressing the region itself strides by this; recv_scales() hides it.
    For callers sizing without a plan: the plan's own info["scaleStride"], asked
    of describe().
    """
    if scale_bytes <= 0:
        return 0
    return _dispatch_info(scale_bytes=int(scale_bytes))["scaleStride"]


def ep_staging_bytes(
    world_size: int, max_recv: int, scale_bytes: int, max_tok_per_rank: int = 0
) -> int:
    """Bytes of the dispatch staging buffer (EpStagingTotalBytes in ep_cfg.hpp): the
    plan's info["stagingBytes"], asked of describe(). 0 where the dispatch does not
    stage (anything but gfx125x).

    ``max_recv`` is EpCfg.maxRecv, where 0 is not a cap of zero but "ask
    EpMaxRecv", which answers world_size * max_tok_per_rank -- and that is the
    Plan default, so a caller sizing the buffer straight from a Cfg hits it.
    Pass ``max_tok_per_rank`` with it; the kernel indexes staging at the cap it
    resolved, so a buffer sized at 0 is not a short allocation that faults but a
    write past the end of whatever the caller handed us.
    """
    recv = max_recv if max_recv > 0 else world_size * max_tok_per_rank
    if recv <= 0:
        raise ValueError(
            "ep_staging_bytes: max_recv is 0 (EpCfg's 'use EpMaxRecv' spelling) and "
            "no max_tok_per_rank was given, so the receive cap is unknown. Pass "
            "max_tok_per_rank, or pass an already-resolved cap as max_recv."
        )
    return _dispatch_info(
        world_size=int(world_size),
        max_recv=int(recv),
        max_tok_per_rank=int(max_tok_per_rank or recv // world_size),
        scale_bytes=int(scale_bytes),
    )["stagingBytes"]


@functools.lru_cache(maxsize=None)
def _dispatch_info_cached(fields: tuple) -> dict:
    return cb.EpDispatchPlan.describe(**dict(fields))


def _dispatch_info(**fields) -> dict:
    return _dispatch_info_cached(tuple(sorted(fields.items())))


@functools.lru_cache(maxsize=None)
def _combine_info_cached(fields: tuple) -> dict:
    return cb.EpCombinePlan.describe(**dict(fields))


# Must match EpXdbFlagSlots in include/mori/ops/dispatch_combine_v2/ep_cfg.hpp.
_XDB_FLAG_SLOTS = 256

PUSH_QGROUP = {"fp4_blockwise": 32, "fp4_blockwise_fp32": 128, "fp8_blockwise": 32}
# The combine request fields each push quant type turns on.
_PUSH_REQUEST = {
    "fp4_blockwise": (("combine_fp4", 1),),
    "fp4_blockwise_fp32": (("combine_fp4", 1), ("combine_fp4_f32_scale", 1)),
    "fp8_blockwise": (("combine_fp8", 1),),
}


def push_wire_nbytes(cfg) -> int:
    """Bytes of one landed row of the push combine (EpCombinePushWireBytes): the
    plan's info["pushWireBytes"], asked of describe(). Reads cfg.hidden_dim and
    cfg.quant_type; the push combine, and so this, is gfx125x only."""
    fields = (("hidden_dim", int(cfg.hidden_dim)), *_PUSH_REQUEST[cfg.quant_type])
    return _combine_info_cached(tuple(sorted(fields)))["pushWireBytes"]


def push_land_rows(cfg) -> int:
    """Landing rows of the push combine, one per token and destination rank
    (EpCombinePushLandRows): the plan's info["pushLandRows"], asked of describe().
    Reads cfg.max_num_inp_token_per_rank, cfg.world_size and
    cfg.num_experts_per_token; gfx125x only, as the push combine is."""
    fields = (
        ("world_size", int(cfg.world_size)),
        ("max_tok_per_rank", int(cfg.max_num_inp_token_per_rank)),
        ("num_expert_per_token", int(cfg.num_experts_per_token)),
        *_PUSH_REQUEST["fp4_blockwise"],
    )
    return _combine_info_cached(tuple(sorted(fields)))["pushLandRows"]


def ll128_layout(cfg):
    if cfg.is_internode or cfg.async_ll:
        return None
    if cfg.quant_type == "none" and cfg.combine_dtype is not torch.bfloat16:
        return None
    if cfg.is_fp4:
        dt = "fp4"
    elif cfg.dispatch_dtype in _FP8_TUNING_DTYPES:
        dt = "fp8"
    elif cfg.dispatch_dtype is torch.bfloat16 and cfg.quant_type == "none":
        dt = "bf16"
    else:
        return None
    from .hip_tuning_configs import ll128

    geom = ll128(
        cfg.world_size,
        cfg.hidden_dim,
        cfg.num_experts_per_token,
        dt,
        cfg.max_num_inp_token_per_rank,
        experts_per_rank=cfg.num_experts_per_rank,
        quant_type=cfg.quant_type,
    )
    if geom is None:
        return None
    cap = cfg.max_num_inp_token_per_rank
    n = geom.get("max_tok", cap)
    return dict(geom, max_tok=n, region_names=_LL_REGIONS if n < cap else _REGIONS)


class TokOffExt:
    peers = None

    @staticmethod
    def wanted(world=1) -> bool:
        return False

    def __init__(self, rank=0, world=1, dev=None):
        pass

    def zero(self):
        pass

    def close(self):
        pass


def _relaxed_ordering_enabled():
    """MORI_IB_ENABLE_RELAXED_ORDERING as rdma.cpp reads it: exactly the value 1."""
    try:
        return int(os.environ.get("MORI_IB_ENABLE_RELAXED_ORDERING", "")) == 1
    except ValueError:
        return False


class EpDispatchCombineOpHip(EpDispatchCombineOp, backend="hip"):
    """C++/JIT-kernel EP op: gather combine, no quant, no replay.

    Dispatch transports bf16/fp32/fp8/fp4, combine reduces in bf16/fp32, and the
    two need not match -- an fp8-in/bf16-out op is just two plans with different
    dtypes. mori does no quantizing here: fp8/fp4 payloads arrive already packed.
    """

    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        self._device_index = torch.cuda.current_device()
        dev = torch.device("cuda", self._device_index)
        self.dev = dev
        self._recv_cap = cfg.effective_max_recv
        self._closed = False
        # Arena views, built once. Every one below is a pure function of cfg and
        # of an arena pointer that never moves, yet dispatch() rebuilt five of
        # them per call -- torch.as_tensor + view showed up in the host profile at
        # ~27us a round, on a path where the host already paces the GPU.
        self._views = {}
        # This round's segmented stride, 0 for the compact layout (see
        # _set_segment_stride). Combine and the views follow the last dispatch.
        self._seg_stride = 0
        # gfx125x routes to the TDM kernel, which needs a superset arena (plan A).
        _arch = getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "") or ""
        self._arch = _arch.split(":")[0]
        self._is1250 = _arch.split(":")[0].startswith("gfx125")
        self._multi_processor_count = torch.cuda.get_device_properties(
            dev
        ).multi_processor_count
        # MORI_EP_DISABLE_RDMA_ATOMICS as the C++ IsEnvVarEnabled reads it: set, and not
        # 0/false/off/no. Compiled into the v2 / v2_ll plans (kConfig.noRdmaAtomics).
        _af = os.environ.get("MORI_EP_DISABLE_RDMA_ATOMICS", "")
        self._no_rdma_atomics = _af.lower() not in ("", "0", "false", "off", "no")
        lay = (
            ll128_layout(cfg) if self._is1250 and cfg.low_latency is not False else None
        )
        split = lay is not None and lay["max_tok"] < cfg.max_num_inp_token_per_rank
        self._ll128_geom = None if split else lay
        self._ll128_split = lay if split else None
        self._ll_recv_cap = cfg.world_size * lay["max_tok"] if split else 0
        self._ll_last = False

        # "auto" resolves before the gate and the arena, from the config alone,
        # so every rank reaches the same answer; the regions depend on it.
        if cfg.async_ll_recv_layout == "auto":
            supported = not self._asyncll_segmented_unsupported(cfg)
            cfg = dataclasses.replace(
                cfg, async_ll_recv_layout="segmented" if supported else "compact"
            )
            self.cfg = cfg

        # Gate FIRST: rejecting a config after taking a symmetric window would
        # leak it (the arena is registered with the communicator), and the whole
        # point of the gate is that an unsupported config never gets that far.
        self._gate(
            KernelSet(dispatch={}, combine={}, unsupported=self._unsupported(cfg))
        )
        if lay is None and not cfg.is_internode and not cfg.async_ll:
            self._warn_unused_low_latency()
        # Active QP counts set_active_qps() may switch to. Each is compiled for
        # every bucket geometry, so a switch never compiles at launch.
        self._selectable_active_qps = tuple(cfg.active_qp_counts or ())
        self._active_qps_override = None

        if self._ll128_geom is not None and cfg.schedule:
            comb_b, comb_w = self._ll128_geom["combine"]
            if "dispatch" in self._ll128_geom:
                disp_b, disp_w = self._ll128_geom["dispatch"]
                cfg.schedule = ((None, disp_b, disp_w, comb_b, comb_w),)
            else:
                cfg.schedule = tuple(
                    (e, db, dw, comb_b, comb_w) for (e, db, dw, _, _) in cfg.schedule
                )
        self._schedule = tuning.resolve_schedule(
            cfg, cu_count=self._multi_processor_count
        )
        self._ll_schedule = tuning.resolve_ll_schedule(
            cfg, cu_count=self._multi_processor_count
        )
        rows = self._schedule + self._ll_schedule
        if cfg.rank == 0 and not cfg.async_ll and logger.isEnabledFor(logging.INFO):
            logger.info("%s", tuning.describe_schedule(cfg, rows))
        self._dispatch_specs = tuple(dict.fromkeys(row.dispatch for row in rows))
        self._combine_specs = tuple(dict.fromkeys(row.combine for row in rows))
        self._bf16_push = self._bf16_push_layout(cfg)
        self._push_specs = tuple(
            dict.fromkeys(
                tuning.KernelConfig("intra", b, w) for _, b, w in self._bf16_push or ()
            )
        )
        # What each intranode plan will report, per variant -- its arena regions,
        # staging, push landing -- asked before any of them is built.
        self._disp_infos, self._comb_infos = self._describe(cfg)
        self.arena = SymmArena(comm, self._regions(cfg))
        try:
            self._build(cfg, comm)
        except BaseException:
            # Everything past SymmArena() takes device resources the interpreter
            # does not own (a symmetric window, QPs, JIT plans); a partial
            # __init__ leaves no object for close() to run on, so unwind here.
            self._close_backend()
            self.arena.close()
            raise
        # _build zeroes this rank's arena, and a peer's first call already writes
        # into it (its signal lands in this rank's recv-count slot). A zero that runs
        # after that write wipes it, and this rank's first dispatch then waits forever
        # for the signal. Every rank finishes its zeroes before any rank can make a
        # call.
        torch.cuda.synchronize(dev)
        comm.barrier()

    def _bf16_push_layout(self, cfg):
        if (
            not self._is1250
            or cfg.is_internode
            or cfg.async_ll
            or cfg.quant_type != "none"
            or (
                cfg.dispatch_dtype is not torch.bfloat16
                and cfg.dispatch_dtype not in _FP8_TUNING_DTYPES
            )
            or cfg.combine_dtype is not torch.bfloat16
            or not cfg.schedule
            or self._ll128_geom is not None
        ):
            return None
        flag = os.environ.get("MORI_EP_BF16_PUSH", "1")
        if flag not in ("0", "1"):
            raise ValueError(f"MORI_EP_BF16_PUSH must be '0' or '1', got {flag!r}")
        if flag == "0":
            return None
        return tuning.bf16_push(
            cfg.world_size,
            cfg.hidden_dim,
            cfg.num_experts_per_token,
            experts_per_rank=cfg.num_experts_per_rank,
        )

    def _build(self, cfg, comm):
        dev = self.dev
        self.arena.zero()

        # The internode passes need a device communicator, and it has to exist
        # before the kernels are bound: the plans take it by value.
        # AsyncLL needs one on a single node too -- ccoSdma hangs off the device
        # communicator, not off the window -- which is why the test is not just
        # is_internode.
        self._dev_comm = (
            self._make_dev_comm(cfg, comm)
            if (cfg.is_internode or cfg.async_ll)
            else None
        )
        # SDMA queue offset (QueueForPe adds it to the destination PE): 0, the
        # shared queue, unless the caller opted in. LSA uses no SDMA queue.
        self._sdma_queue_offset = 0
        self._holds_sdma_queue_offset = False
        if (
            cfg.async_ll
            and cfg.async_ll_transport == "sdma"
            and cfg.async_ll_own_sdma_queue
            and self._dev_comm is not None
            and self._dev_comm.sdma_num_queue > 1
        ):
            self._sdma_queue_offset = _take_sdma_queue_offset(
                self._device_index, self._dev_comm.sdma_num_queue
            )
            self._holds_sdma_queue_offset = True

        # gfx1250 dispatch staging: one dynamically-allocated buffer shared by
        # all dispatch schedule variants.  Must be set before _build_kernels
        # because dispatch plans bind the pointer at construction time.
        self.dispatch_staging = None
        stg_bytes = max((i["stagingBytes"] for i in self._disp_infos), default=0)
        if stg_bytes > 0:
            self.dispatch_staging = torch.zeros(
                stg_bytes, dtype=torch.uint8, device=dev
            )

        self._kernels = self._build_kernels(cfg, self.arena)

        if cfg.is_internode or cfg.async_ll:
            self._alloc_internode_buffers(cfg)
            return

        topk = cfg.num_experts_per_token
        max_tok = cfg.max_num_inp_token_per_rank
        i32 = dict(dtype=torch.int32, device=dev)
        self.token_dest_map = torch.zeros(max_tok * topk, **i32)
        self._null_flat = cfg.world_size * cfg.effective_max_recv
        self.routing_dest_map = torch.full_like(self.token_dest_map, self._null_flat)
        # The send counts, then per destination its reservation base and, from the
        # dispatch signal on, the push combine's landing offset of the rows it
        # returns (EpFp4LandTo; dispatch writes it, combine reads it).
        self.dest_pe_counter = torch.zeros(2 * cfg.world_size, **i32)
        self.total_recv = torch.zeros(1, **i32)
        self._recv = self.total_recv
        self._ll_recv = torch.zeros(1, **i32) if self._ll128_split is not None else None
        self._snapshot_zeroes_count = self._ll128_split is not None or (
            self._ll128_geom is not None and "dispatch" in self._ll128_geom
        )
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
            max_comb_blocks = max(
                spec.block_num for spec in (*self._combine_specs, *self._push_specs)
            )
            if max_comb_blocks > _XDB_FLAG_SLOTS:
                raise ValueError(
                    f"combine block_num {max_comb_blocks} exceeds the {_XDB_FLAG_SLOTS} "
                    "per-block xdb epoch slots the entry barrier owns"
                )
            self.combine_barrier_fan = torch.zeros(max_comb_blocks * 16, **i32)

    # -- internode -------------------------------------------------------
    #
    # Multi-node configs run a sequence of passes (2 for dispatch, 4 for
    # combine) over the internode arena (16 regions, 17 with scales) with a
    # device communicator, instead
    # of two kernels over nine regions. Everything else -- the arena, the plan
    # API, the library -- is the same, which is why this lives here rather than
    # in a backend of its own.

    @staticmethod
    def _make_dev_comm(cfg, comm):
        """The communicator the internode kernels take by value.

        Two requirements the defaults do not meet, made here so no caller has to
        know them:

        * RAIL rather than the CROSSNODE default. The kernels talk only to their
          own local rank on other nodes, so that is the only connection set they
          need, and asking for more costs QPs that never carry traffic.
        * num_qp_per_pe allocated contexts. Each kernel sends on a prefix of
          them (its active count, KernelConfig.active_qps); the count stored in
          the DevComm stays the allocation because it is also the endpoint-array
          stride for world-rank addressing.
        """
        from mori.cco import cco as _cco
        from mori.cco.communicator import DevCommHandle

        requirements = _cco.DevCommRequirements()
        requirements.gda_connection_type = _cco.GDA_CONNECTION_RAIL
        requirements.gda_context_count = max(1, cfg.num_qp_per_pe)
        if cfg.async_ll and cfg.nodes > 1:
            # CROSSNODE, not the RAIL set the v2/v2_ll families ask for. Those
            # aggregate through a same-rail proxy -- each rank only ever talks to
            # its own local rank on another node -- so same-rail QPs are all they
            # can use. AsyncLL addresses peers individually: rank i sends to rank
            # j directly, which is a cross-rail path and needs a QP to every
            # cross-node peer.
            #
            # This also means AsyncLL cannot run across nodes on a rail-only
            # fabric. That failure is silent (QP setup is a TCP exchange and
            # succeeds; no data moves), so confirm the fabric with an ib_write_bw
            # between two DIFFERENT rails before trusting a multi-node run.
            requirements.gda_connection_type = _cco.GDA_CONNECTION_CROSSNODE
        if cfg.async_ll and cfg.async_ll_transport == "sdma":
            # One SDMA queue per peer. The transfer kernels run one issuing warp
            # per destination and the queue is picked as (destPe + the op's
            # sdmaQueueOffset) % sdmaNumQueue, so fewer queues than peers folds
            # two warps onto one queue -- correct, but they then serialise on its
            # in-order commit chain. Eight is the hardware cap; above it cco
            # clamps with a warning.
            #
            # The count is a PERFORMANCE knob, not a correctness one, and asking
            # for more than a pair can hold is fatal rather than degrading:
            # anvil's queue creation is CHECK_HSAKMT_SUCCESS, which exit()s. On
            # 8xMI308X, 8 queues per pair gets HSAKMT_STATUS_NO_MEMORY from
            # hsaKmtCreateQueueExt and every rank leaves the process -- its xGMI
            # engines fold to the even ids {2,4,6} and ROCr's own blit queues
            # take part of that, so three is what fits. With <= 3 the whole
            # AsyncLL suite passes there, which is why this clamps rather than
            # trying to recover: by the time the request is refused the process
            # is already gone.
            requirements.sdma_queue_count = _asyncll_sdma_queues(
                _gpu.detect_model(), cfg.gpu_per_node
            )
        handle = DevCommHandle(comm, requirements=requirements)

        # A rejected handle still owns its QPs, so every exit below its
        # construction has to release it.
        try:
            # The kernels address peers by world rank out of this communicator, so
            # a cfg.world_size that disagrees names ranks that do not exist.
            if handle.world_size != cfg.world_size:
                raise ValueError(
                    f"EP's world_size ({cfg.world_size}) != CCO's world_size "
                    f"({handle.world_size}): the kernels index peers by world rank "
                    "out of this communicator"
                )
            # EP derives "which ranks share a node" from cfg.gpu_per_node; CCO
            # derives its LSA team from the physical topology. Nothing forces the
            # two to agree, and the kernel resolves a same-node peer as
            # `worldPe - lsaBase`, which is only an LSA rank if they do. A mismatch
            # does not fault -- it reads the wrong rank's copy -- so it is checked
            # once, here, where both numbers are visible.
            if handle.lsa_size != cfg.gpu_per_node:
                raise ValueError(
                    f"EP's gpu_per_node ({cfg.gpu_per_node}) != CCO's lsa_size "
                    f"({handle.lsa_size}): EP's idea of a node and the flat-VA team "
                    "are different sets, so a peer-indexed access would resolve to "
                    "the wrong rank"
                )
            expected_lsa_rank = cfg.rank % cfg.gpu_per_node
            if handle.lsa_rank != expected_lsa_rank:
                raise ValueError(
                    f"rank {cfg.rank} has CCO lsa_rank {handle.lsa_rank} but EP's node "
                    f"layout implies {expected_lsa_rank}: world ranks are not laid out "
                    "node-major, so worldPe - lsaBase is not an LSA rank"
                )
            # The env check in _asyncll_unsupported catches MORI_ENABLE_SDMA being
            # unset before anything is built, but it is not the only way to end
            # up with no queues: a library built BUILD_CCO_SDMA=OFF, or peers that
            # do not report SDMA capability, get 0 too. cco only logs that, and
            # every ccoSdma put then moves no bytes, so the receiver spins on a
            # count that never arrives -- a hang, not an error. Ask the handle for
            # what it actually connected. A node of one has no same-node peer and
            # makes no SDMA put at all, so 0 is correct there.
            if (
                cfg.async_ll
                and cfg.async_ll_transport == "sdma"
                and handle.lsa_size > 1
                and handle.sdma_num_queue == 0
            ):
                raise ValueError(
                    "async_ll_transport='sdma' but this communicator has no SDMA "
                    "queues (mori built with BUILD_CCO_SDMA=OFF, MORI_ENABLE_SDMA "
                    "unset, or no peer reports SDMA capability -- the log above "
                    "says which); every transfer would be a no-op and the receiver "
                    "would wait forever. Use async_ll_transport='lsa', which needs "
                    "no engine, or fix the SDMA setup"
                )
        except BaseException:
            handle.close()
            raise
        return handle

    def _alloc_internode_buffers(self, cfg):
        """The local (non-symmetric) buffers the internode passes write.

        Sizes transcribed from EpDispatchCombineHandle's hipMallocs for the same
        config; the kernel indexes them with arithmetic derived from the same
        numbers, so a formula that is merely close overruns silently.
        """
        dev = self.dev
        i32 = dict(dtype=torch.int32, device=dev)
        world_size, gpu_per_node = cfg.world_size, cfg.gpu_per_node
        n_nodes = cfg.nodes
        max_tokens_per_rank = cfg.max_num_inp_token_per_rank
        topk = cfg.num_experts_per_token

        self.lsa_base = (cfg.rank // gpu_per_node) * gpu_per_node

        # The base hands one of these to every dispatch as `dest_map` -- which one
        # depends on whether the caller asked for a routing handle -- and it is
        # what the kernel reads as dispDestTokIdMap. Sized the way v1 sizes that
        # buffer: the kernel indexes it by (token, expert), but v1 allocates the
        # worst case and the region test does not cover it, so this stays
        # conservative rather than clever.
        self.token_dest_map = torch.zeros(
            world_size * max_tokens_per_rank * cfg.num_experts_per_rank, **i32
        )
        self._null_flat = world_size * cfg.effective_max_recv
        self.routing_dest_map = torch.full_like(self.token_dest_map, self._null_flat)

        self.inter_disp_dest_tok_id_map = torch.zeros(
            n_nodes * max_tokens_per_rank * topk, **i32
        )
        self.inter_disp_send_map = torch.zeros(n_nodes * max_tokens_per_rank, **i32)
        # Counts completions. NOT the local half of the symmetric chunk-flag
        # region despite the name: that one is written by dispatch and read+
        # cleared by combine, which enumerates its work from dispatch's leftovers.
        # The v2_ll combine counts one piece per QP of each 64-token chunk.
        self.inter_chunk_flag_combine = torch.zeros(
            n_nodes
            * max(
                max_tokens_per_rank * 2,
                -(-max_tokens_per_rank // 64) * cfg.num_qp_per_pe,
            ),
            **i32,
        )
        self.dest_pe_counter = torch.zeros(world_size, **i32)
        self.block_flag_counter = torch.zeros(n_nodes, **i32)
        # AsyncLL only: recv slot -> the send-buffer slot it arrived in, written by
        # dispatch's RecvCopy and read by combine's SendCopy. One entry per slot
        # this rank can receive into.
        self.disp_receiver_idx_map = (
            torch.zeros(cfg.effective_max_recv, **i32) if cfg.async_ll else None
        )
        self.total_recv = torch.zeros(1, **i32)
        self._recv, self._ll_recv = self.total_recv, None
        self.dispatch_barrier = torch.zeros(world_size, dtype=torch.uint32, device=dev)
        self.combine_barrier = torch.zeros(world_size, dtype=torch.uint32, device=dev)
        self.inter_blocks_barrier = torch.zeros(4, dtype=torch.uint32, device=dev)
        # Zero, not one: the internode kernels start their cross-device epoch at
        # 0 (v1 seeds it that way for exactly these kernel types).
        self.cross_device_flag = torch.zeros(1, dtype=torch.int64, device=dev)

        # Views onto the arena, NOT fresh tensors: EpCombineAll writes the
        # symmetric regions, so a local buffer here would be returned to the
        # caller unwritten. The base slices these to (ct, hidden) / (ct, topk)
        # and never learns which layout it is looking at.
        self.combine_out = from_gpu_ptr(
            self.arena.local_ptr("inter_combine_out"),
            (max_tokens_per_rank * cfg.hidden_dim,),
            cfg.combine_dtype,
        )
        self.combine_out_weights = from_gpu_ptr(
            self.arena.local_ptr("combine_out_weights"),
            (max_tokens_per_rank * topk,),
            torch.float32,
        )

        # moe_sort_block_m: aiter's moe_sorting outputs, sized for the largest
        # tables aiter would allocate for this op (M = every receive row, one
        # more expert than the world has for a trailing mask sentinel).
        bm = cfg.moe_sort_block_m
        self.moe_sort_tables = None
        if bm:
            rows = (
                cfg.effective_max_recv * topk
                + (cfg.world_size * cfg.num_experts_per_rank + 1) * bm
            )
            rows = (rows + bm - 1) // bm * bm
            self.moe_sort_tables = (
                torch.empty(rows, **i32),
                torch.empty(rows, dtype=torch.float32, device=dev),
                torch.empty(rows // bm, **i32),
                torch.zeros(2, **i32),
            )
            # Rows of the combine input the last FC2 accumulated into; dispatch
            # zeroes them (the combine input is fused_moe's output here).
            self.moe_sort_zero_rows = torch.zeros(1, **i32)

        # Pin everything fixed for the op's lifetime, now that every buffer it
        # names exists. The launch path re-fills every field passed through
        # `args` on every call -- only bound values reach the plan's cached
        # struct -- so passing the whole schema each time cost a ctypes field
        # write per field per launch. Binding leaves six.
        static_args = self._internode_static_args()
        for plan in self._plans:
            plan.bind(**static_args)

    def _internode_unsupported(self, cfg) -> tuple[str, ...]:
        bad = []
        # Checked HERE rather than where the plans are built: _build_internode_kernels
        # runs after the arena and the device communicator are taken, and its caller
        # does not gate on what it returns.
        for leg, leg_dtype in (
            ("dispatch", cfg.dispatch_dtype),
            ("combine", cfg.combine_dtype),
        ):
            if leg_dtype not in self._INTERNODE_DTYPE:
                bad.append(
                    f"{leg} dtype {leg_dtype} has no internode kernel "
                    f"(have {', '.join(self._INTERNODE_DTYPE.values())})"
                )
        if cfg.combine_dtype == torch.float4_e2m1fn_x2:
            bad.append("fp4 combine is not supported on the internode path")
        elif cfg.dispatch_dtype == torch.float4_e2m1fn_x2 and cfg.combine_dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            bad.append(
                "fp4 dispatch requires bf16 or fp32 combine on the internode path"
            )
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (the internode kernels gather)")
        if getattr(self, "_no_rdma_atomics", False) and _relaxed_ordering_enabled():
            bad.append(
                "MORI_EP_DISABLE_RDMA_ATOMICS with MORI_IB_ENABLE_RELAXED_ORDERING=1 "
                "(with relaxed ordering a WRITE signal may become visible before its data)"
            )
        if cfg.enable_std_moe:
            bad.append("enable_std_moe (not implemented on the internode path)")
        if cfg.quant_type != "none":
            bad.append(
                f"quant_type={cfg.quant_type!r} (the internode combine is unquantised)"
            )
        if cfg.moe_sort_block_m:
            from .dispatch_combine_op import WAVE

            # Lane topk + j buckets expert slot j; sorted ids keep the row in 24 bits.
            if 2 * cfg.num_experts_per_token > WAVE:
                bad.append(
                    f"moe_sort_block_m with num_experts_per_token="
                    f"{cfg.num_experts_per_token} (needs 2 * topk <= {WAVE})"
                )
            if cfg.effective_max_recv >= 1 << 24:
                bad.append("moe_sort_block_m with 2^24 or more receive rows")
            if cfg.gpu_per_node > 16:
                bad.append("moe_sort_block_m with more than 16 GPUs per node")
            # The finalize reads one expert's counter per lane.
            if cfg.num_experts_per_rank > WAVE:
                bad.append(
                    f"moe_sort_block_m with num_experts_per_rank="
                    f"{cfg.num_experts_per_rank} (needs <= {WAVE})"
                )
        # The internode kernel lays the scale rows down PACKED, and recv_scales()
        # hands them back as an int32 view; a row that is not a whole number of
        # dwords has no such view, and rounding it up runs off the region.
        raw_scale_bytes = cfg.scale_dim * cfg.scale_type_size
        if raw_scale_bytes % 4:
            bad.append(
                f"per-token scale row of {raw_scale_bytes} B "
                f"(scale_dim={cfg.scale_dim} x {cfg.scale_type_size}); "
                "the row must be a whole number of dwords"
            )
        # The same-destination dedup is a ballot with one lane per expert.
        from .dispatch_combine_op import WAVE

        if cfg.num_experts_per_token >= WAVE:
            bad.append(
                f"num_experts_per_token ({cfg.num_experts_per_token}) must be "
                f"smaller than the {WAVE}-wide wavefront"
            )
        # EpCombineAll's per-warp shared pointer arrays are sized topk wide
        # (EpInterNodeCombineSharedBytes) but indexed 0..nNodes -- it clears and
        # fills one slot per node. With topk < nodes a warp writes into its
        # neighbour's slot and the last warp runs off the array, so the fold
        # silently reads the wrong sources. Reject instead of corrupting.
        if cfg.num_experts_per_token < cfg.nodes:
            bad.append(
                f"num_experts_per_token ({cfg.num_experts_per_token}) must be at "
                f"least the node count ({cfg.nodes}): the combine fold indexes "
                "its topk-wide shared pointer arrays by node"
            )
        return tuple(bad)

    # -- backend hooks -----------------------------------------------------

    @staticmethod
    def _scale_i32(cfg) -> int:
        """Dwords in the SOURCE scale row. _unsupported rejects a row that is not
        already dword-sized, so the rounding here never actually rounds."""
        return (cfg.scale_dim * cfg.scale_type_size + 3) // 4

    def scale_stride_bytes(self) -> int:
        """Padded to 128 B on the INTRANODE path; the base returns the row
        unchanged. The internode kernel lays its rows down packed
        (`destTokId * ScaleBytes`, ScaleBytes = scaleDim * scaleTypeSize) and
        internode_regions sizes out_scales that way, so there it is the row itself.
        Before there is an op, EpDispatchPlan.describe(...)["scaleStride"]."""
        # AsyncLL stages scales the same packed way the internode kernels do
        # (destTokId * ScaleBytes), so it reads the row unchanged too.
        if self.cfg.is_internode or self.cfg.async_ll:
            return self._scale_row_bytes()
        return self._disp_infos[0]["scaleStride"]

    def _describe(self, cfg):
        """(dispatch infos, combine infos): every intranode variant, asked of
        describe(), which compiles nothing. Empty on the internode and AsyncLL
        paths, whose regions come from their own formulas."""
        if cfg.is_internode or cfg.async_ll:
            return (), ()
        disp, comb = self._plan_requests(cfg)
        return (
            tuple(
                cb.EpDispatchPlan.describe(
                    **disp,
                    block_num=spec.block_num,
                    warp_per_block=spec.warp_per_block,
                    call_tok_max=cap,
                )
                for spec in self._dispatch_specs
                for cap in self._dispatch_caps(cfg, spec)
            ),
            tuple(
                cb.EpCombinePlan.describe(
                    **comb,
                    block_num=spec.block_num,
                    warp_per_block=spec.warp_per_block,
                )
                for spec in self._combine_specs
            )
            + tuple(
                cb.EpCombinePlan.describe(
                    **comb,
                    combine_bf16_push=True,
                    block_num=spec.block_num,
                    warp_per_block=spec.warp_per_block,
                )
                for spec in self._push_specs
            ),
        )

    def _snapshot_recv_views(self, disp_spec):
        if disp_spec == _LL_SPEC:
            names, cap = ("ll_recv_to_src_token", "ll_out_idx"), self._ll_recv_cap
        else:
            geom = self._ll128_geom
            if geom is None:
                return None
            if "dispatch" not in geom:
                raise ValueError(
                    "snapshot=True on an op whose LL128 combine runs behind the schedule's "
                    "dispatch: that combine reads the slot map peers write into the arena, "
                    "which the next dispatch overwrites"
                )
            names, cap = ("recv_to_src_token", "out_idx"), self._recv_cap
        return (
            from_gpu_ptr(self.arena.local_ptr(names[0]), (cap,), torch.int32),
            from_gpu_ptr(
                self.arena.local_ptr(names[1]),
                (cap * self.cfg.num_experts_per_token,),
                torch.int32,
            ),
        )

    def _reverse_source(self, disp_spec):
        if disp_spec != _LL_SPEC:
            return super()._reverse_source(disp_spec)
        view = self._views.get("ll_reverse")
        if view is None:
            view = self._views["ll_reverse"] = from_gpu_ptr(
                self.arena.local_ptr("ll_recv_to_src_token"),
                (self._ll_recv_cap,),
                torch.int32,
            )
        return view, self._ll_reencode

    def _ll_reencode(self, rev):
        n = self._ll128_split["max_tok"]
        return rev // n * self.cfg.max_num_inp_token_per_rank + rev % n

    def _regions(self, cfg):
        if cfg.async_ll:
            # Same names, per-rank send extents. See asyncll_regions.
            return asyncll_regions(cfg)
        if cfg.is_internode:
            # Sized exactly, region by region. v1 sizes its arena with a 256 MB
            # slack term on a rough estimate; this is the exact sum, which is both
            # the point (an overrun fails instead of landing in the slack) and the
            # risk (a wrong formula corrupts a neighbour). test_internode_regions
            # compares every entry against what v1 allocates for the same config.
            return internode_regions(cfg)
        # Sized by the plans themselves (EpRegion* in ep_cfg.hpp), every variant
        # asked, so the arena covers each one.
        regions = cb.arena_regions(
            *self._disp_infos, *self._comb_infos, region_names=_REGIONS
        )
        ll = self._ll128_requests(cfg)
        if ll is not None:
            sizes = dict(regions)
            for name, n in cb.arena_regions(
                cb.EpDispatchPlan.describe(**ll[0]),
                cb.EpCombinePlan.describe(**ll[1]),
                region_names=self._ll128_split["region_names"],
            ):
                sizes[name] = max(sizes.get(name, 0), n)
            regions = list(sizes.items())
        if any(name == "out_scales" for name, _ in regions):
            # The kernel lays scale rows down at a 128 B pitch, which only lands
            # aligned if the region does; that is another file's constant.
            assert SymmArena._ALIGN % _SCALE_ALIGN == 0, (
                f"SymmArena._ALIGN={SymmArena._ALIGN} does not keep scale regions "
                f"{_SCALE_ALIGN} B-aligned; the padding in EpScaleStride buys nothing"
            )
        return regions

    def _asyncll_unsupported(self, cfg) -> tuple[str, ...]:
        """AsyncLL's own range. Checked before the arena or the communicator is
        taken, like every other gate here."""
        bad = []
        if cfg.moe_sort_block_m:
            bad.append("moe_sort_block_m (v2 / v2_ll only)")
        # Shares the internode kernels' dtype and feature range -- it reduces the
        # same way and stages through the same slots -- except packed fp4, which
        # only the v2/v2_ll dispatch carries.
        dtypes = {k: v for k, v in self._INTERNODE_DTYPE.items() if v != "fp4"}
        for leg, leg_dtype in (
            ("dispatch", cfg.dispatch_dtype),
            ("combine", cfg.combine_dtype),
        ):
            if leg_dtype not in dtypes:
                bad.append(
                    f"{leg} dtype {leg_dtype} has no AsyncLL kernel "
                    f"(have {', '.join(dtypes.values())})"
                )
        if cfg.active_qp_counts:
            bad.append("active_qp_counts (AsyncLL has no set_active_qps)")
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (the AsyncLL combine gathers)")
        if cfg.enable_std_moe:
            bad.append("enable_std_moe (not implemented on the AsyncLL path)")
        if cfg.quant_type != "none":
            bad.append(
                f"quant_type={cfg.quant_type!r} (AsyncLL transports unquantized)"
            )
        if cfg.async_ll_transport not in ("sdma", "lsa"):
            bad.append(
                f"async_ll_transport={cfg.async_ll_transport!r} (have 'sdma', 'lsa')"
            )
        if cfg.nodes > 1 and getattr(self, "_no_rdma_atomics", False):
            bad.append(
                "MORI_EP_DISABLE_RDMA_ATOMICS (cross-node AsyncLL signals with RDMA "
                "atomics; use v2 / v2_ll)"
            )
        if cfg.nodes > 1 and cfg.async_ll_transport == "lsa":
            bad.append(
                "async_ll_transport='lsa' with gpu_per_node="
                f"{cfg.gpu_per_node} < world_size={cfg.world_size}: peer stores "
                "reach only ranks inside this one's flat LSA window. Use 'sdma', "
                "whose cross-node peers go over ccoGda"
            )
        # Without this the communicator comes back with zero SDMA queues and every
        # put is a silent no-op -- the tokens never move and the receiver spins on
        # a count that never arrives. A hang with no message is the worst way to
        # learn about an unset environment variable.
        if cfg.async_ll_transport == "sdma" and os.environ.get(
            "MORI_ENABLE_SDMA", ""
        ) not in ("1", "true", "True"):
            bad.append(
                "MORI_ENABLE_SDMA is not set: cco creates no SDMA queues without "
                "it and every AsyncLL transfer would be a no-op (or use "
                "async_ll_transport='lsa', which needs no engine)"
            )
        if cfg.async_ll_segmented:
            bad.extend(self._asyncll_segmented_unsupported(cfg))
        return tuple(bad)

    def _plan_requests(self, cfg):
        """(dispatch, combine) request fields every plan of this op shares."""
        common = dict(
            world_size=cfg.world_size,
            max_tok_per_rank=cfg.max_num_inp_token_per_rank,
            num_expert_per_rank=cfg.num_experts_per_rank,
            num_expert_per_token=cfg.num_experts_per_token,
            max_recv=cfg.effective_max_recv,
            use_weights=True,
        )
        # The two legs are separate Plans, so each carries its own dtype and its own
        # element count -- which is what makes an asymmetric config (fp8/fp4 in,
        # bf16 out) just two ordinary kernels. hiddenDim is "elements of THIS leg's
        # dtype", so fp4 halves it: 2 e2m1 live in one transported byte.
        disp = dict(
            common,
            hidden_dim=cfg.hidden_dim // 2 if cfg.is_fp4 else cfg.hidden_dim,
            dtype=cfg.dispatch_dtype,
            # Dword-padded: the kernel copies the row as dwords, and EpCfgIsValid
            # rejects a Cfg whose row is not a whole number of them.
            scale_bytes=self._scale_i32(cfg) * 4,
        )
        comb = dict(common, hidden_dim=cfg.hidden_dim, dtype=cfg.combine_dtype)
        if cfg.quant_type in ("fp4_blockwise", "fp4_blockwise_fp32"):
            comb["combine_fp4"] = True
        if cfg.quant_type == "fp4_blockwise_fp32":
            comb["combine_fp4_f32_scale"] = True
        if cfg.quant_type == "fp8_blockwise":
            comb["combine_fp8"] = True
        geom = self._ll128_geom
        if geom is not None and "dispatch" in geom:
            disp.update(
                ll128=True,
                combine_fp4=comb.get("combine_fp4", False),
                combine_fp8=comb.get("combine_fp8", False),
            )
            comb["ll128"] = True
        elif geom is not None:
            comb["ll128_combine_only"] = True
        return disp, comb

    def _ll128_requests(self, cfg):
        split = self._ll128_split
        if split is None:
            return None
        n = split["max_tok"]
        disp, comb = self._plan_requests(cfg)
        ll = dict(max_tok_per_rank=n, max_recv=cfg.world_size * n, ll128=True)
        (disp_b, disp_w), (comb_b, comb_w) = split["dispatch"], split["combine"]
        return (
            dict(
                disp,
                **ll,
                combine_fp4=comb.get("combine_fp4", False),
                combine_fp8=comb.get("combine_fp8", False),
                call_tok_max=n,
                block_num=disp_b,
                warp_per_block=disp_w,
            ),
            dict(comb, **ll, block_num=comb_b, warp_per_block=comb_w),
        )

    def _asyncll_segmented_unsupported(self, cfg) -> list[str]:
        bad = []
        if cfg.async_ll_transport != "sdma" or cfg.nodes > 1:
            bad.append(
                "async_ll_recv_layout='segmented' needs async_ll_transport='sdma' on one node: its "
                "transfer posts one SDMA packet per field to each peer"
            )
        if cfg.world_size < 2:
            bad.append(
                "async_ll_recv_layout='segmented' needs world_size >= 2: an empty row points at "
                "another rank's expert"
            )
        if cfg.world_size > 64:
            bad.append(
                "async_ll_recv_layout='segmented' supports world_size <= 64: the receive "
                "finalize stages one count per source in a 64-entry LDS array"
            )
        if (cfg.scale_dim * cfg.scale_type_size) % 4:
            bad.append(
                "async_ll_recv_layout='segmented' needs a whole number of dwords per scale row"
            )
        try:
            mode, _ = self._asyncll_options(cfg, self._arch)
        except ValueError:
            mode = 0  # reported when the kernels are built
        if mode == 2:
            bad.append(
                "async_ll_recv_layout='segmented' with MORI_ASYNCLL_FUSED_RECV=1: the segmented "
                "receive keeps the separate RecvTransfer"
            )
        return bad

    def _unsupported(self, cfg) -> tuple[str, ...]:
        """Everything this backend cannot do, checked before anything is built."""
        if cfg.async_ll:
            return self._asyncll_unsupported(cfg)
        if cfg.is_internode:
            return self._internode_unsupported(cfg)
        bad = []
        if cfg.dispatch_dtype not in _DISPATCH_DTYPES:
            bad.append(
                f"dispatch dtype {cfg.dispatch_dtype} (have bf16, fp32, fp8, fp4)"
            )
        if cfg.combine_dtype not in _COMBINE_DTYPES:
            bad.append(f"combine dtype {cfg.combine_dtype} (have bf16, fp32)")
        q = cfg.quant_type
        push = q in _PUSH_QUANT_TYPES
        if cfg.is_scatter and not push:
            bad.append(
                "combine_mode='scatter' (gather only, except the push quant types "
                f"{_PUSH_QUANT_TYPES})"
            )
        if q != "none" and not push:
            bad.append(f"quant_type={q!r}")
        if push:
            if not self._is1250:
                bad.append(f"quant_type={q!r} (the push combine is gfx125x only)")
            if cfg.combine_dtype != torch.bfloat16:
                bad.append(
                    f"quant_type={q!r} with combine dtype {cfg.combine_dtype} "
                    "(bf16 only)"
                )
            elif cfg.hidden_dim % 1024:
                bad.append(
                    f"quant_type={q!r} with hidden_dim={cfg.hidden_dim} "
                    "(needs a multiple of 1024)"
                )
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

    _INTERNODE_DTYPE = {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float8_e4m3fnuz: "fp8_fnuz",
        torch.float8_e4m3fn: "fp8_ocp",
        torch.float4_e2m1fn_x2: "fp4",  # packed dispatch only; see _internode_unsupported
    }

    # Pass order is the single source for both plan construction and launch.
    # Common passes share a plan across families at the same geometry.
    _INTERNODE_SEQ = {
        ("dispatch", "v2"): ("copystaging", "dispatch"),
        ("dispatch", "v2_ll"): ("copystaging", "dispatch_ll"),
        ("combine", "v2"): (
            "combinesync",
            "combinesyncbarrier",
            "combine",
            "combineall",
        ),
        ("combine", "v2_ll"): (
            "combinesync",
            "combinesyncbarrier",
            "combine_ll",
            "combineall",
        ),
    }

    def _internode_request(
        self,
        cfg,
        dtype_tag,
        block_num,
        warp_per_block,
        rdma_block_num,
        active_qps,
        num_qp_to_drain,
    ):
        # Each phase uses its own dtype. FP4 dispatch counts packed bytes,
        # while combine and the shared arena keep the logical hidden dimension.
        return dict(
            worldSize=cfg.world_size,
            hiddenDim=cfg.hidden_dim // 2 if dtype_tag == "fp4" else cfg.hidden_dim,
            scaleDim=cfg.scale_dim,
            scaleTypeSize=cfg.scale_type_size,
            maxTokenTypeSize=cfg.max_token_type_size,
            maxNumInpTokenPerRank=cfg.max_num_inp_token_per_rank,
            numExpertPerRank=cfg.num_experts_per_rank,
            numExpertPerToken=cfg.num_experts_per_token,
            maxTotalRecvTokens=cfg.max_total_recv_tokens,
            gpuPerNode=cfg.gpu_per_node,
            # The kernel's ACTIVE count; the allocation lives in the DevComm.
            numQpPerPe=active_qps,
            numQpToDrain=num_qp_to_drain,
            noRdmaAtomics=getattr(self, "_no_rdma_atomics", False),
            moeSort=bool(cfg.moe_sort_block_m),
            quantType=(
                cfg.quant_type.replace("_", "").lower()
                if cfg.quant_type != "none"
                else "none"
            ),
            dtype=dtype_tag,
            blockNum=block_num,
            warpPerBlock=warp_per_block,
            rdmaBlockNum=rdma_block_num,
            mpCount=self._multi_processor_count,
        )

    def _internode_static_args(self):
        """The window handle, the region offsets and the local pointers.

        Everything here is fixed for the life of the op, so it is built once and
        merged with the per-launch values at each call. Memoised because it is
        NOT free at launch scale: thirty-odd arena lookups and data_ptr() calls
        per pass sequence, on a path where the host is already the thing the GPU
        waits for.
        """
        cached = getattr(self, "_internode_static_cache", None)
        if cached is not None:
            return cached
        arena = self.arena
        offset = arena.offset
        has_scales = arena.has("out_scales")

        def ptr(tensor):
            return 0 if tensor is None else tensor.data_ptr()

        self._internode_static_cache = dict(
            window=arena.handle,
            offDispatchInp=offset("inter_dispatch_inp"),
            offCombineInp=offset("inter_combine_inp"),
            offStaging=offset("inter_staging"),
            offDispatchOut=offset("inter_dispatch_out"),
            offCombineOut=offset("inter_combine_out"),
            offDispatchStaging=offset("inter_dispatch_staging"),
            offInpWeights=offset("inp_weights"),
            offDispatchOutWeights=offset("dispatch_out_weights"),
            offCombineOutWeights=offset("combine_out_weights"),
            offOutIndices=offset("out_indices"),
            offRecvTokenNum=offset("recv_token_num"),
            offNodeRecvTokenNum=offset("node_recv_token_num"),
            offDispTokOffset=offset("disp_tok_offset"),
            offDispTokIdToSrcTokId=offset("disp_tok_id_to_src_tok_id"),
            offCrossDeviceBarrier=offset("cross_device_barrier"),
            offChunkFlag=offset("inter_node_chunk_flag"),
            # Absent, not zero-sized, when the transport is off; the kernel gates
            # on cfg.scaleDim and never dereferences it.
            offOutScales=offset("out_scales") if has_scales else 0,
            # AsyncLL's SDMA source word for the small count/epoch writes. Absent
            # from the other families' arenas, where the kernels never read it.
            offSendCount=offset("send_count") if arena.has("send_count") else 0,
            offCombineStaging=(
                offset("inter_combine_staging")
                if arena.has("inter_combine_staging")
                else 0
            ),
            rank=self.cfg.rank,
            lsaBase=self.lsa_base,
            sdmaQueueOffset=self._sdma_queue_offset,
            dispDestTokIdMap=0,  # per-launch: the base passes token_dest_map or routing_dest_map
            interNodeDispDestTokIdMap=ptr(self.inter_disp_dest_tok_id_map),
            interNodeDispSendMap=ptr(self.inter_disp_send_map),
            interNodeChunkFlagCombine=ptr(self.inter_chunk_flag_combine),
            destPeTokenCounter=ptr(self.dest_pe_counter),
            blockFlagCounter=ptr(self.block_flag_counter),
            totalRecvTokenNum=ptr(self.total_recv),
            dispatchGridBarrier=ptr(self.dispatch_barrier),
            interNodeBlocksBarrier=ptr(self.inter_blocks_barrier),
            crossDeviceBarrierFlag=ptr(self.cross_device_flag),
            dispReceiverIdxMap=ptr(self.disp_receiver_idx_map),
            **self._moe_sort_static_args(),
            # The HOST struct, not DevCommHandle.ptr (the device-side copy): the args
            # embed a ccoDevComm by value, so the binding memcpys from host memory.
            devComm=self._dev_comm._dev_comm.host_ptr,
        )
        return self._internode_static_cache

    def _moe_sort_static_args(self):
        if getattr(self, "moe_sort_tables", None) is None:
            return {}
        ids, weights, expert_ids, num_valid = self.moe_sort_tables
        return dict(
            offMoeSort=self.arena.offset("moe_sort"),
            sortBlockM=self.cfg.moe_sort_block_m,
            sortZeroRowBytes=self.cfg.hidden_dim * self.cfg.combine_elem_size,
            sortedTokenIds=ids.data_ptr(),
            sortedWeights=weights.data_ptr(),
            sortedExpertIds=expert_ids.data_ptr(),
            sortNumValid=num_valid.data_ptr(),
            sortZeroState=self.moe_sort_zero_rows.data_ptr(),
            sortZeroOut=self.arena.local_ptr(self._region("out_tok")),
        )

    def moe_sort(self, num_rows=None, num_experts=None):
        """aiter's moe_sorting outputs for the last dispatch (moe_sort_block_m).

        Returns (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf), each sliced to the size aiter's own sort would allocate for
        topk_ids of `num_rows` rows (default: every receive row) and an expert
        mask of `num_experts` entries (default: world_size * experts per rank).
        sorted_expert_ids holds local expert ids, as with expert_mask, and
        num_valid_ids[1] is the received row count. moe_buf is combine_in_view(),
        zero in every row fused_moe can write: pass it as fused_moe's output and
        then to combine(), which then has nothing to copy.
        """
        if getattr(self, "moe_sort_tables", None) is None:
            raise RuntimeError("moe_sort() needs moe_sort_block_m > 0")
        cfg = self.cfg
        bm = cfg.moe_sort_block_m
        m = cfg.effective_max_recv if num_rows is None else num_rows
        e = (
            cfg.world_size * cfg.num_experts_per_rank
            if num_experts is None
            else num_experts
        )
        topk = cfg.num_experts_per_token
        rows = m * topk + e * bm - topk
        rows = (rows + bm - 1) // bm * bm
        ids, weights, expert_ids, num_valid = self.moe_sort_tables
        moe_buf = self.combine_in_view()
        if rows > ids.numel() or m > moe_buf.shape[0]:
            raise ValueError(
                f"moe_sort(num_rows={m}, num_experts={e}) exceeds the tables "
                f"allocated for {moe_buf.shape[0]} rows"
            )
        return (
            ids[:rows],
            weights[:rows],
            expert_ids[: rows // bm],
            num_valid,
            moe_buf[:m],
        )

    # -- AsyncLL ---------------------------------------------------------
    #
    # Nine passes in four groups. The grouping IS the feature: dispatch_send
    # returns as soon as its three kernels are enqueued, and the ccoSdma copies
    # they posted drain while the caller runs whatever comes next.
    # The two transfer passes are named per transport; everything else is shared.
    _ASYNCLL_SEQ_TEMPLATE = {
        "dispatch_send": (
            "dispatch_slotassign",
            "dispatch_sendcopy",
            "dispatch_sendtransfer{sfx}",
        ),
        "dispatch_recv": ("dispatch_recvtransfer{sfx}", "dispatch_recvcopy"),
        "combine_send": ("combine_sendcopy", "combine_sendtransfer{sfx}"),
        "combine_recv": ("combine_recvtransfer{sfx}", "combine_recvcopy"),
    }

    # LSA's dispatch send is two passes, not three. The staging hop exists so
    # ccoSdma has a local contiguous source to hand the engine; a peer store does
    # not need one, so the gather addresses the peer itself and the bulk copy --
    # a whole pass over the payload -- is gone. The fused kernel also publishes
    # the counts, which is all SendTransferLsa did besides copying.
    _ASYNCLL_SEQ_DISPATCH_SEND_LSA = (
        "dispatch_slotassign",
        "dispatch_sendcopy_lsa",
    )

    # Segmented rounds (async_ll_recv_layout, SDMA only): the expert reads the
    # landing segments, so dispatch's RecvCopy becomes a finalize pass, and the
    # segmented expert output is sent in place, so combine has no SendCopy.
    _ASYNCLL_SEG_SEQ = {
        "dispatch_send": (
            "dispatch_slotassign",
            "dispatch_sendcopy_seg",
            "dispatch_sendtransfer_seg",
        ),
        "dispatch_recv": ("dispatch_recvtransfer", "dispatch_recvfinalize_seg"),
        "combine_send": ("combine_sendtransfer_seg",),
        "combine_recv": ("combine_recvtransfer", "combine_recvcopy_seg"),
    }

    @classmethod
    def _asyncll_seg_seq(cls):
        return dict(cls._ASYNCLL_SEG_SEQ)

    @staticmethod
    def _asyncll_options(cfg, arch):
        # Resolve once per op. Shared copy passes also serve LSA, so topology
        # alone cannot select the SDMA notification protocol in native code.
        if cfg.async_ll_transport != "sdma" or cfg.nodes != 1:
            return 0, False

        def flag(name, default):
            value = os.environ.get(name, "1" if default else "0")
            if value not in ("0", "1"):
                raise ValueError(f"{name} must be '0' or '1', got {value!r}")
            return value == "1"

        # On by default where it has been measured to win: gfx1250 (either code
        # target), and gfx950 (MI355X), where it beat legacy at every token count,
        # most with two overlapping streams.
        early = flag(
            "MORI_ASYNCLL_EARLY_NOTIFY", arch in ("gfx950", "gfx1250", "gfx1250-strict")
        )
        fused = flag("MORI_ASYNCLL_FUSED_RECV", False)
        skip_reset = flag("MORI_ASYNCLL_SKIP_RESET", early)
        if (fused or skip_reset) and not early:
            raise ValueError(
                "MORI_ASYNCLL_FUSED_RECV and MORI_ASYNCLL_SKIP_RESET require "
                "MORI_ASYNCLL_EARLY_NOTIFY=1"
            )
        # Must match EpInterNodeCfg::asyncLlMode: legacy, early, fused CTA wait.
        return (2 if fused else int(early)), skip_reset

    @staticmethod
    def _asyncll_fused_recv_ctas(mode, device_index=0, mp_count=0):
        # Grid ceiling for the fused receive copies; 0 lets the native factory
        # pick half the device. MORI_ASYNCLL_FUSED_RECV_CTAS is only a tuning
        # override: the default exists so that fusion cannot starve another
        # stream's sends into a deadlock.
        name = "MORI_ASYNCLL_FUSED_RECV_CTAS"
        value = os.environ.get(name, "0")
        try:
            ctas = int(value)
        except ValueError:
            ctas = -1
        if ctas < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if mode != 2:
            return 0
        if ctas > 0:
            return ctas
        # "Half the device" has to mean half of what this process can use. The
        # native default is mpCount / 2, and mpCount is the whole device even when
        # HSA_CU_MASK leaves the process a fraction of it: at 64 of 256 CUs two
        # 128-CTA fused grids fill every slot, and the ranks deadlock as if there
        # were no ceiling at all.
        usable = _hsa_cu_mask_cus(device_index, mp_count)
        if usable is not None and 0 < usable < mp_count:
            return max(usable // 2, 1)
        return 0

    @classmethod
    def _asyncll_seq(cls, cfg, mode=0):
        lsa = cfg.async_ll_transport == "lsa"
        sfx = "_lsa" if lsa else ""
        seq = {
            group: tuple(n.format(sfx=sfx) for n in names)
            for group, names in cls._ASYNCLL_SEQ_TEMPLATE.items()
        }
        # The unfused LSA sequence is reachable through MORI_ASYNCLL_LSA_FUSED=0
        # so the fusion can be A/B'd against it on one box in one container.
        # Comparing against the pre-fusion numbers recorded elsewhere would be
        # comparing across machines, which on these parts is enough to move a
        # single-digit effect either way.
        if lsa and os.environ.get("MORI_ASYNCLL_LSA_FUSED", "1") != "0":
            seq["dispatch_send"] = cls._ASYNCLL_SEQ_DISPATCH_SEND_LSA
        if mode == 2:
            seq["dispatch_recv"] = ("dispatch_recvcopy",)
            seq["combine_recv"] = ("combine_recvcopy",)
        return seq

    # Which leg's dtype each pass is compiled for, the rule
    # _build_internode_kernels applies per phase: compiling a combine pass with
    # the dispatch dtype is how an fp8 dispatch ends up writing half the bytes a
    # bf16 combine view reads.
    @staticmethod
    def _asyncll_leg(group):
        return "dispatch" if group.startswith("dispatch") else "combine"

    def _asyncll_request(self, cfg, dtype_tag, block_num, warp_per_block):
        # AsyncLL stripes over its active QPs and quiets them itself; the v2
        # drain count is not read by its kernels.
        request = self._internode_request(
            cfg,
            dtype_tag,
            block_num,
            warp_per_block,
            rdma_block_num=0,
            active_qps=tuning.resolve_active_qps(cfg),
            num_qp_to_drain=0,
        )
        # The AsyncLL specs derive every grid themselves -- the copy passes from
        # mpCount, the transfer passes from worldSize -- so the device's CU count
        # has to reach them. The internode request never carries it because its
        # geometry table supplies grids directly.
        request["mpCount"] = self._multi_processor_count
        # AsyncLL has no write-signalling variant; cross-node ones are refused in the gate.
        request["noRdmaAtomics"] = False
        request["moeSort"] = False
        request["asyncLlMode"] = self._asyncll_mode
        request["asyncLlFusedRecvCtas"] = self._asyncll_fused_ctas
        return request

    def _build_asyncll_kernels(self, cfg) -> KernelSet:
        from mori.jit.v2 import plan_api

        leg_dtype = {
            "dispatch": self._INTERNODE_DTYPE[cfg.dispatch_dtype],
            "combine": self._INTERNODE_DTYPE[cfg.combine_dtype],
        }
        block_num = cfg.dispatch_block_num
        warp_per_block = cfg.warp_num_per_block

        plans_by_pass = {}
        # The second value is MORI_ASYNCLL_SKIP_RESET, which no longer selects
        # anything: main dropped the host total_recv reset it skipped (and
        # KernelSet.self_resets_counters with it), and AsyncLL's RecvCopy
        # always assigned the count. Still parsed so existing settings validate.
        self._asyncll_mode, _ = self._asyncll_options(cfg, self._arch)
        self._asyncll_fused_ctas = self._asyncll_fused_recv_ctas(
            self._asyncll_mode, self._device_index, self._multi_processor_count
        )
        sequences = self._asyncll_seq(cfg, self._asyncll_mode)
        seg_sequences = self._asyncll_seg_seq() if cfg.async_ll_segmented else {}
        for group, names in [*sequences.items(), *seg_sequences.items()]:
            for name in names:
                if name in plans_by_pass:
                    continue
                request = self._asyncll_request(
                    cfg, leg_dtype[self._asyncll_leg(group)], block_num, warp_per_block
                )
                plan = cb.asyncll_plan(name)(**request)
                # Neither varies for AsyncLL: there is one geometry per op, and
                # this backend has no replay path. Bound rather than passed so the
                # value is written into the plan's struct once instead of per
                # launch.
                plan.bind(rdmaBlockNum=0, replayMode=0)
                plans_by_pass[name] = plan
                self._plans.append(plan)
        self._asyncll_groups = {
            group: plan_api.make_launch_group([plans_by_pass[n] for n in names])
            for group, names in sequences.items()
        }
        self._asyncll_groups_seg = {
            group: plan_api.make_launch_group([plans_by_pass[n] for n in names])
            for group, names in seg_sequences.items()
        }
        self._seg_layout = asyncll_seg_layout(cfg) if cfg.async_ll_segmented else None

        # AsyncLL compiles one geometry -- its specs derive each pass's grid from
        # mpCount and worldSize, so (block, warp) only sets the block width.
        # resolve_schedule gives it a single entry, so each table holds that one
        # spec and _pick returns it for every token count.
        def table(group):
            specs = (
                self._dispatch_specs
                if group.startswith("dispatch")
                else self._combine_specs
            )
            fn = self._wrap_asyncll(group)
            return {spec: fn for spec in specs}

        both = {
            "dispatch": self._wrap_asyncll("dispatch_send", "dispatch_recv"),
            "combine": self._wrap_asyncll("combine_send", "combine_recv"),
        }
        return KernelSet(
            # dispatch()/combine() stay available and simply run both halves back
            # to back, so a caller that does not want the seam sees the ordinary
            # API and the parity tests need no special case.
            dispatch={spec: both["dispatch"] for spec in self._dispatch_specs},
            combine={spec: both["combine"] for spec in self._combine_specs},
            dispatch_send=table("dispatch_send"),
            dispatch_recv=table("dispatch_recv"),
            combine_send=table("combine_send"),
            combine_recv=table("combine_recv"),
            dispatch_replay=None,
            stages_in_kernel=True,
            capabilities=frozenset({"gather", "scales", "async_ll"}),
        )

    def _wrap_asyncll(self, *groups):
        """Launch one or more AsyncLL pass groups against one filled arg struct.

        Same two-field memory as _wrap_internode and for the same reason: combine
        reads `tokenIndices` and the weights region that DISPATCH was handed, but
        the base's combine signature carries neither.
        """

        def run(*, input, num_tokens, dest_map, **kwargs):
            phase = "dispatch" if groups[0].startswith("dispatch") else "combine"
            stride = self._seg_stride
            if stride and "combine_send" in groups:
                input = self._seg_combine_input(input, stride)
            if phase == "dispatch":
                if groups[0] == "dispatch_recv":
                    # dispatch_recv() launches this half with only input/dest_map/
                    # num_tokens, but its copy pass gates the weight unpack on
                    # weightsBuf: reuse what the send half was handed, or the recv
                    # weights view is never written.
                    weights = getattr(self, "_asyncll_weights", None)
                    indices = getattr(self, "_asyncll_indices", None)
                else:
                    weights, indices = kwargs.get("weights"), kwargs.get("indices")
                    self._asyncll_weights = weights
                self._asyncll_has_weights = weights is not None
                self._asyncll_indices = indices
                weights_ptr = 0 if weights is None else weights.data_ptr()
                if (
                    weights_ptr == 0
                    and weights is not None
                    and "dispatch_recv" in groups
                ):
                    # A rank with no tokens of its own (empty weights, null data_ptr)
                    # still receives its peers' tokens, and the recv copy pass only
                    # uses weightsBuf as the gate for unpacking their weights. Keep
                    # it non-null; the send half reads nothing with 0 tokens.
                    weights_ptr = self.recv_weights().data_ptr()
            else:
                # AsyncLL does not fold weights on combine, and says so rather
                # than returning a zero tensor that looks like an answer.
                #
                # The fold the internode family does (EpCombineAll's second
                # WarpAccum) reads the weights back out of the COMBINE staging,
                # which carries topk floats per token alongside the hidden row.
                # AsyncLL's combine payload is hidden-only by design -- see
                # asyncll_regions -- so there is nothing to accumulate, and
                # widening it would add topk*4 bytes per token to the one
                # transfer this kernel family exists to make cheap. v1's AsyncLL
                # makes the same trade: low_latency_async.cpp writes weights on
                # dispatch and never on combine.
                if kwargs.get("want_weights", False):
                    raise NotImplementedError(
                        "AsyncLL combine does not produce output weights: its "
                        "combine payload carries no per-token metadata (v1's "
                        "AsyncLL is the same). Call combine() with weights=None, "
                        "or use a non-AsyncLL kernel family if you need the fold."
                    )
                weights_ptr = 0
                indices = getattr(self, "_asyncll_indices", None)

            # Only what varies; everything fixed for the op's lifetime was bound
            # onto the plans in _alloc_internode_buffers.
            args = dict(
                curRankNumToken=num_tokens,
                dispDestTokIdMap=dest_map.data_ptr(),
                tokenIndices=0 if indices is None else indices.data_ptr(),
                inpTokenBuf=input.data_ptr(),
                weightsBuf=weights_ptr,
                scalesBuf=(
                    kwargs["scales"].data_ptr()
                    if kwargs.get("scales") is not None
                    else 0
                ),
            )
            if self.cfg.async_ll_segmented:
                # Bit 63 marks a segmented-capable op, so compact rounds on it
                # still exchange the mode word (asyncll::SegModeWord).
                args["segStride"] = SEG_CAPABLE_BIT | stride
            table = self._asyncll_groups_seg if stride else self._asyncll_groups
            stream = _raw_stream(self._device_index)
            for group in groups:
                table[group].launch(stream, **args)

        return run

    def _reachable_schedule(self, cfg):
        """Schedule rows a launch can select: those starting below the capacity."""
        rows = []
        for schedule in (self._schedule, self._ll_schedule):
            lower = 0
            for row in schedule:
                if lower >= cfg.max_num_inp_token_per_rank:
                    break
                rows.append(row)
                lower = row.max_tokens if row.max_tokens is not None else lower
        return rows

    def _set_segment_stride(self, stride, num_tokens):
        if not stride:
            self._seg_stride = 0
            return
        cfg = self.cfg
        if not cfg.async_ll_segmented:
            raise ValueError(
                "segment_stride needs an op built with async_ll_recv_layout='segmented'"
            )
        # Only the receive cap is checked here. It is the same on every rank, so a
        # stride past it fails on every rank before any of them launches -- as long
        # as S itself is the same on every rank, which the contract requires: compute
        # it from values every rank shares (e.g. the max token count after the DP
        # all-gather). A rank that alone passes a stride past the cap raises here
        # while the others enter the round and wait for it. The
        # field blocks are sized by max_num_inp_token_per_rank rows per source and
        # the reverse map and the expert output by effective_max_recv_per_rank,
        # which is at most that, so the cap is the binding bound.
        #
        # A stride below this rank's token count is NOT refused: the other ranks
        # would launch the round and wait in the receive forever. The send pass
        # drops the rows past the stride instead and counts the round as an
        # overflow (segment_error_counts()), so the caller sees a wrong round, not
        # a hang.
        cap = cfg.effective_max_recv_per_rank
        if not 0 < stride <= cap:
            raise ValueError(
                f"segment_stride={stride} must be positive and within the per-source "
                f"receive cap {cap}"
            )
        self._seg_stride = int(stride)

    def _seg_combine_input(self, input, stride):
        """The expert output a segmented combine sends: world_size * stride rows
        in the dispatch's segment layout, sent from the arena view."""
        rows = self.cfg.world_size * stride
        if input.shape[0] != rows:
            raise ValueError(
                f"a segmented combine takes the segmented expert output: "
                f"{rows} rows (world_size x stride {stride}), got {input.shape[0]}"
            )
        view = self.combine_in_view()
        if input.data_ptr() != view.data_ptr():
            view.copy_(input)
        return view

    def _seg_words_view(self, key, first, n):
        view = self._views.get(key)
        if view is None:
            ptr = self.arena.local_ptr("send_count") + first * 8
            view = from_gpu_ptr(ptr, (n,), torch.int64)
            self._views[key] = view
        return view

    def segment_error_counts(self):
        """(overflow, mismatch): cumulative counts of each segmented-layout error on
        this rank since the op was built or last reset(), counted per peer and
        round.

        overflow: a segmented round where this rank had more rows for a peer than
        the stride, i.e. it was called with segment_stride below its own token
        count. The rows past the stride were dropped. Only the rank that dropped
        rows counts it; its peers see a short segment and count nothing. mismatch: a
        peer's round, segmented or compact, did not use this rank's mode and stride,
        so rows were read at the wrong offsets. Every rank counts that.

        So the counts, and segment_error_flags(), differ between ranks: reduce them
        across ranks (e.g. an all-reduce max) before acting on them, such as
        falling back to compact, or the ranks diverge in the next round.

        Use this for totals (e.g. a periodic health check that diffs two reads);
        segment_error_flags() answers "anything new since I last looked". Reads
        device memory, so it synchronizes."""
        if not self.cfg.async_ll_segmented:
            return (0, 0)
        first = asyncll_seg_words(self.cfg.world_size)["errors"]
        overflow, mismatch = self._seg_words_view("seg_errors", first, 2).tolist()
        return (int(overflow), int(mismatch))

    def segment_error_flags(self, clear=False):
        """Error bits since the last clear, 0 when none: SEGMENT_ERR_OVERFLOW (1)
        and SEGMENT_ERR_MISMATCH (2), see segment_error_counts(). clear=True
        moves the baseline to now. The device only ever adds to the counters, so
        nothing set while this runs is lost. Synchronizes."""
        counts = self.segment_error_counts()
        base = getattr(self, "_seg_error_base", (0, 0))
        flags = 0
        if counts[0] > base[0]:
            flags |= SEGMENT_ERR_OVERFLOW
        if counts[1] > base[1]:
            flags |= SEGMENT_ERR_MISMATCH
        if clear:
            self._seg_error_base = counts
        return flags

    def segment_counts(self):
        """Device int64 [world_size]: n_s, the rows source s sent this rank in the
        last segmented round (valid rows [s * S, s * S + n_s) of the recv views).
        Meaningful after dispatch_recv() of a segmented round; None on an op
        without the segmented layout."""
        if not self.cfg.async_ll_segmented:
            return None
        first = asyncll_seg_words(self.cfg.world_size)["counts"]
        return self._seg_words_view("seg_counts", first, self.cfg.world_size)

    def _build_internode_kernels(self, cfg) -> KernelSet:
        from mori.jit.v2 import plan_api

        # set_active_qps() swaps the count of whatever spec a bucket picks, so
        # each selectable count is built for every spec.
        specs_by_phase = {
            phase: tuple(
                dict.fromkeys(
                    variant
                    for spec in specs
                    for variant in (
                        spec,
                        *(
                            spec._replace(active_qps=count)
                            for count in self._selectable_active_qps
                        ),
                    )
                )
            )
            for phase, specs in (
                ("dispatch", self._dispatch_specs),
                ("combine", self._combine_specs),
            )
        }
        # Dispatch drains every QP a launch of this op can send on; buckets above
        # the token capacity are never launched, so they do not count.
        num_qp_to_drain = max(
            [
                spec.active_qps
                for row in self._reachable_schedule(cfg)
                for spec in (row.dispatch, row.combine)
            ]
            + list(self._selectable_active_qps)
        )
        plans = {}
        kernels = {"dispatch": {}, "combine": {}}
        for phase, dtype in (
            ("dispatch", cfg.dispatch_dtype),
            ("combine", cfg.combine_dtype),
        ):
            for spec in specs_by_phase[phase]:
                request = self._internode_request(
                    cfg,
                    self._INTERNODE_DTYPE[dtype],
                    spec.block_num,
                    spec.warp_per_block,
                    spec.rdma_block_num,
                    spec.active_qps,
                    num_qp_to_drain,
                )
                group_plans = []
                for pass_name in self._INTERNODE_SEQ[phase, spec.family]:
                    # Common passes share a plan across families at one geometry.
                    # Only the ordinary combine pass compiles the load-first
                    # gather, so only its key and request carry the flag.
                    load_first = pass_name == "combine" and spec.load_first
                    key = (
                        pass_name,
                        spec.block_num,
                        spec.warp_per_block,
                        spec.rdma_block_num,
                        spec.active_qps,
                        load_first,
                    )
                    if key not in plans:
                        plan = cb.EP_INTERNODE_PLANS[pass_name](
                            **request, combineLoadFirst=load_first
                        )
                        self._plans.append(plan)
                        plan.bind(rdmaBlockNum=spec.rdma_block_num, replayMode=0)
                        plans[key] = plan
                    group_plans.append(plans[key])
                group = plan_api.make_launch_group(group_plans)
                kernels[phase][spec] = self._wrap_internode(phase, group)

        return KernelSet(
            dispatch=kernels["dispatch"],
            combine=kernels["combine"],
            dispatch_replay=None,
            stages_in_kernel=True,
            capabilities=frozenset({"gather", "scales", "internode"}),
        )

    def kernel_config(self, num_tokens, max_tokens=None) -> tuning.ScheduleEntry:
        """Selected HIP configurations for both phases, using this call's tokens.

        This reads the schedule prepared at construction: no table reload,
        compilation or device query. Intra and inter both select from it. On
        gfx125x, a call the low_latency rule sends to LL128 launches the LL128
        kernels instead (except a snapshot call under a push quant type), which
        this does not report. AsyncLL reports active_qps as None; its kernels
        run with resolve_active_qps(cfg). A set_active_qps() override replaces
        both phases' active count.
        """
        schedule = self._schedule
        if self._ll_schedule:
            family = tuning.internode_kernel_family(self.cfg, max_tokens=max_tokens)
            if family == "v2_ll":
                schedule = self._ll_schedule
        selected = tuning.pick_bucket(schedule, num_tokens)
        override = getattr(self, "_active_qps_override", None)
        if override is None:
            return selected
        return selected._replace(
            dispatch=selected.dispatch._replace(active_qps=override),
            combine=selected.combine._replace(active_qps=override),
        )

    def _pick(self, num_tokens, snapshot=False, max_tokens=None):
        split = getattr(self, "_ll128_split", None)
        if (
            split is not None
            and max_tokens is not None
            and max_tokens <= split["max_tok"]
            and not (snapshot and self.cfg.quant_type in _PUSH_QUANT_TYPES)
        ):
            return _LL_SPEC, _LL_SPEC
        selected = self.kernel_config(num_tokens, max_tokens)
        return selected.dispatch, selected.combine

    def _peak_limit(self, snapshot):
        split = getattr(self, "_ll128_split", None)
        if split is not None and not (
            snapshot and self.cfg.quant_type in _PUSH_QUANT_TYPES
        ):
            return split["max_tok"]
        if getattr(self, "_ll_schedule", ()):
            return self.cfg.internode_auto_ll_max_tokens
        return None

    def _wrap_internode(self, phase, group):
        """One ABI crossing for the whole pass sequence.

        Every pass shares the schema and the same filled struct, so the arguments
        are written once and each plan launches against them in order: one ABI
        crossing for the whole sequence instead of one per pass.
        """

        def run(*, input, num_tokens, dest_map, **kwargs):
            # Two of the kernel's arguments are set by dispatch and READ AGAIN by
            # combine, but the base only hands them to dispatch -- combine's
            # signature carries no indices and treats `weights` as a request
            # rather than an input. v1 does not notice because its handle keeps
            # both across the pair; here they have to be remembered explicitly.
            #
            #   tokenIndices  EpCombineAllGeneric dereferences it unconditionally
            #                 (args.tokenIndices[tokenId * topk + laneId]), so a
            #                 null is a fault, not a skipped branch.
            #   weightsBuf    it is a DIFFERENT tensor in each phase, and its
            #                 NULLNESS is the kernel's only gate on the fold. It
            #                 also sets `combXferBytes = hidden + (weights ? wt :
            #                 0)`, but only the combine passes index that, so the
            #                 phases need not agree on null-ness.
            #
            # Dispatch's weightsBuf is the caller's per-input-token weights and is
            # indexed by source token id. Combine's is indexed by RECEIVED token
            # id (`CombineSync` stages `weightsBuf + tokenId * topk` for tokenId in
            # [0, totalRecvTokenNum)), so it must be the weights dispatch just
            # delivered -- the `dispatch_out_weights` region, which is exactly the
            # `dispatch_weights` output v1's callers hand back to `combine()`.
            # Passing the input weights here does not fault: it folds the right
            # number of node slots holding other tokens' weights.
            if phase == "dispatch":
                weights, indices = kwargs.get("weights"), kwargs.get("indices")
                if weights is None and self.cfg.moe_sort_block_m:
                    raise ValueError("moe_sort_block_m needs dispatch weights")
                self._internode_has_weights = weights is not None
                # The TENSOR, not just its address: combine dereferences this
                # pointer on a LATER call, and nothing else keeps the caller's
                # indices alive that long.
                self._internode_indices = indices
                weights_ptr = 0 if weights is None else weights.data_ptr()
            elif kwargs.get("want_weights", False):
                # The fold's source is what dispatch delivered, so there is nothing
                # to fold if it delivered none -- and the region still holds the
                # previous round's values, which would be returned as this one's.
                if not getattr(self, "_internode_has_weights", False):
                    raise ValueError(
                        "combine(weights=...) asks for the weight fold, but the "
                        "preceding dispatch carried no weights: the internode "
                        "combine folds the weights dispatch delivered, not the "
                        "argument"
                    )
                weights_ptr = self.arena.local_ptr("dispatch_out_weights")
            else:
                weights_ptr = 0
            held_indices = getattr(self, "_internode_indices", None)
            indices_ptr = 0 if held_indices is None else held_indices.data_ptr()

            # Static pointers, rdmaBlockNum and replayMode are already bound.
            args = dict(
                curRankNumToken=num_tokens,
                dispDestTokIdMap=dest_map.data_ptr(),
                tokenIndices=indices_ptr,
                inpTokenBuf=input.data_ptr(),
                weightsBuf=weights_ptr,
                scalesBuf=(
                    kwargs["scales"].data_ptr()
                    if kwargs.get("scales") is not None
                    else 0
                ),
            )
            group.launch(_raw_stream(self._device_index), **args)

        return run

    def _build_kernels(self, cfg, arena) -> KernelSet:
        bad = self._unsupported(cfg)
        if bad:
            return KernelSet(dispatch={}, combine={}, unsupported=bad)

        self._plans = []
        if cfg.async_ll:
            return self._build_asyncll_kernels(cfg)
        if cfg.is_internode:
            return self._build_internode_kernels(cfg)

        disp_req, comb_req = self._plan_requests(cfg)
        bound = dict(arena=arena, region_names=_REGIONS)
        # One plan per (block, warp) the schedule can select. Compilation happens
        # here and only here, so _pick never touches the compiler.
        dispatch, combine = {}, {}
        for spec in self._dispatch_specs:
            sized = []
            for cap in self._dispatch_caps(cfg, spec):
                plan = cb.EpDispatchPlan(
                    **disp_req,
                    **bound,
                    block_num=spec.block_num,
                    warp_per_block=spec.warp_per_block,
                    call_tok_max=cap,
                )
                plan.bind(rank=cfg.rank)
                if self.dispatch_staging is not None:
                    plan.bind(staging_base=self.dispatch_staging)
                self._plans.append(plan)
                sized.append((cap, plan))
            dispatch[spec] = self._wrap_dispatch(sized)
        for spec in self._combine_specs:
            plan = cb.EpCombinePlan(
                **comb_req,
                **bound,
                block_num=spec.block_num,
                warp_per_block=spec.warp_per_block,
            )
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            combine[spec] = self._wrap_combine(plan)
        push = {}
        for spec in self._push_specs:
            plan = cb.EpCombinePlan(
                **comb_req,
                **bound,
                combine_bf16_push=True,
                block_num=spec.block_num,
                warp_per_block=spec.warp_per_block,
            )
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            push[spec] = self._wrap_combine(plan)
        if push:
            combine = {
                spec: self._pull_or_push(run, push) for spec, run in combine.items()
            }
        ll = self._ll128_requests(cfg)
        if ll is not None:
            ll_bound = dict(arena=arena, region_names=self._ll128_split["region_names"])
            plan = cb.EpDispatchPlan(**ll[0], **ll_bound)
            plan.bind(rank=cfg.rank)
            if self.dispatch_staging is not None:
                plan.bind(staging_base=self.dispatch_staging)
            self._plans.append(plan)
            n = self._ll128_split["max_tok"]
            dispatch[_LL_SPEC] = self._wrap_dispatch([(n, plan)], ll=True)
            plan = cb.EpCombinePlan(**ll[1], **ll_bound)
            plan.bind(rank=cfg.rank)
            self._plans.append(plan)
            combine[_LL_SPEC] = self._wrap_combine(plan, ll=True)

        return KernelSet(
            dispatch=dispatch,
            combine=combine,
            dispatch_replay=None,  # no replay path in this backend
            # The combine kernel stages into out_tok itself (and skips the copy
            # when the caller already wrote there), so the op must not do it.
            stages_in_kernel=True,
            capabilities=frozenset(
                {"gather", "scales"} | ({"scatter"} if cfg.is_scatter else set())
            ),
        )

    def reset(self):
        super().reset()
        self._recv.zero_()
        if self._ll_recv is not None:
            self._ll_recv.zero_()
        # The arena zero above also cleared the segment error counters, so the
        # segment_error_flags() baseline restarts with them.
        self._seg_error_base = (0, 0)

    def _close_backend(self):
        for plan in getattr(self, "_plans", ()):
            plan.close()
        # After the plans: they carry staging_base by value, so nothing may still be
        # able to launch when the tensor behind it goes.
        self.dispatch_staging = None
        # After the plans: they embed the ccoDevComm by value and their kernels
        # dereference its QPs. The static args cache the host struct's address, so
        # it goes too -- nothing may re-read it once the handle is gone.
        dev_comm = getattr(self, "_dev_comm", None)
        if dev_comm is not None:
            self._dev_comm = None
            self._internode_static_cache = None
            dev_comm.close()
        if getattr(self, "_holds_sdma_queue_offset", False):
            self._holds_sdma_queue_offset = False
            _give_back_sdma_queue_offset(self._device_index, self._sdma_queue_offset)

    # -- views (same contract as the FlyDSL backend) -----------------------

    # Region names for the two arena layouts. The views below are one contract
    # over both, so the only thing that varies is which name each maps to.
    _VIEW_REGION = {
        "disp_out": ("disp_out", "inter_dispatch_out"),
        "out_tok": ("out_tok", "inter_combine_inp"),
        "out_wts": ("out_wts", "dispatch_out_weights"),
        "out_idx": ("out_idx", "out_indices"),
        "out_scales": ("out_scales", "out_scales"),
        "recv_to_src_token": ("recv_to_src_token", "disp_tok_id_to_src_tok_id"),
    }

    # AsyncLL overrides where it differs from the internode layout. Only one
    # region does: the combine input needs to be somewhere peers never write.
    # See asyncll_regions.
    _ASYNCLL_VIEW_REGION = {"out_tok": "inter_dispatch_staging"}

    def _region(self, name):
        intranode, internode = self._VIEW_REGION[name]
        # AsyncLL's arena is the internode one with different extents, so it uses
        # the internode names even on a single node.
        if self.cfg.async_ll:
            return self._ASYNCLL_VIEW_REGION.get(name, internode)
        return internode if self.cfg.is_internode else intranode

    def _recv_region(self, name):
        if self._ll_last:
            return "ll_" + name, self._ll_recv_cap
        return self._region(name), self._recv_cap

    def _seg_view(self, key, region, offset, cols, dtype, per_source=None):
        """Rows [0, world_size * stride) of a segmented field block of
        per_source rows per source (default max_num_inp_token_per_rank)."""
        view = self._views.get(key)
        if view is None:
            per_source = per_source or self.cfg.max_num_inp_token_per_rank
            rows = self.cfg.world_size * per_source
            ptr = self.arena.local_ptr(region) + offset
            view = from_gpu_ptr(ptr, (rows, cols), dtype)
            self._views[key] = view
        return view[: self.cfg.world_size * self._seg_stride]

    def recv_tokens(self):
        if self._seg_stride:
            offset, cols = self._seg_layout["hidden"], self.cfg.hidden_dim
            dtype = self.cfg.dispatch_dtype
            return self._seg_view(
                "seg_tokens", "inter_dispatch_inp", offset, cols, dtype
            )
        key = "ll_recv_tokens" if self._ll_last else "recv_tokens"
        view = self._views.get(key)
        if view is not None:
            return view
        # fp4 packs 2 e2m1 per element of the torch dtype -> last dim is hidden/2.
        cols = self.cfg.hidden_dim // 2 if self.cfg.is_fp4 else self.cfg.hidden_dim
        region, rows = self._recv_region("disp_out")
        view = from_gpu_ptr(
            self.arena.local_ptr(region), (rows, cols), self.cfg.dispatch_dtype
        )
        self._views[key] = view
        return view

    def combine_in_view(self):
        if self._seg_stride:
            cols, dtype = self.cfg.hidden_dim, self.cfg.combine_dtype
            # inter_dispatch_staging holds effective_max_recv_per_rank rows per
            # source, the same as on a compact op (internode_regions.py).
            return self._seg_view(
                "seg_combine_in",
                "inter_dispatch_staging",
                0,
                cols,
                dtype,
                per_source=self.cfg.effective_max_recv_per_rank,
            )
        view = self._views.get("combine_in")
        if view is None:
            view = from_gpu_ptr(
                self.arena.local_ptr(self._region("out_tok")),
                (self._recv_cap, self.cfg.hidden_dim),
                self.cfg.combine_dtype,
            )
            self._views["combine_in"] = view
        return view

    def recv_weights(self):
        if self._seg_stride:
            offset, topk = self._seg_layout["weight"], self.cfg.num_experts_per_token
            region = "inter_dispatch_inp"
            return self._seg_view("seg_weights", region, offset, topk, torch.float32)
        key = "ll_recv_weights" if self._ll_last else "recv_weights"
        view = self._views.get(key)
        if view is None:
            region, rows = self._recv_region("out_wts")
            view = from_gpu_ptr(
                self.arena.local_ptr(region),
                (rows, self.cfg.num_experts_per_token),
                torch.float32,
            )
            self._views[key] = view
        return view

    def recv_indices(self):
        if self._seg_stride:
            offset, topk = self._seg_layout["index"], self.cfg.num_experts_per_token
            region = "inter_dispatch_inp"
            return self._seg_view("seg_indices", region, offset, topk, torch.int32)
        key = "ll_recv_indices" if self._ll_last else "recv_indices"
        view = self._views.get(key)
        if view is None:
            region, rows = self._recv_region("out_idx")
            view = from_gpu_ptr(
                self.arena.local_ptr(region),
                (rows, self.cfg.num_experts_per_token),
                torch.int32,
            )
            self._views[key] = view
        return view

    def recv_scales(self):
        """The forwarded scale rows, or None when the transport is off -- the same
        answer FlyDSL gives, and the same (recv_cap, dwords) int32 view, so a caller
        cannot tell the backends apart.

        The rows sit scale_stride_bytes() apart -- 128 B-padded on the intranode
        path, packed on the internode one. Anything reading the region by pointer
        needs that pitch, not this shape.
        """
        n_i32 = self._scale_i32(self.cfg)
        if not n_i32:
            return None
        if self._seg_stride:
            # Packed rows, a whole number of dwords (the segmented gate checks).
            offset = self._seg_layout["scale"]
            region = "inter_dispatch_inp"
            return self._seg_view("seg_scales", region, offset, n_i32, torch.int32)
        key = "ll_recv_scales" if self._ll_last else "recv_scales"
        view = self._views.get(key)
        if view is not None:
            return view
        stride_i32 = self.scale_stride_bytes() // 4
        region, n = self._recv_region("out_scales")
        rows = from_gpu_ptr(
            self.arena.local_ptr(region),
            (n, stride_i32),
            torch.int32,
        )
        view = rows[:, :n_i32]
        self._views[key] = view
        return view

    def local_expert_count(self):
        raise NotImplementedError(
            "local_expert_count is flydsl-only; use backend='flydsl'"
        )

    def convert_dispatch_output(self):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    def convert_combine_input(self, routing):
        raise NotImplementedError("StdMoE is flydsl-only; use backend='flydsl'")

    # -- ops ---------------------------------------------------------------

    def set_active_qps(self, active_qps):
        """Override the active QP count of every bucket, both phases, for the
        launches that follow; None goes back to what each bucket resolved.

        Only counts listed in cfg.active_qp_counts can be selected -- they are
        the ones compiled for every bucket geometry. Rank-local: ranks, and the
        dispatch and combine of one pair, may use different counts, because a
        combine's barrier contribution does not depend on it (see
        kCombineBarrierMarkerTotal). A captured graph keeps the plans it was
        captured with.
        """
        if not self.cfg.is_internode:
            raise ValueError("active_qps requires the internode HIP backend")
        if active_qps is not None and (
            type(active_qps) is not int or active_qps not in self._selectable_active_qps
        ):
            raise ValueError(
                f"active_qps={active_qps!r} was not compiled; selectable counts "
                f"are active_qp_counts={self._selectable_active_qps}"
            )
        self._active_qps_override = active_qps

    # -- kernel adapters: the ctypes plan -> the base's named convention --

    def _dispatch_caps(self, cfg, spec):
        cap = int(cfg.max_num_inp_token_per_rank)
        if (
            not self._is1250
            or not cfg.schedule
            or cfg.world_size < 16
            or (self._ll128_geom is not None and "dispatch" in self._ll128_geom)
        ):
            return [cap]
        caps = {
            cap if row.max_tokens is None else min(int(row.max_tokens), cap)
            for row in self._schedule
            if row.dispatch == spec
        }
        if not caps or self._schedule[-1].dispatch == spec:
            caps.add(cap)
        return sorted(caps)

    def _wrap_dispatch(self, sized, ll=False):
        def run(
            *,
            input,
            indices,
            weights,
            scales,
            dest_map,
            num_tokens,
        ):
            plan = next((p for c, p in sized if num_tokens <= c), None)
            if plan is None:
                raise ValueError(
                    f"dispatch of {num_tokens} tokens is past this variant's largest "
                    f"call size {sized[-1][0]}"
                )
            count = self._ll_recv if ll else self._recv
            plan.launch(
                stream=torch.cuda.current_stream().cuda_stream,
                token_indices=indices,
                inp_token_buf=input,
                weights_buf=weights,
                scales_buf=scales,
                disp_dest_tok_id_map=dest_map,
                dest_pe_token_counter=self.dest_pe_counter,
                total_recv_token_num=count,
                grid_barrier=self.dispatch_barrier,
                num_tokens=num_tokens,
            )
            self.total_recv = count
            self._ll_last = ll

        return run

    def _wrap_combine(self, plan, ll=False):
        push = self.cfg.quant_type in _PUSH_QUANT_TYPES
        src_ptr = self.arena.local_ptr(
            "ll_recv_to_src_token" if ll else "recv_to_src_token"
        )
        idx_ptr = self.arena.local_ptr("ll_out_idx" if ll else "out_idx")
        clears = ll or self._ll128_geom is not None

        def run(
            *,
            input,
            dest_map,
            total_recv,
            num_tokens,
            want_weights=False,
            recv_to_src=None,
            recv_idx=None,
        ):
            if want_weights and push:
                raise NotImplementedError(
                    f"quant_type={self.cfg.quant_type!r} folds no weights; call combine "
                    "with weights=None"
                )
            own = total_recv is self._recv or total_recv is self._ll_recv
            if clears and not own:
                total_recv = total_recv.clone()
            plan.launch(
                stream=torch.cuda.current_stream().cuda_stream,
                inp_token_buf=input,
                out_token_buf=self.combine_out,
                # Null == "skip the weight fold" (the kernel's only gate on it).
                out_weights_buf=self.combine_out_weights if want_weights else None,
                disp_dest_tok_id_map=dest_map,
                dest_pe_token_counter=self.dest_pe_counter,
                total_recv_token_num=total_recv,
                grid_barrier=self.combine_barrier,
                xdb_flag=self.cross_device_flag,
                combine_barrier_fan=self.combine_barrier_fan,  # None on non-gfx1250 -> 0
                recv_to_src=src_ptr if recv_to_src is None else recv_to_src.data_ptr(),
                recv_idx=idx_ptr if recv_idx is None else recv_idx.data_ptr(),
                num_tokens=num_tokens,
            )

        return run

    def _pull_or_push(self, pull, push):
        routes = tuple(
            (mx, push[tuning.KernelConfig("intra", b, w)])
            for mx, b, w in self._bf16_push
        )

        def run(
            *,
            input,
            dest_map,
            total_recv,
            num_tokens,
            want_weights=False,
            recv_to_src=None,
            recv_idx=None,
        ):
            if want_weights or total_recv is not self._recv:
                return pull(
                    input=input,
                    dest_map=dest_map,
                    total_recv=total_recv,
                    num_tokens=num_tokens,
                    want_weights=want_weights,
                    recv_to_src=recv_to_src,
                    recv_idx=recv_idx,
                )
            return tuning.pick_bucket(routes, num_tokens)[1](
                input=input,
                dest_map=dest_map,
                total_recv=total_recv,
                num_tokens=num_tokens,
            )

        return run

    def push_landing(self):
        if self.cfg.quant_type not in _PUSH_QUANT_TYPES and not self._push_specs:
            raise ValueError(
                f"no landing rows: quant_type {self.cfg.quant_type!r} is not a push one"
            )
        view = self._views.get("push_landing")
        if view is None:
            info = next((i for i in self._comb_infos if "pushLandRows" in i), None)
            if info is None:
                raise RuntimeError(
                    f"no landing rows: none of this op's {len(self._comb_infos)} "
                    "combine plans reports a push landing (info['pushLandRows'])"
                )
            view = from_gpu_ptr(
                self.arena.local_ptr("out_tok") + info["pushLandOff"],
                (info["pushLandRows"], info["pushWireBytes"]),
                torch.uint8,
            )
            self._views["push_landing"] = view
        return view
