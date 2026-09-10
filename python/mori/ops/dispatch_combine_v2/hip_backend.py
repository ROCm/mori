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
  leg (no fp4). "v2" and "v2_ll" are two distinct kernel families rather than
  a runtime branch: ``cfg.internode_kernel`` (auto | v2 | v2_ll) says which are
  compiled, and only "auto" compiles both and chooses per launch, at
  ``cfg.internode_auto_ll_max_tokens``. Geometry is a compile-time identity on this
  path, so every bucket of ``internode_tuning_configs`` is built up front.
  No other backend implements internode.

Imports ``ep_plans`` (the C++/JIT plans) but never flydsl, so it works where
FlyDSL is not installed.
"""

from __future__ import annotations

import os

import torch

from mori.tensor_utils import from_gpu_ptr

from . import ep_plans as cb
from .dispatch_combine_op import EpDispatchCombineOp, KernelSet
from .internode_regions import internode_regions
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
}

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


def _geometry_from_env(name):
    """``"block,rdma,warp"`` from the environment, or None.

    Sweep-only. Read at build time (see _internode_geometry_buckets): the
    internode plans are compiled per geometry, so a geometry cannot be chosen at
    launch and a sweep has to pin it before the op is constructed.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    parts = tuple(int(x) for x in raw.replace(" ", "").split(","))
    if len(parts) != 3:
        raise ValueError(f"{name}={raw!r}: want three ints, block,rdma,warp")
    return parts


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
        # gfx125x routes to the TDM kernel, which needs a superset arena (plan A).
        _arch = getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "") or ""
        self._is1250 = _arch.split(":")[0].startswith("gfx125")
        self._multi_processor_count = torch.cuda.get_device_properties(
            dev
        ).multi_processor_count

        # Gate FIRST: rejecting a config after taking a symmetric window would
        # leak it (the arena is registered with the communicator), and the whole
        # point of the gate is that an unsupported config never gets that far.
        self._gate(
            KernelSet(dispatch={}, combine={}, unsupported=self._unsupported(cfg))
        )

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

    def _build(self, cfg, comm):
        dev = self.dev
        self.arena.zero()

        self._dispatch_specs, self._combine_specs = self._specs_from(cfg)
        # The internode passes need a device communicator, and it has to exist
        # before the kernels are bound: the plans take it by value.
        self._dev_comm = self._make_dev_comm(cfg, comm) if cfg.is_internode else None
        self._kernels = self._build_kernels(cfg, self.arena)

        if cfg.is_internode:
            self._alloc_internode_buffers(cfg)
            return

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
        * a context count of num_qp_per_pe. ccoGda picks its QP as
          ``contextId % numQpPerPe``, so fewer contexts than QPs silently
          collapses the stripes onto a subset of them -- no error, just less
          bandwidth than the config asked for.
        """
        from mori.cco import cco as _cco
        from mori.cco.communicator import DevCommHandle

        requirements = _cco.DevCommRequirements()
        requirements.gda_connection_type = _cco.GDA_CONNECTION_RAIL
        requirements.gda_context_count = max(1, cfg.num_qp_per_pe)
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
        self.inter_chunk_flag_combine = torch.zeros(
            n_nodes * max_tokens_per_rank * 2, **i32
        )
        self.dest_pe_counter = torch.zeros(world_size, **i32)
        self.block_flag_counter = torch.zeros(n_nodes, **i32)
        self.total_recv = torch.zeros(1, **i32)
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

        # Pin everything fixed for the op's lifetime, now that every buffer it
        # names exists. The launch path re-fills every field passed through
        # `args` on every call -- only bound values reach the plan's cached
        # struct -- so passing the whole schema each time cost a ctypes field
        # write per field per launch. Binding leaves six.
        static_args = self._internode_static_args()
        for plans_by_pass in self._internode_plans.values():
            for plan in plans_by_pass.values():
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
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (the internode kernels gather)")
        if cfg.enable_std_moe:
            bad.append("enable_std_moe (not implemented on the internode path)")
        if cfg.quant_type != "none":
            bad.append(
                f"quant_type={cfg.quant_type!r} (the internode combine is unquantised)"
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
        Use the module-level function when there is no op yet (sizing an arena)."""
        if self.cfg.is_internode:
            return self._scale_row_bytes()
        return scale_stride_bytes(self._scale_row_bytes())

    @classmethod
    def _scale_stride_i32(cls, cfg) -> int:
        """Dwords per DESTINATION scale row, 0 when the transport is off."""
        return scale_stride_bytes(cls._scale_i32(cfg) * 4) // 4

    def _regions(self, cfg):
        if cfg.is_internode:
            # Sized exactly, region by region. v1 sizes its arena with a 256 MB
            # slack term on a rough estimate; this is the exact sum, which is both
            # the point (an overrun fails instead of landing in the slack) and the
            # risk (a wrong formula corrupts a neighbour). test_internode_regions
            # compares every entry against what v1 allocates for the same config.
            return internode_regions(cfg)
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
        return regions

    def _unsupported(self, cfg) -> tuple[str, ...]:
        """Everything this backend cannot do, checked before anything is built."""
        if cfg.is_internode:
            return self._internode_unsupported(cfg)
        bad = []
        if cfg.dispatch_dtype not in _DISPATCH_DTYPES:
            bad.append(
                f"dispatch dtype {cfg.dispatch_dtype} (have bf16, fp32, fp8, fp4)"
            )
        if cfg.combine_dtype not in _COMBINE_DTYPES:
            bad.append(f"combine dtype {cfg.combine_dtype} (have bf16, fp32)")
        if cfg.is_scatter:
            bad.append("combine_mode='scatter' (gather only)")
        if cfg.quant_type != "none":
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

    _INTERNODE_DTYPE = {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float8_e4m3fnuz: "fp8_fnuz",
        torch.float8_e4m3fn: "fp8_ocp",
    }

    # Which leg each pass belongs to. The two legs are separate Plans and carry
    # DIFFERENT element types, exactly as the intranode path does: the kernel's
    # `T` is "elements of this leg's dtype", and `hiddenBytes = hiddenDim *
    # sizeof(T)` sizes the dispatch payload in one and the combine output in the
    # other. Compiling all eight with the dispatch dtype is what made an fp8
    # dispatch write hidden*sizeof(fp8) bytes where the bf16 combine_out view
    # reads hidden*sizeof(bf16) -- half the output left as whatever was there.
    _INTERNODE_LEG = {
        "copystaging": "dispatch",
        "dispatch": "dispatch",
        "dispatch_ll": "dispatch",
        "combinesync": "combine",
        "combinesyncbarrier": "combine",
        "combine": "combine",
        "combine_ll": "combine",
        "combineall": "combine",
    }

    # (phase, low_latency) -> the pass sequence, in launch order. Two for
    # dispatch, four for combine; the LL and non-LL members of each pair are
    # separate compiled entries rather than a runtime branch.
    _INTERNODE_SEQ = {
        ("dispatch", False): ("copystaging", "dispatch"),
        ("dispatch", True): ("copystaging", "dispatch_ll"),
        ("combine", False): (
            "combinesync",
            "combinesyncbarrier",
            "combine",
            "combineall",
        ),
        ("combine", True): (
            "combinesync",
            "combinesyncbarrier",
            "combine_ll",
            "combineall",
        ),
    }

    def _internode_request(
        self, cfg, dtype_tag, block_num, warp_per_block, rdma_block_num
    ):
        return dict(
            worldSize=cfg.world_size,
            hiddenDim=cfg.hidden_dim,
            scaleDim=cfg.scale_dim,
            scaleTypeSize=cfg.scale_type_size,
            maxTokenTypeSize=cfg.max_token_type_size,
            maxNumInpTokenPerRank=cfg.max_num_inp_token_per_rank,
            numExpertPerRank=cfg.num_experts_per_rank,
            numExpertPerToken=cfg.num_experts_per_token,
            maxTotalRecvTokens=cfg.max_total_recv_tokens,
            gpuPerNode=cfg.gpu_per_node,
            numQpPerPe=cfg.num_qp_per_pe,
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
            rank=self.cfg.rank,
            lsaBase=self.lsa_base,
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
            # The HOST struct, not DevCommHandle.ptr (the device-side copy): the args
            # embed a ccoDevComm by value, so the binding memcpys from host memory.
            devComm=self._dev_comm._dev_comm.host_ptr,
        )
        return self._internode_static_cache

    def _internode_variants(self, phase):
        """The kernel names of `phase` this config needs compiled.

        "v2" and "v2_ll" are separate JIT modules, so naming one in the config
        compiles one; only "auto" needs both, because only "auto" chooses per
        launch. Returned as a tuple so it can be concatenated with the passes
        that are common to both.
        """
        kernel_family = self.cfg.internode_kernel
        plain, low_latency = phase, phase + "_ll"
        if kernel_family == "v2":
            return (plain,)
        if kernel_family == "v2_ll":
            return (low_latency,)
        return (plain, low_latency)

    def _internode_use_ll(self, num_tokens):
        """Which of the two families this launch runs.

        Fixed by the config unless it is "auto", where the token count decides at
        the configured crossover. An explicit choice cannot fall back: the other
        family was never compiled.
        """
        kernel_family = self.cfg.internode_kernel
        if kernel_family == "v2":
            return False
        if kernel_family == "v2_ll":
            return True
        return num_tokens <= self.cfg.internode_auto_ll_max_tokens

    _PIN_FIELDS = {
        "dispatch": (
            "dispatch_block_num",
            "dispatch_rdma_block_num",
            "warp_num_per_block",
        ),
        "combine": (
            "combine_block_num",
            "combine_rdma_block_num",
            "combine_warp_num_per_block",
        ),
    }

    def _overlay_pinned_geometry(self, cfg, phase, geometry):
        """Overlay the caller's pinned geometry on a tuned triple, field by field.

        A pinned field is a manual override and beats the table; an unpinned one
        keeps what the table chose. cfg._pinned_geometry is the caller's original
        input -- by this point every field holds a number, because
        _resolve_geometry() filled the unpinned ones with the untuned fallback.
        """
        pinned = getattr(cfg, "_pinned_geometry", frozenset())
        return tuple(
            getattr(cfg, name) if name in pinned else tuned_value
            for name, tuned_value in zip(self._PIN_FIELDS[phase], geometry)
        )

    @staticmethod
    def _fit_internode_geometry(geometry):
        """Force rdma_block_num < block_num.

        The kernel splits the grid: blocks below rdmaBlockNum take the RDMA leg,
        the rest take the intra-node one. rdma >= block leaves the intra-node
        half with no blocks AND makes the dispatch fan-in wait for
        rdmaBlockNum * warpNum arrivals that can never happen -- every peer then
        spins forever, with no host-side error. internode_tuning_configs.lookup()
        clamps on the table-hit path; this is the choke point that covers the
        env-pin and untuned paths too.
        """
        block, rdma, warp = geometry
        return (block, min(rdma, max(1, block - 1)), warp)

    def _internode_geometry_buckets(self, cfg):
        return [
            (
                max_tokens,
                self._fit_internode_geometry(dispatch_geometry),
                self._fit_internode_geometry(combine_geometry),
            )
            for max_tokens, dispatch_geometry, combine_geometry in (
                self._internode_geometry_buckets_raw(cfg)
            )
        ]

    def _internode_geometry_buckets_raw(self, cfg):
        """``[(max_tokens | None, dispatch_geometry, combine_geometry)]``, coarsest
        last.

        Compilation happens at build time, so every geometry a launch could pick
        has to be known now -- which is why this returns the whole schedule and
        not just the one for the current token count. Same rule as the intranode
        schedule: `_pick` must never be able to trigger a compile.

        The table keeps dispatch and combine on DIFFERENT rdma/warp values for the
        same token count, which a single geometry per bucket cannot express; that
        is why a bucket carries two triples rather than one.
        """
        from .internode_tuning_configs import _TABLE, _device_key, lookup

        # Sweep hook: pin one geometry for every token count, bypassing the table.
        # A geometry is a compile-time identity here, so a sweep cannot select one
        # at launch the way the intranode path can -- it has to be fixed before
        # the plans are built, which is before the op exists. Hence an env var
        # rather than a CLI flag threaded through the config.
        pinned_dispatch = _geometry_from_env("MORI_EP_DISP_GEOM")
        pinned_combine = _geometry_from_env("MORI_EP_COMB_GEOM")
        if pinned_dispatch or pinned_combine:
            config_dispatch = (
                cfg.dispatch_block_num,
                cfg.dispatch_rdma_block_num,
                cfg.warp_num_per_block,
            )
            config_combine = (
                cfg.combine_block_num,
                cfg.combine_rdma_block_num,
                cfg.combine_warp_num_per_block,
            )
            return [
                (
                    None,
                    pinned_dispatch or config_dispatch,
                    pinned_combine or config_combine,
                )
            ]

        dtype = "fp8" if cfg.dispatch_dtype in _FP8_TUNING_DTYPES else "bf16"
        key = (_device_key(), cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token)
        entry = _TABLE.get(key)
        if not entry:
            # Untuned shape: one bucket, whatever the config resolved to.
            return [
                (
                    None,
                    (
                        cfg.dispatch_block_num,
                        cfg.dispatch_rdma_block_num,
                        cfg.warp_num_per_block,
                    ),
                    (
                        cfg.combine_block_num,
                        cfg.combine_rdma_block_num,
                        cfg.combine_warp_num_per_block,
                    ),
                )
            ]
        schedule = entry.get(dtype) or entry.get("fp8")
        buckets = []
        for row in schedule:
            max_tokens = row[0]
            geometry = lookup(
                cfg.world_size,
                cfg.hidden_dim,
                cfg.num_experts_per_token,
                max_tokens if max_tokens is not None else 1 << 30,
                dtype=dtype,
            )
            buckets.append(
                (
                    max_tokens,
                    self._overlay_pinned_geometry(
                        cfg, "dispatch", geometry["dispatch"]
                    ),
                    self._overlay_pinned_geometry(cfg, "combine", geometry["combine"]),
                )
            )
        return buckets

    def _build_internode_kernels(self, cfg) -> KernelSet:
        # One tag per leg, not one for the op: see _INTERNODE_LEG. Both keys are
        # present -- _internode_unsupported rejected the config otherwise.
        leg_dtype = {
            "dispatch": self._INTERNODE_DTYPE[cfg.dispatch_dtype],
            "combine": self._INTERNODE_DTYPE[cfg.combine_dtype],
        }

        self._internode_buckets = self._internode_geometry_buckets(cfg)
        self._plans = []
        # Keyed by the geometry triple so two buckets that resolve to the same
        # geometry share one compile -- the shipped table does exactly that for
        # tokens 4 and 8.
        self._internode_plans = {}
        # Collect the passes each geometry has to serve BEFORE building any, and
        # union them: a geometry is not owned by one phase. The shipped table
        # gives tokens 16 the same triple for dispatch and combine, so keying on
        # the triple alone and skipping an already-present one leaves that
        # geometry holding only whichever phase was seen first -- a KeyError on
        # the other phase's first launch, at run time, not build time.
        passes_by_geometry = {}
        for _, dispatch_geometry, combine_geometry in self._internode_buckets:
            for geometry, names in (
                (
                    dispatch_geometry,
                    ("copystaging",) + self._internode_variants("dispatch"),
                ),
                (
                    combine_geometry,
                    ("combinesync", "combinesyncbarrier")
                    + self._internode_variants("combine")
                    + ("combineall",),
                ),
            ):
                passes_by_geometry.setdefault(geometry, set()).update(names)
        for geometry, names in passes_by_geometry.items():
            block, rdma, warp = geometry
            plans_by_pass = {}
            for pass_name in sorted(names):
                request = self._internode_request(
                    cfg, leg_dtype[self._INTERNODE_LEG[pass_name]], block, warp, rdma
                )
                plans_by_pass[pass_name] = cb.EP_INTERNODE_PLANS[pass_name](**request)
                # Two launch arguments that never vary for THIS plan, so they
                # belong on the plan rather than in every launch's dict:
                #   rdmaBlockNum  a plan is compiled per geometry and lives in
                #                 exactly one `_internode_plans[geometry]`, so the
                #                 value a launch could pass is always geometry[1].
                #   replayMode    this backend has no replay path (KernelSet
                #                 carries dispatch_replay=None); it is always 0.
                # bind() stores ints in the plan's cached struct, and _launch_buf
                # re-writes only the per-call names and the non-int defaults --
                # so a bound int is written once at bind time and never again,
                # while a passed one costs a _set_arg every launch (~1.2us each,
                # x2 args x2 launches per round).
                plans_by_pass[pass_name].bind(rdmaBlockNum=rdma, replayMode=0)
            self._internode_plans[geometry] = plans_by_pass
            self._plans.extend(plans_by_pass.values())

        # One bound launch group per (geometry, phase, low-latency). The set of
        # plans a launch fires is fixed by that triple, so the validation and the
        # ctypes handle array belong here rather than on the launch path. A
        # geometry only carries the passes it actually serves, hence the subset
        # test -- the shipped table gives dispatch and combine the same triple at
        # 16 tokens but different ones elsewhere.
        from mori.jit.v2 import plan_api

        self._internode_groups = {}
        for geometry, plans_by_pass in self._internode_plans.items():
            for sequence_key, names in self._INTERNODE_SEQ.items():
                if all(name in plans_by_pass for name in names):
                    self._internode_groups[(geometry, sequence_key)] = (
                        plan_api.make_launch_group(
                            [plans_by_pass[name] for name in names]
                        )
                    )

        dispatch_spec = (cfg.dispatch_block_num, cfg.warp_num_per_block)
        combine_spec = (cfg.combine_block_num, cfg.combine_warp_num_per_block)
        return KernelSet(
            dispatch={dispatch_spec: self._wrap_internode("dispatch")},
            combine={combine_spec: self._wrap_internode("combine")},
            dispatch_replay=None,
            stages_in_kernel=True,
            # copystaging zeroes total_recv as its first act, so the host does not
            # have to -- that zero_() was a fill kernel enqueued ahead of the
            # dispatch sequence, delaying the kernels it protected.
            self_resets_counters=True,
            capabilities=frozenset({"gather", "scales", "internode"}),
        )

    def _internode_geom_for(self, phase, num_tokens):
        """The tuned geometry for this token count. Buckets are ordered, coarsest
        last, and a schedule with no None sentinel falls back to the last one --
        exactly as `_pick` walks the intranode schedule."""
        bucket = self._internode_buckets[-1]
        for row in self._internode_buckets:
            if row[0] is None or num_tokens <= row[0]:
                bucket = row
                break
        return bucket[1] if phase == "dispatch" else bucket[2]

    def _wrap_internode(self, phase):
        """One ABI crossing for the whole pass sequence.

        Every pass shares the schema and the same filled struct, so the arguments
        are written once and each plan launches against them in order: one ABI
        crossing for the whole sequence instead of one per pass.
        """

        def run(*, input, num_tokens, dest_map, **kwargs):
            use_low_latency = self._internode_use_ll(num_tokens)
            geometry = self._internode_geom_for(phase, num_tokens)
            group = self._internode_groups[(geometry, (phase, use_low_latency))]

            # Two of the kernel's arguments are set by dispatch and READ AGAIN by
            # combine, but the base only hands them to dispatch -- combine's
            # signature carries no indices and treats `weights` as a request
            # rather than an input. v1 does not notice because its handle keeps
            # both across the pair; here they have to be remembered explicitly.
            #
            #   tokenIndices  EpCombineAll dereferences it unconditionally
            #                 (args.tokenIndices[tokenId * topk + laneId]), so a
            #                 null is a fault, not a skipped branch.
            #   weightsBuf    it is a DIFFERENT tensor in each phase, and its
            #                 NULLNESS is the kernel's only gate on the fold. It
            #                 also sets `combXferBytes = hidden + (weights ? wt :
            #                 0)`, but only the four combine passes index that, so
            #                 the phases need not agree on null-ness.
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

            # Only what varies. The rest is bound on the plan; see
            # _build_internode_kernels -- rdmaBlockNum and replayMode used to be
            # passed here and are bound there now, since neither can differ
            # between two launches that reach the same plan.
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
        if not bad and cfg.is_internode:
            return self._build_internode_kernels(cfg)
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
        comb_cfg = dict(hidden_dim=cfg.hidden_dim, dtype=cfg.combine_dtype)
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
            capabilities=frozenset({"gather", "scales"}),
        )

    def _close_backend(self):
        for plan in getattr(self, "_plans", ()):
            plan.close()
        # After the plans: they embed the ccoDevComm by value and their kernels
        # dereference its QPs. The static args cache the host struct's address, so
        # it goes too -- nothing may re-read it once the handle is gone.
        dev_comm = getattr(self, "_dev_comm", None)
        if dev_comm is not None:
            self._dev_comm = None
            self._internode_static_cache = None
            dev_comm.close()

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

    def _region(self, name):
        intranode, internode = self._VIEW_REGION[name]
        return internode if self.cfg.is_internode else intranode

    def recv_tokens(self):
        view = self._views.get("recv_tokens")
        if view is not None:
            return view
        # fp4 packs 2 e2m1 per element of the torch dtype -> last dim is hidden/2.
        cols = self.cfg.hidden_dim // 2 if self.cfg.is_fp4 else self.cfg.hidden_dim
        view = from_gpu_ptr(
            self.arena.local_ptr(self._region("disp_out")),
            (self._recv_cap, cols),
            self.cfg.dispatch_dtype,
        )
        self._views["recv_tokens"] = view
        return view

    def combine_in_view(self):
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
        view = self._views.get("recv_weights")
        if view is None:
            view = from_gpu_ptr(
                self.arena.local_ptr(self._region("out_wts")),
                (self._recv_cap, self.cfg.num_experts_per_token),
                torch.float32,
            )
            self._views["recv_weights"] = view
        return view

    def recv_indices(self):
        view = self._views.get("recv_indices")
        if view is None:
            view = from_gpu_ptr(
                self.arena.local_ptr(self._region("out_idx")),
                (self._recv_cap, self.cfg.num_experts_per_token),
                torch.int32,
            )
            self._views["recv_indices"] = view
        return view

    def recv_scales(self):
        """The forwarded scale rows, or None when the transport is off -- the same
        answer FlyDSL gives, and the same (recv_cap, dwords) int32 view, so a caller
        cannot tell the backends apart.

        The rows sit scale_stride_bytes() apart -- 128 B-padded on the intranode
        path, packed on the internode one. Anything reading the region by pointer
        needs that pitch, not this shape.
        """
        view = self._views.get("recv_scales")
        if view is not None:
            return view
        n_i32 = self._scale_i32(self.cfg)
        if not n_i32:
            return None
        stride_i32 = self.scale_stride_bytes() // 4
        rows = from_gpu_ptr(
            self.arena.local_ptr(self._region("out_scales")),
            (self._recv_cap, stride_i32),
            torch.int32,
        )
        view = rows[:, :n_i32]
        self._views["recv_scales"] = view
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
