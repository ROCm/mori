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
from mori import cpp as mori_cpp
from mori.tensor_utils import from_gpu_ptr, dtype_to_int
import logging
import os
from dataclasses import dataclass
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

TOPK_IDX_DTYPE = torch.int32

# Threads per wavefront. gfx1250 (MI450/MI455) is wave32; gfx9xx (MI300/MI355)
# is wave64. This MUST NOT be hard-coded: the device kernels recompute
# warpNum = blockDim.x / warpSize at runtime, so the host block dim
# (warpSize * warp_num_per_block) and the combine dynamic-LDS sizing must use
# the SAME wavefront size the hardware uses. Hard-coding 64 on a wave32 device
# doubles the device warp count and overflows combine's shared-mem pointer
# arrays -> garbage src pointers -> the cross-device spin barrier never
# completes -> the EP job hangs. Detected per device below.
_DEFAULT_WAVE_SIZE = 64


def _detect_warp_size():
    """Return this device's wavefront size (32 or 64).

    Order matches the FlyDSL v2 path (dispatch_combine_v2/intranode_kernels.py):
    env override first, then the ARCH STRING. We deliberately do NOT trust the
    runtime device property first: on gfx1250 get_warp_size / warp_size wrongly
    reports 64 (see the FlyDSL _detect_wave_size note), which would silently
    re-introduce the wave32 launch hang. Keep MORI_WAVE_SIZE in sync with v2 so
    both paths agree.
    """
    v = os.environ.get("MORI_WAVE_SIZE")
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    try:
        from mori.jit.config import detect_gpu_arch

        # gfx12xx (MI400/gfx1250) is wave32; gfx9xx (MI300/MI355) is wave64.
        if str(detect_gpu_arch()).startswith("gfx12"):
            return 32
        return 64
    except Exception:
        pass
    # Last resort only if arch detection is unavailable.
    try:
        dev = torch.cuda.current_device()
        ws = getattr(torch.cuda.get_device_properties(dev), "warp_size", None)
        if ws in (32, 64):
            return int(ws)
    except Exception:
        pass
    return _DEFAULT_WAVE_SIZE


# Process-global CCO communicator, reused across ops (creating one per op would
# re-run the socket bootstrap every time). Used when _ep_comm() resolves to cco, which
# routes the C++ EP handle's symmetric buffers through a cco LSA window instead of
# the mori-shmem heap (enables intra-node EP on archs without shmem support).
_CCO_COMM = None


def _maybe_get_cco_comm(config):
    """Return a cached cco Communicator when the resolved backend is cco, else None.

    Bootstraps once via a torch.distributed broadcast of the cco unique-id (the
    same pattern the v2/FlyDSL path uses), sized for this config's symmetric
    buffers. Reused for the lifetime of the process.
    """
    if _ep_comm() != "cco":
        return None
    global _CCO_COMM
    if _CCO_COMM is not None:
        return _CCO_COMM
    from mori.cco import Communicator

    if not dist.is_initialized():
        raise RuntimeError(
            "MORI_EP_COMM=cco requires an initialized torch.distributed group"
        )
    # Per-rank VMM must hold every symmetric buffer this rank allocates. The
    # dispatch/combine token buffers dominate (~world * max_tok * hidden * dtype);
    # use generous headroom for the ~18 buffers + alignment slack.
    big = (
        int(config.world_size)
        * int(config.max_num_inp_token_per_rank)
        * int(config.hidden_dim)
        * int(config.max_token_type_size)
    )
    per_rank_vmm = 6 * big + (1 << 30)
    uid = Communicator.get_unique_id() if config.rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0)
    uid = objs[0]
    _CCO_COMM = Communicator.init(
        int(config.world_size), int(config.rank), uid, per_rank_vmm=per_rank_vmm
    )
    logger.info(
        "MORI_EP_COMM=cco: created cco communicator rank=%d/%d per_rank_vmm=%d",
        config.rank,
        config.world_size,
        per_rank_vmm,
    )
    return _CCO_COMM


class EpDispatchCombineKernelType(mori_cpp.EpDispatchCombineKernelType):
    def __str__(self):
        return self.name


class EpDispatchCombineQuantType(mori_cpp.EpDispatchCombineQuantType):
    def __str__(self):
        return self.name


_QUANT_TYPE_MAP = {
    "none": EpDispatchCombineQuantType.None_,
    "fp8_direct_cast": EpDispatchCombineQuantType.Fp8DirectCast,
    "fp8_blockwise": EpDispatchCombineQuantType.Fp8BlockwiseQuant,
    # Blockwise FP4 (E2M1) combine is its own quant type. It shares the blockwise staging/scale
    # layout with FP8 but transports packed FP4 (0.5 byte/elem) and uses half-sized staging slots.
    "fp4_blockwise": EpDispatchCombineQuantType.Fp4BlockwiseQuant,
}

# Blockwise combine quant types share the staging/scale layout and kernel launch config; only the
# element codec (and staging slot size) differ, so kernel selection treats them together and then
# swaps the codec token (fp8_blockwise <-> fp4_blockwise) in the kernel name.
_BLOCKWISE_COMBINE_QUANT_TYPES = (
    EpDispatchCombineQuantType.Fp8BlockwiseQuant,
    EpDispatchCombineQuantType.Fp4BlockwiseQuant,
)

# The FP4 blockwise combine kernels registered in ep_intranode.hip. Kernel-name selection derives
# an fp4_blockwise name from the fp8_blockwise one; the result is asserted against this set so a mismatch fails
# loudly instead of launching a non-existent symbol.
_FP4_COMBINE_KERNELS = frozenset(
    {
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4_blockwise",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4_blockwise_noweight_block128_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4_blockwise_noweight_block256_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4_blockwise_noweight_block128_vec8_top9",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4_blockwise_noweight_block256_vec8_top9",
    }
)


def _normalize_quant_type(quant_type):
    if isinstance(quant_type, EpDispatchCombineQuantType):
        return quant_type
    if isinstance(quant_type, str):
        key = quant_type.strip().lower()
        if key in _QUANT_TYPE_MAP:
            return _QUANT_TYPE_MAP[key]
    raise ValueError(
        f"invalid quant_type '{quant_type}', expected one of {list(_QUANT_TYPE_MAP.keys())}"
    )


def _current_stream():
    return torch.cuda.current_stream().cuda_stream


@dataclass
class EpDispatchRoutingHandle:
    """Per-call routing snapshot from cache-routing dispatch, replayed by combine / replay-routing dispatch."""

    disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_send_map: torch.Tensor
    total_recv_token_num: torch.Tensor
    disp_tok_id_to_src_tok_id_local: torch.Tensor
    cur_rank_num_token: int = 0

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


@dataclass
class EpDispatchCombineConfig:
    """Configuration for :class:`EpDispatchCombineOp`.

    Args:
        data_type: Deprecated. Tensor dtype kept only for backward
            compatibility with tests and examples. Kernel launch dtype is
            inferred from the runtime input tensor instead of this field.
        rank: Rank of the current process in the expert-parallel group.
        world_size: Total number of ranks participating in the dispatch/combine
            operation.
        hidden_dim: Hidden dimension of each token embedding.
        scale_dim: Number of scale values stored per token for quantized paths.
            Describes caller-provided dispatch scales (e.g. FP4 input).
            ``quant_type="fp8_blockwise"`` combine uses its own internal
            scale_dim driven by ``MORI_FP8_COMBINE_SCALE_DIM`` (default 56).
        scale_type_size: Size in bytes of each scale element.
        max_token_type_size: Maximum size in bytes for the token element type.
        max_num_inp_token_per_rank: Maximum number of input tokens each rank
            can process.
        num_experts_per_rank: Number of local experts hosted on each rank.
        num_experts_per_token: Number of experts selected for each token.
        warp_num_per_block: Number of warps per GPU block for the kernel launch.
        block_num: Number of GPU blocks to launch for the main kernel.
        max_total_recv_tokens: Optional cap used to derive the maximum number
            of tokens a rank can receive, which also affects memory
            consumption. A value of ``0`` disables the cap. If the actual
            received token count exceeds the derived limit, the kernel
            currently asserts.
        use_external_inp_buf: Whether the operator expects the input buffer to
            be managed externally.
        kernel_type: Dispatch/combine kernel implementation to use.
        gpu_per_node: Number of GPUs per node. This affects all kernel types.
        rdma_block_num: Number of RDMA blocks for inter-node kernels.
        num_qp_per_pe: Number of queue pairs per processing element.
        quant_type: Quantization mode. Supported string values are ``"none"``,
            ``"fp8_direct_cast"``, and ``"fp8_blockwise"``.
    """

    data_type: (
        torch.dtype
    )  # Deprecated for kernel launch (runtime dtype inferred from input tensor); retained for test/example compatibility
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    max_total_recv_tokens: int = 0
    use_external_inp_buf: bool = True
    kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"


def _cpp_dispatch_combine_factory(entity_name, allow_missing=False):
    if allow_missing:
        return getattr(mori_cpp, entity_name, None)
    return getattr(mori_cpp, entity_name)


# ---------------------------------------------------------------------------
# Kernel type → .hsaco compilation unit mapping
# ---------------------------------------------------------------------------
_KERNEL_TYPE_TO_HIP = {
    EpDispatchCombineKernelType.IntraNode: "ep_intranode",
    EpDispatchCombineKernelType.IntraNodeLL: "ep_intranode",
    EpDispatchCombineKernelType.InterNode: "ep_internode",
    EpDispatchCombineKernelType.InterNodeV1: "ep_internode_v1",
    EpDispatchCombineKernelType.InterNodeV1LL: "ep_internode_v1ll",
    EpDispatchCombineKernelType.AsyncLL: "ep_async_ll",
}

# dtype → kernel name suffix
_DTYPE_SUFFIX = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
}
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fn] = "fp8_ocp"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fnuz] = "fp8_fnuz"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float4_e2m1fn_x2] = "fp4"
except AttributeError:
    pass

# pointer size on device for shared memory calculation (sizeof(T**) and sizeof(float**))
_PTR_SIZE = 8

# Dynamic LDS a block may reserve on gfx125x, measured. Must equal MORI_COMB_LDS_BUDGET in
# src/ops/dispatch_combine/intranode_1250x.hpp: the kernel picks its combine transport by testing
# its tiles against that number, and this function reserves for whichever one it will pick.
_COMB_LDS_BUDGET = 327680

# The QUAD gather's tile-buffer count, i.e. MORI_COMB_QUAD in that same header. Repeated here rather
# than derived because the reservation below has to describe the layout the DEVICE compiled, and
# this decides its size to the byte.
_QUAD_DEPTH = 2


def _is_gfx125x():
    """Whether this device takes the TDM code paths.

    Single source of truth on the host side, because three things have to agree with the kernel's
    own `#if defined(__gfx1250__) || defined(__gfx1251__)` and with each other: which dispatch body
    runs, how much dynamic LDS is reserved for it, and what launch geometry it is given. They used
    to be keyed on the MORI_DISP_TDM env instead, which let all three disagree with the compiled
    kernel whenever the variable was unset.

    jit.core owns the arch string because the JIT is what compiles for it. Any failure degrades to
    False, i.e. the portable body, never a crash on the launch path.
    """
    try:
        from mori.jit.core import _FASTPATH_ARCH_PREFIX, _target_arch

        return _target_arch().startswith(_FASTPATH_ARCH_PREFIX)
    except Exception:
        return False


def _ep_comm():
    """Resolved backend for the EP handle's symmetric buffers: "cco" or "shmem".

    gfx125x defaults to cco, everything else keeps mori-shmem. An explicit MORI_EP_COMM wins either
    way, so the variable is still there for anyone who needs to force the other one.

    This is an arch default for the same reason the dispatch body is: which backend can carry the
    symmetric buffers is a property of the hardware, and a caller who has to know that in order to
    set an environment variable will eventually not know it. cco routes the buffers through an LSA
    window instead of the shmem heap, which is what makes intra-node EP work on an arch without
    shmem support.

    NOT VERIFIED on any non-gfx125x device -- there is none on this machine -- which is exactly why
    the non-125x branch keeps the behaviour it already had rather than being changed blind.
    """
    val = os.environ.get("MORI_EP_COMM", "").strip().lower()
    if val:
        return val
    return "cco" if _is_gfx125x() else "shmem"


def warmup_jit_kernels(kernel_type):
    """Pre-compile kernels for a kernel_type. Call from main process before spawning workers."""
    from mori.ops._jit_loader import ensure_compiled, _compiled_hsaco

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    hip_name = _KERNEL_TYPE_TO_HIP[kernel_type]
    ensure_compiled(hip_name)
    return _compiled_hsaco.get(hip_name)


def _ensure_jit_kernels(kernel_type):
    """Ensure the required kernels for this kernel_type are JIT-compiled."""
    from mori.ops._jit_loader import ensure_compiled

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    ensure_compiled(_KERNEL_TYPE_TO_HIP[kernel_type])


def _load_hip_modules(kernel_type, init_shmem=True):
    """Load HipModule for the given kernel_type and init shmem gpu states."""
    from mori.ops._jit_loader import load_hip_module

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    return load_hip_module(_KERNEL_TYPE_TO_HIP[kernel_type], init_shmem=init_shmem)


class EpDispatchCombineOp:
    @staticmethod
    def supports_sbo_tile_signal_v1() -> bool:
        return hasattr(mori_cpp, "build_args_sbo")

    def __init__(self, config):
        self.config = config
        # Wavefront size of THIS device (32 on gfx1250, 64 on gfx9xx). Used for
        # every kernel launch's block dim; keep consistent with the device-side
        # warpNum = blockDim.x / warpSize so combine LDS sizing stays in bounds.
        self._warp_size = _detect_warp_size()
        logger.debug("EpDispatchCombineOp: detected warp_size=%d", self._warp_size)
        _ensure_jit_kernels(config.kernel_type)

        if dist.is_initialized():
            dist.barrier()

        handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
        self._cpp_config = mori_cpp.EpDispatchCombineConfig(
            rank=config.rank,
            world_size=config.world_size,
            hidden_dim=config.hidden_dim,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            max_token_type_size=config.max_token_type_size,
            max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
            num_experts_per_rank=config.num_experts_per_rank,
            num_experts_per_token=config.num_experts_per_token,
            warp_num_per_block=config.warp_num_per_block,
            block_num=config.block_num,
            use_external_inp_buf=config.use_external_inp_buf,
            kernel_type=config.kernel_type,
            gpu_per_node=config.gpu_per_node,
            rdma_block_num=config.rdma_block_num,
            num_qp_per_pe=config.num_qp_per_pe,
            quant_type=_normalize_quant_type(config.quant_type),
            max_total_recv_tokens=config.max_total_recv_tokens,
        )

        self._cco_comm = _maybe_get_cco_comm(config)
        _cco_ptr = self._cco_comm.ptr if self._cco_comm is not None else 0
        self._handle = handle_class(self._cpp_config, cco_comm_ptr=_cco_ptr)
        # cco backend is shmem-free: skip shmem_module_init (asserts when mori-shmem
        # is uninitialized; on gfx1250 shmem init can hang). Intra-node cco kernels
        # use cco LSA peer pointers + pointer-based device waits, not the shmem heap.
        self._hip_module = _load_hip_modules(
            config.kernel_type, init_shmem=self._cco_comm is None
        )
        self._handle_info = mori_cpp.get_handle_info(self._handle)

        self._fp8_blockwise_combine_scale_dim = self._handle_info[
            "fp8_blockwise_combine_scale_dim"
        ]
        self._fp8_blockwise_combine_scale_type_size = self._handle_info[
            "fp8_blockwise_combine_scale_type_size"
        ]
        # Detect the fp4_blockwise combine so we can fail fast on unsupported archs at construction.
        # (Kernel selection keys off the Fp4BlockwiseQuant enum directly.)
        self._combine_is_fp4 = (
            isinstance(config.quant_type, str)
            and config.quant_type.strip().lower() == "fp4_blockwise"
        )
        if self._combine_is_fp4:
            # The packed-FP4 combine relies on the gfx950 OCP FP4 conversion instructions
            # (cvt_scalef32_pk_f32_fp4). On other archs there is no hardware path, so fail fast
            # instead of silently selecting an fp4 kernel that would fall back to slow software.
            from mori.jit.config import detect_gpu_arch

            _arch = str(detect_gpu_arch())
            if "gfx950" not in _arch:
                raise ValueError(
                    f"quant_type='fp4_blockwise' combine requires a gfx950 GPU (OCP FP4 "
                    f"conversion instructions); detected arch '{_arch}'. Use 'fp8_blockwise' "
                    f"instead on this device."
                )

        # Dispatch metadata staging capacity, gfx125x only. The scratch in intranode_1250x.hpp is a
        # fixed pool (CUSPLIT_POOL_SLOTS) split by world_size at runtime, and destTokId spans the
        # destination peer's WHOLE recv space, so the slice has to cover MaxNumTokensToRecv().
        #
        # The arch gate matters: the portable dispatch body has no such pool -- it indexes
        # dispTokIdToSrcTokId, which is allocated for MaxNumTokensToRecv() outright -- so without it
        # this rejects configs the other archs serve fine. Ungated, it failed the world_size=8
        # 65536-token/rank intranode test on gfx950.
        #
        # Checked here rather than on the device because an over-capacity config has no correct
        # device-side answer -- the kernel can only drop the metadata -- and because it used to be
        # answered SILENTLY: the stride was a fixed 16384 slots, exactly world_size *
        # max_num_inp_token_per_rank for EP4 at 4096 tokens, and past that the write side dropped
        # slots while the read side (gated on MaxNumTokensToRecv instead) read the NEXT peer's
        # region. Still inside the allocation, so no fault and no OOM -- just a corrupt
        # dispTokIdToSrcTokId and a dispatch correctness assert at 8192 and 16384 tokens/rank.
        #
        # Keep CUSPLIT_POOL_SLOTS in sync with the .hpp. Drifting high here only over-restricts,
        # which fails closed.
        if (
            config.kernel_type == EpDispatchCombineKernelType.IntraNode
            and _is_gfx125x()
        ):
            _cusplit_pool_slots = 8 * 32768
            _slots_per_peer = _cusplit_pool_slots // max(config.world_size, 1)
            if config.max_total_recv_tokens > 0:
                _recv_per_rank = min(
                    -(-config.max_total_recv_tokens // config.world_size),
                    config.max_num_inp_token_per_rank,
                )
            else:
                _recv_per_rank = config.max_num_inp_token_per_rank
            _slots_needed = config.world_size * _recv_per_rank
            if _slots_needed > _slots_per_peer:
                raise ValueError(
                    f"dispatch metadata staging holds {_slots_per_peer} slots/peer at "
                    f"world_size={config.world_size}, but this config needs {_slots_needed} "
                    f"(max_num_inp_token_per_rank={config.max_num_inp_token_per_rank}, "
                    f"max_total_recv_tokens={config.max_total_recv_tokens}). Raise "
                    f"CUSPLIT_POOL_SLOTS in src/ops/dispatch_combine/intranode_1250x.hpp and this "
                    f"check, or lower max_num_inp_token_per_rank."
                )

        self._dispatch_out_ptrs = mori_cpp.get_dispatch_output_ptrs(self._handle, True)
        self._combine_out_ptrs = mori_cpp.get_combine_output_ptrs(self._handle, True)

        # Output tensor views over the two pointer sets above. Read the opt-out
        # once here: probing the environment per call would reintroduce exactly
        # the kind of host-side cost this cache removes.
        self._view_cache = (
            None if os.environ.get("MORI_EP_DISABLE_VIEW_CACHE", "0") == "1" else {}
        )

        self.local_expert_count = torch.zeros(
            config.num_experts_per_rank, dtype=torch.int32, device="cuda"
        )

        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            "get_dispatch_src_token_pos"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_factory(
            "get_cur_rank_num_token"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_sender_token_idx_map"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_receiver_token_idx_map"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_factory(
            "get_registered_combine_input_buffer"
        )

        self.launch_config_mode = os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "MANUAL")
        if self.launch_config_mode == "AUTO":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            try:
                from mori.ops.tuning_config import (
                    TuningConfigManager,
                    kernel_type_to_config_str,
                    quant_type_to_config_str,
                    detect_gpu_model,
                )
                from mori.jit.config import detect_gpu_arch

                gpu_arch = detect_gpu_arch()
                gpu_model = detect_gpu_model()
                kt_str = kernel_type_to_config_str(config.kernel_type)
                self._qt_str = quant_type_to_config_str(config.quant_type)
                mgr = TuningConfigManager.get_instance(
                    gpu_arch,
                    kt_str,
                    config.world_size,
                    gpu_model,
                )
                self._dispatch_rules = mgr.dispatch_rules or None
                self._combine_rules = mgr.combine_rules or None
                if logger.isEnabledFor(logging.DEBUG):
                    if self._dispatch_rules is None and self._combine_rules is None:
                        logger.debug(
                            "AUTO tuning: no config for %s_%s_%s_ep%d; "
                            "using hard-coded fallback.",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                        )
                    else:
                        d_dtypes = sorted(
                            {r["dtype"] for r in (self._dispatch_rules or [])}
                        )
                        c_dtypes = sorted(
                            {r["dtype"] for r in (self._combine_rules or [])}
                        )
                        logger.debug(
                            "AUTO tuning: %s_%s_%s_ep%d — "
                            "dispatch(%d rules, dtypes=%s) combine(%d rules, dtypes=%s)",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                            len(self._dispatch_rules or []),
                            d_dtypes,
                            len(self._combine_rules or []),
                            c_dtypes,
                        )
            except Exception as exc:
                logger.warning(
                    "AUTO tuning: failed to load config (%s); "
                    "using hard-coded fallback.",
                    exc,
                )

            if (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (96, 64, 8)
            elif (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1LL.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (256, 128, 8)
            else:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (128, 0, 16)
        elif self.launch_config_mode == "MANUAL":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            self.auto_block_num, self.auto_rdma_block_num, self.auto_warp_per_block = (
                None,
                None,
                None,
            )
        else:
            raise ValueError(
                f"invalid MORI_EP_LAUNCH_CONFIG_MODE, must be ['MANUAL', 'AUTO'], got '{self.launch_config_mode}'"
            )

        # Buffers are zeroed as they are allocated, and peers do not wait for that: once
        # a peer reaches dispatch, its kernel writes straight into our buffers. A rank a
        # few milliseconds behind then zeroes a buffer a peer already wrote to:
        #
        #   rank 0  [barrier] alloc+zero, dispatch kernel --+
        #                                                   |  writes rank 1's
        #                                                   v  recvTokenNum, payload
        #   rank 1  [barrier] alloc+zero . . . . . . . . . -X  zeroed here, write lost
        #
        # A lost recvTokenNum hangs rank 1, and everyone waiting on it. The barrier at
        # the top of __init__ runs before the allocation, so it only makes ranks start
        # together; the sync finishes our zeroing, the barrier makes peers wait for it.
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

    # ------------------------------------------------------------------
    # Kernel launch helpers
    # ------------------------------------------------------------------
    def _intranode_dispatch_default_launch(self):
        """Per-body default (block_num, warp_per_block) for the IntraNode dispatch kernel.

        Split on the ARCH, matching the #if that picks the body in intranode_entry.hpp. gfx125x
        runs EpDispatchIntraNodeKernel_1250x_body and takes 64x8; every other arch runs the
        portable EpDispatchIntraNodeKernel_body in intranode.hpp, which interleaves scattered
        per-token metadata with the payload and needs the wide grid to hide that, so it keeps its
        historical 256x16.

        The gfx125x body batches metadata into one TDM copy per (block, peer) run. That argues for
        a narrow grid where each block owns a large contiguous run, and 64 blocks came from that
        reasoning -- correctly, as it turns out, but the 8 warps that came with it left the payload
        phase starved of concurrent TDM.

        Device facts (measured, torch.cuda.get_device_properties on gfx1250): CU=256,
        LDS=327680B per CU AND per block, max 2048 threads/CU, warpSize=32. At wpb=8 the tile is
        8 * 7168 * 2 = 114KB, so TWO THIRDS of a block's LDS budget sat unused.

        wpb sweep at DBN=64, EP4-4K bf16 hidden 7168, noTIMING mean_algo_bw, one bw run per point:

            wpb    8      12     16     20     22
            GB/s  1278   1218   1365   1255   1239

        16 is a sharp peak, not a plateau, and the reason is the token partition rather than
        occupancy: _tpi = warpSize/topk = 4 tokens per warp-iteration, so one round consumes
        DBN * wpb * 4 tokens. At wpb=16 that is exactly 4096 -- one round, 4 tokens per warp, no
        remainder. wpb=8 needs two rounds; wpb=12 leaves a second round holding only 1024 tokens
        (three quarters of the warps idle in it); wpb=20/22 finish in one round but hand a large
        share of warps no tokens at all while still costing their LDS. So this default is tied to
        max_num_inp_token_per_rank=4096: the rule is wpb such that DBN * wpb * (warpSize/topk)
        divides the token count.

        Widening the GRID reaches the same ceiling: DBN=128/wpb=8 measures 1366 GB/s, identical to
        DBN=64/wpb=16, but spends 128 CUs instead of 64. Both put 1024 warps in flight, which is what
        actually bounds the payload phase -- concurrent TDM operations, not compute. (DBN sweep at
        wpb=8: 32:882 48:1073 64:1278 96:1239 112:1160 128:1354 160:1356 256:1352.)

        DEFAULT DELIBERATELY STAYS AT 64/8. Both 1366 GB/s points buy their gain with more physical
        resource -- wpb=16 doubles blockDim to 512 and the LDS reservation to 229KB, DBN=128 doubles
        the CUs -- so neither is a kernel improvement, and the mandate here is to reach 1.3TB/s
        without changing the footprint. They are recorded because they bound what the payload phase
        can do when TDM concurrency is not the constraint.

        Which body runs used to be an env choice (-DMORI_DISP_CLEAN) on top of another env choice
        (-DMORI_DISP_TDM). Both are gone: an environment variable could pick the empty body on
        gfx125x or the wide-grid body's geometry on hardware running the narrow-grid one.
        """
        if not _is_gfx125x():
            return 256, 16
        return 64, 8

    def _intranode_combine_default_launch(self, is_push=False):
        """Per-body default (block_num, warp_per_block) for the IntraNode combine kernel.

        64x8 on gfx125x, the same geometry dispatch defaults to, so an intra-node caller with no
        opinion gets ONE shape for both phases instead of two unrelated ones. (0, 0) elsewhere means
        "no opinion, use the config defaults", which is the behaviour every arch had before the QUAD
        transport existed; the width only matters where the TDM transports exist at all.

        PUSH is the exception and gets 16, because the two transports scale oppositely with width
        and 8 is PULL's number. MEASURED 2026-08-04, EP4 bf16 hidden 7168, 64 blocks, check armed
        (rc=0 on every row):

            transport                wpb 8     wpb 16
            PUSH (caller-owned inp)  288.2     221.2      <- 959.9 GB/s
            zero copy PULL           168.5     cannot

        "cannot" is not untried: PULL's gather needs one LDS tile per source, so 16 warps want
        458 KB against a 320 KB budget, the reservation declines and the kernel falls back to the
        lane gather at 1166.4us. PUSH scales instead because its fold ALIASES the send tile, one
        per warp, and because half its work is a LOCAL read at 2793 GB/s rather than a cross-card
        one: 8 -> 16 warps buys the push phase 7.9% and the fold phase 42.9%.

        Zero copy is still 24% faster and still the right thing for a caller who can hand over a
        registered buffer; this is only about the caller who cannot.

        QUAD needs one whole-token tile per warp, double buffered, plus an output tile: at hidden
        7168 bf16 that is 287,872 B at 8 warps against a 327,680 B budget, and 574,976 at 16. So the
        width is not a tuning choice here, it is the difference between the fast transport running
        and the kernel falling back to WarpAccumLF. 64 blocks with 8 warps is where it was measured:
        168.9us for 212.3 MB = 1255 GB/s, rc=0 on the bench's per-element check.

        The width is applied even where QUAD is off (external input buffer, blockwise quant, or the
        gates turned down), because the fallback it hands those configs is worse: with no per-body
        default they land on config.block_num / config.warp_num_per_block, which is 80x8 and was
        measured at 104.6 GB/s.

        Only a default. An explicit block_num/warp_per_block from the caller, a tuning-config hit,
        and AUTO mode all still win, and each of those paths goes through the LDS budget in
        _combine_shared_mem(), which falls back to a transport that fits rather than failing.
        Verified both ways, rc=0 each time: a caller passing block_num=-1/warp_per_block=-1 (no
        opinion, which is what this default is for) lands on 64x8 and 168.9us / 1255 GB/s, and the
        same workload forced to 16 warps runs 1171.2us -- both TDM transports decline on LDS, the
        gather takes over, and nothing crashes or corrupts.
        """
        if not _is_gfx125x():
            return 0, 0
        return (64, 16) if is_push else (64, 8)

    def _resolve_launch_params(
        self,
        block_num,
        rdma_block_num,
        warp_per_block,
        *,
        num_tokens=0,
        hidden_dim=0,
        dtype=None,
        tuning_rules=None,
        zero_copy=None,
        quant_type=None,
        is_intranode_dispatch=False,
        is_intranode_combine=False,
        is_push_transport=False,
    ):
        if tuning_rules and dtype is not None:
            from mori.ops.tuning_config import TuningConfigManager

            params = TuningConfigManager.lookup(
                tuning_rules,
                dtype,
                num_tokens,
                hidden_dim,
                zero_copy,
                quant_type,
                topk=self.config.num_experts_per_token,
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        bn = self.auto_block_num if self.auto_block_num else block_num
        rbn = self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num
        wpb = self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block
        def_bn, def_wpb = (0, 0)
        if self.config.kernel_type == EpDispatchCombineKernelType.IntraNode:
            if is_intranode_dispatch:
                def_bn, def_wpb = self._intranode_dispatch_default_launch()
            elif is_intranode_combine:
                def_bn, def_wpb = self._intranode_combine_default_launch(
                    is_push=is_push_transport
                )
        actual_bn = (def_bn or self.config.block_num) if bn <= 0 else bn
        actual_rbn = self.config.rdma_block_num if rbn <= 0 else rbn
        actual_wpb = (def_wpb or self.config.warp_num_per_block) if wpb <= 0 else wpb
        return actual_bn, actual_rbn, actual_wpb

    def _get_func(self, name):
        return self._hip_module.get_function(name)

    def _dispatch_shared_mem(self, warp_per_block):
        """Shared memory for dispatch kernels (worldSize + numExpertPerRank per warp + numExpertPerRank) * sizeof(index_t)."""
        base = (
            self.config.world_size * warp_per_block
            + self.config.num_experts_per_rank * warp_per_block
            + self.config.num_experts_per_rank
        ) * 4  # sizeof(index_t)
        # On gfx125x the IntraNode dispatch body stages each token's hidden-dim payload through ONE
        # per-warp LDS tile, so it needs warp_per_block * hiddenDim * elemSize bytes of dynamic
        # shared. Here warp_per_block is the DEVICE warp count per block (block = warpSize*wpb).
        # Everywhere else the WarpCopy body runs, which stages nothing and needs only `base`.
        #
        # This tests the ARCH, matching the #if that selects the body in intranode_entry.hpp. It
        # used to test MORI_DISP_TDM, which meant an unset env reserved `base` for a kernel that
        # stages 14KB tiles per warp into that reservation.
        if _is_gfx125x():
            # One FULL token tile per warp = hidden*elemSize bytes (14KB at hidden 7168 bf16), so
            # wpb<=16 stays inside the 320KB gfx1250 LDS budget. A second tile per warp for payload
            # double-buffering is not worth its 229KB (measured 1280.8 vs 1280.7 GB/s, see the drain
            # comment in intranode_1250x.hpp's payload loop).
            # The gfx125x dispatch body reuses this same per-warp tile for its batched metadata
            # send (see the tokCapM computation in intranode_1250x.hpp), so no extra budget is
            # needed.
            tile = (
                warp_per_block
                * int(self.config.hidden_dim)
                * int(self.config.max_token_type_size)
            )
            return max(base, tile)
        return base

    def _intranode_dispatch_kernel(self, sfx, stdmoe=False):
        """Intra-node dispatch. One launch symbol; which body it reaches is decided on the device
        side by EpDispatchIntraNodeKernel_entry in intranode_entry.hpp."""
        name = f"EpDispatchIntraNodeKernel_{sfx}"
        if stdmoe:
            name += "_stdmoe"
        return name

    def _combine_shared_mem(self, warp_per_block, use_weights=True):
        """Shared memory for combine kernels."""
        quant_type = _normalize_quant_type(self.config.quant_type)
        num_ptr_arrays = 1 + int(bool(use_weights))
        if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
            num_ptr_arrays += 1
        base = (
            warp_per_block
            * self.config.num_experts_per_token
            * num_ptr_arrays
            * _PTR_SIZE
        )
        # The combine token goes through TDM on gfx125x, which needs per-warp LDS tiles holding one
        # chunk of a token. The kernel places them right AFTER the pointer arrays above, so this is a
        # sum and not a max, and chunk_elems must match _cTileElems/_cPullTileElems in
        # intranode_1250x.hpp: chunk rounded up to a whole 128B TDM row.
        #
        # The two transports need different tile counts, and which one is compiled follows the same
        # flag that picks the kernel: use_external_inp_buf=True -> _nop2p -> PUSH (one tile per warp,
        # staged then TDM-stored to the peer); False -> _p2p -> PULL (one tile per source, all topk
        # loads issued before the wait).
        #
        # Blockwise-quant combine is excluded, and this is a layout requirement rather than a
        # tuning choice: intranode_entry.hpp sends every quantizing instantiation to the portable
        # body, which has no LDS tiles at all. Reserving them here would shrink occupancy for
        # nothing and could trip the budget check below on a kernel that never had a tile.
        #
        # MORI_COMB_TDM in intranode_1250x.hpp, the chunk count. Reserving for the tiles requires
        # knowing whether the device compiled them at all, which is the arch test in
        # intranode_entry.hpp and nothing else -- see _is_gfx125x for why all three host-side
        # answers come from one place.
        chunks = 2 if _is_gfx125x() else 0
        if chunks and quant_type not in _BLOCKWISE_COMBINE_QUANT_TYPES:
            elem = int(self.config.max_token_type_size)
            row_elems = 128 // elem
            hidden = int(self.config.hidden_dim)
            topk = int(self.config.num_experts_per_token)
            world = int(self.config.world_size)
            if self.config.use_external_inp_buf:
                # PUSH never splits a token and never holds more than one: one warp sends one whole
                # token to its one destination PE, so the tile is exactly hiddenDim elements and
                # MORI_COMB_TDM only gates TDM on/off there.
                tile_elems = hidden
                tiles_per_warp = 1
            else:
                # PULL still chunks, because a warp holds one tile per SOURCE rather than one tile
                # total, so whole tokens would not fit. The source count is min(topk, world_size), not
                # topk: dispatch dedups a token's experts by destination PE and combine compacts the
                # survivors, so one survives per distinct PE. Must match _cPullSrcMax in
                # intranode_1250x.hpp, including the world_size <= 4 condition -- that is where the
                # compaction that makes the tile indices dense runs.
                tile_elems = -(-hidden // chunks)
                tile_elems = -(-tile_elems // row_elems) * row_elems
                tiles_per_warp = world if (world <= 4 and world < topk) else topk
                # QUAD (MORI_COMB_QUAD in intranode_1250x.hpp) turns the decomposition 90 degrees:
                # one warp owns one SOURCE and reads that source's token, so a warp needs one tile
                # instead of one per source. It gets _QUAD_DEPTH of them, each a whole token:
                # D*hidden*elem*wpb bytes buy D-1 reads in flight at hidden*elem bytes each.
                #
                # Getting this wrong in either direction is a silent layout mismatch, not a
                # slowdown: host and kernel must agree on the tile size to the byte.
                _qd = _QUAD_DEPTH if chunks else 0
                if _qd >= 2:
                    tile_elems = hidden
                    tiles_per_warp = _qd
                    tile_bytes = warp_per_block * tiles_per_warp * tile_elems * elem
                    # ... plus one int per (warp, buffer) for the source-count ring that follows the
                    # tiles (it is in LDS because holding it in registers spills), and two more per
                    # (group, buffer). Those last are unread, and are reserved anyway because the
                    # kernel's _qLdsNeed still counts them and still places _qOut past them; the two
                    # numbers are one layout and may only move together.
                    _qgroups = max(1, warp_per_block // world)
                    tile_bytes += (warp_per_block + 2 * _qgroups) * tiles_per_warp * 4
                    # The fold writes its output into LDS and the TDM engine stores that out, which
                    # costs one more buffer set of the fold's SLICE of a tile -- 1/world of the
                    # tiles -- 128B-aligned past the counters.
                    tile_bytes = (tile_bytes + 127) & ~127
                    tile_bytes += warp_per_block * _qd * (tile_elems // world) * elem
                    total = ((base + 127) & ~127) + tile_bytes
                    # Mirrors the _qLdsNeed guard in intranode_1250x.hpp, which declines QUAD at a
                    # width whose tiles do not fit and lets the token fall through to the chunked
                    # gather.
                    # Reserving QUAD's footprint here while the kernel runs the chunked path (or the
                    # reverse) is a silent layout mismatch, so the two predicates are the same one.
                    if total <= _COMB_LDS_BUDGET:
                        return total
                    # A width the tiles do not fit: fall through to the chunked tiles below, which
                    # are what the kernel will use.
                    tile_elems = -(-hidden // chunks)
                    tile_elems = -(-tile_elems // row_elems) * row_elems
                    tiles_per_warp = world if (world <= 4 and world < topk) else topk
            tile_bytes = warp_per_block * tiles_per_warp * tile_elems * elem
            # The kernel rounds past the pointer arrays to 128B (TDM row) before the first tile.
            total = ((base + 127) & ~127) + tile_bytes
            if total <= _COMB_LDS_BUDGET:
                base = total
            elif self.config.use_external_inp_buf:
                # PUSH has no fallback in the kernel -- its fold aliases the send tile rather than
                # allocating one -- so an overflow there is fatal. PULL is the opposite: _cPullOk
                # declines the tiles and gathers without them, so raising on that would turn a
                # fallback into a crash, which is how a 16-warp PULL with two tile sets per warp
                # behaves -- 458 KB of tiles against a 320 KB budget.
                raise ValueError(
                    f"The combine TDM push needs {total} B of LDS "
                    f"(warp_per_block={warp_per_block}, tiles/warp={tiles_per_warp}, "
                    f"tile_elems={tile_elems}) but the budget is {_COMB_LDS_BUDGET} B. "
                    f"Lower warp_per_block."
                )
            # Else: arch default at a width the tiles do not fit. _cPullOk in intranode_1250x.hpp
            # fails the same test and the gather falls back to WarpAccumLF, which needs no tiles,
            # so the reservation stays at the pointer arrays.
        return base

    def _cached_view(self, key, ptr, shape, dtype):
        """Memoized ``from_gpu_ptr`` over a handle-owned shmem buffer.

        dispatch()/combine() hand back views over buffers whose addresses are
        snapshotted once in __init__ (self._dispatch_out_ptrs /
        self._combine_out_ptrs) and whose shapes are fixed by the config, yet a
        fresh view was built on every call. Each rebuild walks
        __cuda_array_interface__ and queries the current device; six of them per
        dispatch+combine round measured ~60us of host time, and at small token
        counts the host submission path is what paces the GPU.

        The cache key includes the pointer so that shape/dtype/pointer together
        determine the entry. It is not protection against reallocation: the
        pointers themselves are snapshotted once in __init__ and never
        refreshed, so a realloc would leave both the key and the view stale.
        That assumption is pre-existing -- the buffers are handle-owned and have
        no realloc path -- and this cache does not weaken it. Entries are
        bounded by the number of distinct (shape, dtype) a given op is called
        with -- one, for a fixed model.

        The returned tensor aliases exactly the memory a fresh view would, and
        the kernels overwrite it in place either way, so callers see no change
        beyond object identity.
        """
        if self._view_cache is None:
            return from_gpu_ptr(ptr, shape, dtype)
        ck = (key, ptr, tuple(shape), dtype)
        t = self._view_cache.get(ck)
        if t is None:
            t = from_gpu_ptr(ptr, shape, dtype)
            self._view_cache[ck] = t
        return t

    def _launch(self, func_name, grid, block, shared_mem, stream, args_ptr):
        func = self._get_func(func_name)
        func.launch_struct(grid, block, shared_mem, stream, args_ptr)

    def _launch_multi(self, func_names, grids, blocks, shared_mems, stream, args_ptr):
        from mori.jit.hip_driver import launch_multi

        funcs = [self._get_func(name)._func for name in func_names]
        launch_multi(funcs, grids, blocks, shared_mems, stream, args_ptr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_launch_config(
        self, is_dispatch=True, block_num=-1, rdma_block_num=-1, warp_per_block=-1
    ):
        """Resolve saved launch params. Not the path dispatch()/combine() take.

        Those call _resolve_launch_params with ``dtype=input.dtype``, i.e. each
        phase is looked up with the dtype of its own input tensor. This method
        has no such argument and uses ``config.data_type`` for both, which is
        the *dispatch* dtype (and is itself deprecated -- the kernel infers the
        real dtype from the input tensor). So for a combine rule saved under a
        different combine dtype, this returns None where the runtime matches.

        Nothing in the repository calls this; it is kept as public API. A
        caller that wants the combine geometry for a mixed-dtype config should
        look the rule up directly with the combine dtype.
        """
        rules = self._dispatch_rules if is_dispatch else self._combine_rules
        if rules:
            from mori.ops.tuning_config import TuningConfigManager

            zc = not self.config.use_external_inp_buf if not is_dispatch else None
            qt = self._qt_str if not is_dispatch else None
            params = TuningConfigManager.lookup(
                rules,
                self.config.data_type,
                self.config.max_num_inp_token_per_rank,
                self.config.hidden_dim,
                zero_copy=zc,
                quant_type=qt,
                topk=self.config.num_experts_per_token,
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        return (
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def max_num_tokens_to_recv(self):
        return self._cpp_config.max_num_tokens_to_recv()

    def max_num_tokens_to_recv_per_rank(self):
        return self._cpp_config.max_num_tokens_to_recv_per_rank()

    def max_num_tokens_to_send(self):
        return self._cpp_config.max_num_tokens_to_send()

    def max_num_tokens_to_send_per_rank(self):
        return self._cpp_config.max_num_tokens_to_send_per_rank()

    def decode_send_flat_idx(self, flat_idx):
        """Decode a flat send index into (rank, local_token_id)."""
        stride = self.max_num_tokens_to_send()
        return int(flat_idx) // stride, int(flat_idx) % stride

    def get_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        ptr, shape0, shape1 = self._get_registered_combine_input_buffer(
            self._handle, hidden_dim
        )
        return from_gpu_ptr(ptr, (shape0, shape1), dtype)

    def _alloc_routing_handle(self) -> EpDispatchRoutingHandle:
        cfg = self.config
        world_size = cfg.world_size
        m_send = cfg.max_num_inp_token_per_rank
        e = cfg.num_experts_per_token
        r = cfg.num_experts_per_rank
        n_nodes = max(1, world_size // cfg.gpu_per_node)

        device = "cuda"
        return EpDispatchRoutingHandle(
            disp_dest_tok_id_map=torch.zeros(
                world_size * m_send * r, dtype=torch.int32, device=device
            ),
            inter_node_disp_dest_tok_id_map=torch.zeros(
                n_nodes * m_send * e, dtype=torch.int32, device=device
            ),
            inter_node_disp_send_map=torch.zeros(
                n_nodes * m_send, dtype=torch.int32, device=device
            ),
            total_recv_token_num=torch.zeros(1, dtype=torch.int32, device=device),
            disp_tok_id_to_src_tok_id_local=torch.zeros(
                world_size * m_send * r, dtype=torch.int32, device=device
            ),
        )

    def _supports_routing_handle(self) -> bool:
        kt = self.config.kernel_type.value
        return kt in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
        )

    @staticmethod
    def _routing_source_token_count(routing: EpDispatchRoutingHandle) -> int:
        """Return the cache-routing source token count stored in a routing handle."""
        n = int(routing.cur_rank_num_token)
        if n <= 0:
            raise ValueError(
                "routing handle has cur_rank_num_token <= 0; use a handle from "
                "dispatch(..., return_routing=True) or pass a positive "
                "cur_rank_num_token to EpDispatchRoutingHandle.from_tensors(...)"
            )
        return n

    @staticmethod
    def _check_combine_indices(indices, cur_n: int) -> None:
        """Reject recv-slot-layout indices in combine.

        The InterNodeV1 combine indexes tokenIndices by this rank's own token id,
        the same key as interNodeDispSendMap, so anything but this rank's own
        [num_token, topk] routing reduces cross-node tokens against an unrelated
        token's routing and silently corrupts the result (ROCm/mori#475).
        """
        n = int(indices.size(0))
        if n != cur_n:
            raise ValueError(
                f"combine() indices has {n} rows but this rank dispatched {cur_n} "
                f"tokens. Pass this rank's own [num_tokens, topk] routing -- the "
                f"tensor given to dispatch() -- not dispatch()'s returned "
                f"out_idx (ROCm/mori#475)."
            )

    def _build_args_routing(
        self,
        routing,
        *,
        rdma_block_num,
        hidden_dim,
        replay_mode,
        use_external_inp_buf=-1,
    ):
        return mori_cpp.build_args_with_routing(
            self._handle,
            rdma_block_num=rdma_block_num,
            hidden_dim=hidden_dim,
            use_external_inp_buf=use_external_inp_buf,
            replay_mode=replay_mode,
            disp_dest_tok_id_map_ptr=routing.disp_dest_tok_id_map.data_ptr(),
            inter_node_disp_dest_tok_id_map_ptr=routing.inter_node_disp_dest_tok_id_map.data_ptr(),
            inter_node_disp_send_map_ptr=routing.inter_node_disp_send_map.data_ptr(),
            total_recv_token_num_ptr=routing.total_recv_token_num.data_ptr(),
            disp_tok_id_to_src_tok_id_local_ptr=routing.disp_tok_id_to_src_tok_id_local.data_ptr(),
        )

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
        *,
        routing: "EpDispatchRoutingHandle | None" = None,
        return_routing: bool = False,
    ):
        if routing is not None and return_routing:
            raise ValueError(
                "pass either `routing=` (replay) or `return_routing=True` "
                "(new layout), not both"
            )
        use_routing_handle = routing is not None or return_routing
        is_replay = routing is not None
        if use_routing_handle and not self._supports_routing_handle():
            raise NotImplementedError(
                f"routing handle path not supported for kernel_type="
                f"{self.config.kernel_type}; only IntraNode and InterNodeV1 "
                "expose cache/replay routing dispatch."
            )
        if return_routing:
            routing = self._alloc_routing_handle()

        num_tokens = int(input.size(0))
        if is_replay:
            replay_n = self._routing_source_token_count(routing)
            if num_tokens != replay_n:
                raise ValueError(
                    f"replay dispatch input has {num_tokens} tokens but "
                    f"routing.cur_rank_num_token={replay_n}"
                )
            if int(indices.size(0)) != replay_n:
                raise ValueError(
                    f"replay dispatch indices has {int(indices.size(0))} tokens but "
                    f"routing.cur_rank_num_token={replay_n}"
                )

        hidden_dim = input.size(1)
        weight_ptr = weights.data_ptr() if weights is not None else 0
        has_scales = scales is not None and self.config.scale_dim > 0
        scale_ptr = scales.data_ptr() if has_scales else 0
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
            is_intranode_dispatch=True,
        )
        self._cached_dispatch_launch = (actual_bn, actual_rbn, actual_wpb)
        stream = _current_stream()
        self._dispatch_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
        )
        if use_routing_handle:
            args_ptr = self._build_args_routing(
                routing,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                replay_mode=is_replay,
            )
        else:
            args_ptr = mori_cpp.build_args(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
            )

        grid = (actual_bn,)
        block = (self._warp_size * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpDispatchInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1Kernel_{sfx}",
                ],
                [mp, actual_bn],
                [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1KernelLowLatency_{sfx}",
                ],
                [mp, actual_bn],
                [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                self._intranode_dispatch_kernel(sfx),
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNodeLL.value:
            self._launch(
                f"EpDispatchIntraNodeLLKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = self._warp_size * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncSendCopySlotAssign_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendCopyMultiBlock_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendTransfer_{sfx}",
                ],
                [mp_aligned, mp_aligned, self.config.world_size],
                [mb_block, mb_block, self._warp_size * actual_wpb],
                [0, 0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(f"Unsupported dispatch kernel_type: {kt}")

        if call_local_expert_count and kt != EpDispatchCombineKernelType.AsyncLL.value:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            if use_routing_handle:
                total_ptr = routing.total_recv_token_num.data_ptr()
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

        out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr = self._dispatch_out_ptrs
        max_recv = self._cpp_config.max_num_tokens_to_recv()
        out = self._cached_view(
            "disp_out", out_ptr, (max_recv, hidden_dim), input.dtype
        )
        out_weights = self._cached_view(
            "disp_w",
            outW_ptr,
            (max_recv, self.config.num_experts_per_token),
            torch.float32,
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = self._cached_view(
                "disp_s", outS_ptr, (max_recv, self.config.scale_dim), scales.dtype
            )
        out_indices = self._cached_view(
            "disp_i",
            outI_ptr,
            (max_recv, self.config.num_experts_per_token),
            TOPK_IDX_DTYPE,
        )
        total_recv = (
            routing.total_recv_token_num
            if use_routing_handle
            else self._cached_view("disp_total", total_ptr, (1,), TOPK_IDX_DTYPE)
        )

        if return_routing:
            mori_cpp.snapshot_disp_tok_id_to_src_tok_id_local(
                self._handle,
                routing.disp_tok_id_to_src_tok_id_local.data_ptr(),
                stream=stream,
            )
            routing.cur_rank_num_token = int(input.size(0))

        base = (out, out_weights, out_scales, out_indices, total_recv)
        if return_routing:
            return base + (routing,)
        return base

    def dispatch_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        return self.dispatch(
            input,
            weights,
            scales,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def dispatch_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        if hasattr(self, "_cached_dispatch_launch"):
            _, _, actual_wpb = self._cached_dispatch_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_dispatch_dtype"
        ), "dispatch_recv requires a prior dispatch/dispatch_send call"
        sfx = _DTYPE_SUFFIX[self._dispatch_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding dispatch_send/dispatch call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = self._warp_size * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncRecvTransfer_{sfx}",
                    f"EpDispatchLowLatencyAsyncRecvCopyMultiBlock_{sfx}",
                ],
                [self.config.world_size, mp_aligned],
                [self._warp_size * actual_wpb, mb_block],
                [0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                f"dispatch_recv only supports AsyncLL, got kernel_type={kt}"
            )

        if call_local_expert_count:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
        *,
        routing: "EpDispatchRoutingHandle | None" = None,
        sbo_route_tiles: "torch.Tensor | None" = None,
        sbo_tile_state: "torch.Tensor | None" = None,
        sbo_expected_n_tiles: int = 0,
    ):
        """Reduce post-expert tokens back onto this rank's tokens.

        ``indices`` is this rank's own [num_token, topk] routing, the same
        tensor handed to ``dispatch()`` -- not the received-token indices that
        ``dispatch()`` returned.
        """
        if routing is not None and not self._supports_routing_handle():
            raise NotImplementedError(
                f"routing handle path not supported for kernel_type="
                f"{self.config.kernel_type}; only IntraNode and InterNodeV1 "
                "currently consume routing handles in combine."
            )

        hidden_dim = input.size(1)
        weight_ptr = (
            weights.data_ptr() if weights is not None and weights.size(0) != 0 else 0
        )
        actual_use_ext = (
            use_external_inp_buf
            if use_external_inp_buf >= 0
            else int(self.config.use_external_inp_buf)
        )
        is_zero_copy = not actual_use_ext
        sbo_enabled = sbo_route_tiles is not None or sbo_tile_state is not None
        if sbo_enabled:
            if sbo_route_tiles is None or sbo_tile_state is None:
                raise ValueError("MORI SBO requires both route_tiles and tile_state")
            if routing is not None:
                raise ValueError("MORI SBO v1 does not support routing handles")
            if (
                sbo_route_tiles.dtype != torch.int32
                or sbo_tile_state.dtype != torch.int32
                or not sbo_route_tiles.is_contiguous()
                or not sbo_tile_state.is_contiguous()
            ):
                raise ValueError("MORI SBO buffers must be contiguous int32 tensors")
            if (
                sbo_route_tiles.device != input.device
                or sbo_tile_state.device != input.device
            ):
                raise ValueError("MORI SBO buffers must be on the combine input device")
            if sbo_route_tiles.ndim != 2 or sbo_route_tiles.shape[0] != input.shape[0]:
                raise ValueError(
                    "MORI SBO route_tiles must be [recv_capacity, topk], "
                    f"got {tuple(sbo_route_tiles.shape)} for input {tuple(input.shape)}"
                )
            if sbo_route_tiles.shape[1] != self.config.num_experts_per_token:
                raise ValueError(
                    "MORI SBO route width must match configured top-k "
                    f"{self.config.num_experts_per_token}, got "
                    f"{sbo_route_tiles.shape[1]}"
                )
            # v1 permits sort_block_m up to 128 with an M32 Stage2 producer:
            # reserve four counter slots per global expert for padding gaps.
            min_tile_capacity = (
                sbo_route_tiles.numel() + 31
            ) // 32 + (
                self.config.num_experts_per_rank * self.config.world_size * 4
            )
            if sbo_tile_state.ndim != 1 or sbo_tile_state.numel() < min_tile_capacity:
                raise ValueError(
                    "MORI SBO tile_state must cover per-expert sorted padding: "
                    f"need at least {min_tile_capacity}, got {sbo_tile_state.numel()}"
                )
            expected_n_tiles = (input.shape[1] + 127) // 128
            if sbo_expected_n_tiles != expected_n_tiles:
                raise ValueError(
                    f"MORI SBO expected_n_tiles must be {expected_n_tiles}, "
                    f"got {sbo_expected_n_tiles}"
                )
            if (
                self.config.kernel_type != EpDispatchCombineKernelType.IntraNode
                or _normalize_quant_type(self.config.quant_type)
                != EpDispatchCombineQuantType.Fp8BlockwiseQuant
                or input.dtype != torch.bfloat16
                or not actual_use_ext
            ):
                raise ValueError(
                    "MORI SBO v1 requires IntraNode bf16 input with external "
                    "FP8-blockwise combine"
                )
        cur_n = (
            self._routing_source_token_count(routing)
            if routing is not None
            else self._get_cur_rank_num_token(self._handle)
        )
        self._check_combine_indices(indices, cur_n)
        # The width default follows the TRANSPORT. Same three conditions the _nop2p suffix is
        # chosen by at the launch below, and deliberately not "actual_use_ext" alone -- blockwise
        # and direct_cast own their input too, reach the gather by other routes, and have no
        # 16-warp measurement behind them.
        is_push = bool(
            actual_use_ext
            and self.config.kernel_type.value
            == EpDispatchCombineKernelType.IntraNode.value
            and _normalize_quant_type(self.config.quant_type)
            == EpDispatchCombineQuantType.None_
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=cur_n,
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=is_zero_copy,
            quant_type=self._qt_str,
            is_intranode_combine=True,
            is_push_transport=is_push,
        )
        self._cached_combine_launch = (actual_bn, actual_rbn, actual_wpb)
        if sbo_enabled and actual_bn >= self._handle_info["multi_processor_count"]:
            raise ValueError(
                "MORI SBO combine must reserve at least one CU for its producer: "
                f"blocks={actual_bn}, CUs={self._handle_info['multi_processor_count']}"
            )
        stream = _current_stream()
        self._combine_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=cur_n,
            weight_ptr=weight_ptr,
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        if routing is not None:
            args_ptr = self._build_args_routing(
                routing,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                replay_mode=False,
                use_external_inp_buf=use_external_inp_buf,
            )
        elif sbo_enabled:
            args_ptr = mori_cpp.build_args_sbo(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                use_external_inp_buf=use_external_inp_buf,
                route_tiles_ptr=sbo_route_tiles.data_ptr(),
                tile_state_ptr=sbo_tile_state.data_ptr(),
                topk=sbo_route_tiles.shape[1],
                expected_n_tiles=sbo_expected_n_tiles,
            )
        else:
            args_ptr = mori_cpp.build_args(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                use_external_inp_buf=use_external_inp_buf,
            )

        grid = (actual_bn,)
        block = (self._warp_size * actual_wpb,)
        kt = self.config.kernel_type.value
        quant_type = _normalize_quant_type(self.config.quant_type)
        shared_mem = self._combine_shared_mem(actual_wpb)

        if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
            label = (
                "fp4_blockwise"
                if quant_type == EpDispatchCombineQuantType.Fp4BlockwiseQuant
                else "fp8_blockwise"
            )
            # fp8 blockwise also runs on AsyncLL; fp4 blockwise is IntraNode-only.
            allowed_kts = [
                EpDispatchCombineKernelType.IntraNode.value,
                EpDispatchCombineKernelType.IntraNodeLL.value,
            ]
            if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                allowed_kts.append(EpDispatchCombineKernelType.AsyncLL.value)
            if kt not in allowed_kts:
                supported = (
                    "IntraNode/IntraNodeLL/AsyncLL"
                    if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant
                    else "IntraNode/IntraNodeLL"
                )
                raise ValueError(
                    f"{label} combine currently only supports {supported} combine"
                )
            if sfx != "bf16":
                raise ValueError(f"{label} combine only supports bf16, got {sfx}")
            if not actual_use_ext:
                raise ValueError(
                    f"{label} combine currently requires --zero-copy 0 "
                    "(useExternalInpBuffer=True). P2P read path not yet implemented."
                )
            if self._fp8_blockwise_combine_scale_dim <= 0:
                raise ValueError(
                    f"{label} combine requires internal combine scale_dim > 0"
                )

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpCombineInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = self._warp_size * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1Kernel_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, self._warp_size, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = self._warp_size * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1KernelLowLatency_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, self._warp_size, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.IntraNodeLL.value,
        ):
            if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
                # Mirror of the AccumNum=8/9 + VecBytes=8 specialization gating in
                # LaunchCombine() / launch.cpp. top-k==9 covers shared-expert fusion
                # (8 routed + 1 fused shared). Keep in sync.
                fp8_scale_dim = self._fp8_blockwise_combine_scale_dim
                block_elems = (hidden_dim + fp8_scale_dim - 1) // fp8_scale_dim
                base_vec8_top8_eligible = (
                    weight_ptr == 0
                    and (hidden_dim % 512) == 0
                    and self.config.num_experts_per_token in (8, 9)
                    and self.config.world_size > 4
                )
                top9 = self.config.num_experts_per_token == 9
                # PUSH on every arch. gfx125x used to pick the _p2p symbol here so blockwise could
                # ride the TDM tile fold, which was worth 3646.2us -> 1410.9us; that fold is gone
                # (intranode_entry.hpp keeps quantization out of the TDM body), so the _p2p symbol
                # no longer buys anything and cost the weightless top8/top9 vec8 kernels below,
                # which have no _p2p variant.
                kernel_name = "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise"
                use_vec8_top8 = False
                if base_vec8_top8_eligible:
                    if block_elems == 128:
                        kernel_name = (
                            "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise_noweight_block128_vec8_top9"
                            if top9
                            else "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise_noweight_block128_vec8"
                        )
                        use_vec8_top8 = True
                    elif block_elems == 256:
                        kernel_name = (
                            "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise_noweight_block256_vec8_top9"
                            if top9
                            else "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise_noweight_block256_vec8"
                        )
                        use_vec8_top8 = True
                # Blockwise FP4: select the packed-FP4 kernel variants (identical launch config to
                # the fp8_blockwise variants; only the in-kernel quant/dequant math differs). Assert the
                # derived name is a registered fp4_blockwise symbol so a naming mismatch fails loudly.
                if quant_type == EpDispatchCombineQuantType.Fp4BlockwiseQuant:
                    kernel_name = kernel_name.replace(
                        "_fp8_blockwise", "_fp4_blockwise"
                    )
                    assert (
                        kernel_name in _FP4_COMBINE_KERNELS
                    ), f"fp4_blockwise combine selected unregistered kernel '{kernel_name}'"
                if sbo_enabled:
                    if (
                        quant_type
                        != EpDispatchCombineQuantType.Fp8BlockwiseQuant
                        or kernel_name
                        != "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise"
                    ):
                        raise ValueError(
                            "MORI SBO v1 requires generic IntraNode FP8-blockwise combine"
                        )
                    kernel_name += "_sbo"
                shared_mem = self._combine_shared_mem(
                    actual_wpb, use_weights=not use_vec8_top8
                )
                self._last_combine_kernel_name = kernel_name
                self._launch(
                    kernel_name,
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
            elif actual_use_ext:
                if (
                    sfx == "bf16"
                    and quant_type == EpDispatchCombineQuantType.Fp8DirectCast
                ):
                    self._launch(
                        "EpCombineIntraNodeKernel_bf16_nop2p_fp8cast",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
                else:
                    self._launch(
                        f"EpCombineIntraNodeKernel_{sfx}_nop2p",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
            else:
                self._launch(
                    f"EpCombineIntraNodeKernel_{sfx}_p2p",
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncSendCopy_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncSendTransfer_bf16_fp8cast",
                    ],
                    [mp_aligned, self.config.world_size],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
            elif quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncSendCopy_bf16_fp8_blockwise",
                        "EpCombineLowLatencyAsyncSendTransfer_bf16_fp8_blockwise",
                    ],
                    [mp_aligned, self.config.world_size],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncSendCopy_{sfx}",
                        f"EpCombineLowLatencyAsyncSendTransfer_{sfx}",
                    ],
                    [mp_aligned, self.config.world_size],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(f"Unsupported combine kernel_type: {kt}")

        out_ptr, outW_ptr = self._combine_out_ptrs
        out = self._cached_view(
            "comb_out",
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = self._cached_view(
                "comb_w",
                outW_ptr,
                (
                    self.config.max_num_inp_token_per_rank,
                    self.config.num_experts_per_token,
                ),
                weights.dtype,
            )

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def combine_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.combine(
            input,
            weights,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def combine_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        if hasattr(self, "_cached_combine_launch"):
            _, _, actual_wpb = self._cached_combine_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_combine_dtype"
        ), "combine_recv requires a prior combine/combine_send call"
        sfx = _DTYPE_SUFFIX[self._combine_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding combine_send/combine call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        shared_mem = self._combine_shared_mem(actual_wpb)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            quant_type = _normalize_quant_type(self.config.quant_type)
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncRecvTransfer_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncRecvCopy_bf16_fp8cast",
                    ],
                    [self.config.world_size, mp_aligned],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
            elif quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncRecvTransfer_bf16",
                        "EpCombineLowLatencyAsyncRecvCopy_bf16_fp8_blockwise",
                    ],
                    [self.config.world_size, mp_aligned],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncRecvTransfer_{sfx}",
                        f"EpCombineLowLatencyAsyncRecvCopy_{sfx}",
                    ],
                    [self.config.world_size, mp_aligned],
                    [self._warp_size * actual_wpb, self._warp_size * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(
                f"combine_recv only supports AsyncLL, got kernel_type={kt}"
            )

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "dispatch_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        hidden_dim = input.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=input.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=input.device
        )

        set_fn(self._handle, packed_recv_x.data_ptr(), packed_recv_src_info.data_ptr())

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=(weights.data_ptr() if weights is not None else 0),
            scale_ptr=(
                scales.data_ptr()
                if scales is not None and self.config.scale_dim > 0
                else 0
            ),
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (self._warp_size * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(
                f"EpDispatchCopyToStaging_{sfx}", (mp,), block, 0, stream, args_ptr
            )
            self._launch(
                f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                self._intranode_dispatch_kernel(sfx, stdmoe=True),
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "dispatch_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def combine_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "combine_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        hidden_dim = input.size(2)
        cur_n = self._get_cur_rank_num_token(self._handle)
        self._check_combine_indices(indices, cur_n)
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=cur_n,
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=False,
            quant_type=self._qt_str,
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        set_fn(self._handle, input.data_ptr(), 0)

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=cur_n,
            weight_ptr=(
                weights.data_ptr()
                if weights is not None and weights.size(0) != 0
                else 0
            ),
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (self._warp_size * actual_wpb,)
        shared_mem = self._combine_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(f"EpCombineSync_{sfx}", (mp,), block, 0, stream, args_ptr)
            self._launch(
                f"EpCombineSyncBarrier_{sfx}",
                (1,),
                (self._warp_size,),
                0,
                stream,
                args_ptr,
            )
            self._launch(
                f"EpCombineInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
            self._launch(
                f"EpCombineAll_{sfx}", (mp,), block, shared_mem, stream, args_ptr
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpCombineIntraNodeKernel_{sfx}_p2p_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "combine_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        out_ptr = self._combine_out_ptrs[0]
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def convert_dispatch_output(
        self,
        dispatch_out_x: torch.Tensor,
        dispatch_out_topk_idx: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_dispatch_output_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_dispatch_output is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = dispatch_out_x.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=dispatch_out_x.dtype,
            device=dispatch_out_x.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=dispatch_out_x.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=dispatch_out_x.device
        )

        args_ptr = build_fn(
            self._handle,
            dispatch_out_x.data_ptr(),
            dispatch_out_topk_idx.data_ptr(),
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
        )
        try:
            grid = (actual_bn,)
            block = (self._warp_size * actual_wpb,)
            self._launch(
                "mori_ConvertDispatchOutputKernel", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def convert_combine_input(
        self,
        packed_recv_x: torch.Tensor,
        packed_recv_src_info: torch.Tensor,
        packed_recv_layout_range: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_combine_input_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_combine_input is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = packed_recv_x.size(2)
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[packed_recv_x.dtype]

        args_ptr = build_fn(
            self._handle,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
        )
        try:
            grid = (actual_bn,)
            block = (self._warp_size * actual_wpb,)
            self._launch(
                f"ConvertCombineInputKernel_{sfx}", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        max_recv = self._cpp_config.max_num_tokens_to_recv()
        combine_input_ptr = mori_cpp.get_combine_input_ptr(self._handle)
        return from_gpu_ptr(
            combine_input_ptr, (max_recv, hidden_dim), packed_recv_x.dtype
        )

    def reset(self):
        self._reset_func(self._handle, _current_stream())

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.IntraNodeLL.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
            EpDispatchCombineKernelType.InterNodeV1LL.value,
            EpDispatchCombineKernelType.AsyncLL.value,
        ):
            ptr, size = self._get_dispatch_src_token_pos_func(self._handle)
            return from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_sender_token_idx_map_func(self._handle)
        dispatch_sender_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_receiver_token_idx_map_func(self._handle)
        dispatch_receiver_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for i in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            src_token_pos.append(src_pe * max_num_token_to_send_per_rank + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

    def get_debug_time_buf(self):
        """Get the debug time buffer as a torch.Tensor (int64)."""
        ptr, size = mori_cpp.get_debug_time_buf(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int64)

    def get_debug_time_offset(self):
        """Get the debug time offset buffer as a torch.Tensor (int32)."""
        ptr, size = mori_cpp.get_debug_time_offset(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int32)
