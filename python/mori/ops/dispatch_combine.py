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
        raise RuntimeError("MORI_EP_COMM=cco requires an initialized torch.distributed group")
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
def _comb_qpull() -> bool:
    """Whether blockwise combine moves its bytes by PULL gather rather than PUSH staging.

    gfx125x defaults to PULL, every other arch keeps PUSH, and MORI_COMB_QPULL wins either way.

    The two differ only in who carries the bytes, not in what is computed. PUSH has the producer
    quantise straight into the peer's staging slot, which is a per-lane cross-card scatter that
    cannot use a TDM bulk store. PULL has it quantise into its own buffer and lets each consumer
    read peer fp8 plus scales in bulk. MEASURED 64x8 EP4 bf16 fp8_blockwise, correctness check
    armed, rc=0 on both: PUSH 3646.2us / 58.3 GB/s against PULL 1410.9us / 150.5 GB/s.

    Chunk count is not part of the win and the recorded 1466.4us figure misattributed it:
    MORI_COMB_TDM=4 on its own moves nothing (3651.7us against 3646.2us), and on top of PULL it
    COSTS 3.9% (1466.7us against 1410.9us). PULL at the default chunk count is the fast one.

    PULL gives up the weightless top8/top9 vec8 kernels, which have no _p2p variant. That is free at
    EP4, where those need worldSize > 4 to be eligible at all, and is NOT measured above it.
    """
    val = os.environ.get("MORI_COMB_QPULL", "").strip().lower()
    if val:
        return val in ("1", "true", "on", "yes")
    return _is_gfx125x()


def _comb_pull_mode() -> str:
    """How a CALLER-OWNED (use_external_inp_buf) combine input reaches the PULL gather.

    PULL is the fast transport -- each consumer bulk-reads its peers' buffers instead of having
    every producer scatter per-lane writes across the fabric -- but the kernel can only read a
    peer's buffer if the bytes are IN a symmetric buffer, and a caller-owned tensor is not one.
    That is the whole reason use_external_inp_buf has been pinned to PUSH. The bytes therefore
    have to be staged into the registered buffer first, and this picks who does it:

      "host"   -- a d2d copy on the caller's stream before launch. The kernel then runs the
                  ordinary zero-copy _p2p path unchanged, so correctness is inherited from the
                  path that already passes. MEASURED 64x8 EP4 bf16, check armed, rc=0:
                  236.7us / 896.9 GB/s against PUSH 318.8us / 666.0 GB/s. The 67.7us over the
                  169.0us zero-copy reference is the copy, 424 MB of local traffic at 6.3 TB/s.
      "kernel" -- KNOWN WRONG, kept only to name what was tried. The kernel stages the token
                  itself and then reads peers, which is what blockwise does under QPULL, and it
                  fails the correctness check at EP4 (rc=1, max diff 3.75 against tol 0.159 on
                  3 of 4 ranks). Two candidate causes are already excluded by measurement rather
                  than argument: MORI_COMB_RELFENCE (a per-block release fence) does not fix it,
                  and MORI_COMB_FASTPATH=0 does not either, so it is neither cross-card
                  visibility nor the QUAD/TDM gather.
      "both"   -- DIAGNOSTIC: host copy AND the in-kernel staging, which rewrites the same bytes.
                  Isolates the staging loop, the one thing that differs between host and kernel.
      "off"    -- keep PUSH.

    Default is set by measurement in _comb_pull_default(); MORI_COMB_PULL overrides.
    """
    val = os.environ.get("MORI_COMB_PULL", "").strip().lower()
    if val in ("host", "kernel", "both", "off"):
        return val
    if val in ("1", "true", "on", "yes"):
        return "host"
    if val in ("0", "false", "no"):
        return "off"
    return _comb_pull_default()


def _comb_pull_default() -> str:
    """Transport for a caller-owned combine input when MORI_COMB_PULL is unset.

    gfx125x stages and pulls; everywhere else keeps PUSH, since the win is measured on the TDM
    gather and nothing says it carries to an arch without one. Owning your input buffer used to
    cost 25.8% of combine (318.8us against 236.7us at 64x8 EP4) for a property of the allocation
    rather than anything about the data.
    """
    return "host" if _is_gfx125x() else "off"


def _comb_env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if val.isdigit() and int(val) > 0:
        return int(val)
    return default


def _comb_qpre_mode() -> str:
    """Whether the blockwise quantise pass runs as its OWN kernel before the combine kernel.

    This is the blockwise counterpart of _comb_pull_mode()="host", and it is here for the same
    reason: staging and gathering want different launch widths, and one fused kernel can only have
    one. The gather is bound by peer reads in flight and by the LDS its tiles need; the quantise is
    a pure local stream over the caller's bf16 with no cross-card edge in it at all. Fusing them
    made one width serve both. MEASURED EP4 fp8_blockwise, check armed, rc=0 on every row:

        inline, combine 64x8                        1011.3us
        split, pre 256x8, combine 64x8               856.4
        split, pre 256x8, combine 256x8              428.2
        split, pre 256x8, combine 256x16             367.6
        bf16 zero-copy PULL 64x8 (the bar)           169.2   / 1254.7 GB/s

    Only the PULL blockwise path can use it: PUSH quantises straight into the peer's slot, which is
    the transport itself and not a separable pass.

      "on"  / 1  -- launch it.
      "off" / 0  -- keep the pass inside the combine kernel.
      "noq"      -- DIAGNOSTIC, WRONG RESULTS: take the split path but never launch the pre-kernel,
                    so the combine kernel is the gather and nothing else. This is the deletion
                    pricing for the pass, and it exists as its own mode because the -D that used to
                    do that job (MORI_COMB_NOQUANT) changes the build, and a new -D set costs a
                    ~890s JIT rebuild on this node. Read in Python, so it is free.

    MORI_COMB_QPRE_BN and MORI_COMB_QPRE_WPB set the pre-pass geometry (default: 4 blocks per CU,
    8 warps each). MEASURED at combine 64x8, pre-grid alone moving: 1024x8 1143.5us, 512x8 940.7,
    128x8 876.0, 512x4 860.4, 256x8 856.4.
    """
    val = os.environ.get("MORI_COMB_QPRE", "").strip().lower()
    if val in ("1", "true", "on", "yes"):
        return "on"
    if val in ("0", "false", "no", "off"):
        return "off"
    if val == "noq":
        return "noq"
    return "on" if _is_gfx125x() else "off"


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
# src/ops/dispatch_combine/intranode.hpp: the kernel picks its combine transport by testing its
# tiles against that number, and this function reserves for whichever one it will pick.
_COMB_LDS_BUDGET = 327680


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

        self._dispatch_out_ptrs = mori_cpp.get_dispatch_output_ptrs(self._handle, True)
        self._combine_out_ptrs = mori_cpp.get_combine_output_ptrs(self._handle, True)

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

    # ------------------------------------------------------------------
    # Kernel launch helpers
    # ------------------------------------------------------------------
    def _intranode_dispatch_default_launch(self):
        """Per-body default (block_num, warp_per_block) for the IntraNode dispatch kernel.

        Split on the ARCH, matching the #if that picks the body in intranode.hpp. gfx125x runs
        EpDispatchIntraNodeKernel_body and takes 64x8; every other arch runs the WarpCopy body,
        which interleaves scattered per-token metadata with the payload and needs the wide grid to
        hide that, so it keeps its historical 256x16.

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

    def _intranode_combine_default_launch(self):
        """Per-body default (block_num, warp_per_block) for the IntraNode combine kernel.

        64x8 on gfx125x, the same geometry dispatch defaults to, so an intra-node caller with no
        opinion gets ONE shape for both phases instead of two unrelated ones. (0, 0) elsewhere means
        "no opinion, use the config defaults", which is the behaviour every arch had before the QUAD
        transport existed; the width only matters where the TDM transports exist at all.

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
        return 64, 8

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
    ):
        if tuning_rules and dtype is not None:
            from mori.ops.tuning_config import TuningConfigManager

            params = TuningConfigManager.lookup(
                tuning_rules, dtype, num_tokens, hidden_dim, zero_copy, quant_type
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
                def_bn, def_wpb = self._intranode_combine_default_launch()
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
        # This tests the ARCH, matching the #if that selects the body in intranode.hpp. It used to
        # test MORI_DISP_TDM, which meant an unset env reserved `base` for a kernel that stages
        # 14KB tiles per warp into that reservation.
        if _is_gfx125x():
            # One FULL token tile per warp = hidden*elemSize bytes (14KB at hidden 7168 bf16), so
            # wpb<=16 stays inside the 320KB gfx1250 LDS budget. A second tile per warp for payload
            # double-buffering is not worth its 229KB (measured 1280.8 vs 1280.7 GB/s, see the drain
            # comment in intranode.hpp's payload loop).
            # The default dispatch body reuses this same per-warp tile for its batched metadata
            # send (see the tokCapM computation in intranode.hpp), so no extra budget is needed.
            tile = (
                warp_per_block
                * int(self.config.hidden_dim)
                * int(self.config.max_token_type_size)
            )
            return max(base, tile)
        return base

    def _intranode_dispatch_kernel(self, sfx, stdmoe=False):
        """Intra-node dispatch. One launch symbol, one body: EpDispatchIntraNodeKernel_body in
        intranode.hpp."""
        name = f"EpDispatchIntraNodeBatchKernel_{sfx}"
        if stdmoe:
            name += "_stdmoe"
        return name

    def _combine_shared_mem(self, warp_per_block, use_weights=True, force_pull=False):
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
        # MORI_COMB_TDM=N routes the combine token through TDM, which needs per-warp LDS tiles holding
        # one chunk of a token. The kernel places them right AFTER the pointer arrays above, so this
        # is a sum and not a max, and chunk_elems must match _cTileElems/_cPullTileElems in
        # intranode.hpp: chunk rounded up to a whole 128B TDM row.
        #
        # The two transports need different tile counts, and which one is compiled follows the same
        # flag that picks the kernel: use_external_inp_buf=True -> _nop2p -> PUSH (one tile per warp,
        # staged then TDM-stored to the peer); False -> _p2p -> PULL (one tile per source, all topk
        # loads issued before the wait).
        #
        # Blockwise-quant combine is excluded because the kernel gates TDM on
        # !UseFp8BlockwiseQuant -- allocating a tile there would only shrink occupancy, and could
        # trip the budget check below on a configuration that never runs TDM at all.
        #
        # The exclusion is total, and that is fine rather than merely tolerable: blockwise quant
        # requires PUSH (bench and launch.cpp both reject --zero-copy 1 for it) while QUAD requires
        # PULL, so the two can never be asked for at once. MEASURED with the arch defaults live,
        # fp8_blockwise EP4 bf16 hidden 7168 4096 tokens at 64x8, check armed: 6037.6us rc=0, and
        # the same run with MORI_COMB_FASTPATH=0 gives 6173.5us rc=0 -- within 2%, i.e. the gates
        # are genuinely inert here, not silently changing a quantised result. (That 34 GB/s is its
        # own problem and has nothing to do with these gates.)
        from mori.jit.core import _comb_env_set, _comb_tdm_chunks

        chunks = _comb_tdm_chunks()
        # ...that exclusion held only while blockwise was PUSH-only. Under MORI_COMB_QPULL the
        # kernel runs _p2p and _cPullBwq lets blockwise onto the tile path, so the tiles have to be
        # reserved here or _cPullOk fails its budget check and silently drops back to the lane
        # gather -- the slow thing this is meant to remove.
        _qpull_bwq = _comb_qpull() and (
            quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant
        )
        if chunks and (quant_type not in _BLOCKWISE_COMBINE_QUANT_TYPES or _qpull_bwq):
            # The tile holds TokT, which is 1-byte fp8 under blockwise rather than the token dtype,
            # so both the row width and the byte total below follow sizeof(TokT). Mirrors
            # _cPullRowElems = 128 / sizeof(TokT).
            elem = 1 if _qpull_bwq else int(self.config.max_token_type_size)
            row_elems = 128 // elem
            hidden = int(self.config.hidden_dim)
            topk = int(self.config.num_experts_per_token)
            world = int(self.config.world_size)
            # _qpull_bwq keeps use_external_inp_buf True (that is what makes the kernel quantise
            # locally) while compiling UseP2PRead=true, so the transport here follows the KERNEL,
            # not the buffer flag. This is the one case where the two disagree.
            # force_pull is the same disagreement as _qpull_bwq from the other direction: the
            # caller owns the input buffer but its bytes were staged into the registered one
            # before launch, so the kernel is a _p2p kernel and needs PULL tiles.
            if self.config.use_external_inp_buf and not _qpull_bwq and not force_pull:
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
                # intranode.hpp, including the world_size <= 4 condition -- that is where the
                # compaction that makes the tile indices dense runs.
                tile_elems = -(-hidden // chunks)
                tile_elems = -(-tile_elems // row_elems) * row_elems
                tiles_per_warp = world if (world <= 4 and world < topk) else topk
                # MORI_COMB_QUAD turns the decomposition 90 degrees: one warp owns one SOURCE and
                # reads that source's token, so a warp needs one tile instead of one per source.
                # QUAD=D gives it D of them, QSPLIT=S cuts each to hidden/S elements, and the two
                # trade off exactly: D*(hidden/S)*elem*wpb bytes buy D-1 reads in flight at
                # (hidden/S)*elem bytes each.
                from mori.jit.core import _comb_quad_depth, _comb_quad_split

                # QUAD now serves blockwise too (the kernel gate takes _cPullBwq), and `elem` above
                # is already 1 there, so the tiles reserved here are the fp8 ones the kernel lays
                # out. Getting this wrong in either direction is a silent layout mismatch, not a
                # slowdown: host and kernel must agree on the tile size to the byte -- which is why
                # the depth-4 floor below is repeated from intranode.hpp rather than assumed. See
                # there for why blockwise refuses depth 2 (wrong results, cause not yet found).
                _qd = _comb_quad_depth()
                if _qpull_bwq and _qd < 4:
                    _qd = 0
                if _qd >= 2:
                    _qs = _comb_quad_split()
                    tile_elems = hidden // _qs
                    tiles_per_warp = _qd
                    tile_bytes = warp_per_block * tiles_per_warp * tile_elems * elem
                    # ... plus one int per (warp, buffer) for the source-count ring that follows
                    # the tiles (it is in LDS because holding it in registers spills), and two
                    # more per (group, buffer) for MORI_COMB_QFLAG's arrival/departure counters.
                    _qgroups = max(1, warp_per_block // world)
                    tile_bytes += (warp_per_block + 2 * _qgroups) * tiles_per_warp * 4
                    # MORI_COMB_QLOC needs the local source's POINTER to survive from the launch of
                    # a unit to its fold, which is the same ring problem as the count: two more ints
                    # per (warp, buffer).
                    from mori.jit.core import _comb_qloc

                    if _comb_qloc():
                        tile_bytes += 2 * warp_per_block * tiles_per_warp * 4
                    # MORI_COMB_QTST folds into LDS and lets the TDM engine store that out, which
                    # costs one more buffer set of the fold's SLICE of a tile -- 1/world of the
                    # tiles -- 128B-aligned past the counters.
                    from mori.jit.core import _comb_qob, _comb_qtst

                    if _comb_qtst():
                        _qob = max(_comb_qob(), _qd)
                        tile_bytes = (tile_bytes + 127) & ~127
                        tile_bytes += warp_per_block * _qob * (tile_elems // world) * elem
                    total = ((base + 127) & ~127) + tile_bytes
                    # Mirrors the _qLdsNeed guard in intranode.hpp, which declines QUAD at a width
                    # whose tiles do not fit and lets the token fall through to the chunked gather.
                    # Reserving QUAD's footprint here while the kernel runs the chunked path (or the
                    # reverse) is a silent layout mismatch, so the two predicates are the same one.
                    if total <= _COMB_LDS_BUDGET:
                        return total
                    if _comb_env_set("MORI_COMB_QUAD"):
                        raise ValueError(
                            f"MORI_COMB_QUAD={_qd} MORI_COMB_QSPLIT={_qs} needs {total} B of LDS "
                            f"for combine (warp_per_block={warp_per_block}) but the budget is "
                            f"{_COMB_LDS_BUDGET} B. Lower warp_per_block, raise MORI_COMB_QSPLIT, "
                            f"or unset MORI_COMB_QUAD to let the arch default pick a transport "
                            f"that fits."
                        )
                    # Arch default, not a request: fall through to the chunked tiles below, which
                    # are what the kernel will use.
                    tile_elems = -(-hidden // chunks)
                    tile_elems = -(-tile_elems // row_elems) * row_elems
                    tiles_per_warp = world if (world <= 4 and world < topk) else topk
                # MORI_COMB_PIPE=N gives the PULL gather N tile sets per warp so N-1 chunks of peer
                # reads stay in flight while one is folded. Must match _cPullBufs in intranode.hpp.
                _pipe = os.environ.get("MORI_COMB_PIPE", "").strip().lower()
                _pipe_n = (
                    int(_pipe) if _pipe.isdigit() else (2 if _pipe in ("true", "on", "yes") else 0)
                )
                if _pipe_n == 1:
                    _pipe_n = 2
                if _pipe_n >= 2:
                    tiles_per_warp *= _pipe_n
            tile_bytes = warp_per_block * tiles_per_warp * tile_elems * elem
            # The kernel rounds past the pointer arrays to 128B (TDM row) before the first tile.
            total = ((base + 127) & ~127) + tile_bytes
            if total <= _COMB_LDS_BUDGET:
                base = total
            elif _comb_env_set("MORI_COMB_TDM") or (
                self.config.use_external_inp_buf and not _qpull_bwq and not force_pull
            ):
                # PUSH has no fallback in the kernel -- its fold aliases the send tile rather than
                # allocating one -- so an overflow there is still fatal, as is one the caller asked
                # for by naming MORI_COMB_TDM. The buffer flag alone does not mean PUSH any more:
                # under _qpull_bwq and force_pull the caller owns the input while the kernel
                # compiled is _p2p, and _cPullOk there declines the tiles and gathers without them.
                # Raising on those turned a fallback into a crash, which is how MORI_COMB_PIPE=2 at
                # 16 warps behaves -- 458 KB of tiles against a 320 KB budget.
                raise ValueError(
                    f"MORI_COMB_TDM={chunks} needs {total} B of LDS for combine "
                    f"(warp_per_block={warp_per_block}, tiles/warp={tiles_per_warp}, "
                    f"tile_elems={tile_elems}) but the budget is {_COMB_LDS_BUDGET} B. "
                    f"Raise MORI_COMB_TDM (smaller PULL chunks) or lower warp_per_block."
                )
            # Else: arch default at a width the tiles do not fit. _cPullOk in intranode.hpp fails
            # the same test and the gather falls back to WarpAccumLF, which needs no tiles, so the
            # reservation stays at the pointer arrays.
        return base

    _launch_traced = set()

    def _launch(self, func_name, grid, block, shared_mem, stream, args_ptr):
        # MORI_EP_TRACE_LAUNCH=1 prints each distinct (symbol, geometry, LDS) once. Which symbol a
        # config lands on, and how much LDS it asked for, decide which in-kernel transport is
        # eligible -- the tile paths all carry a runtime budget check and fall back silently when
        # it fails -- and reading that off the source means tracking arch defaults through three
        # files. One line of output settles it instead.
        if os.environ.get("MORI_EP_TRACE_LAUNCH", "").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        ):
            key = (func_name, grid[0], block[0], shared_mem)
            if key not in EpDispatchCombineOp._launch_traced:
                EpDispatchCombineOp._launch_traced.add(key)
                print(
                    f"[EPLAUNCH] {func_name} grid={grid[0]} block={block[0]} "
                    f"lds={shared_mem}",
                    flush=True,
                )
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
        out = from_gpu_ptr(out_ptr, (max_recv, hidden_dim), input.dtype)
        out_weights = from_gpu_ptr(
            outW_ptr, (max_recv, self.config.num_experts_per_token), torch.float32
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = from_gpu_ptr(
                outS_ptr, (max_recv, self.config.scale_dim), scales.dtype
            )
        out_indices = from_gpu_ptr(
            outI_ptr, (max_recv, self.config.num_experts_per_token), TOPK_IDX_DTYPE
        )
        total_recv = (
            routing.total_recv_token_num
            if use_routing_handle
            else from_gpu_ptr(total_ptr, (1,), TOPK_IDX_DTYPE)
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
    ):
        if routing is not None and not self._supports_routing_handle():
            raise NotImplementedError(
                f"routing handle path not supported for kernel_type="
                f"{self.config.kernel_type}; only IntraNode and InterNodeV1 "
                "currently consume routing handles in combine."
            )

        if routing is not None:
            routing_n = self._routing_source_token_count(routing)
            if int(indices.size(0)) != routing_n:
                raise ValueError(
                    f"combine indices has {int(indices.size(0))} tokens but "
                    f"routing.cur_rank_num_token={routing_n}"
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
        # "PULL + caller-owned input", host route: stage the caller's bytes into the registered
        # buffer here, on their stream, then run the ordinary zero-copy _p2p kernel untouched.
        # The stream orders the copy before the launch, so the kernel's existing producer/consumer
        # barrier remains the only cross-card edge -- exactly the situation true zero copy is in,
        # which is why this inherits its correctness rather than needing a new argument for it.
        # Restricted to unquantised intranode combine: blockwise reaches PULL through
        # _comb_qpull instead, and fp8_direct_cast has no _p2p kernel to fall into.
        force_pull = False
        if (
            actual_use_ext
            and self.config.kernel_type.value
            == EpDispatchCombineKernelType.IntraNode.value
            and _normalize_quant_type(self.config.quant_type)
            == EpDispatchCombineQuantType.None_
        ):
            _pm = _comb_pull_mode()
            if _pm in ("host", "both"):
                reg = self.get_registered_combine_input_buffer(input.dtype, hidden_dim)
                reg[: input.size(0)].copy_(input)
                force_pull = True
                if _pm == "host":
                    # From here on this call is indistinguishable from a zero-copy one, which is
                    # the point: no kernel, no argument and no tile size differs from the path
                    # that already passes. "both" deliberately leaves the flag set instead, so
                    # the kernel also stages, and the only live difference is that loop.
                    use_external_inp_buf = 0
                    actual_use_ext = 0
                    is_zero_copy = True
            elif _pm == "kernel":
                # The buffer flag stays set -- that is what makes the kernel run its own staging
                # arm -- while the kernel compiled is the PULL one, so the transport follows the
                # kernel and not the flag. Same disagreement _qpull_bwq already relies on.
                force_pull = True
        # Blockwise PULL: run the quantise pass as its own kernel and tell the combine kernel the
        # input buffer is no longer external, which is the one flag its staging arm reads. Decided
        # here rather than at the launch below because the flag has to reach build_args.
        qpre_mode = _comb_qpre_mode()
        qpre = (
            qpre_mode != "off"
            and actual_use_ext
            and self.config.kernel_type.value
            == EpDispatchCombineKernelType.IntraNode.value
            and _normalize_quant_type(self.config.quant_type)
            == EpDispatchCombineQuantType.Fp8BlockwiseQuant
            and _comb_qpull()
        )
        kernel_ext_flag = 0 if qpre else use_external_inp_buf
        cur_n = (
            self._routing_source_token_count(routing)
            if routing is not None
            else self._get_cur_rank_num_token(self._handle)
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
        )
        self._cached_combine_launch = (actual_bn, actual_rbn, actual_wpb)
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
                use_external_inp_buf=kernel_ext_flag,
            )
        else:
            args_ptr = mori_cpp.build_args(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                use_external_inp_buf=kernel_ext_flag,
            )

        grid = (actual_bn,)
        block = (self._warp_size * actual_wpb,)
        kt = self.config.kernel_type.value
        quant_type = _normalize_quant_type(self.config.quant_type)
        shared_mem = self._combine_shared_mem(actual_wpb, force_pull=force_pull)

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
                # PULL, the gfx125x default (_comb_qpull), routes blockwise over the gather instead
                # of the PUSH staging. useExternalInpBuffer stays true either way -- that is what
                # makes the kernel quantise into its own combineInp rather than expecting bf16
                # already there -- so this only swaps which half of the kernel moves the bytes.
                # It costs the weightless top8/top9 vec8 kernels below, which have no _p2p variant;
                # they need worldSize > 4 to be eligible, so at EP4 there is nothing to give up.
                # fp4 is excluded: the kernel's _cPullBwq gate refuses UseFp4Combine (packed 2-per-
                # byte indexing is not what the tile fold assumes) and no _p2p_fp4_blockwise is registered.
                _qpull = _comb_qpull() and (
                    quant_type != EpDispatchCombineQuantType.Fp4BlockwiseQuant
                )
                kernel_name = (
                    "EpCombineIntraNodeKernel_bf16_p2p_fp8_blockwise"
                    if _qpull
                    else "EpCombineIntraNodeKernel_bf16_nop2p_fp8_blockwise"
                )
                use_vec8_top8 = False
                if base_vec8_top8_eligible and not _qpull:
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
                    kernel_name = kernel_name.replace("_fp8_blockwise", "_fp4_blockwise")
                    assert (
                        kernel_name in _FP4_COMBINE_KERNELS
                    ), f"fp4_blockwise combine selected unregistered kernel '{kernel_name}'"
                shared_mem = self._combine_shared_mem(
                    actual_wpb, use_weights=not use_vec8_top8
                )
                # MORI_COMB_NOQUANT deletes the quantise pass to price it, and the pass moved out
                # of the kernel the -D lives in, so honour it here or deletion pricing silently
                # stops deleting anything. MORI_COMB_QPRE=noq is the same deletion without the
                # rebuild. WRONG RESULTS either way, same as it always was.
                if (
                    qpre
                    and qpre_mode != "noq"
                    and not os.environ.get("MORI_COMB_NOQUANT", "").strip()
                ):
                    # Its own width, and deliberately not the combine geometry: this pass has no
                    # cross-card edge and no LDS tiles, so the only thing it wants is enough waves
                    # to hide the loads. Grid-strided over a token count that only the device
                    # knows, so the grid is a latency-hiding choice and not a partition.
                    q_wpb = _comb_env_int("MORI_COMB_QPRE_WPB", 8)
                    q_bn = _comb_env_int(
                        "MORI_COMB_QPRE_BN",
                        4 * int(self._handle_info["multi_processor_count"]),
                    )
                    self._launch(
                        "EpCombineQuantizeInputKernel_bf16",
                        (q_bn,),
                        (self._warp_size * q_wpb,),
                        0,
                        stream,
                        args_ptr,
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
                    # force_pull here means MORI_COMB_PULL=kernel: caller still owns the input
                    # buffer, but the kernel stages it into combineInp itself and the peers read
                    # that, so the _p2p symbol is the right one despite the flag.
                    _tr = "_p2p" if force_pull else "_nop2p"
                    self._launch(
                        f"EpCombineIntraNodeKernel_{sfx}{_tr}",
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
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = from_gpu_ptr(
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
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=self._get_cur_rank_num_token(self._handle),
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
            num_tokens=self._get_cur_rank_num_token(self._handle),
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
                f"EpCombineSyncBarrier_{sfx}", (1,), (self._warp_size,), 0, stream, args_ptr
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
