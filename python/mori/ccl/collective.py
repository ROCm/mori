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

import ctypes
import os
import time

import torch
from mori import cpp as mori_cpp
from typing import Optional


# ---------------------------------------------------------------------------
# JIT compilation for CCL kernels
# ---------------------------------------------------------------------------
_ccl_hip_module = None


def _ensure_ccl_jit():
    """JIT compile ccl_kernels.hip and load the HipModule (once)."""
    global _ccl_hip_module
    if _ccl_hip_module is not None:
        return
    from mori.jit.core import compile_genco
    from mori.ops._jit_loader import _compiled_hsaco, load_hip_module

    if "ccl_kernels" not in _compiled_hsaco:
        _compiled_hsaco["ccl_kernels"] = compile_genco(
            "ccl_kernels", source_dir="src/collective/kernels"
        )
    _ccl_hip_module = load_hip_module("ccl_kernels", init_shmem=True)


def _get_ccl_func(name: str):
    """Get a kernel function from the JIT-compiled CCL module."""
    return _ccl_hip_module.get_function(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sdma_env(class_name: str) -> None:
    raw = os.environ.get("MORI_ENABLE_SDMA", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        raise RuntimeError(
            f"{class_name} requires MORI_ENABLE_SDMA=1 in the process "
            f"environment (got MORI_ENABLE_SDMA={os.environ.get('MORI_ENABLE_SDMA')!r}). "
            f"Export it before launch on every rank."
        )


# ---------------------------------------------------------------------------
# AllReduce / All2all transport selection
# ---------------------------------------------------------------------------
# MORI_CCL_ALLREDUCE_BACKEND and MORI_CCL_ALL2ALL_BACKEND pick which engine
# carries the cross-GPU bytes. Default is sdma on every architecture, including
# gfx125x, except for All2all on gfx125x, which runs on tdm only
# (_resolve_all2all_backend). tdm requires gfx125x because the TDM engine does
# not exist elsewhere, and because ccl_kernels.hip only emits the Tdm* symbols
# under that arch guard -- asking for them on another target would fail at
# kernel lookup rather than here.
_CCL_BACKENDS = ("sdma", "tdm")


def _detect_arch_str() -> str:
    """The arch the JIT will compile for. Deliberately the same source as
    --offload-arch (mori.jit.config.detect_gpu_arch) so a host-side decision
    here cannot disagree with what ends up in the code object."""
    try:
        from mori.jit.config import detect_gpu_arch

        return str(detect_gpu_arch())
    except Exception:
        return ""


def _resolve_backend(env_name: str) -> str:
    raw = os.environ.get(env_name, "").strip().lower()
    if raw in ("", "sdma"):
        return "sdma"
    if raw not in _CCL_BACKENDS:
        raise ValueError(f"{env_name}={raw!r} is not one of {_CCL_BACKENDS}")
    arch = _detect_arch_str()
    if not arch.startswith("gfx125"):
        raise RuntimeError(
            f"{env_name}=tdm requires a gfx125x GPU (detected "
            f"arch {arch!r}); the TDM engine does not exist on this target."
        )
    return "tdm"


def _resolve_all2all_backend() -> str:
    if not _detect_arch_str().startswith("gfx125"):
        return _resolve_backend("MORI_CCL_ALL2ALL_BACKEND")
    raw = os.environ.get("MORI_CCL_ALL2ALL_BACKEND", "").strip().lower()
    if raw not in ("", "tdm"):
        raise RuntimeError(
            f"MORI_CCL_ALL2ALL_BACKEND={raw!r}: All2all on gfx125x runs on the "
            f"TDM engine only and does not use SDMA."
        )
    return "tdm"


# Must equal MORI_CCL_TDM_TILE_BYTES in
# include/mori/collective/allreduce/twoshot_tdm_kernel.hpp. The kernel indexes
# its LDS by warp against this number, so reserving less is an out-of-bounds LDS
# access, which on this hardware means an unkillable process rather than an error.
_TDM_TILE_BYTES = 256 * 8 * 4
# gfx125x is wave32. Not read from the device: get_warp_size() is known to
# misreport 64 on gfx1250 (src/jit/v2/toolchain.cpp), and this path is
# gfx125x-only anyway.
_TDM_WAVE_SIZE = 32
# 64 blocks x 16 wave32 puts 1024 warps in flight. That number, not the CU
# count, is what bounds a TDM payload phase: the EP dispatch sweep reaches the
# same 1366 GB/s at 64 blocks x 16 warps and at 128 blocks x 8 warps, and only
# 1278 at 64 x 8 (python/mori/ops/dispatch_combine.py, _resolve_launch_params).
# Also stays well under the CU count, which the kernel's grid rendezvous needs
# because it spins on blocks being co-resident.
_TDM_MAX_BLOCKS = 64
_TDM_THREADS = 512


def _tdm_shared_mem(threads: int) -> int:
    warps = (threads + _TDM_WAVE_SIZE - 1) // _TDM_WAVE_SIZE
    return warps * _TDM_TILE_BYTES


# Must equal MORI_CCL_TDM_A2A_TILE_BYTES in
# include/mori/collective/all2all/oneshot_all2all_tdm_kernel.hpp, for the same
# reason as _TDM_TILE_BYTES. 16 warps x 16 KiB is 256 KiB of LDS a block.
_TDM_A2A_TILE_BYTES = 256 * 16 * 4
# 64 blocks x 16 wave32 is the 1024 warps in flight the AllReduce launch uses.
# Must not exceed MORI_CCL_TDM_A2A_MAX_BLOCKS: every block owns a slot in the
# sync array and, without copy_output_to_user, a word per rank in the output
# transit buffer.
_TDM_A2A_BLOCKS = 64
# MORI_CCL_ALL2ALL_BLOCKS overrides the block count per handle, for comparison
# only: 64 is what ships. Must equal MORI_CCL_TDM_A2A_MAX_BLOCKS.
_TDM_A2A_MAX_BLOCKS = 128
_TDM_A2A_SHARED_MEM = (_TDM_THREADS // _TDM_WAVE_SIZE) * _TDM_A2A_TILE_BYTES
# Must equal MORI_CCL_TDM_A2A_SYNC_WORDS: a slot per block, the go word, the
# two masks of the registered-output handshake and the counter of its push.
_TDM_A2A_SYNC_WORDS = _TDM_A2A_MAX_BLOCKS + 4
# With copy_output_to_user, calls of at least this many bytes a rank
# (count * npes * 4) push every tile first, exchange one word a block and then
# copy (CclTdmA2aCopyAfterPush in the kernel header); smaller ones raise a word
# per tile and copy while they push. The kernel decides; this must equal
# kCclTdmA2aCopyAfterPushMinBytes there.
_TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES = 64 * 1024 * 1024
# sizeof(CclAll2allArgs<uint32_t>), the struct prepare_sync fills. The TDM
# kernel takes those bytes followed by the sync pointer, the caller's output,
# the 4-byte elements in a transit half, the 4-byte elements before the words,
# the 4-byte elements before the LL128 area and in a half of it, and the two
# pointers of the registered output's symmetric object, zero when the output is
# not registered, and the word that says which registration it is
# (CclTdmAll2allArgs, which static_asserts every offset).
_TDM_A2A_BASE_ARGS_BYTES = 72
_TDM_A2A_ARGS_BYTES = _TDM_A2A_BASE_ARGS_BYTES + 72
# Each half of the output transit buffer starts on a 128 B boundary, the
# alignment TDM needs.
_TDM_A2A_HALF_ALIGN = 128
# With copy_output_to_user, peers raise a word for every tile they write, and a
# peer's consecutive words sit this far apart. Must equal kCclTdmA2aWordGap * 8
# in the kernel header. The words start on a boundary of the same size.
_TDM_A2A_WORD_GAP_BYTES = 256
# Calls of at most this many bytes a rank (count * npes * 4) move as LL128
# lines instead of TDM tiles (CclTdmA2aLl128 in the kernel header). Measured on
# 4 gfx1250 GPUs with graphs of 100 calls: 9.87 us a call at 512 KiB against
# 13.36 us through TDM, and 14.91 against 13.57 at 1 MiB.
# MORI_CCL_ALL2ALL_LL128_MAX_BYTES overrides it per handle and 0 turns the path
# off; every rank must use the same value. Receivers poll the lines, which has
# been verified only on GPUs whose local memory the driver maps MTYPE_RW.
_TDM_A2A_LL128_MAX_BYTES = 512 * 1024
# Must equal kCclLlUnitBytes and kCclLlUnitPayload in the kernel header.
_TDM_A2A_LL128_UNIT_BYTES = 512
_TDM_A2A_LL128_UNIT_PAYLOAD = 480
# Each half of the LL128 area starts on a page, which is also a whole number of
# 128 B lines.
_TDM_A2A_LL128_ALIGN = 4096


def _resolve_all2all_ll128_max_bytes() -> int:
    raw = os.environ.get("MORI_CCL_ALL2ALL_LL128_MAX_BYTES", "").strip()
    if not raw:
        return _TDM_A2A_LL128_MAX_BYTES
    try:
        value = int(raw, 0)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(
            f"MORI_CCL_ALL2ALL_LL128_MAX_BYTES must be a byte count >= 0, got {raw!r}"
        )
    return value


# MORI_CCL_ALL2ALL_AUTO_REGISTER=1 makes a handle register, on its own, every
# output a large call is given (see All2allSdma.register_output_buffer), as
# RCCL does with its user buffers. Off by default: the registration is
# collective, so it is only safe when every rank passes a new output in the
# same call, and it is skipped while a graph is being captured. The handle
# keeps the last few registered this way.
_TDM_A2A_AUTO_REGISTER_KEEP = 4


# Data pointers of the outputs some handle of this process holds registered. Two
# registrations of the same memory would share one entry of the symmetric
# memory pool, and dropping either would take the other's mappings with it.
_TDM_A2A_REGISTERED_PTRS = set()


def _env_ranks(name: str) -> set:
    # Test hooks: a comma separated list of ranks.
    raw = os.environ.get(name, "").strip()
    return {int(t) for t in raw.split(",") if t.strip()}


def _resolve_all2all_auto_register() -> bool:
    raw = os.environ.get("MORI_CCL_ALL2ALL_AUTO_REGISTER", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# Without copy_output_to_user the result used to land in the output transit
# buffer, in the symmetric heap. That heap is uncached memory by default, and
# blocks of this rank copying into it while the peers push into it are slow:
# with 5 of 64 blocks copying the kept chunk during the push a call took 808 us
# at 1 GiB a rank on 4 gfx1250 GPUs, against 558 on a heap of ordinary memory
# and 594 with the copy after the push. So the handle now keeps the result in
# ordinary device memory of its own, registered like a caller's output: the
# peers write it directly and the kernel runs as it does for a registered
# output. get_output_transit_buffer returns that memory. The output transit
# buffer is then twice the room the caller asked for, so that a rank whose
# memory the peers cannot reach still receives through a half of it, and the
# handle takes three times that room in all, as one with copy_output_to_user
# does with the caller's output. MORI_CCL_ALL2ALL_RESULT_IN_HEAP=1 restores
# the result in the output transit buffer.
def _resolve_all2all_result_in_heap() -> bool:
    raw = os.environ.get("MORI_CCL_ALL2ALL_RESULT_IN_HEAP", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _resolve_all2all_blocks() -> int:
    raw = os.environ.get("MORI_CCL_ALL2ALL_BLOCKS", "").strip()
    if not raw:
        return _TDM_A2A_BLOCKS
    try:
        value = int(raw, 0)
    except ValueError:
        value = 0
    if not 1 <= value <= _TDM_A2A_MAX_BLOCKS:
        raise ValueError(
            f"MORI_CCL_ALL2ALL_BLOCKS must be a block count from 1 to "
            f"{_TDM_A2A_MAX_BLOCKS}, got {raw!r}"
        )
    return value


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _tdm_a2a_words_bytes(
    npes: int, half_bytes: int, copy_out: bool, blocks: int = _TDM_A2A_BLOCKS
) -> int:
    # copy_output_to_user: a word per rank and tile of the largest call a half
    # holds, half_bytes // (4 * npes) elements a destination. Otherwise a u64
    # per rank and block.
    if not copy_out:
        return npes * blocks * 8
    tile_elems = _TDM_A2A_TILE_BYTES // 4
    tiles = (half_bytes // 4 // npes + tile_elems - 1) // tile_elems
    return npes * tiles * _TDM_A2A_WORD_GAP_BYTES


def _tdm_ll128_half_bytes(npes: int, max_bytes: int) -> int:
    # Room for the largest LL128 call, max_bytes // (4 * npes) elements a
    # destination: npes chunks of whole 512 B units.
    chunk = max_bytes // (4 * npes) * 4
    units = (chunk + _TDM_A2A_LL128_UNIT_PAYLOAD - 1) // _TDM_A2A_LL128_UNIT_PAYLOAD
    return _align_up(npes * units * _TDM_A2A_LL128_UNIT_BYTES, _TDM_A2A_LL128_ALIGN)


_TORCH_DTYPE_TO_NUMPY = {
    torch.uint32: "<u4",
    torch.int32: "<i4",
    torch.float16: "<f2",
    torch.bfloat16: "<u2",
    torch.float32: "<f4",
}


def _stream_to_int(stream) -> int:
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    return stream.cuda_stream


class _GpuBufferView:
    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 2,
        }


def _ptr_to_tensor(ptr: int, size_bytes: int, dtype=torch.uint32, device=None):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_size
    typestr = _TORCH_DTYPE_TO_NUMPY.get(dtype, "<u4")
    buf = _GpuBufferView(ptr, (num_elements,), typestr)
    device = _normalize_cuda_device(device) or torch.device("cuda")
    if dtype == torch.bfloat16:
        raw = torch.as_tensor(buf, device=device).view(torch.bfloat16)
    else:
        raw = torch.as_tensor(buf, device=device)
    return raw


def _normalize_cuda_device(device):
    if device is None:
        return None
    if isinstance(device, torch.Tensor):
        device = device.device
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise TypeError("device must be a CUDA tensor, torch.device, int, str, or None")
    if device.type != "cuda":
        raise ValueError("device must refer to a CUDA device")
    return device


def _resolve_transit_view_args(dtype, device, fallback_dtype):
    if device is None and dtype is not None and not isinstance(dtype, torch.dtype):
        device = dtype
        dtype = None
    if isinstance(device, torch.Tensor) and dtype is None:
        dtype = device.dtype
    if dtype is None:
        dtype = fallback_dtype
    return dtype, _normalize_cuda_device(device)


# ---------------------------------------------------------------------------
# All2allSdma
# ---------------------------------------------------------------------------


class All2allSdma:
    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
        _gather: bool = False,
    ):
        # Resolved once, like AllreduceSdma's: the two transports read the same
        # flags differently, so ranks must agree on it for the life of the handle.
        self._backend = _resolve_all2all_backend()
        # An all-gather on the same kernel: every rank is sent the first count
        # elements of the input instead of the chunk the input holds for it
        # (AllgatherSdma on gfx125x).
        self._gather = _gather
        # The TDM kernel carries every size itself and never touches an SDMA
        # queue, unlike AllreduceSdma's TDM backend, which keeps small messages
        # on SDMA.
        if self._backend != "tdm":
            _require_sdma_env("All2allSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        # How the kernel is driven: with an output to fill, the caller's or, for a
        # handle without copy_output_to_user on the TDM transport, the handle's
        # own (see _resolve_all2all_result_in_heap).
        own_result = (
            self._backend == "tdm"
            and not copy_output_to_user
            and not _resolve_all2all_result_in_heap()
        )
        self._copy_out = copy_output_to_user or own_result
        self._tdm_result = None
        # The TDM kernel copies into the caller's output itself, so the C++
        # handle never copies: its copy is a hipMemcpyAsync between two host
        # synchronizations.
        handle_copy = copy_output_to_user and self._backend != "tdm"
        handle_class = getattr(mori_cpp, "All2allSdmaHandle")
        # The TDM kernel keeps its words in the output transit buffer, after the
        # room the caller asked for and on a _TDM_A2A_WORD_GAP_BYTES boundary,
        # and after those the two halves of its LL128 area, from the next page on.
        extra = 0
        ll_half = 0
        if self._backend == "tdm":
            self._tdm_ll_max = _resolve_all2all_ll128_max_bytes()
            self._tdm_blocks = _resolve_all2all_blocks()
            ll_half = _tdm_ll128_half_bytes(npes, self._tdm_ll_max)
            # The room the caller asked for, split as below.
            if input_buffer_size is not None and output_buffer_size is not None:
                room = output_buffer_size
            elif transit_buffer_size is not None:
                room = transit_buffer_size // 2
            else:
                room = 512 * 1024 * 1024 // 2
            # The kernel is told halves of at most this size (below), so the
            # words cover every tile it can be asked to move.
            result_bytes = room
            if own_result:
                # A half as large as the result the caller asked room for.
                room *= 2
                if output_buffer_size is not None:
                    output_buffer_size *= 2
                elif transit_buffer_size is not None:
                    transit_buffer_size = transit_buffer_size // 2 * 3
                else:
                    transit_buffer_size = 512 * 1024 * 1024 // 2 * 3
            half_cap = room // 2 // _TDM_A2A_HALF_ALIGN * _TDM_A2A_HALF_ALIGN
            words_bytes = _tdm_a2a_words_bytes(
                npes, half_cap, self._copy_out, self._tdm_blocks
            )
            extra = words_bytes + _TDM_A2A_WORD_GAP_BYTES
            if ll_half:
                extra += _TDM_A2A_LL128_ALIGN + 2 * ll_half
            else:
                self._tdm_ll_max = 0

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size + extra, handle_copy
            )
        elif extra:
            # How the C++ handle splits a single size.
            total = transit_buffer_size
            if total is None:
                total = 512 * 1024 * 1024
            self._handle = handle_class(
                my_pe,
                npes,
                total // 3 if own_result else total // 2,
                (total // 3 * 2 if own_result else total // 2) + extra,
                handle_copy,
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, transit_buffer_size, handle_copy)
        else:
            self._handle = handle_class(my_pe, npes, 512 * 1024 * 1024, handle_copy)
        self._output_transit_bytes = self._handle.get_output_transit_buffer()[1] - extra
        if self._backend == "tdm":
            if npes > _TDM_WAVE_SIZE:
                raise ValueError(
                    f"the TDM all2all signals its peers from one wave32 and takes at "
                    f"most {_TDM_WAVE_SIZE} ranks, got npes={npes}"
                )
            # One per handle, so two handles never share block slots or call
            # numbers. Ordinary device memory: only this rank's blocks use it.
            self._tdm_sync = torch.zeros(
                _TDM_A2A_SYNC_WORDS,
                dtype=torch.int64,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            # With copy_output_to_user, consecutive calls alternate between the
            # two halves of the output transit buffer.
            half = min(self._output_transit_bytes // 2, half_cap)
            self._tdm_half_bytes = half // _TDM_A2A_HALF_ALIGN * _TDM_A2A_HALF_ALIGN
            # Peers raise the words and store LL128 lines from the first call on;
            # a zeroed line shows no call number. Offsets count from the start of
            # the buffer, which ShmemMalloc puts on a 256 B boundary.
            words_at = _align_up(self._output_transit_bytes, _TDM_A2A_WORD_GAP_BYTES)
            ll_at = _align_up(words_at + words_bytes, _TDM_A2A_LL128_ALIGN)
            self._tdm_ll_elems = ll_at // 4 if ll_half else 0
            _ptr_to_tensor(
                self._handle.get_output_transit_buffer()[0] + words_at,
                extra - _TDM_A2A_WORD_GAP_BYTES,
                torch.int32,
                self._tdm_sync.device,
            ).zero_()
            # The first launch may be on another stream than the fills, and no
            # peer may raise a word before its owner has zeroed it.
            torch.cuda.synchronize()
            mori_cpp.shmem_barrier_all()
            self._tdm_args = ctypes.create_string_buffer(_TDM_A2A_ARGS_BYTES)
            # Outputs registered for direct delivery, by data pointer: the two
            # pointers of the symmetric object, the size in bytes, the tensor,
            # the number of the registration and whether the peers reach this
            # rank's buffer. _tdm_reg_count is the number of registrations so
            # far, the same on every rank.
            self._tdm_registered = {}
            self._tdm_reg_count = 0
            # Test hooks: on the listed ranks a registered output counts as not
            # exportable (MORI_CCL_ALL2ALL_TEST_NO_DIRECT) or is launched as if
            # it were not registered (MORI_CCL_ALL2ALL_TEST_NO_REG), which
            # exercises the mixed cases the start tokens sort out.
            self._tdm_test_no_direct = my_pe in _env_ranks(
                "MORI_CCL_ALL2ALL_TEST_NO_DIRECT"
            )
            self._tdm_test_no_reg = my_pe in _env_ranks("MORI_CCL_ALL2ALL_TEST_NO_REG")
            # Outputs this handle registered by itself, oldest first.
            self._tdm_auto = copy_output_to_user and _resolve_all2all_auto_register()
            self._tdm_auto_outputs = []
            ctypes.c_uint64.from_buffer(
                self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES
            ).value = self._tdm_sync.data_ptr()
            ctypes.c_uint64.from_buffer(
                self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 16
            ).value = (self._tdm_half_bytes // 4)
            ctypes.c_uint64.from_buffer(
                self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 24
            ).value = (words_at // 4)
            ctypes.c_uint64.from_buffer(
                self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 40
            ).value = (ll_half // 4)
            if own_result:
                self._tdm_result = torch.empty(
                    result_bytes // 4, dtype=torch.int32, device=self._tdm_sync.device
                )
                self.register_output_buffer(self._tdm_result)

    def _check_tdm_count(self, count: int) -> None:
        # The TDM kernel writes count * npes uint32 into every rank's output
        # transit buffer, into one half of it with copy_output_to_user, and
        # nothing on the device checks the bound.
        need = count * self.npes * 4
        room = self._tdm_half_bytes if self._copy_out else self._output_transit_bytes
        if need > room:
            raise ValueError(
                f"count={count} needs {need} bytes of output transit buffer per "
                f"rank, but it has {room}"
                + (" (half of it, with copy_output_to_user)" if self._copy_out else "")
            )

    def _tdm_auto_register(self, output_data, count: int) -> None:
        # With MORI_CCL_ALL2ALL_AUTO_REGISTER: a large call into an output that
        # is not registered registers it first, on every rank at this same
        # call, and drops the oldest one registered this way beyond a few.
        need = count * self.npes * 4
        if not self._tdm_auto or need < _TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES:
            return
        if torch.cuda.is_current_stream_capturing():
            # A registration allocates and copies on the device, which a stream
            # that is being captured into a graph refuses: register before.
            return
        reg = self._tdm_registered.get(output_data.data_ptr())
        if reg is not None and reg[2] >= need:
            return
        while len(self._tdm_auto_outputs) >= _TDM_A2A_AUTO_REGISTER_KEEP:
            self.deregister_output_buffer(self._tdm_auto_outputs.pop(0))
        self.register_output_buffer(output_data)
        self._tdm_auto_outputs.append(output_data)

    def _launch_tdm(self, args, output_ptr: int, count: int, s: int) -> None:
        # hipModuleLaunchKernel copies the argument bytes at launch, and a graph
        # capture keeps that copy, so the buffer is reused for every call.
        ctypes.memmove(self._tdm_args, args, _TDM_A2A_BASE_ARGS_BYTES)
        ctypes.c_uint64.from_buffer(
            self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 8
        ).value = (output_ptr if self._copy_out else 0)
        ll = count * self.npes * 4 <= self._tdm_ll_max
        ctypes.c_uint64.from_buffer(
            self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 32
        ).value = (self._tdm_ll_elems if ll else 0)
        # A registered output big enough for this call: the kernel lets the
        # peers that hold the same registration write large calls straight
        # into it. Once a handle has registered anything, large calls start
        # with a handshake in which the ranks tell each other which
        # registration their output belongs to (bit 1 of the word).
        reg = self._tdm_registered.get(output_ptr) if self._copy_out else None
        if reg is not None and (
            count * self.npes * 4 > reg[2] or self._tdm_test_no_reg
        ):
            reg = None
        ctypes.c_uint64.from_buffer(
            self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 48
        ).value = (reg[0] if reg else 0)
        ctypes.c_uint64.from_buffer(
            self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 56
        ).value = (reg[1] if reg else 0)
        word = 2 if self._tdm_reg_count else 0
        if self._gather:
            word |= 1 << 63
        if reg:
            word |= (reg[4] << 2) | int(reg[5])
        ctypes.c_uint64.from_buffer(
            self._tdm_args, _TDM_A2A_BASE_ARGS_BYTES + 64
        ).value = word
        _get_ccl_func("TdmAll2allKernel_u32").launch_struct(
            (self._tdm_blocks,),
            (_TDM_THREADS,),
            _TDM_A2A_SHARED_MEM,
            s,
            ctypes.addressof(self._tdm_args),
        )

    def __call__(self, input_data, output_data, count: int, stream=None) -> float:
        if self._tdm_result is not None:
            output_data = self._tdm_result
        s = _stream_to_int(stream)
        t0 = time.perf_counter()
        if self._backend == "tdm":
            self._check_tdm_count(count)
        args = self._handle.prepare_sync(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        if self._backend == "tdm":
            # Ordered on the stream like any kernel: no host synchronization,
            # so it can be captured into a graph.
            self._tdm_auto_register(output_data, count)
            self._launch_tdm(args, output_data.data_ptr(), count, s)
            return time.perf_counter() - t0
        _get_ccl_func("OneShotAll2allSdmaKernel_u32").launch_struct(
            (1,), (64,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), count, s)
        return time.perf_counter() - t0

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        if self._tdm_result is not None:
            output_data = self._tdm_result
        s = _stream_to_int(stream)
        if self._backend == "tdm":
            self._check_tdm_count(count)
        args = self._handle.prepare_async_start(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        if self._backend == "tdm":
            self._tdm_auto_register(output_data, count)
            self._launch_tdm(args, output_data.data_ptr(), count, s)
        else:
            _get_ccl_func("OneShotAll2allSdmaAsyncPutKernel_u32").launch_struct(
                (1,), (64,), 0, s, args
            )
        self._handle.after_async_start()
        return True

    def wait_async(self, stream=None) -> float:
        s = _stream_to_int(stream)
        if self._backend == "tdm":
            # The TDM kernel waits for every peer and fills the caller's output
            # itself; only the host side is left.
            return self._handle.finish_async_wait(s)
        args = self._handle.prepare_async_wait(s)
        _get_ccl_func("OneShotAll2allSdmaAsyncWaitKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return self._handle.finish_async_wait(s)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def register_output_buffer(self, tensor) -> None:
        """Let the peers write results straight into ``tensor`` (TDM transport,
        copy_output_to_user).

        Collective: every rank calls it at the same point with its own tensor of
        the same size, and then passes that tensor, from its first element, as
        the output of the same calls. Calls of at least
        ``_TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES`` a rank into a registered output
        skip the output transit buffer and the copy out of it; smaller calls
        are not affected, and a tensor smaller than that is not registered at
        all. A peer writes into the tensor only while this rank is
        inside the call. A rank whose tensor the runtime cannot export to the
        peers still takes part: it receives through the output transit buffer
        as before and sends directly. The handle holds the tensor until
        deregister_output_buffer, so its memory is not reused while peers can
        still reach it.
        """
        if self._backend != "tdm" or not self._copy_out:
            raise RuntimeError(
                "register_output_buffer needs the TDM transport and "
                "copy_output_to_user=True"
            )
        size = tensor.numel() * tensor.element_size()
        if size < _TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES:
            # No call into it could be large enough to land directly, and small
            # tensors come out of allocator blocks that cannot always be exported
            # to the peers (hipIpcGetMemHandle then fails, which ends the process).
            return
        ptr = tensor.data_ptr()
        if ptr in _TDM_A2A_REGISTERED_PTRS and ptr not in self._tdm_registered:
            raise RuntimeError(
                "this output is registered with another All2allSdma handle; "
                "deregister it there first"
            )
        cpu, gpu, number, direct = self._handle.register_output_buffer(
            tensor.data_ptr(), size
        )
        self._tdm_reg_count = max(self._tdm_reg_count, number)
        direct = direct and not self._tdm_test_no_direct
        stale = [
            p
            for p, r in self._tdm_registered.items()
            if p + r[2] > ptr and ptr + size > p
        ]
        for p in stale:
            del self._tdm_registered[p]
            _TDM_A2A_REGISTERED_PTRS.discard(p)
        # Kept alive until it is deregistered: a peer must never write into
        # memory the allocator has handed to something else.
        _TDM_A2A_REGISTERED_PTRS.add(ptr)
        self._tdm_registered[tensor.data_ptr()] = (
            cpu,
            gpu,
            size,
            tensor,
            number,
            direct,
        )

    def deregister_output_buffer(self, tensor) -> None:
        """Undo register_output_buffer. Collective, like it."""
        if self._tdm_registered.pop(tensor.data_ptr(), None) is not None:
            self._handle.deregister_output_buffer(tensor.data_ptr())
            _TDM_A2A_REGISTERED_PTRS.discard(tensor.data_ptr())

    def __del__(self):
        for p in getattr(self, "_tdm_registered", {}):
            _TDM_A2A_REGISTERED_PTRS.discard(p)

    def get_output_transit_buffer(self, dtype=None, device=None):
        # On the TDM transport with copy_output_to_user, the result is in the
        # caller's output and this buffer is the kernel's scratch. The TDM
        # kernel's words and LL128 area after the caller's room are left out.
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        if self._tdm_result is not None:
            return _ptr_to_tensor(
                self._tdm_result.data_ptr(),
                self._tdm_result.numel() * 4,
                dtype,
                device,
            )
        ptr = self._handle.get_output_transit_buffer()[0]
        return _ptr_to_tensor(ptr, self._output_transit_bytes, dtype, device)


# ---------------------------------------------------------------------------
# AllgatherSdma
# ---------------------------------------------------------------------------


class AllgatherSdma:
    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
    ):
        # gfx125x has no SDMA all-gather: the TDM all-to-all kernel carries it,
        # every rank sending the same chunk to all the others. The handle is an
        # All2allSdma in that mode, and what this class does not define itself
        # (register_output_buffer, start_async, get_output_transit_buffer, ...)
        # is that handle's.
        self._a2a = None
        if _detect_arch_str().startswith("gfx125"):
            self.my_pe = my_pe
            self.npes = npes
            self._a2a = All2allSdma(
                my_pe,
                npes,
                input_buffer_size,
                output_buffer_size,
                transit_buffer_size,
                copy_output_to_user,
                _gather=True,
            )
            a2a = self._a2a

            def start_async(input_data, output_data, count, stream=None):
                u32_count = (count * input_data.element_size() + 3) // 4
                return a2a.start_async(input_data, output_data, u32_count, stream)

            def wait_async(stream=None, capturing=False):
                return a2a.wait_async(stream)

            def unsupported(*args, **kwargs):
                raise NotImplementedError(
                    "AllgatherSdma on gfx125x (TDM) has no param-contiguous form"
                )

            # Instance attributes, so they take the place of the SDMA methods below.
            self.start_async = start_async
            self.wait_async = wait_async
            self.enqueue_param_contiguous = unsupported
            self.start_async_param_contiguous = unsupported
            for name in (
                "is_async_in_progress",
                "cancel_async",
                "reset_flags",
                "get_output_transit_buffer",
                "register_output_buffer",
                "deregister_output_buffer",
            ):
                setattr(self, name, getattr(a2a, name))
            return
        _require_sdma_env("AllgatherSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        handle_class = getattr(mori_cpp, "AllgatherSdmaHandle")

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, transit_buffer_size, copy_output_to_user
            )
        else:
            self._handle = handle_class(
                my_pe, npes, 512 * 1024 * 1024, copy_output_to_user
            )

    def __call__(
        self, input_data, output_data, count: int, stream=None, capturing: bool = False
    ) -> bool:
        """Run a blocking AllGather.

        ``capturing=True`` skips the host-side stream syncs so the call is legal
        inside ``hipStreamBeginCapture``. The completion generation is derived
        on the device, so a captured graph is safe to replay.
        """
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        if self._a2a is not None:
            self._a2a(input_data, output_data, u32_count, stream)
            return True
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(
            input_data.data_ptr(), output_data.data_ptr(), u32_count, s
        )
        _get_ccl_func("OneShotAllGatherSdmaKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(
            output_data.data_ptr(), u32_count, s, capturing=capturing
        )
        return True

    def __getattr__(self, name):
        a2a = self.__dict__.get("_a2a")
        if a2a is None:
            raise AttributeError(name)
        return getattr(a2a, name)

    def enqueue_param_contiguous(
        self,
        input_data,
        output_data,
        count: int,
        split_sizes,
        split_offsets,
        stream=None,
        capturing: bool = False,
    ) -> bool:
        if split_sizes.numel() != split_offsets.numel():
            raise ValueError("split_sizes and split_offsets must have the same length")
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_param_contiguous(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            split_sizes.data_ptr(),
            split_offsets.data_ptr(),
            split_sizes.numel(),
            s,
        )
        _get_ccl_func("OneShotAllGatherSdmaParamContiguousKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(
            output_data.data_ptr(), u32_count, s, capturing=capturing
        )
        return True

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_start(
            input_data.data_ptr(), output_data.data_ptr(), u32_count, s
        )
        _get_ccl_func("OneShotAllGatherSdmaAsyncPutKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.after_async_start()
        return True

    def start_async_param_contiguous(
        self,
        input_data,
        output_data,
        count: int,
        split_sizes,
        split_offsets,
        stream=None,
    ) -> bool:
        if split_sizes.numel() != split_offsets.numel():
            raise ValueError("split_sizes and split_offsets must have the same length")
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_start_param_contiguous(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            split_sizes.data_ptr(),
            split_offsets.data_ptr(),
            split_sizes.numel(),
            s,
        )
        _get_ccl_func(
            "OneShotAllGatherSdmaParamContiguousAsyncPutKernel_u32"
        ).launch_struct((1,), (512,), 0, s, args)
        self._handle.after_async_start()
        return True

    def wait_async(self, stream=None, capturing: bool = False) -> float:
        """Enqueue the completion wait.

        ``capturing=True`` skips the host-side stream syncs: the wait kernel
        and the copy-out are stream-ordered, so a consumer enqueued on the same
        stream still sees complete data, and the launch thread stays free. It
        returns ``-1.0`` rather than a duration, nothing having waited for the
        transfer to land.
        """
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_wait(s)
        _get_ccl_func("OneShotAllGatherSdmaAsyncWaitKernel_u32").launch_struct(
            (1,), (64,), 0, s, args
        )
        return self._handle.finish_async_wait(s, capturing=capturing)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)

    def register_output_buffer(self, tensor):
        self._handle.register_output_buffer(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )

    def deregister_output_buffer(self, tensor):
        self._handle.deregister_output_buffer(tensor.data_ptr())

    def is_output_registered(self, tensor) -> bool:
        return self._handle.is_output_registered(tensor.data_ptr())


# ---------------------------------------------------------------------------
# InterNodeRingAllgather — inter-node RDMA ring
# ---------------------------------------------------------------------------


class InterNodeRingAllgather:
    """Inter-node AllGather over the shmem ring (P2P intra-node, RDMA inter-node).

    This is the inter-node phase of the hierarchical cross-node AllGather. Each
    participating PE contributes one ``count``-element chunk; after the ring
    every PE holds all ``npes`` chunks concatenated in PE order -- identical to
    ``torch.distributed.all_gather_into_tensor`` when every PE is a participant.

    The ring schedule (CPU-validated by ``inter_node_ring_reference``) is run on
    device by the JIT kernel ``InterNodeRingAllGatherKernel_u32``, which moves
    raw bytes so a single u32 kernel serves bf16/fp16/fp32/int32.

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``
    matching this handle.
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        ring_buffer_bytes: Optional[int] = None,
        ring_size: int = -1,
        ring_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
        num_qp: int = 1,
        num_blocks: int = 1,
    ):
        # The inter-node phase uses the RDMA/P2P shmem transport, not the SDMA
        # copy engines (those drive the intra-node phase), so MORI_ENABLE_SDMA is
        # not required here; forcing it on would route same-node puts through the
        # intra-node SDMA multi-queue path, which is not what the ring needs.
        #
        # Sub-group ring: when ``ring_size`` >= 0 the ring runs over the
        # arithmetic sub-group of global PEs
        # ``{pe_base, pe_base+pe_stride, ..., pe_base+(ring_size-1)*pe_stride}``
        # and this PE is at position ``ring_pos`` within it. The output holds the
        # ``ring_size`` chunks in ring order. The default (``ring_size=-1``) is
        # the flat whole-world ring (ring_size=npes, ring_pos=my_pe).
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.ring_size = ring_size if ring_size >= 0 else npes
        handle_class = getattr(mori_cpp, "InterNodeRingAllgatherHandle")
        if ring_buffer_bytes is None:
            ring_buffer_bytes = 512 * 1024 * 1024
        # num_qp>1 fans the per-round ring put across that many RDMA QPs (the
        # kernel applies it only to true cross-node neighbours; same-node P2P/SDMA
        # neighbours stay single-warp). Default 1 == unchanged single-QP put.
        self.num_qp = num_qp
        # num_blocks>1 launches the ring as that many CTAs ("channels"), each
        # driving a disjoint chunk sub-range on its own QP. The kernel engages
        # it only for true RDMA neighbours; same-node sims fall back to a single
        # working block. Default 1 == single-block ring.
        self.num_blocks = num_blocks if num_blocks and num_blocks >= 1 else 1
        self._handle = handle_class(
            my_pe,
            npes,
            ring_buffer_bytes,
            ring_size,
            ring_pos,
            pe_base,
            pe_stride,
            num_qp,
            self.num_blocks,
        )

    def __call__(
        self,
        input_data,
        output_data,
        count: int,
        stream=None,
        chunk_in_place: bool = False,
        stream_ring: bool = False,
        defer_inter_fin: bool = False,
    ) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        # stream_ring=True uses the on-device ShmemBarrierOnStream prepare/finish
        # (no host hipStreamSynchronize / host ShmemBarrierAll), keeping the whole
        # op enqueued on the stream. Same byte moves and global fencing as the
        # host-synced path; removes 2 CPU<->GPU round-trips per op.
        if chunk_in_place:
            # This PE's chunk is already in its ring slot (the upstream intra
            # gather wrote there via ``slot_tensor``), so skip the prepare_sync
            # copy-IN.
            if stream_ring:
                args = self._handle.prepare_stream_in_place(u32_count, s)
            else:
                args = self._handle.prepare_sync_in_place(u32_count, s)
        else:
            if stream_ring:
                args = self._handle.prepare_stream(input_data.data_ptr(), u32_count, s)
            else:
                args = self._handle.prepare_sync(input_data.data_ptr(), u32_count, s)
        _get_ccl_func("InterNodeRingAllGatherKernel_u32").launch_struct(
            (self.num_blocks,), (512,), 0, s, args
        )
        if stream_ring:
            # defer_inter_fin defers the ring-reuse fence to the next op's
            # prepare_stream barrier -- the copy-OUT stays stream-ordered so
            # the collection is correct; only cross-PE ring reuse needs the
            # fence, which the successor op provides.
            self._cu_finish_copyout(
                output_data, u32_count, s, barrier=not defer_inter_fin
            )
        else:
            self._handle.finish_sync(output_data.data_ptr(), u32_count, s)
        return True

    def _cu_finish_copyout(
        self, output_data, u32_count: int, s: int, barrier: bool = True
    ) -> None:
        """CU-domain ring finish copy-OUT + (optional deferred) reuse barrier.

        Copies the gathered ring buffer (``self._handle.buf_ptr()``) into
        ``output_data`` with the ``RingFinishCopyKernel_u32`` COMPUTE-UNIT
        kernel rather than the copy-engine hipMemcpyAsync, so the drain observes
        the RDMA-landed remote-half bytes the ring kernel fenced (the
        receiver/copy-out completion residual). ``barrier`` mirrors the
        finish_stream reuse fence (deferred when defer_inter_fin)."""
        total_u32 = int(u32_count) * int(self.ring_size)
        block = 256
        grid = (total_u32 + block - 1) // block
        if grid < 1:
            grid = 1
        if grid > 4096:
            grid = 4096
        _get_ccl_func("RingFinishCopyKernel_u32").launch(
            (grid,),
            (block,),
            0,
            s,
            output_data.data_ptr(),
            self._handle.buf_ptr(),
            total_u32,
        )
        if barrier:
            self._handle.finish_stream_no_copy(s)

    def prepare_stream_only(self, input_data, count: int, stream=None):
        """Issue ONLY the stream-ordered ring prepare (the global on-stream
        ShmemBarrierOnStream entry barrier + the per-PE copy-IN of ``input_data``
        into the ring slot) and return ``(args, u32_count, s)`` WITHOUT launching
        the ring kernel.

        This splits the monolithic ``__call__`` (prepare -> kernel -> finish) so
        the caller can interleave INDEPENDENT work (e.g. the slice path's local
        node-block SDMA reassembly gather, which reads only this rank's own input
        and has no ring dependency) on a SIDE stream between the entry barrier and
        the ring kernel -- the side work then overlaps the ring kernel while the
        ring's prepare barrier remains the SOLE global entry fence (the side work
        runs barrier-free, so only one global on-stream fence is ever in flight).
        Stream-ring only (the on-device barrier path).
        """
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_stream(input_data.data_ptr(), u32_count, s)
        return args, u32_count, s

    def finish_ring_stream(
        self,
        output_data,
        count: int,
        stream=None,
        barrier: bool = True,
        cu_copyout: Optional[bool] = None,
    ) -> bool:
        """Stream-ordered ring finish copy-OUT ONLY -- the ring kernel was already
        launched ELSEWHERE (e.g. by the fused ``FusedRingLocalGatherKernel_u32``
        that runs the ring concurrently with the local-block SDMA gather in one
        grid). Issues just the copy-OUT of the gathered ring buffer into
        ``output_data`` + the (optionally deferred) ShmemBarrierOnStream reuse
        fence. Pairs with ``prepare_stream_only`` + the external fused kernel
        launch. ``barrier`` mirrors ``defer_inter_fin``.

        ``cu_copyout`` selects the copy-OUT engine for THIS call: the CU
        RingFinishCopyKernel_u32 moves the whole ring buffer on compute units,
        which burns CUs on bulk all-gather bytes. On the fuse_local path bulk
        bytes must ride SDMA/RDMA only (CUs stay free for the GEMM), so the
        fuse_local caller passes cu_copyout=False. None = CU copy-out.
        """
        byte_count = count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        use_cu = True if cu_copyout is None else bool(cu_copyout)
        if use_cu:
            self._cu_finish_copyout(output_data, u32_count, s, barrier=barrier)
        else:
            self._handle.finish_stream(
                output_data.data_ptr(), u32_count, s, barrier=barrier
            )
        return True

    def slot_tensor(self, count: int, dtype, device=None):
        """Torch view of this PE's ring slot (``count`` elements of ``dtype``).

        Write this PE's chunk here (e.g. as the intra gather's output) and then
        call ``__call__(..., chunk_in_place=True)`` to run the ring without the
        prepare_sync copy-IN. ``count`` is in elements of ``dtype``; the slot is
        sized in u32 lanes, so ``count*element_size`` must be a multiple of 4.
        """
        byte_count = count * torch.tensor([], dtype=dtype).element_size()
        u32_count = (byte_count + 3) // 4
        ptr = self._handle.slot_ptr(u32_count)
        return _ptr_to_tensor(ptr, u32_count * 4, dtype, device)[:count]


# ---------------------------------------------------------------------------
# IntraNodeSubGroupAllgatherSdma — intra-node SDMA gather over a
# sub-group of local ranks (the intra-node phase of the hierarchical AllGather)
# ---------------------------------------------------------------------------


class IntraNodeSubGroupAllgatherSdma:
    """Intra-node SDMA AllGather over an arithmetic sub-group of local ranks.

    The ``group_size`` ranks ``{pe_base, pe_base+pe_stride, ...}`` (this PE at
    position ``group_pos``) gather their ``count``-element shards over the SDMA
    copy engines (XGMI); after the call every member holds the ``group_size``
    shards concatenated in group-position order -- its node's contiguous block.
    The SDMA-side building block of the hierarchical cross-node AllGather
    (intra-node phase). The default (``group_size=-1``) is the flat whole-world
    SDMA gather (group_size=npes, group_pos=my_pe).

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``.
    Requires ``MORI_ENABLE_SDMA=1`` (the SDMA copy-engine path).
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        out_buffer_bytes: Optional[int] = None,
        group_size: int = -1,
        group_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
    ):
        _require_sdma_env("IntraNodeSubGroupAllgatherSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.group_size = group_size if group_size >= 0 else npes
        handle_class = getattr(mori_cpp, "IntraNodeSubGroupAllgatherSdmaHandle")
        if out_buffer_bytes is None:
            out_buffer_bytes = 512 * 1024 * 1024
        self._handle = handle_class(
            my_pe, npes, out_buffer_bytes, group_size, group_pos, pe_base, pe_stride
        )

    def __call__(
        self,
        input_data,
        output_data,
        count: int,
        stream=None,
        barrier: bool = True,
        prepare_barrier: bool = True,
    ) -> bool:
        # ``barrier=False`` skips the trailing ShmemBarrierAll in finish_sync.
        # Safe only when an immediately-following global barrier synchronizes all
        # PEs before any remote read (the PUSH gather's in-kernel flag-wait
        # already makes this PE's node-block complete on kernel return). Used by
        # HierAllGather's fused-barrier path.
        # ``prepare_barrier=False`` additionally skips the ENTRY ShmemBarrierAll
        # in prepare_sync. Safe only when the PREVIOUS pipeline iteration ended
        # with a global barrier (so every peer's out_ transit is free) AND this
        # is not the first call (out_ already registered). The caller
        # (HierAllGather) enforces both via a first-call guard.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(
            input_data.data_ptr(), u32_count, s, prepare_barrier
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def gather_kernel(
        self,
        input_data,
        count: int,
        dst_base_offset: int = 0,
        stream=None,
        prepare_barrier: bool = True,
        dst_slot_stride: int = 0,
    ) -> bool:
        # Launch ONLY the gather kernel (no copy-OUT), writing this gather's
        # groupSize-slot block into ``out_`` at element offset ``dst_base_offset``
        # (of the input dtype). Used by the fused sliced path to stack the N
        # reassembly gathers into disjoint regions of one enlarged transit; a
        # single ``finish_batch`` then copies them all out at once.
        # ``dst_slot_stride`` (in elements of the input dtype, 0 == contiguous)
        # decouples the per-peer destination slot stride from the copy ``count``,
        # so a CHUNK of a slice can land at its strided position inside a
        # full-size block -- the chunked inter/intra pipeline enabler.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_base_offset_bytes = dst_base_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(
            input_data.data_ptr(),
            u32_count,
            s,
            prepare_barrier,
            dst_base_offset_bytes,
            dst_slot_stride_bytes,
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return True

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)

    def finish_batch(
        self, output_data, total_count: int, stream=None, barrier: bool = True
    ) -> bool:
        # One bulk copy-OUT of ``total_count`` elements (the full
        # N*groupSize*chunk stacked by ``gather_kernel``) from ``out_`` to the
        # user output, then one barrier. Replaces N per-gather finish copies.
        byte_count = total_count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        self._handle.finish_batch(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def finish_batch_stream(
        self, output_data, total_count: int, stream=None, barrier: bool = True
    ) -> bool:
        # STREAM-ORDERED bulk copy-OUT. Same bytes as finish_batch but the
        # trailing global fence is an on-device ShmemBarrierOnStream(stream)
        # instead of host hipStreamSynchronize + ShmemBarrierAll. Removes the last
        # host CPU<->GPU round-trip in the fused sliced Phase-B so the whole op
        # stays enqueued on ``stream``. Pairs with the stream-ordered inter ring.
        # ``barrier=False`` DEFERS the trailing fence to the next op's inter-ring
        # prepare ShmemBarrierOnStream (redundant back-to-back across the op
        # boundary). The copy-OUT stays stream-ordered so the output is correct
        # regardless.
        byte_count = total_count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        self._handle.finish_batch_stream(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def register_output_buffer(self, tensor) -> bool:
        # Register a user output tensor for DIRECT-TO-OUTPUT gathers (collective;
        # cached). No-op if already registered.
        self._handle.register_output_buffer(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )
        return True

    def deregister_output_buffer(self, tensor) -> bool:
        self._handle.deregister_output_buffer(tensor.data_ptr())
        return True

    def deregister_output_buffer_ptr(self, ptr: int) -> bool:
        # Deregister by raw base pointer (the live-registration tracker holds an
        # int, not the original tensor, after the buffer was freed). Collective --
        # must be called in lockstep on every PE; the C++ side looks up the stored
        # extent so only the ptr is needed.
        self._handle.deregister_output_buffer(ptr)
        return True

    def is_output_registered(self, tensor) -> bool:
        return self._handle.is_output_registered(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )

    def gather_kernel_direct(
        self,
        input_data,
        output_data,
        count: int,
        dst_block_offset: int = 0,
        stream=None,
        prepare_barrier: bool = True,
        dst_slot_stride: int = 0,
        flag_slot_base: int = 0,
    ) -> bool:
        # DIRECT gather -- SDMA-PUSH each member's slice straight into the
        # (registered) ``output_data`` at element offset ``dst_block_offset`` (of
        # the input dtype), no internal transit + no copy-OUT. ``output_data``
        # MUST have been registered via register_output_buffer.
        # ``dst_slot_stride`` matches gather_kernel. ``flag_slot_base`` gives a
        # CONCURRENT lane a disjoint flag-slot region [flag_slot_base, +groupSize)
        # so concurrent reassembly gathers never race on the shared flag slots
        # (0=off, the single-stream default).
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_block_offset_bytes = dst_block_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct(
            input_data.data_ptr(),
            u32_count,
            s,
            prepare_barrier,
            output_data.data_ptr(),
            dst_block_offset_bytes,
            dst_slot_stride_bytes,
            flag_slot_base,
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return True

    def gather_kernel_direct_param_contiguous(
        self,
        input_data,
        output_data,
        block_stride: int,
        num_blocks: int,
        world_size: int,
        split_sizes_u32,
        split_offsets_u32,
        stream=None,
        prepare_barrier: bool = True,
        first_block: int = 0,
    ) -> bool:
        # FUSED param-contiguous direct gather -- ONE launch scatters ALL node
        # blocks * param splits from the Phase-A ``input_data`` collection
        # straight into the (registered) ``output_data`` in PARAM-CONTIGUOUS
        # layout. Replaces the per-(block, param) gather_kernel_direct loop. All
        # size arguments are in u32-lane units. ``split_sizes_u32`` /
        # ``split_offsets_u32`` are int64 DEVICE tensors (E_s / O_s per param,
        # u32 lanes).
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct_param_contiguous(
            input_data.data_ptr(),
            s,
            prepare_barrier,
            output_data.data_ptr(),
            block_stride,
            num_blocks,
            world_size,
            split_sizes_u32.data_ptr(),
            split_offsets_u32.data_ptr(),
            split_sizes_u32.numel(),
            0,
            first_block,
        )
        _get_ccl_func(
            "OneShotAllGatherSdmaSubGroupParamContiguousKernel_u32"
        ).launch_struct((1,), (512,), 0, s, args)
        return True

    def prepare_direct_only(
        self,
        input_data,
        output_data,
        count: int,
        dst_block_offset: int = 0,
        stream=None,
        prepare_barrier: bool = True,
        dst_slot_stride: int = 0,
    ) -> int:
        # Build the DIRECT-gather jit_args (prime the per-peer flag slots +
        # optional entry barrier + register the strided output dst) and RETURN the
        # int64 args pointer WITHOUT launching the kernel -- so the fused
        # ``FusedRingLocalGatherKernel_u32`` can run this gather as a single
        # designated block of a larger grid (blockLocal=true) concurrently with
        # the inter-node RDMA ring. Mirrors ``gather_kernel_direct`` minus the
        # launch_struct; the returned ptr is the handle's jit_args_ member (kept
        # alive until the next prepare on this handle, exactly as the launch path
        # relies on). ``output_data`` MUST already be registered.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_block_offset_bytes = dst_block_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct(
            input_data.data_ptr(),
            u32_count,
            s,
            prepare_barrier,
            output_data.data_ptr(),
            dst_block_offset_bytes,
            dst_slot_stride_bytes,
        )
        return args

    def finish_direct_stream(self, stream=None, barrier: bool = True) -> bool:
        # Completion fence for the DIRECT path (no copy-OUT; gathers already
        # pushed into the user output). On-device global fence.
        s = _stream_to_int(stream)
        self._handle.finish_direct_stream(s, barrier)
        return True


def launch_fused_ring_local_gather(
    ring_args: int, gather_args: int, ring_blocks: int, s: int
) -> bool:
    """Merge a prepared inter-node ring's jit_args with a prepared intra-node
    local-block direct-gather's jit_args (via the
    ``build_fused_ring_local_gather_args`` C++ glue) and launch the fused
    ``FusedRingLocalGatherKernel_u32`` ONCE on stream ``s`` with ``ring_blocks +
    1`` CTAs: blocks ``[0, ring_blocks)`` run the RDMA ring (Phase A, over the
    NIC) and the last block runs the local node-block SDMA reassembly gather
    (Phase B, m == node_id -- the half independent of the ring) over XGMI.
    Replaces the two serial kernel launches + host ``wait_stream`` merge of the
    overlap path with one concurrent grid (NIC ring || XGMI gather).
    """
    rb = ring_blocks if ring_blocks and ring_blocks >= 1 else 1
    fused = mori_cpp.build_fused_ring_local_gather_args(ring_args, gather_args, rb)
    _get_ccl_func("FusedRingLocalGatherKernel_u32").launch_struct(
        (rb + 1,), (512,), 0, s, fused
    )
    return True


def launch_fused_ring_remote_gather(
    ring_args: int,
    gather_args: int,
    ring_blocks: int,
    chunk_ready_flags_ptr: int,
    num_nodes: int,
    node_id: int,
    s: int,
    reassembly_blocks: int = 0,
    reasm_deep_sq: int = 0,
    deep_pipe: int = 1,
    deep_pipe_quiet: int = 0,
) -> bool:
    """Launch the FUSED, PIPELINED ``FusedRingRemoteGatherKernel_u32`` ONCE on
    stream ``s`` with ``2*ring_blocks + 1`` CTAs. Blocks ``[0, ring_blocks)`` run
    the RDMA ring (Phase A) and each publishes ``chunkReadyFlags[bid]`` on
    landing; block ``[ring_blocks]`` runs the ring-independent LOCAL node-block
    SDMA gather; blocks ``(ring_blocks, 2*ring_blocks]`` each spin on
    ``chunkReadyFlags[j]`` and, the instant sub-range ``j`` lands, SDMA-push that
    sub-range of every remote block from this PE's ring buffer straight into the
    registered output -- overlapping ring channel ``j+1`` still crossing the NIC
    (closes the two-serial-phases gap). The ``gather_args`` must be the LOCAL
    block's prepared direct-gather jit_args (dst_slot_stride == count,
    dst_block_offset == node_id*block_count); the remote blocks derive their
    offsets from the ring buffer. ``chunk_ready_flags_ptr`` is a device uint64
    buffer of >= ring_blocks entries, ZEROED before this launch.
    """
    rb = ring_blocks if ring_blocks and ring_blocks >= 1 else 1
    reasm = reassembly_blocks if reassembly_blocks and reassembly_blocks >= 1 else rb
    fused = mori_cpp.build_fused_ring_remote_gather_args(
        ring_args,
        gather_args,
        rb,
        chunk_ready_flags_ptr,
        num_nodes,
        node_id,
        reasm,
        reasm_deep_sq,
        deep_pipe,
        deep_pipe_quiet,
    )
    _reasm_ctas = reasm
    _get_ccl_func("FusedRingRemoteGatherKernel_u32").launch_struct(
        (rb + 1 + _reasm_ctas,), (512,), 0, s, fused
    )
    return True


# ---------------------------------------------------------------------------
# IntraNodeSubGroupBroadcastSdma — intra-node SDMA broadcast over a
# sub-group of local ranks (the placement phase of the leader-only hierarchical
# AllGather: leader's full N*G output is fanned to its G local ranks over XGMI).
# ---------------------------------------------------------------------------


class IntraNodeSubGroupBroadcastSdma:
    """Intra-node SDMA broadcast over an arithmetic sub-group of local ranks.

    The ``group_size`` ranks ``{pe_base, pe_base+pe_stride, ...}`` (this PE at
    position ``group_pos``) receive the full ``count``-element buffer held by the
    root (``group_pos == 0``) over the SDMA copy engines (XGMI). After the call
    every member's output equals the root's input. The placement-side building
    block of the leader-only hierarchical cross-node AllGather (intra-node
    phase). The default (``group_size=-1``) broadcasts from rank 0 over the whole
    world.

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``.
    Requires ``MORI_ENABLE_SDMA=1`` (the SDMA copy-engine path).
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        out_buffer_bytes: Optional[int] = None,
        group_size: int = -1,
        group_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
    ):
        _require_sdma_env("IntraNodeSubGroupBroadcastSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.group_size = group_size if group_size >= 0 else npes
        handle_class = getattr(mori_cpp, "IntraNodeSubGroupBroadcastSdmaHandle")
        if out_buffer_bytes is None:
            out_buffer_bytes = 512 * 1024 * 1024
        self._handle = handle_class(
            my_pe, npes, out_buffer_bytes, group_size, group_pos, pe_base, pe_stride
        )

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        # ``count`` is the number of elements in the broadcast payload (the full
        # buffer), not a per-rank shard. On non-root members ``input_data`` is
        # ignored; pass the user output tensor (any same-dtype buffer) as input.
        byte_count = count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        in_ptr = input_data.data_ptr() if input_data is not None else 0
        args = self._handle.prepare_sync(in_ptr, u32_count, s)
        _get_ccl_func("OneShotBroadcastSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), u32_count, s)
        return True


# ---------------------------------------------------------------------------
# AllreduceSdma
# ---------------------------------------------------------------------------


def _sdma_env_enabled() -> bool:
    raw = os.environ.get("MORI_ENABLE_SDMA", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


class _AllreduceTdm:
    """AllReduce on gfx125x without SDMA, on the TDM All2all kernel.

    On the links an all-reduce is an all-to-all followed by an all-gather: every
    rank sends chunk j of its input to rank j, each rank sums the npes chunks it
    then holds, and sends that sum to all the others. So the handle is two
    All2allSdma handles and a sum between them, three kernels on the caller's
    stream and no host synchronization, which a graph can capture:

    * reduce-scatter: an all-to-all whose result stays in memory of the handle;
    * the sum of its npes chunks, written into this rank's chunk of the output;
    * all-gather of that chunk into the output of every rank.

    Buffers of at most _ONESHOT_MAX_BYTES take one transfer instead of two:
    every rank sends its whole buffer to all the others and each sums locally.

    A count that does not split into npes chunks of whole 4-byte words goes
    through padded copies in memory of the handle.
    """

    _ONESHOT_MAX_BYTES = 1 << 20

    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int],
        output_buffer_size: Optional[int],
        transit_buffer_size: Optional[int],
        dtype: torch.dtype,
    ):
        self.my_pe = my_pe
        self.npes = npes
        self.dtype = dtype
        # torch has no arithmetic on uint32; the sum of the bits is the same.
        self._sum_dtype = torch.int32 if dtype == torch.uint32 else dtype
        self._element_size = torch.empty((), dtype=dtype).element_size()
        if input_buffer_size is not None:
            room = input_buffer_size
        elif transit_buffer_size is not None:
            room = transit_buffer_size // 2
        else:
            room = 512 * 1024 * 1024 // 2
        # Whole pages a rank: the TDM kernel wants its halves on 128 B.
        chunk = -(-room // (npes * 4096)) * 4096
        room = chunk * npes
        self._room = room
        one = min(room, self._ONESHOT_MAX_BYTES)
        one = -(-one // 4096) * 4096
        self._one_room = one
        # One-shot: an all-gather of the whole buffer; the result holds npes of them.
        self._one = All2allSdma(
            my_pe,
            npes,
            input_buffer_size=one,
            output_buffer_size=npes * one,
            copy_output_to_user=False,
            _gather=True,
        )
        self._rs = None
        self._ag = None
        if room > self._ONESHOT_MAX_BYTES:
            self._rs = All2allSdma(
                my_pe,
                npes,
                input_buffer_size=room,
                output_buffer_size=room,
                copy_output_to_user=False,
            )
            # Into the caller's output, so two halves of the largest buffer.
            self._ag = All2allSdma(
                my_pe,
                npes,
                input_buffer_size=chunk,
                output_buffer_size=2 * room,
                copy_output_to_user=True,
                _gather=True,
            )
        self._pad_in = None
        self._pad_out = None

    def _typed(self, tensor):
        return (
            tensor if tensor.dtype == self._sum_dtype else tensor.view(self._sum_dtype)
        )

    def _reduce(self, parts, rows: int, cols: int, out) -> None:
        torch.sum(self._typed(parts)[: rows * cols].view(rows, cols), dim=0, out=out)

    def _run(self, input_data, output_data, count: int, stream) -> None:
        n = self.npes
        es = self._element_size
        nbytes = count * es
        out = self._typed(output_data)
        if nbytes <= self._ONESHOT_MAX_BYTES or self._rs is None:
            if nbytes > self._one_room:
                raise ValueError(
                    f"count={count} needs {nbytes} bytes a rank, the handle has "
                    f"{self._one_room}"
                )
            self._one(input_data, None, nbytes // 4, stream)
            parts = self._one.get_output_transit_buffer(dtype=self._sum_dtype)
            self._reduce(parts, n, count, out[:count])
            return
        if nbytes > self._room:
            raise ValueError(
                f"count={count} needs {nbytes} bytes a rank, the handle has {self._room}"
            )
        chunk = count // n
        u32 = chunk * es // 4
        self._rs(input_data, None, u32, stream)
        parts = self._rs.get_output_transit_buffer(dtype=self._sum_dtype)
        own = out[self.my_pe * chunk : (self.my_pe + 1) * chunk]
        self._reduce(parts, n, chunk, own)
        self._ag(own, output_data, u32, stream)

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        n = self.npes
        es = self._element_size
        # Elements a chunk must hold whole 4-byte words; a one-shot call only whole words.
        step = 4 // es
        one_shot = count * es <= self._ONESHOT_MAX_BYTES or self._rs is None
        unit = step if one_shot else n * step
        padded = -(-count // unit) * unit
        ctx = torch.cuda.stream(stream) if stream is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            if padded == count:
                self._run(input_data, output_data, count, stream)
                return True
            if self._pad_in is None or self._pad_in.numel() < padded:
                dev = input_data.device
                self._pad_in = torch.zeros(padded, dtype=self._sum_dtype, device=dev)
                self._pad_out = torch.zeros(padded, dtype=self._sum_dtype, device=dev)
            self._pad_in[:count].copy_(self._typed(input_data)[:count])
            self._pad_in[count:padded].zero_()
            self._run(self._pad_in, self._pad_out, padded, stream)
            self._typed(output_data)[:count].copy_(self._pad_out[:count])
            return True
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    def register_output_buffer(self, tensor) -> None:
        """Let the peers write the result straight into ``tensor``; collective,
        like All2allSdma.register_output_buffer, whose rules apply."""
        if self._ag is not None:
            self._ag.register_output_buffer(tensor)

    def deregister_output_buffer(self, tensor) -> None:
        if self._ag is not None:
            self._ag.deregister_output_buffer(tensor)


_DTYPE_TO_SUFFIX = {
    torch.uint32: "u32",
    torch.int32: "i32",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}

_HANDLE_MAP = {
    torch.uint32: "AllreduceSdmaHandle",
    torch.int32: "AllreduceSdmaHandleInt32",
    torch.float32: "AllreduceSdmaHandleFp32",
    torch.float16: "AllreduceSdmaHandleFp16",
    torch.bfloat16: "AllreduceSdmaHandleBf16",
}


class AllreduceSdma:
    _ONESHOT_MAX_BYTES = 1 << 20

    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
        dtype: torch.dtype = torch.uint32,
        mode: str = "eager",
    ):
        # gfx125x without SDMA: the TDM All2all kernels carry the all-reduce at every
        # size (_AllreduceTdm), and nothing below is built. With MORI_ENABLE_SDMA=1
        # the handle is the SDMA one, whose large calls MORI_CCL_ALLREDUCE_BACKEND=tdm
        # moves to the older two-kernel TDM path.
        self._tdm_impl = None
        if _detect_arch_str().startswith("gfx125") and not _sdma_env_enabled():
            self.my_pe = my_pe
            self.npes = npes
            self.dtype = dtype
            self.mode = mode
            self._backend = "tdm"
            if dtype not in _DTYPE_TO_SUFFIX:
                raise ValueError(
                    f"Unsupported dtype {dtype}. Supported: {list(_DTYPE_TO_SUFFIX.keys())}"
                )
            impl = _AllreduceTdm(
                my_pe,
                npes,
                input_buffer_size,
                output_buffer_size,
                transit_buffer_size,
                dtype,
            )
            self._tdm_impl = impl

            def run_sync(
                input_data, output_data, count, stream=None, force_copy_output=False
            ):
                return impl(input_data, output_data, count, stream)

            # Instance attributes, so they take the place of the SDMA methods below. The
            # kernels wait for every peer themselves, so the async pair is the call.
            self._run_sync = run_sync
            self.start_async = run_sync
            self.wait_async = lambda stream=None: 0.0
            self.is_async_in_progress = lambda: False
            self.cancel_async = lambda: None
            self.reset_flags = lambda: None
            self.register_output_buffer = impl.register_output_buffer
            self.deregister_output_buffer = impl.deregister_output_buffer
            return
        _require_sdma_env("AllreduceSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.dtype = dtype
        self._element_size = torch.empty((), dtype=dtype).element_size()
        self._async_oneshot = False
        self.mode = mode
        # Resolved once, in the constructor, so every call in a process uses the
        # same transport: the flag protocol is shared state, and two ranks
        # disagreeing about which kernel is running would hang rather than fail.
        self._backend = _resolve_backend("MORI_CCL_ALLREDUCE_BACKEND")
        self._type_suffix = _DTYPE_TO_SUFFIX.get(dtype)
        if self._type_suffix is None:
            raise ValueError(
                f"Unsupported dtype {dtype}. Supported: {list(_DTYPE_TO_SUFFIX.keys())}"
            )

        handle_name = _HANDLE_MAP.get(dtype)
        if handle_name is None:
            raise ValueError(f"Unsupported dtype {dtype}")
        handle_class = getattr(mori_cpp, handle_name)

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, transit_buffer_size, copy_output_to_user
            )
        else:
            self._handle = handle_class(
                my_pe, npes, 512 * 1024 * 1024, copy_output_to_user
            )

    def _use_tdm(self, count: int) -> bool:
        """TDM carries only the two-shot sizes. Below the one-shot threshold the
        LDS round trip every TDM tile pays is not amortized, and that range is
        where SDMA's one-descriptor-per-chunk form is already efficient, so small
        messages stay on SDMA even when the TDM backend is selected."""
        return (
            self._backend == "tdm"
            and count * self._element_size > self._ONESHOT_MAX_BYTES
        )

    def _tdm_grid(self, count: int):
        """Blocks are capped at _TDM_MAX_BLOCKS and scaled down for messages too
        small to occupy it, sized against both phases: one warp per tile for the
        scatter, one thread per 16-byte pack for the local reduce."""
        threads = _TDM_THREADS
        warps_per_block = threads // _TDM_WAVE_SIZE
        per_rank_elems = count // self.npes
        per_rank_bytes = per_rank_elems * self._element_size
        tiles = (per_rank_bytes + _TDM_TILE_BYTES - 1) // _TDM_TILE_BYTES
        pack_size = 16 // self._element_size
        packed_per_rank = (per_rank_elems + pack_size - 1) // pack_size
        need = max(
            (tiles + warps_per_block - 1) // warps_per_block,
            (packed_per_rank + threads - 1) // threads,
        )
        blocks = max(1, min(_TDM_MAX_BLOCKS, need))
        return blocks, threads

    def _run_sync(
        self,
        input_data,
        output_data,
        count: int,
        stream=None,
        force_copy_output: bool = False,
    ) -> bool:
        s = _stream_to_int(stream)
        sfx = self._type_suffix

        if self._use_tdm(count):
            blocks, threads = self._tdm_grid(count)
            lds = _tdm_shared_mem(threads)
            args = self._handle.prepare_reduce_scatter(
                input_data.data_ptr(), output_data.data_ptr(), count, s
            )
            _get_ccl_func(f"TdmReduceScatterKernel_{sfx}").launch_struct(
                (blocks,), (threads,), lds, s, args
            )
            args = self._handle.prepare_allgather(count, s)
            _get_ccl_func(f"TdmAllGatherKernel_{sfx}").launch_struct(
                (blocks,), (threads,), lds, s, args
            )
            self._handle.finish_sync(
                output_data.data_ptr(), count, s, force_copy_output
            )
            return True

        # Step 1: ReduceScatter
        args = self._handle.prepare_reduce_scatter(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        blocks, threads = self._handle.get_reduce_scatter_grid(count)
        oneshot = count * self._element_size <= self._ONESHOT_MAX_BYTES
        kernel = "SdmaOneShotAllreduceKernel" if oneshot else "SdmaReduceScatterKernel"
        _get_ccl_func(f"{kernel}_{sfx}").launch_struct(
            (blocks,), (threads,), 0, s, args
        )
        if oneshot:
            self._handle.finish_sync(
                output_data.data_ptr(), count, s, force_copy_output, True
            )
            return True
        # Step 2: AllGather
        args = self._handle.prepare_allgather(count, s)
        _get_ccl_func(f"AllGatherSdmaFromProtectedKernel_{sfx}").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), count, s, force_copy_output)
        return True

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        return self._run_sync(input_data, output_data, count, stream)

    def allreduce_inplace(self, data, count: int, stream=None) -> bool:
        return self._run_sync(data, data, count, stream, force_copy_output=True)

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        s = _stream_to_int(stream)
        sfx = self._type_suffix
        # ReduceScatter
        args = self._handle.prepare_async_reduce_scatter(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        self._async_oneshot = count * self._element_size <= self._ONESHOT_MAX_BYTES

        if self._use_tdm(count):
            blocks, threads = self._tdm_grid(count)
            lds = _tdm_shared_mem(threads)
            _get_ccl_func(f"TdmReduceScatterKernel_{sfx}").launch_struct(
                (blocks,), (threads,), lds, s, args
            )
            args = self._handle.prepare_async_allgather_put(count, s)
            _get_ccl_func(f"TdmAllGatherKernel_{sfx}").launch_struct(
                (blocks,), (threads,), lds, s, args
            )
            capturing = torch.cuda.is_current_stream_capturing()
            self._handle.after_async_start(capturing)
            return True

        blocks, threads = self._handle.get_reduce_scatter_grid(count)
        kernel = (
            "SdmaOneShotAllreduceKernel"
            if self._async_oneshot
            else "SdmaReduceScatterKernel"
        )
        _get_ccl_func(f"{kernel}_{sfx}").launch_struct(
            (blocks,), (threads,), 0, s, args
        )
        if not self._async_oneshot:
            args = self._handle.prepare_async_allgather_put(count, s)
            _get_ccl_func(f"AllGatherSdmaFromProtectedKernel_{sfx}").launch_struct(
                (1,), (512,), 0, s, args
            )
        capturing = torch.cuda.is_current_stream_capturing()
        self._handle.after_async_start(capturing)
        return True

    def wait_async(self, stream=None) -> float:
        s = _stream_to_int(stream)
        capturing = torch.cuda.is_current_stream_capturing()
        if self._async_oneshot:
            return self._handle.finish_async_wait(s, capturing, True)
        return self._handle.finish_async_wait(s, capturing)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()
