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
"""Python wrapper for the LocalExpertCount kernel.

Uses the shared two-step JIT pattern from ``mori.ops._jit_loader``:
  1. ``ensure_compiled("ep_local_expert_count")``  — compile ``.hip`` → ``.hsaco``
  2. ``load_hip_module("ep_local_expert_count", init_shmem=False)`` — load into HIP
     runtime.  Shmem is not needed for this kernel.

The loaded module is cached globally in ``_jit_loader._loaded_modules`` and
shared with any other ops that load the same kernel name.

The kernel is launched entirely from Python via ``HipModule.launch_struct``,
mirroring the dispatch/combine pattern and avoiding the C++ pybind path.
"""
import ctypes

_KERNEL_NAME = "ep_local_expert_count"
_WARP_SIZE = 64


class _LocalExpertCountArgs(ctypes.Structure):
    """Mirror of ``mori::moe::LocalExpertCountArgs`` in dispatch_combine.hpp."""

    _fields_ = [
        ("indices", ctypes.c_void_p),
        ("totalRecvTokenNum", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("numExpertPerRank", ctypes.c_int),
        ("numExpertPerToken", ctypes.c_int),
        ("localExpertCount", ctypes.c_void_p),
    ]


def _ensure_kernel():
    """Compile and load the LocalExpertCount kernel on first call."""
    from mori.ops._jit_loader import ensure_compiled, load_hip_module

    ensure_compiled(_KERNEL_NAME)
    load_hip_module(_KERNEL_NAME, init_shmem=False)


def launch_local_expert_count(
    config,
    indices_ptr,
    total_recv_token_num_ptr,
    local_expert_count_ptr,
    block_num=-1,
    warp_per_block=-1,
    stream=0,
):
    """Count how many tokens are routed to each local expert.

    Args:
        config: ``EpDispatchCombineConfig`` (pybind object) with ``rank``,
            ``num_experts_per_rank``, ``num_experts_per_token``,
            ``warp_num_per_block``, and ``block_num`` set.
        indices_ptr: Device pointer (int) to a flat ``int32`` array of expert
            indices, shape ``[total_recv_tokens, num_experts_per_token]``.
        total_recv_token_num_ptr: Device pointer (int) to a 1-element ``int32``
            tensor holding the number of received tokens.
        local_expert_count_ptr: Device pointer (int) to an ``int32`` output
            array of length ``num_experts_per_rank``.  Zeroed before counting.
        block_num: Grid size override; uses ``config.block_num`` when -1.
        warp_per_block: Warps-per-block override; uses ``config.warp_num_per_block`` when -1.
        stream: HIP stream handle (raw ``hipStream_t`` as int).
    """
    from mori.jit.hip_driver import _get_hip_lib
    from mori.ops._jit_loader import _loaded_modules

    _ensure_kernel()
    mod = _loaded_modules.get(_KERNEL_NAME)
    if mod is None:
        raise RuntimeError("[mori] LocalExpertCountKernel failed to load")

    wpb = config.warp_num_per_block if warp_per_block <= 0 else warp_per_block
    bn = config.block_num if block_num <= 0 else block_num
    if wpb <= 0 or bn <= 0:
        raise RuntimeError(
            "LaunchLocalExpertCount requires positive block and warp settings"
        )

    hip = _get_hip_lib()
    err = hip.hipMemsetAsync(
        ctypes.c_void_p(local_expert_count_ptr),
        ctypes.c_int(0),
        ctypes.c_size_t(config.num_experts_per_rank * 4),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(
            f"HIP error {err}: hipMemsetAsync failed for LocalExpertCount"
        )

    args = _LocalExpertCountArgs(
        indices=indices_ptr,
        totalRecvTokenNum=total_recv_token_num_ptr,
        rank=config.rank,
        numExpertPerRank=config.num_experts_per_rank,
        numExpertPerToken=config.num_experts_per_token,
        localExpertCount=local_expert_count_ptr,
    )

    func = mod.get_function("LocalExpertCountKernel")
    func.launch_struct(
        (bn,),
        (_WARP_SIZE * wpb,),
        0,
        stream,
        ctypes.addressof(args),
    )
