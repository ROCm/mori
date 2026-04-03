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

Follows the same JIT-load pattern as ``EpDispatchCombineOp``:
  1. ``compile_genco()``  — JIT-compile ``.hip`` → ``.hsaco``
  2. ``HipModule(hsaco)`` — load into the HIP runtime
  3. ``shmem_module_init`` — initialise ``globalGpuStates`` in that module
  4. ``module.get_function()`` — obtain ``hipFunction_t``
  5. ``launch_local_expert_count_direct()`` — launch via ``hipModuleLaunchKernel``

Imports are deferred to function bodies to avoid the circular-import that
arises from ``mori.cpp`` ↔ ``mori.ops`` mutual loading during package init.
"""

_hip_module = None  # HipModule — kept alive to prevent GC / hipModuleUnload
_hip_function = None  # HipFunction for "LocalExpertCountKernel"


def _ensure_kernel():
    """JIT-compile and load ep_local_expert_count on first call."""
    global _hip_module, _hip_function
    if _hip_function is not None:
        return
    try:
        import sys
        from mori.jit.core import compile_genco
        from mori.jit.hip_driver import HipModule

        hsaco = compile_genco("ep_local_expert_count")
        _hip_module = HipModule(hsaco)
        sys.modules["libmori_pybinds"].shmem_module_init(_hip_module._module.value)
        _hip_function = _hip_module.get_function("LocalExpertCountKernel")
    except Exception as e:
        import warnings

        warnings.warn(f"[mori] ep_local_expert_count JIT skipped: {e}")


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

    Uses the same HipModule/hipModuleLaunchKernel pattern as
    ``EpDispatchCombineOp``: the kernel is JIT-compiled once, loaded into HIP,
    and launched via ``launch_local_expert_count_direct`` which accepts a raw
    ``hipFunction_t`` instead of looking one up from ``KernelRegistry``.

    Args:
        config: ``EpDispatchCombineConfig`` with ``rank``, ``num_experts_per_rank``,
            ``num_experts_per_token``, ``warp_num_per_block``, and ``block_num`` set.
        indices_ptr: Device pointer (int) to a flat ``int32`` array of expert
            indices, shape ``[total_recv_tokens, num_experts_per_token]``.
        total_recv_token_num_ptr: Device pointer (int) to a 1-element ``int32``
            tensor holding the number of received tokens.
        local_expert_count_ptr: Device pointer (int) to an ``int32`` output
            array of length ``num_experts_per_rank``.  Zeroed by the kernel
            before counting.
        block_num: Grid size override; uses ``config.block_num`` when -1.
        warp_per_block: Warps-per-block override; uses ``config.warp_num_per_block`` when -1.
        stream: HIP stream handle (raw ``hipStream_t`` as int).
    """
    import sys

    _ensure_kernel()
    if _hip_function is None:
        raise RuntimeError("[mori] LocalExpertCountKernel failed to load")
    sys.modules["libmori_pybinds"].launch_local_expert_count_direct(
        config,
        indices_ptr,
        total_recv_token_num_ptr,
        local_expert_count_ptr,
        _hip_function._func.value,
        block_num=block_num,
        warp_per_block=warp_per_block,
        stream=stream,
    )
