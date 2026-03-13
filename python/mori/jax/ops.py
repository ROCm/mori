# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
#
# Based on PR #173 by Chao Chen <cchen104@amd.com>
# Adapted for refactored architecture with JIT kernel compilation.
"""High-level JAX wrappers for mori MoE dispatch/combine operations."""

from mori.jax import _ffi_registry

# Kernel type constants (mirrors mori::moe::KernelType)
INTRA_NODE = 0
INTER_NODE = 1
INTER_NODE_V1 = 2
INTER_NODE_V1_LL = 3
ASYNC_LL = 4

_KERNEL_TYPE_TO_HIP = {
    INTRA_NODE: "ep_intranode",
    INTER_NODE: "ep_internode",
    INTER_NODE_V1: "ep_internode_v1",
    INTER_NODE_V1_LL: "ep_internode_v1ll",
    ASYNC_LL: "ep_async_ll",
}


class EpDispatchCombineOp:
    """Manages an EpDispatchCombine handle and exposes dispatch/combine as
    JAX-compatible operations via XLA FFI custom calls.

    Usage::

        op = EpDispatchCombineOp(
            rank=0, world_size=8, hidden_dim=4096,
            num_experts_per_rank=8, num_experts_per_token=2,
            max_num_inp_token_per_rank=128)
        # ... use op.handle_id as an FFI attribute in custom calls ...
        op.close()
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        scale_dim: int = 0,
        scale_type_size: int = 0,
        max_token_type_size: int = 4,
        max_num_inp_token_per_rank: int = 128,
        num_experts_per_rank: int = 1,
        num_experts_per_token: int = 2,
        warp_num_per_block: int = 1,
        block_num: int = 1,
        kernel_type: int = INTRA_NODE,
        gpu_per_node: int = 8,
        rdma_block_num: int = 0,
        num_qp_per_pe: int = 1,
    ):
        self._ensure_kernels(kernel_type)
        self.handle_id = _ffi_registry.create_handle(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=max_token_type_size,
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            gpu_per_node=gpu_per_node,
            rdma_block_num=rdma_block_num,
            num_qp_per_pe=num_qp_per_pe,
        )
        self._kernel_type = kernel_type
        self._closed = False

    @staticmethod
    def _ensure_kernels(kernel_type: int):
        """JIT-compile and register .hsaco kernels for the given kernel_type.

        Also initializes globalGpuStates in the loaded module so that shmem
        device functions work correctly. Requires shmem to be initialized.
        """
        from mori.jit.core import compile_genco

        if kernel_type not in _KERNEL_TYPE_TO_HIP:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
        hip_name = _KERNEL_TYPE_TO_HIP[kernel_type]
        hsaco_path = compile_genco(hip_name)
        _ffi_registry.register_kernel_module(kernel_type, hsaco_path)
        # globalGpuStates is now initialized automatically inside
        # KernelManager::RegisterModule via ShmemModuleInit.
        # If shmem was initialized after module registration, call
        # shmem_module_init_for_kernel to re-initialize.
        try:
            _ffi_registry.shmem_module_init_for_kernel(kernel_type)
        except Exception:
            pass

    def close(self):
        if not self._closed:
            _ffi_registry.destroy_handle(self.handle_id)
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
