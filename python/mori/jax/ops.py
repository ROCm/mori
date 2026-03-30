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
import gc
from mori import shmem
from mori import cpp
import jax
import jax.numpy as jnp

from jax._src.lib import _jax
import numpy as np


def get_distributed_client():  # -> _jax.DistributedRuntimeClient:
    from jax._src.distributed import global_state

    assert isinstance(global_state.client, _jax.DistributedRuntimeClient)
    return global_state.client


def shmem_init_attr(rank, world_size, sync_name="mori/unique_id", timeout_ms=5_000):

    client = get_distributed_client()
    if rank == 0:
        unique_id = shmem.shmem_get_unique_id()
        client.key_value_set_bytes(sync_name, unique_id)
    else:
        unique_id = client.blocking_key_value_get_bytes(sync_name, timeout_ms)

    shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id
    )
    # print(f"{rank} unique ID initattr OK", flush=True)
    cpp.preload_kernels()


def shmem_finalize():
    jax.clear_caches()
    cpp.clear_ep_handle_cache()
    gc.collect()
    shmem.shmem_finalize()


class EpDispatchCombineOp:
    def __init__(self, config: cpp.EpDispatchCombineConfig):
        self.config = config

    # output tuple (weights and scales could be None):
    # (out, out_indices, total_recv_token_num, out_weights (optional), out_scales (optional))
    def dispatch(
        self,
        input,
        weights,
        scales,
        indices,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        n_tokens = self.config.max_num_tokens_to_recv()
        has_scales = scales is not None and self.config.scale_dim > 0
        has_weights = weights is not None

        outputs = [
            # out
            jax.ShapeDtypeStruct((n_tokens, self.config.hidden_dim), input.dtype),
            # out_indices
            jax.ShapeDtypeStruct(
                (n_tokens, self.config.num_experts_per_token), jnp.int32
            ),
            # total_recv_token_num
            jax.ShapeDtypeStruct((), jnp.int32),
        ]
        if has_weights:
            outputs.append(
                jax.ShapeDtypeStruct(
                    (n_tokens, self.config.num_experts_per_token), jnp.float32
                ),
            )
        if has_scales:
            outputs.append(
                jax.ShapeDtypeStruct((n_tokens, self.config.scale_dim), scales.dtype),
            )
        args = [input, indices]
        if has_weights:
            args.append(weights)
        if has_scales:
            args.append(scales)

        output = jax.ffi.ffi_call("mori_ep", outputs)(
            *args,
            ep_config=np.asarray(self.config.to_packed_array(), dtype=np.int32),
            dispatch_op=True,
            has_scales=np.int32(has_scales),
            block_num=np.int32(block_num),
            rdma_block_num=np.int32(rdma_block_num),
            warp_per_block=np.int32(warp_per_block),
        )
        # Normalize output to always be:
        # [out, out_indices, total_recv_token_num, out_weights, out_scales]
        if has_weights:
            if not has_scales:
                return (*output, None)
            return output
        if has_scales:
            out, out_indices, total, out_scales = output
            return (out, out_indices, total, None, out_scales)
        return (*output, None, None)

    # output order:
    # out, out_weights (optional)
    def combine(
        self,
        input,
        weights,
        indices,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        n_tokens = self.config.max_num_inp_token_per_rank
        has_weights = weights is not None
        outputs = [
            jax.ShapeDtypeStruct((n_tokens, self.config.hidden_dim), input.dtype)
        ]
        if has_weights:
            outputs.append(
                jax.ShapeDtypeStruct(
                    (n_tokens, self.config.num_experts_per_token), jnp.float32
                )
            )

        args = [input, indices]
        if has_weights:
            args.append(weights)

        output = jax.ffi.ffi_call("mori_ep", outputs)(
            *args,
            ep_config=np.asarray(self.config.to_packed_array(), dtype=np.int32),
            combine_op=True,
            block_num=np.int32(block_num),
            rdma_block_num=np.int32(rdma_block_num),
            warp_per_block=np.int32(warp_per_block),
        )
        if call_reset:
            jax.ffi.ffi_call("mori_ep", (), has_side_effect=True)(
                ep_config=np.asarray(self.config.to_packed_array(), dtype=np.int32),
                reset_op=True,
            )
        return output if has_weights else (*output, None)

    def get_dispatch_src_token_pos(self, total_recv_token_num):
        if self.config.kernel_type.value in (
            cpp.EpDispatchCombineKernelType.IntraNode.value,
            cpp.EpDispatchCombineKernelType.InterNodeV1.value,
            cpp.EpDispatchCombineKernelType.InterNodeV1LL.value,
        ):
            # here we need to allocate enough space to accomodate handle->totalRecvTokenNum[0] items
            n_tokens = self.config.max_num_tokens_to_recv()
            return jax.ffi.ffi_call(
                "mori_ep",
                (jax.ShapeDtypeStruct((n_tokens,), jnp.int32)),
            )(
                total_recv_token_num,
                ep_config=np.asarray(self.config.to_packed_array(), dtype=np.int32),
                get_src_token_id=True,
            )

        raise NotImplementedError
