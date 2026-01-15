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
import mori

from dataclasses import dataclass
#import pickle
import numpy as np
import jax
import jax.numpy as jnp

# this is for jax 0.6.0
#import jaxlib.xla_extension as xe 
from jax._src.lib import _jax

class EpDispatchCombineKernelType(mori.cpp.EpDispatchCombineKernelType):
  def __str__(self):
    return self.name
  
def get_distributed_client(): # -> _jax.DistributedRuntimeClient:
  from jax._src.distributed import global_state
  assert isinstance(global_state.client, _jax.DistributedRuntimeClient)
  return global_state.client
  
def mori_shmem_init_attr(rank, world_size, sync_name="mori/unique_id", 
            timeout_ms=5_000):
  
  client = get_distributed_client()
  if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()
    client.key_value_set_bytes(
       sync_name, unique_id
       #devs.key, pickle.dumps(nccl_id)
    )
  else:
    unique_id = client.blocking_key_value_get_bytes(
      sync_name, timeout_ms)
  
  #print(f"{rank} unique ID initattr {unique_id}", flush=True)
  mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
                   rank, world_size, unique_id)
  print(f"{rank} unique ID initattr OK", flush=True)

@dataclass
class EpDispatchCombineConfig:
  data_type: jnp.dtype
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
  use_external_inp_buf: bool = True
  kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
  gpu_per_node: int = 8
  rdma_block_num: int = 0
  num_qp_per_pe: int = 1

class EpDispatchCombineOp:
  def __init__(self, config):
    handle_class = mori.cpp.EpDispatchCombineHandle
    self.config = mori.cpp.EpDispatchCombineConfig(
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
        num_qp_per_pe=config.num_qp_per_pe)
    self.handle_ = handle_class(self.config)

  def dispatch(self, input, weights, scales, indices,
          has_weights = True, has_scales = True, 
          block_num: int = -1, warp_per_block: int = -1):
    n_tokens = self.config.MaxNumTokensToRecv()
    has_scales = has_scales and self.config.scale_dim > 0
    return jax.ffi.ffi_call(
      "launch_dispatch", (
        # out
        jax.ShapeDtypeStruct((n_tokens, self.config.hidden_dim), input.dtype),
        # out_weights
        jax.ShapeDtypeStruct((n_tokens, self.config.num_experts_per_token) if has_weights else (1,1), jnp.float32),
        # out_scales
        jax.ShapeDtypeStruct((n_tokens, self.config.scale_dim) if has_scales else (1,1), scales.dtype),
        # out_indices
        jax.ShapeDtypeStruct((n_tokens, self.config.num_experts_per_token), jnp.int32),
        # total_recv_token_num
        jax.ShapeDtypeStruct((), jnp.int32)),
    )(
      input,
      weights,
      scales,
      indices,
      handle_ptr=np.int64(self.handle_.ptr()),
      kernel_type=np.int32(self.config.kernel_type.value),
      block_num=np.int32(block_num),
      warp_per_block=np.int32(warp_per_block),
      has_scales=np.int32(has_scales),
      has_weights=np.int32(has_weights),
    )
    
  def combine(
    self, input, weights, indices,
    has_weights = True,
    block_num: int = -1,
    warp_per_block: int = -1,
    call_reset: bool = False,
  ):
    n_tokens = self.config.max_num_inp_token_per_rank
    output = jax.ffi.ffi_call(
      "launch_combine", (
        # out
        jax.ShapeDtypeStruct((n_tokens, self.config.hidden_dim), input.dtype),
        # out_weights
        jax.ShapeDtypeStruct((n_tokens, self.config.num_experts_per_token) if has_weights else (1,1), jnp.float32)),
    )(
      input,
      weights,
      indices,
      handle_ptr=np.int64(self.handle_.ptr()),
      kernel_type=np.int32(self.config.kernel_type.value),
      block_num=np.int32(block_num),
      warp_per_block=np.int32(warp_per_block),
      has_weights=np.int32(has_weights),
    )
    if call_reset:
      jax.ffi.ffi_call(
         "launch_reset", ())(
          handle_ptr=np.int64(self.handle_.ptr()),
      )
    return output

  def get_dispatch_src_token_pos(self, total_recv_token_num):
    if self.config.kernel_type.value in (
      EpDispatchCombineKernelType.IntraNode.value,
      EpDispatchCombineKernelType.InterNodeV1.value,
      EpDispatchCombineKernelType.InterNodeV1LL.value,
    ):
      # here we need to allocate enough space to accomodate handle->totalRecvTokenNum[0] items
      n_tokens = self.config.MaxNumTokensToRecv()
      return jax.ffi.ffi_call(
            "get_dispatch_src_token_id", (
            jax.ShapeDtypeStruct((n_tokens,), jnp.int32)),
        )(
            total_recv_token_num, 
            handle_ptr=np.int64(self.handle_.ptr()),
         )
  
    raise NotImplementedError
