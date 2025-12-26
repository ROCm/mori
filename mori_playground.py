import jax
import jax.numpy as jnp

import mori
import os
import time

import torch
import torch.distributed as dist

class PlaygroundTestCase:
    def __init__(self, rank, world_size, dtype=torch.bfloat16):
        self.rank = rank
        self.world_size = world_size
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=7168,
            # scale_dim=32,
            scale_dim=0,
            scale_type_size=torch.tensor(
                [], dtype=torch.float8_e4m3fnuz
            ).element_size(),
            max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
            max_num_inp_token_per_rank=4096,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            use_external_inp_buf=False,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            gpu_per_node=1,
        )
        
        op = mori.ops.EpDispatchCombineOp(self.config)
        
        @jax.jit
        def gen_inputs():
          w,h = 256, 768
          input = jnp.linspace(-1, 1, w*h).reshape([w, h])
          weights = jnp.linspace(-1, 1, w*h).reshape([w, h])
          scales = jnp.linspace(-1, 1, w*h).reshape([w, h])
          indices = jnp.linspace(0, 100, w*h, dtype=jnp.int32).reshape([w, h])
          return input, weights, scales, indices
        
        S = gen_inputs()
        res = op.dispatch_jax(*S)
        
        
if __name__ == "__main__":
  print("Start")
  test_case = PlaygroundTestCase(rank=110, world_size=1)