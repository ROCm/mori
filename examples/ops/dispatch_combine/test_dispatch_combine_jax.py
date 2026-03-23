import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
import numpy as np

import gc, random, sys
import os
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")
import mori
import argparse, time, functools
from functools import partial
from jax.experimental.multihost_utils import sync_global_devices

def init_distributed():
  parser = argparse.ArgumentParser()
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--coordination_service", type=str, default="")
  args, _ = parser.parse_known_args()

  jax.distributed.initialize(
      coordinator_address=args.coordination_service or "localhost:12345",
      num_processes=args.world_size,
      process_id=args.rank)
  print(f"[rank {args.rank}] local_dev: {jax.local_devices()}", flush=True)
  
  return (args.rank, args.world_size)

class EpDispatchCombineTestCase:
    def __init__(self, rank, world_size, dtype=jnp.bfloat16):
        self.dtype = dtype
        self.jax_scale_dtype = jnp.float8_e4m3fnuz
        self.rank = rank
        self.world_size = world_size
        
        self.config = mori.cpp.EpDispatchCombineConfig(
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=7168,
            scale_dim=32,
            scale_type_size=jnp.dtype(self.jax_scale_dtype).itemsize,
            max_token_type_size=jnp.dtype(jnp.float32).itemsize,
            max_num_inp_token_per_rank=4096,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            warp_num_per_block=8,
            block_num=80,
            use_external_inp_buf=True,
            kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
            gpu_per_node=8,
            rdma_block_num=16,
            num_qp_per_pe=1,
            quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
        )
        
    def setup(self):
        mori.jax.shmem_init_attr(self.rank, self.world_size)

    def cleanup(self):
        jax.clear_caches()
        mori.cpp.clear_ep_handle_cache()
        gc.collect()
        mori.shmem.shmem_finalize()
        
  
    def gen_local_data(self, num_tokens):
        """Generate random test data using NumPy (avoids JAX/ROCm PRNG bugs)."""
        total_experts = self.config.num_experts_per_rank * self.config.world_size
        rng = np.random.RandomState(777 + self.rank)

        indices_np = np.zeros((num_tokens, self.config.num_experts_per_token), dtype=np.int32)
        for t in range(num_tokens):
            indices_np[t] = rng.choice(total_experts, self.config.num_experts_per_token, replace=False)
        indices = jnp.array(indices_np)

        weights = jnp.array(rng.uniform(size=(num_tokens, self.config.num_experts_per_token)).astype(np.float32))

        if self.config.scale_dim != 0:
            scales_fp32 = jnp.array(rng.uniform(size=(num_tokens, self.config.scale_dim)).astype(np.float32))
        else:
            scales_fp32 = jnp.zeros((1, 1), dtype=jnp.float32)

        input_fp32 = jnp.array(rng.standard_normal((num_tokens, self.config.hidden_dim)).astype(np.float32))
        return indices, weights, scales_fp32, input_fp32

    def all_gather_data(self, indices, weights, scales_fp32, input_fp32):
        """Gather local data from all ranks using shard_map."""
        max_tokens = self.config.max_num_inp_token_per_rank

        def do_gather(indices, weights, scales_fp32, input_fp32):
            def _gather(x):
                padded = jnp.pad(x,
                    [(0, max_tokens - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
                gathered = jax.lax.all_gather(padded, axis_name="i")
                return gathered.reshape(-1, *gathered.shape[2:])
            return _gather(indices), _gather(weights), _gather(scales_fp32), _gather(input_fp32)

        return do_gather(indices, weights, scales_fp32, input_fp32)

    def gen_test_data(self, num_tokens, mesh):
        """Generate local data, then all_gather across ranks."""
        indices, weights, scales_fp32, input_fp32 = self.gen_local_data(num_tokens)

        gather_fn = shard_map(
            self.all_gather_data,
            mesh=mesh,
            in_specs=(P(), P(), P(), P()),
            out_specs=(P(), P(), P(), P()),
            check_rep=False)
        jitted_gather = jax.jit(gather_fn)
        indices_list, weights_list, scales_list_fp32, input_list = jitted_gather(
            indices, weights, scales_fp32, input_fp32)
        scales_list = scales_list_fp32.astype(self.jax_scale_dtype)
        input_list = input_list.astype(self.dtype)

        print(f"num_tokens {num_tokens}  hidden: {self.config.hidden_dim} "
              f"indices: {indices.shape}/{indices.dtype}", flush=True)

        return (indices, weights,
            scales_fp32.astype(self.jax_scale_dtype),
            input_fp32.astype(self.dtype),
            indices_list,
            weights_list,
            scales_list,
            input_list)
        
    def run_test_once(self, op, num_tokens, test_data):
        (indices, weights, scales, inputs,
         indices_list, weights_list, scales_list, input_list) = test_data

        @jax.jit
        def ffi_calls(inputs, weights, scales, indices):
            (dispatch_output, 
            dispatch_indices, 
            num,
            dispatch_weights,
            dispatch_scales) = op.dispatch(
                inputs, weights, scales, indices)
            src_token_pos = op.get_dispatch_src_token_pos(num)

            (combine_output, 
            combine_weights) = op.combine(
                dispatch_output.astype(self.dtype), 
                dispatch_weights, indices)
            
            return (dispatch_output, 
                    dispatch_indices, 
                    num,
                    dispatch_weights, 
                    dispatch_scales), src_token_pos, combine_output, combine_weights

        print(f"{self.rank} pre-compiling ffi_calls ...", flush=True)
        compiled = ffi_calls.lower(inputs, weights, scales, indices).compile()

        sync_global_devices("all_compiled")
        print(f"{self.rank} all compiled, executing ...", flush=True)

        (dispatch_output, 
            dispatch_indices, 
            dispatch_recv_num_token,
            dispatch_weights, 
            dispatch_scales), src_token_pos, combine_output, combine_weights = compiled(inputs, weights, scales, indices)

        combine_output.block_until_ready()
        print(f"{self.rank} ffi_calls done", flush=True)

        num_recv = int(dispatch_recv_num_token)
        print(f"{self.rank} sent {num_tokens} / received {num_recv} tokens", flush=True)

        # --- dispatch validation (tokens, weights, scales, indices) ---
        @jax.jit
        def validate_dispatch(num, src_pos, base_list, base_out,
                              w_list, w_out, s_list, s_out, i_list, i_out,
                              has_scales):
          Y = base_list[src_pos]
          N = Y.shape[0]
          mask = jnp.arange(N) < num
          mask2D = mask[:,None]
          tok_ok = jnp.all((Y == base_out) | (~mask2D))
          w_ok = jnp.all((w_list[src_pos] == w_out) | (~mask2D))
          i_ok = jnp.all((i_list[src_pos] == i_out) | (~mask2D))
          s_ok = jnp.where(has_scales,
              jnp.all((s_list[src_pos] == s_out) | (~mask2D)),
              True)
          return tok_ok & w_ok & i_ok & s_ok

        res = validate_dispatch(dispatch_recv_num_token, src_token_pos,
                    input_list, dispatch_output,
                    weights_list, dispatch_weights,
                    scales_list, dispatch_scales if self.config.scale_dim != 0 else jnp.zeros_like(dispatch_weights),
                    indices_list, dispatch_indices,
                    self.config.scale_dim != 0)
        assert res, f"{self.rank} validate_dispatch FAILED!"
        print(f"{self.rank} dispatch tokens ok (num_recv={num_recv})", flush=True)

        # --- combine validation (numpy to avoid multi-process JIT shape deadlock) ---
        co = np.array(combine_output)[:num_tokens].astype(np.float32)
        inp = np.array(inputs).astype(np.float32)
        idx = np.array(indices)

        pes = idx // self.config.num_experts_per_rank
        pes_sorted = np.sort(pes, axis=-1)
        unique_pes = 1 + np.sum(pes_sorted[:, 1:] != pes_sorted[:, :-1], axis=-1)
        expected_output = inp * unique_pes[:, None]

        diff_out = np.abs(co - expected_output)
        tol_out = 1e-2 + 1e-2 * np.abs(expected_output)
        assert np.all(diff_out <= tol_out), \
            f"{self.rank} validate_combine output FAILED! max_diff={np.max(diff_out)}"

        if combine_weights is not None:
            cw = np.array(combine_weights)[:num_tokens].astype(np.float32)
            w = np.array(weights).astype(np.float32)
            expected_w = w * unique_pes[:, None]
            diff_w = np.abs(cw - expected_w)
            tol_w = 1e-5 + 1e-5 * np.abs(expected_w)
            assert np.all(diff_w <= tol_w), \
                f"{self.rank} validate_combine weights FAILED! max_diff={np.max(diff_w)}"

        print(f"{self.rank} combine tokens ok (max_diff={np.max(diff_out):.4f})", flush=True)

    def test_dispatch_combine(self):
        op = mori.jax.EpDispatchCombineOp(self.config)
        random.seed(333 + self.rank)
        max_tokens = self.config.max_num_inp_token_per_rank
        num_tokens = int(random.randint(1, max_tokens + 1))
        
        devices = np.array(jax.devices())
        mesh = jax.sharding.Mesh(devices, axis_names=("i",))
        
        for _ in range(1):
            test_data = self.gen_test_data(num_tokens, mesh)
            print(f"{self.rank} gen_test_data OK", flush=True)
            self.run_test_once(op, num_tokens, test_data)
        del op

def test_dispatch_combine(rank, world_size):
    test_case = EpDispatchCombineTestCase(rank, world_size, jnp.bfloat16)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()

if __name__ == "__main__":
    world_config = init_distributed()
    test_dispatch_combine(*world_config)
