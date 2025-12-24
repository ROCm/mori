import jax
import jax.numpy as jnp
import numpy as np

import random
import mori
import argparse, os, time
from functools import partial

import torch
import torch.distributed as dist

def init_distributed():
  parser = argparse.ArgumentParser()
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--coordination_service", type=str, default="")
  args, _ = parser.parse_known_args()

#   if args.jax_distributed:
#     dist.initialize(
#       coordinator_address=args.coordination_service or "localhost:1234",
#       num_processes=args.process_count,
#       process_id=args.process_index,
#     )
  print(f"[rank {args.rank}] devices = {jax.local_devices()}")
  return (args.rank, args.world_size)


class EpDispatchCombineTestCase:
    def __init__(self, rank, world_size, dtype=torch.bfloat16, jax_dtype=jnp.bfloat16):
        self.jax_dtype = jax_dtype
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
        
    def setup(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.cuda.set_device(0) #self.rank)
        self.device = torch.device("cuda", 0) #self.rank)
        
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )
        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        # self.rng = torch.Generator(device=self.device)
        # self.rng.manual_seed(int(time.time()) + self.rank)
                # simple rng keyed by time + rank to vary per rank
        # seed = int(time.time()) + self.rank
        
        def set_seed():
          return jax.random.PRNGKey(777)
        self.rng =jax.jit(set_seed)()

    def cleanup(self):
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()
        
    def allgather_padded(self, inp: jnp.ndarray, max_token_num: int): # -> List[jnp.ndarray]:
        """
        Emulate all_gather with padding by returning a list of length world_size
        where each entry is padded to [max_token_num, ...] and is a copy of local input.
        This keeps the test logic unchanged while avoiding needing to rework mori internals.
        """
        shape = list(inp.shape)
        pad_len = max_token_num - shape[0]
        if pad_len < 0:
            raise ValueError("inp has more tokens than max_token_num")

        if pad_len > 0:
            pad_shape = [pad_len] + shape[1:]
            padded = jnp.concatenate([inp, jnp.zeros(pad_shape, dtype=inp.dtype)], axis=0)
        else:
            padded = inp

        # Return a copy for each "rank"
        return [padded.copy() for _ in range(self.world_size)]

    #@partial(jax.jit, static_argnums=(0,))
    def gen_test_data(self):
        max_tokens = self.config.max_num_inp_token_per_rank
        num_tokens = int(random.randint(1, max_tokens + 1))
        # indices: shape [num_tokens, num_experts_per_token]
        total_experts = self.config.num_experts_per_rank * self.config.world_size

        # Use numpy for per-row permutation sampling
        indices_np = np.empty((num_tokens, self.config.num_experts_per_token), dtype=np.int64)
        for i in range(num_tokens):
            perm = np.random.permutation(total_experts)
            indices_np[i, :] = perm[: self.config.num_experts_per_token]
        indices = jnp.array(indices_np, dtype=jnp.int32)

        indices_list = self.allgather_padded(indices, self.config.max_num_inp_token_per_rank)
        indices_list = [jnp.array(x, dtype=jnp.int32) for x in indices_list]

        # weights: [num_tokens, num_experts_per_token], float32
        weights = jax.random.uniform(self.rng, (num_tokens, self.config.num_experts_per_token), dtype=jnp.float32)
        weights_list = self.allgather_padded(weights, self.config.max_num_inp_token_per_rank)

        # scales (scale_dim == 0 in config; still create shapes)
        if self.config.scale_dim != 0:
            scales_fp32 = jax.random.uniform(self.rng, (num_tokens, self.config.scale_dim), dtype=jnp.float32)
            scales_list = self.allgather_padded(scales_fp32, self.config.max_num_inp_token_per_rank)
        else:
            scales_fp32 = jnp.zeros((1, 1), dtype=jnp.float32)
            scales_list = [jnp.zeros((self.config.max_num_inp_token_per_rank, 0), dtype=jnp.float32) for _ in range(self.world_size)]

        # input: [num_tokens, hidden_dim]
        input_fp32 = jax.random.normal(self.rng, (num_tokens, self.config.hidden_dim), 
                            dtype=jnp.float32)
        input_list = self.allgather_padded(input_fp32.astype(self.jax_dtype), self.config.max_num_inp_token_per_rank)
        input_list = [jnp.array(x, dtype=self.jax_dtype) for x in input_list]
        
        print(f"num_tokens {num_tokens}  hidden: {self.config.hidden_dim} indices: {indices.shape} tp {indices.dtype}")
        print(f"weights: {weights.shape} {weights.dtype}")
        print(f"scales_fp32: {scales_fp32.shape} {scales_fp32.dtype}")

        return (num_tokens,
            indices,
            weights,
            scales_fp32,
            input_fp32.astype(self.jax_dtype),
            indices_list,
            weights_list,
            scales_list,
            input_list)
        
    def run_test_once(self, op, test_data):
        (num_tokens,
            indices,
            weights,
            scales,
            input_arr,
            indices_list,
            weights_list,
            scales_list,
            input_list) = test_data

        (dispatch_output, 
         dispatch_weights, 
         dispatch_scales, 
         dispatch_indices, 
         dispatch_recv_num_token) = op.dispatch_jax(
            input_arr, weights, scales, indices, block_num=80, warp_per_block=16,
            has_scales=True, has_weights=True,
        )

        src_token_pos = op.get_dispatch_src_token_pos()
        print(f"------------ recv num: {dispatch_recv_num_token}")
        print(
            f"rank {self.rank} got {num_tokens} tokens received {src_token_pos.size(0)} tokens"
        )
        # Validate dispatch outputs against gathered inputs
        for i, pos in enumerate(src_token_pos):
            pos_i = int(pos)
            src_rank = pos_i // self.config.max_num_inp_token_per_rank
            src_id = pos_i % self.config.max_num_inp_token_per_rank

            left = np.array(input_list[src_rank][src_id])
            right = np.array(dispatch_output[i])
            assert np.array_equal(left, right), f"dispatch_output mismatch at token {i} (rank {self.rank})"

            left_w = np.array(weights_list[src_rank][src_id])
            right_w = np.array(dispatch_weights[i])
            assert np.array_equal(left_w, right_w), f"dispatch_weights mismatch at token {i} (rank {self.rank})"

            if scales_list is not None and self.config.scale_dim != 0:
                left_s = np.array(scales_list[src_rank][src_id])
                right_s = np.array(dispatch_scales[i])
                assert np.array_equal(left_s, right_s), f"dispatch_scales mismatch at token {i} (rank {self.rank})"

            left_idx = np.array(indices_list[src_rank][src_id])
            right_idx = np.array(dispatch_indices[i])
            if not np.array_equal(left_idx, right_idx):
                print("ops indices mismatch")
                print(f"lhs = {left_idx} -- {left} -- {left_w}")
                print(f"rhs = {right_idx} -- {right} -- {right_w}")
            assert np.array_equal(left_idx, right_idx), f"dispatch_indices mismatch at token {i} (rank {self.rank})"

        assert len(np.unique(np.array(src_token_pos))) == len(src_token_pos)
        assert len(src_token_pos) == int(np.array(dispatch_recv_num_token)[0])

        if self.config.rank == 0:
            print("[rank 0] Dispatch Pass")

        
    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        for _ in range(5):
            test_data = self.gen_test_data()
            print(f"{self.rank} en_test_data OK")
            self.run_test_once(op, test_data)
        del op

        # op = mori.ops.EpDispatchCombineOp(self.config)
        
        # @jax.jit
        # def gen_inputs():
        #   w,h = 256, 768
        #   input = jnp.linspace(-1, 1, w*h).reshape([w, h])
        #   weights = jnp.linspace(-1, 1, w*h).reshape([w, h])
        #   scales = jnp.linspace(-1, 1, w*h).reshape([w, h])
        #   indices = jnp.linspace(0, 100, w*h, dtype=jnp.int32).reshape([w, h])
        #   return input, weights, scales, indices
        
        # S = gen_inputs()
        # res = op.dispatch_jax(*S)

# if __name__ == "__main__":
#   print("Start")
#   test_case = PlaygroundTestCase(rank=110, world_size=1)


def test_dispatch_combine(rank, world_size):
    # test_case = EpDispatchCombineTestCase(rank, world_size, torch.float8_e4m3fnuz)
    test_case = EpDispatchCombineTestCase(rank, world_size, 
                                            torch.bfloat16, jnp.bfloat16)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()

if __name__ == "__main__":
    world_config = init_distributed()
    test_dispatch_combine(*world_config)
    
    # world_size = 8
    # torch.multiprocessing.spawn(
    #     test_dispatch_combine, args=(world_size,), nprocs=world_size, join=True
    # )