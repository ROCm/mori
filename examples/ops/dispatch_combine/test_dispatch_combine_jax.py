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
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
import numpy as np
import random
from functools import partial

import mori
import argparse

NUM_PEERS=4
RUN_MULTITHREADED=True

def init_distributed():
    if RUN_MULTITHREADED:
        return (0, NUM_PEERS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--coordination_service", type=str, default="")
    args, _ = parser.parse_known_args()

    jax.distributed.initialize(
        coordinator_address=args.coordination_service or "localhost:12345",
        num_processes=args.world_size,
        process_id=args.rank,
    )
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
            num_experts_per_token=56,
            warp_num_per_block=8,
            block_num=80,
            use_external_inp_buf=True,
            kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
            gpu_per_node=NUM_PEERS,
            rdma_block_num=16,
            num_qp_per_pe=1,
            quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
        )

    def setup(self):
        # for JAX multi-threaded mode, SHMEM is initialized in the thread
        if not RUN_MULTITHREADED:
            mori.jax.shmem_init_attr(self.rank, self.world_size)
        def set_seed():
          return jax.random.PRNGKey(777 + self.rank)
        self.rng = jax.jit(set_seed)()

    def cleanup(self):
        mori.jax.shmem_finalize()
    
    def gen_test_data(self, num_tokens):
        max_tokens = self.config.max_num_inp_token_per_rank

        def all_gather(inp):
            padded = jnp.pad(inp,
                [(0, max_tokens - inp.shape[0])] + [(0, 0)] * (inp.ndim - 1),
            )
            gathered = jax.lax.all_gather(padded, axis_name="i")
            gathered = gathered.reshape(-1, *gathered.shape[2:])
            return gathered
        
        # indices: shape [num_tokens, num_experts_per_token]
        total_experts = self.config.num_experts_per_rank * self.config.world_size
        # print(f"----- rank {self.rank} #tokens {num_tokens} #experts {total_experts}")
       
        keys = jax.random.split(self.rng, num_tokens)
        perms = jax.vmap(
            lambda k: jax.random.permutation(k, total_experts)
        )(keys)
        indices = perms[:,:self.config.num_experts_per_token]
        indices_list = all_gather(indices)
                
        weights = jax.random.uniform(self.rng, (num_tokens, self.config.num_experts_per_token), dtype=jnp.float32)
        weights_list = all_gather(weights)

        if self.config.scale_dim != 0:
            scales_fp32 = jax.random.uniform(self.rng, (num_tokens, self.config.scale_dim), dtype=jnp.float32)
        else:
            scales_fp32 = jnp.zeros((1, 1), dtype=jnp.float32)
        # cast to target type after gather
        scales_list = all_gather(scales_fp32).astype(self.jax_scale_dtype)

        # input: [num_tokens, hidden_dim] - this 
        input_fp32 = jax.random.normal(self.rng, (num_tokens, self.config.hidden_dim), dtype=jnp.float32)
        input_list = all_gather(input_fp32).astype(self.dtype)
        
        print(f"num_tokens {num_tokens}  hidden: {self.config.hidden_dim} indices: {indices.shape}/{indices.dtype}")
        print(f"weights: {weights.shape}/{weights.dtype}")
        print(f"scales_fp32: {scales_fp32.shape}/{scales_fp32.dtype}", flush=True)

        return (indices, weights,
            scales_fp32.astype(self.jax_scale_dtype),
            input_fp32.astype(self.dtype),
            indices_list,
            weights_list,
            scales_list,
            input_list)

    def run_test_once(self, op, num_tokens, test_data, mesh):
        (
            indices,
            weights,
            scales,
            inputs,
            indices_list,
            weights_list,
            scales_list,
            input_list,
        ) = test_data

        def ffi_calls(inputs, weights, scales, indices):
            (
                dispatch_output,
                dispatch_indices,
                num,
                dispatch_weights,
                dispatch_scales,
            ) = op.dispatch(inputs, weights, scales, indices)
            src_token_pos = op.get_dispatch_src_token_pos(num)

            (combine_output, combine_weights) = op.combine(
                dispatch_output.astype(self.dtype), dispatch_weights, dispatch_indices
            )

            return (
                (
                    dispatch_output,
                    dispatch_indices,
                    num,
                    dispatch_weights,
                    dispatch_scales,
                ),
                src_token_pos,
                combine_output,
                combine_weights,
            )

        jitted_ffi_calls = jax.jit(shard_map(
            ffi_calls,
            mesh=mesh,
            in_specs=(P(), P(), P(), P()),
            out_specs=(
                (P(), P(), P(), P(), P()),
                P(),
                P(),
                P(),
            ),
            check_rep=False,
        ))

        (
            (
                dispatch_output,
                dispatch_indices,
                dispatch_recv_num_token,
                dispatch_weights,
                dispatch_scales,
            ),
            src_token_pos,
            combine_output,
            combine_weights,
        ) = jitted_ffi_calls(inputs, weights, scales, indices)

        print(f"src_token_pos: {src_token_pos} {src_token_pos.shape}", flush=True)
        # num_tokens = [1..max_num_inp_tokens_per_rank]
        # indices [num_tokes x num_experts_per_token]
        # this basically maps each local token to some set of experts
        # expert IDs go from 0 to total_experts-1
        # where total_experts = num_experts_per_rank * world_size

        # inputs [num_tokens x hidden_dim] - these are actual tokens of size hidden_dim
        # weights: [num_tokens, num_experts_per_token], float32

        # combine_output: [max_num_inp_tokens_per_rank, hidden_dim]
        # combine_weights: [max_num_inp_tokens_per_rank, num_experts_per_token]

        num_recv = dispatch_recv_num_token
        print(
            f"rank {self.rank} got {num_tokens} / received {num_recv} tokens",
            flush=True,
        )

        src_num_recv_token_pos = np.array(src_token_pos)[:num_recv]
        # print(f"src_token_pos sz {src_token_pos.size} shape {src_token_pos.shape}")
        # print(f"baze size: {input_list[src_token_pos].shape[0]} === {input_list.shape}")

        print(
            f"src_num_recv_token_pos size: {src_num_recv_token_pos.size} vs dispatch_recv_num_token {dispatch_recv_num_token}"
        )
        assert src_num_recv_token_pos.size == int(dispatch_recv_num_token)

        @jax.jit
        def validate_dispatch(num, src_pos, base_list, base_out, *args):
            Y = base_list[src_pos]
            N = Y.shape[0]
            mask = jnp.arange(N) < num
            mask2D = mask[:, None]  # expand to [N,?]
            x = jnp.all((Y == base_out) | (~mask2D))
            for x_list, x_out in args:
                if x_out is not None:
                    x = x & jnp.all((x_list[src_pos] == x_out) | (~mask2D))
            # we make an assumption that maxv does not collide with src_pos
            # checking that src_pos[:N] contains only unique values
            maxv = jnp.iinfo(src_pos.dtype).max
            S_masked = jnp.where(mask, src_pos, maxv)
            S_sorted = jnp.sort(S_masked)
            eq_adjacent = S_sorted[1:] == S_sorted[:-1]
            valid = (S_sorted[1:] != maxv) & (S_sorted[:-1] != maxv)
            x = x & ~jnp.any(eq_adjacent & valid)
            return x

        # Validate dispatch outputs against gathered inputs
        res = validate_dispatch(
            dispatch_recv_num_token,
            src_token_pos,
            input_list,
            dispatch_output,
            (weights_list, dispatch_weights),
            (scales_list, dispatch_scales if self.config.scale_dim != 0 else None),
            (indices_list, dispatch_indices),
        )
        assert res, f"{self.rank} validate_dispatch failed!"

        # print(f"input_list {input_list.shape} ss {ss.shape} at {src_token_pos.shape} vs {vv.shape}")
        print(f"{self.rank} dispatch tokens ok", flush=True)

        @jax.jit
        def validate_combine(
            combine_output,
            combine_weights,
            inputs,
            weights,
            indices,
            num_experts_per_rank,
            num_tokens,
        ):
            # NOTE: If `num_tokens` is truly a runtime (traced) value, you cannot slice to
            # `[:num_tokens, :]` under `jit` because it would create a dynamic output shape.
            # The standard pattern is: keep arrays padded to a fixed `max_tokens` and use a mask.
            max_tokens = combine_output.shape[0]
            mask_1d = jnp.arange(max_tokens) < num_tokens

            def masked_allclose(a, b, mask, *, atol, rtol):
                # `mask` is 1D over tokens; broadcast across remaining dims.
                broad_mask = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
                diff = jnp.abs(a - b)
                tol = atol + rtol * jnp.abs(b)
                ok = diff <= tol
                return jnp.all(ok | (~broad_mask))

            pes = indices // num_experts_per_rank
            pes_sorted = jnp.sort(pes, axis=-1)
            unique_pes = 1 + jnp.sum(pes_sorted[:, 1:] != pes_sorted[:, :-1], axis=-1)

            Xinputs = inputs.astype(self.dtype) * unique_pes[:, None]

            # Pad `inputs/weights/indices` up to `max_tokens` so the validation is shape-stable.
            # (This is only fully "runtime-num_tokens-jittable" if upstream already provides
            # fixed-shape tensors; otherwise you'll still recompile when shapes change.)
            inputs_buf = jnp.zeros((max_tokens, Xinputs.shape[1]), dtype=Xinputs.dtype)
            inputs_buf = jax.lax.dynamic_update_slice(inputs_buf, Xinputs, (0, 0))
            expected_output = inputs_buf
            ok_output = masked_allclose(
                combine_output.astype(jnp.float32),
                expected_output.astype(jnp.float32),
                mask_1d,
                atol=1e-2,
                rtol=1e-2,
            )

            ok_weight = True
            if weights is not None:
                Xweights = weights * unique_pes[:, None]
                weights_buf = jnp.zeros(
                    (max_tokens, Xweights.shape[1]), dtype=Xweights.dtype
                )
                weights_buf = jax.lax.dynamic_update_slice(
                    weights_buf, Xweights, (0, 0)
                )
                expected_weight = weights_buf
                ok_weight = masked_allclose(
                    combine_weights, expected_weight, mask_1d, atol=1e-5, rtol=1e-5
                )
            return ok_output & ok_weight

        res = validate_combine(
            combine_output,
            combine_weights,
            inputs,
            weights,
            indices,
            self.config.num_experts_per_rank,
            num_tokens,
        )
        assert res, f"{self.rank} validate_combine failed!"

        print(f"{self.rank} combine tokens ok", flush=True)
        # print(f"combine_output: {combine_output} / expected {expected_output}")

    def test_dispatch_combine(self):
        op = mori.jax.EpDispatchCombineOp(self.config)
        random.seed(333 + self.rank)
        max_tokens = self.config.max_num_inp_token_per_rank
        num_tokens = int(random.randint(1, max_tokens + 1))
        
        devices = np.array(jax.devices())
        mesh = jax.sharding.Mesh(devices, axis_names=("i",))
        
        #@jax.jit
        #@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P(), check_rep=False)
        jitted_gen = jax.jit(shard_map(partial(self.gen_test_data, num_tokens),
                    mesh=mesh, in_specs=(), out_specs=(P(), # indices
                                                       P(), # weights
                                                       P(), # scales
                                                       P(), # input
                                                       P(), # indices_list
                                                       P(), # weights_list
                                                       P(), # scales_list
                                                       P(),), # input_list
                    check_rep=False), static_argnums=())
        for _ in range(1):
            test_data = jitted_gen() 
            print(f"{self.rank} en_test_data OK")
            self.run_test_once(op, num_tokens, test_data, mesh)
        del op


def test_dispatch_combine(rank, world_size):
    test_case = EpDispatchCombineTestCase(rank, world_size, jnp.bfloat16)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()


if __name__ == "__main__":
    world_config = init_distributed()
    test_dispatch_combine(*world_config)
