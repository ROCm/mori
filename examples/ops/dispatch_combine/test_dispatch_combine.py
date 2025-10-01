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
import os
import time

import torch
import torch.distributed as dist


class EpDispatchCombineTestCase:
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
        )

    def setup(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)

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

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(time.time()) + self.rank)

    def cleanup(self):
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def gen_test_data(self):
        # gen num_tokens
        # if self.config.rank < 4:
        if False:
            num_tokens = 0
        else:
            num_tokens = int(
                torch.randint(
                    1,
                    self.config.max_num_inp_token_per_rank + 1,
                    [1],
                    generator=self.rng,
                    device=self.device,
                ).item()
            )

        # gen indices
        indices = torch.empty(
            num_tokens,
            self.config.num_experts_per_token,
            dtype=torch.int64,
            # device=self.device,
        )
        for i in range(num_tokens):
            perm = torch.randperm(
                self.config.num_experts_per_rank * self.config.world_size,
                generator=self.rng,
                device=self.device,
            )
            indices[i] = perm[: self.config.num_experts_per_token]
        indices_list = self._allgather_with_token_num_padding(
            indices.cpu(), self.config.max_num_inp_token_per_rank
        )
        indices_list = [
            tensor.to(self.device).to(torch.int32) for tensor in indices_list
        ]
        indices = indices.to(self.device).to(torch.int32)

        # gen weights
        weights = torch.rand(
            num_tokens,
            self.config.num_experts_per_token,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )
        weights_list = self._allgather_with_token_num_padding(
            weights, self.config.max_num_inp_token_per_rank
        )

        # gen scales
        scales_fp32 = torch.rand(
            num_tokens,
            self.config.scale_dim,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )
        scales_list = self._allgather_with_token_num_padding(
            scales_fp32, self.config.max_num_inp_token_per_rank
        )
        scales_list = [tensor.to(torch.float8_e4m3fnuz) for tensor in scales_list]

        # gen input & output
        # some functions such as randn and cat are not implemented for fp8
        input_fp32 = torch.randn(
            num_tokens,
            self.config.hidden_dim,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )
        input_list = self._allgather_with_token_num_padding(
            input_fp32, self.config.max_num_inp_token_per_rank
        )
        input_list = [tensor.to(self.config.data_type) for tensor in input_list]

        return (
            num_tokens,
            indices,
            weights,
            # None,
            # scales_fp32,
            scales_fp32.to(torch.float8_e4m3fnuz),
            input_fp32.to(self.config.data_type),
            indices_list,
            weights_list,
            # None,
            scales_list,
            input_list,
        )

    def run_test_once(self, op, test_data):
        (
            num_tokens,
            indices,
            weights,
            scales,
            input,
            indices_list,
            weights_list,
            scales_list,
            input_list,
        ) = test_data
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            input,
            weights,
            scales,
            indices,
            block_num=80,
            warp_per_block=16,
        )
        torch.cuda.synchronize()

        src_token_pos = op.get_dispatch_src_token_pos()
        print(
            f"rank {self.rank} got {num_tokens} tokens received {src_token_pos.size(0)} tokens"
        )

        for i, pos in enumerate(src_token_pos):
            src_rank = int(pos) // self.config.max_num_inp_token_per_rank
            src_id = int(pos) % self.config.max_num_inp_token_per_rank
            assert torch.equal(input_list[src_rank][src_id], dispatch_output[i])
            assert torch.equal(weights_list[src_rank][src_id], dispatch_weights[i])
            if scales_list is not None and self.config.scale_dim != 0:
                assert torch.equal(scales_list[src_rank][src_id], dispatch_scales[i])
            assert torch.equal(indices_list[src_rank][src_id], dispatch_indices[i])
        assert len(torch.unique(src_token_pos)) == len(src_token_pos)
        assert len(src_token_pos) == dispatch_recv_num_token[0]

        if self.config.rank == 0:
            print("Dispatch Pass")

        total_recv_num_token = dispatch_recv_num_token[0].item()
        combine_input = op.get_registered_combine_input_buffer(self.config.data_type)
        combine_input[:total_recv_num_token, :].copy_(
            dispatch_output[:total_recv_num_token, :]
        )

        combine_input = dispatch_output
        combine_input_weight = dispatch_weights

        combine_output, combine_output_weight = op.combine(
            combine_input.to(torch.bfloat16),
            combine_input_weight,
            indices,
            block_num=80,
            warp_per_block=8,
        )
        torch.cuda.synchronize()

        for i in range(num_tokens):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in indices[i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            # got, expected = combine_output[i], (
            #     input[i].to(torch.float32) * unique_pes
            # ).to(self.config.data_type)
            got, expected = combine_output[i], input[i].to(torch.bfloat16) * unique_pes

            assert torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)

            got_weight, expected_weight = (
                combine_output_weight[i],
                weights[i] * unique_pes,
            )
            weight_match = torch.allclose(
                got_weight, expected_weight, atol=1e-5, rtol=1e-5
            )
            if not weight_match and self.config.rank == 0:
                print(f"Weight mismatch for token {i}:")
                print(f"  indices[{i}]: {indices[i].cpu().tolist()}")
                print(f"  pes: {pes}")
                print(f"  unique_pes: {unique_pes}")
                print(f"  got_weight: {got_weight}")
                print(
                    f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                )
                print(f"  original weights[{i}]: {weights[i]}")
                print(f"  diff: {torch.abs(got_weight - expected_weight)}")
                print(f"  max_diff: {torch.abs(got_weight - expected_weight).max()}")

            assert weight_match, f"Weight assertion failed for token {i}"

        if self.config.rank == 0:
            print("Combine Pass")

    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(5):
            test_data = self.gen_test_data()
            self.run_test_once(op, test_data)
        del op


def test_dispatch_combine(rank, world_size):
    # test_case = EpDispatchCombineTestCase(rank, world_size, torch.float8_e4m3fnuz)
    test_case = EpDispatchCombineTestCase(rank, world_size, torch.bfloat16)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()


if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(
        test_dispatch_combine, args=(world_size,), nprocs=world_size, join=True
    )
