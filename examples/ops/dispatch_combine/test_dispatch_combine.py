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
            max_num_inp_token_per_rank=128,
            num_experts_per_rank=32,
            num_experts_per_token=8,
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
        num_tokens = int(
            torch.randint(
                1,
                self.config.max_num_inp_token_per_rank + 1,
                [1],
                generator=self.rng,
                device=self.device,
            ).item()
        )

        # gen indicies
        indicies = torch.empty(
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
            indicies[i] = perm[: self.config.num_experts_per_token]
        indicies_list = self._allgather_with_token_num_padding(
            indicies.cpu(), self.config.max_num_inp_token_per_rank
        )
        indicies_list = [
            tensor.to(self.device).to(torch.uint32) for tensor in indicies_list
        ]
        indicies = indicies.to(self.device).to(torch.uint32)

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

        # input_list = [
        #     torch.zeros(
        #         (self.config.max_num_inp_token_per_rank, self.config.hidden_dim),
        #         dtype=self.config.data_type,
        #         device=self.device,
        #     )
        #     for _ in range(self.world_size)
        # ]
        # padded_input = torch.cat(
        #     [
        #         input_fp32,
        #         torch.zeros(
        #             self.config.max_num_inp_token_per_rank - num_tokens,
        #             self.config.hidden_dim,
        #             dtype=torch.float32,
        #             device=self.device,
        #         ),
        #     ],
        #     0,
        # ).to(self.config.data_type)
        # dist.all_gather(input_list, padded_input)

        return (
            num_tokens,
            indicies,
            weights,
            input_fp32.to(self.config.data_type),
            indicies_list,
            weights_list,
            input_list,
        )

    def run_test_once(self, op, test_data):
        (
            num_tokens,
            indicies,
            weights,
            input,
            indicies_list,
            weights_list,
            input_list,
        ) = test_data
        dispatch_output, dispatch_weights, dispatch_indicies = op.dispatch(
            input, weights, indicies
        )
        torch.cuda.synchronize()

        src_token_pos = op.get_dispatch_src_token_pos()
        print(
            f"rank {self.rank} got {num_tokens} tokens received {src_token_pos.size(0)} tokens"
        )

        for i, pos in enumerate(src_token_pos):
            src_rank = int(pos) // self.config.max_num_inp_token_per_rank
            src_id = int(pos) % self.config.max_num_inp_token_per_rank
            is_equal = torch.equal(input_list[src_rank][src_id], dispatch_output[i])
            if not is_equal:
                print(
                    f"dispatch rank {self.rank} i {i} pos {pos} src_rank {src_rank} src_id {src_id} {input_list[src_rank][src_id]}, {dispatch_output[i]}"
                )
            assert is_equal

            is_equal = torch.equal(weights_list[src_rank][src_id], dispatch_weights[i])
            if not is_equal:
                print(
                    f"dispatch rank {self.rank} i {i} pos {pos} src_rank {src_rank} src_id {src_id} {weights_list[src_rank][src_id]}, {dispatch_weights[i]}"
                )
            assert is_equal

            is_equal = torch.equal(
                indicies_list[src_rank][src_id], dispatch_indicies[i]
            )
            if not is_equal:
                print(
                    f"dispatch rank {self.rank} i {i} pos {pos} src_rank {src_rank} src_id {src_id} {indicies_list[src_rank][src_id]}, {dispatch_indicies[i]}"
                )
            assert is_equal

        assert len(torch.unique(src_token_pos)) == len(src_token_pos)

        if self.config.rank == 0:
            print("Dispatch Pass")

        combine_output = op.combine(dispatch_output, weights, indicies)
        torch.cuda.synchronize()

        for i in range(num_tokens):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in indicies[i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            got, expected = combine_output[i], (
                input[i].to(torch.float32) * unique_pes
            ).to(self.config.data_type)

            is_equal = torch.allclose(
                got.float(), expected.float(), atol=1e-2, rtol=1e-2
            )
            if not is_equal:
                print(
                    f"combine rank {self.rank} i {i} src_rank {src_rank} src_id {src_id} unique_pes {unique_pes} {indicies[i]} {pes} {got} {expected}"
                )
            assert is_equal
        if self.config.rank == 0:
            print("Combine Pass")

    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(10000):
            test_data = self.gen_test_data()
            self.run_test_once(op, test_data)
        del op


def test_dispatch_combine(rank, world_size):
    test_case = EpDispatchCombineTestCase(rank, world_size, torch.float8_e4m3fnuz)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()


if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(
        test_dispatch_combine, args=(world_size,), nprocs=world_size, join=True
    )
