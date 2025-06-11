import mori
import os
import time

import torch
import torch.distributed as dist


class EpDispatchCombineTestCase:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        self.config = mori.EpDispatchCombineConfig(
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=7168,
            max_num_inp_token_per_rank=128,
            num_expert_per_rank=32,
            num_expert_per_token=8,
            warp_num_per_block=4,
            block_num=256,
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
        mori.shmem_torch_process_group_init("default")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(time.time()) + self.rank)

    def cleanup(self):
        mori.shmem_finalize()
        dist.destroy_process_group()

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
            dtype=torch.uint32,
            device=self.device,
        )
        for i in range(num_tokens):
            perm = torch.randperm(
                self.config.num_experts_per_rank * self.config.world_size,
                generator=self.rng,
                device=self.device,
            )
            indicies[i] = perm[: self.config.num_experts_per_token]

        # gen weights
        weights = torch.rand(
            num_tokens,
            self.config.num_experts_per_token,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )

        # gen input & output
        input = torch.randn(
            num_tokens,
            self.config.hidden_dim,
            dtype=torch.bfloat16,
            generator=self.rng,
            device=self.device,
        )

        input_list = [
            torch.zeros(
                (self.config.max_num_inp_token_per_rank, self.config.hidden_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    self.config.max_num_inp_token_per_rank - num_tokens,
                    self.config.hidden_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            ],
            0,
        )
        dist.all_gather(input_list, padded_input)

        return (num_tokens, indicies, weights, input, input_list)

    def run_test_once(self, handle, test_data):
        num_tokens, indicies, weights, input, input_list = test_data
        dispatch_output = mori.launch_intra_node_dispatch_Bf16(
            handle,
            input,
            weights,
            indicies,
        )
        torch.cuda.synchronize()

        src_token_pos = mori.get_dispatch_src_token_pos_Bf16(handle)
        print(
            f"rank {self.rank} got {num_tokens} tokens received {src_token_pos.size(0)} tokens"
        )

        for i, pos in enumerate(src_token_pos):
            src_rank = int(pos) // self.config.max_num_inp_token_per_rank
            src_id = int(pos) % self.config.max_num_inp_token_per_rank
            is_equal = torch.equal(input_list[src_rank][src_id], dispatch_output[i])
            if not is_equal:
                print(
                    f"dispatch rank {self.rank} i {i} pos {pos} src_rank {src_rank} src_id {src_id}"
                )
            assert is_equal

        assert len(torch.unique(src_token_pos)) == len(src_token_pos)

        combine_output = mori.launch_intra_node_combine_Bf16(
            handle,
            dispatch_output,
            weights,
            indicies,
        )
        torch.cuda.synchronize()

        for i in range(num_tokens):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in indicies[i].cpu().tolist()
            ]
            unique_pes = len(set(pes))
            is_equal = torch.allclose(
                combine_output[i], input[i] * unique_pes, atol=1e-2, rtol=1e-2
            )
            if not is_equal:
                print(
                    f"combine rank {self.rank} i {i} src_rank {src_rank} src_id {src_id} unique_pes {unique_pes} {indicies[i]} {pes} {combine_output[i]} {input[i] * unique_pes}"
                )
            assert is_equal

        mori.launch_reset_Bf16(handle)
        torch.cuda.synchronize()

    def test_dispatch_combine(self):
        handle = mori.EpDispatchCombineHandleBf16(self.config)
        for _ in range(1000):
            test_data = self.gen_test_data()
            self.run_test_once(handle, test_data)
        del handle


def test_dispatch_combine(rank, world_size):
    test_case = EpDispatchCombineTestCase(rank, world_size)
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()


if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(
        test_dispatch_combine, args=(world_size,), nprocs=world_size, join=True
    )
