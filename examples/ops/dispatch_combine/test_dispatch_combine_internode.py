import mori
import os
import time

import torch
import torch.distributed as dist


class EpDispatchCombineTestCase:
    def __init__(self, rank, gpu_per_node, world_size, dtype=torch.bfloat16):
        self.rank = rank
        self.gpu_per_node = gpu_per_node
        self.world_size = world_size
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=7168,
            max_num_inp_token_per_rank=4,
            num_experts_per_rank=32,
            num_experts_per_token=8,
        )

    def setup(self):
        local_rank = self.rank % self.gpu_per_node
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )

        print("init process group done")
        world_group = torch.distributed.group.WORLD
        assert world_group is not None

        print("process group ok")
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")

        self.rng = torch.Generator(device=self.device)
        # self.rng.manual_seed(int(time.time()) + self.rank)
        self.rng.manual_seed(123)

    def cleanup(self):
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

    def gen_test_data(self):
        # gen num_tokens
        num_token = torch.randint(
            1,
            self.config.max_num_inp_token_per_rank + 1,
            [self.world_size],
            generator=self.rng,
            device=self.device,
        )
        print(num_token, num_token.sum())

        # gen indicies
        all_rank_indicies = []
        for r in range(self.world_size):
            indicies = torch.empty(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.int64,
                # device=self.device,
            )
            for i in range(num_token[r]):
                perm = torch.randperm(
                    self.config.num_experts_per_rank * self.config.world_size,
                    generator=self.rng,
                    device=self.device,
                )
                indicies[i] = perm[: self.config.num_experts_per_token]
            all_rank_indicies.append(indicies.to(torch.uint32).to(self.device))
        print(all_rank_indicies)

        # gen weights
        all_rank_weights = [
            torch.rand(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.world_size)
        ]
        # print(all_rank_weights)

        # gen input & output
        # some functions such as randn and cat are not implemented for fp8
        all_rank_input = []
        num_token_off = 0
        for r in range(self.world_size):
            input = torch.cat(
                [
                    torch.ones(
                        [1, self.config.hidden_dim],
                        dtype=self.config.data_type,
                        device=self.device,
                    )
                    * (t + num_token_off)
                    for t in range(num_token[r])
                ]
            )
            num_token_off += num_token[r]
            all_rank_input.append(input.to(self.device))
        print(all_rank_input)

        return (num_token, all_rank_indicies, all_rank_input, all_rank_weights)

    def run_test_once(self, op, test_data):
        (all_rank_num_token, all_rank_indicies, all_rank_input, all_rank_weights) = (
            test_data
        )
        print(
            self.rank,
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_indicies[self.rank],
        )
        (
            dispatch_output,
            dispatch_weights,
            dispatch_indicies,
            dispatch_recv_num_token,
        ) = op.dispatch_internode(
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_indicies[self.rank],
        )
        torch.cuda.synchronize()

        print(self.rank, dispatch_output.shape)

        if self.config.rank == 0:
            print("Dispatch Pass")

    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(1):
            test_data = self.gen_test_data()
            self.run_test_once(op, test_data)
        del op


def test_dispatch_combine(local_rank, num_node, gpu_per_node):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    test_case = EpDispatchCombineTestCase(
        global_rank, gpu_per_node, world_size, torch.bfloat16
    )
    test_case.setup()
    test_case.test_dispatch_combine()
    test_case.cleanup()


if __name__ == "__main__":
    gpu_per_node = os.environ.get("GPU_PER_NODE", None)
    gpu_per_node = int(gpu_per_node) if gpu_per_node is not None else 8
    num_node = int(os.environ["WORLD_SIZE"])

    world_size = num_node * gpu_per_node
    torch.multiprocessing.spawn(
        test_dispatch_combine,
        args=(
            num_node,
            gpu_per_node,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
