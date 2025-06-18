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
            max_num_inp_token_per_rank=128,
            num_experts_per_rank=16,
            num_experts_per_token=8,
            warp_num_per_block=4,
            block_num=64,
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
        # print(all_rank_indicies)

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
        for r in range(self.world_size):
            all_rank_input.append(
                torch.randn(
                    num_token[r],
                    self.config.hidden_dim,
                    dtype=torch.float32,
                    generator=self.rng,
                    device=self.device,
                ).to(self.config.data_type)
            )

        return (num_token, all_rank_indicies, all_rank_input, all_rank_weights)

    def run_test_once(self, op, test_data):
        (all_rank_num_token, all_rank_indicies, all_rank_input, all_rank_weights) = (
            test_data
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

        dispatch_sender_token_id_map = op.get_dispatch_sender_token_id_map()
        dispatch_receiver_token_id_map = op.get_dispatch_receiver_token_id_map()

        max_num_token_to_send_per_rank = (
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token
        )
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        reverse_sender_token_id_map = {}
        for r in range(self.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : all_rank_num_token.tolist()[r] * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )

        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            assert recv_mapped_id in reverse_sender_token_id_map
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            assert torch.equal(dispatch_output[i], all_rank_input[src_pe][src_tok_id])
            # print(dispatch_output[i], all_rank_input[src_pe][src_tok_id])
        assert len(dispatch_receiver_token_id_map.tolist()) == len(
            reverse_sender_token_id_map
        )

        if self.config.rank == 0:
            print("Dispatch Pass")

        op.reset()
        torch.cuda.synchronize()

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
