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
            scale_dim=32,
            scale_type_size=4,
            max_num_inp_token_per_rank=2048,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            warp_num_per_block=8,
            block_num=80,
            max_token_type_size=2,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNode,
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

    def gen_test_data(self, use_max_token_num=False):
        # gen num_tokens
        if use_max_token_num:
            num_token = torch.tensor(
                [self.config.max_num_inp_token_per_rank for i in range(self.world_size)]
            ).to(self.device)
        else:
            num_token = torch.randint(
                1,
                self.config.max_num_inp_token_per_rank + 1,
                [self.world_size],
                generator=self.rng,
                device=self.device,
            )

        # gen indices
        all_rank_indices = []
        for r in range(self.world_size):
            indices = torch.empty(
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
                indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

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

        # gen weights
        all_rank_scales = [
            torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.world_size)
        ]

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

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def run_test_once(self, op, test_data):
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        dist.barrier()
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
        )
        torch.cuda.synchronize()
        dist.barrier()

        src_token_pos = op.get_dispatch_src_token_pos().tolist()
        max_num_token_to_send_per_rank = (
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token
        )
        print(f"rank {self.rank} recv {len(src_token_pos)} tokens")
        for i, src_token_id in enumerate(src_token_pos):
            src_pe = src_token_id // max_num_token_to_send_per_rank
            src_tok_id = src_token_id % max_num_token_to_send_per_rank
            is_pass = torch.equal(
                dispatch_output[i], all_rank_input[src_pe][src_tok_id]
            )
            if not is_pass:
                print(
                    f"rank {self.rank} token {i} assert {is_pass} expected { all_rank_input[src_pe][src_tok_id]} got {dispatch_output[i]}"
                )
                assert False
            # assert torch.equal(
            #     dispatch_weights[i], all_rank_weights[src_pe][src_tok_id]
            # )
            # assert torch.equal(
            #     dispatch_indicies[i], all_rank_indicies[src_pe][src_tok_id]
            # )
            # TODO: test output scales

        if self.config.rank == 0:
            print("Dispatch Pass")

        dist.barrier()

        combine_output = op.combine(
            dispatch_output,
            all_rank_weights[self.rank],
            all_rank_indices[self.rank],
        )
        torch.cuda.synchronize()

        for i in range(all_rank_num_token[self.rank]):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            got, expected = combine_output[i], (
                all_rank_input[self.rank][i].to(torch.float32) * unique_pes
            ).to(self.config.data_type)

            ok = torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
            if not ok:
                print(self.rank, "got: ", got)
                print(self.rank, "expected: ", expected)
                assert False

        if self.config.rank == 0:
            print("Combine Pass")

    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(500):
            if self.rank == 0:
                print(f"Round {i} begin")
            test_data = self.gen_test_data()
            self.run_test_once(op, test_data)

        del op

    def run_bench_once(self, op, test_data):
        (
            all_rank_num_token,
            all_rank_indicies,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indicies,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indicies[self.rank],
        )
        end_event.record()
        torch.cuda.synchronize()
        disp_duration = start_event.elapsed_time(end_event)

        dist.barrier()
        total_recv_num_token = dispatch_recv_num_token[0].item()
        max_num_token_to_send_per_rank = (
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token
        )

        my_node = self.rank // self.gpu_per_node
        total_rdma_recv_num_token = 0
        src_token_pos = op.get_dispatch_src_token_pos().cpu().tolist()
        for i, src_token_id in enumerate(src_token_pos):
            src_pe = src_token_id // max_num_token_to_send_per_rank
            src_node = src_pe // self.gpu_per_node
            if src_node != my_node:
                total_rdma_recv_num_token += 1
        print(
            f"rank {self.rank} recv {total_recv_num_token} tokens {total_rdma_recv_num_token} rdma tokens"
        )

        element_size = all_rank_input[self.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        disp_bandwidth = total_bytes / (1024**3) / (disp_duration / (10**3))

        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        combine_output = op.combine(
            dispatch_output,
            all_rank_weights[self.rank],
            all_rank_indicies[self.rank],
            call_reset=False,
        )
        end_event.record()
        torch.cuda.synchronize()
        comb_duration = start_event.elapsed_time(end_event)
        comb_bandwidth = total_bytes / (1024**3) / (comb_duration / (10**3))

        op.reset()
        torch.cuda.synchronize()
        return disp_duration, disp_bandwidth, comb_duration, comb_bandwidth

    def bench_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(use_max_token_num=True)

        disp_duration_us_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_bandwidth_GB_list = []

        for i in range(50):
            if self.rank == 0:
                print(f"Round {i} begin")
            disp_duration, disp_bandwidth, comb_duration, comb_bandwidth = (
                self.run_bench_once(op, test_data)
            )

            disp_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]

            dist.all_gather(disp_duration_output, torch.tensor([disp_duration * 1000]))
            dist.all_gather(disp_bandwidth_output, torch.tensor([disp_bandwidth]))
            dist.all_gather(comb_duration_output, torch.tensor([comb_duration * 1000]))
            dist.all_gather(comb_bandwidth_output, torch.tensor([comb_bandwidth]))

            disp_duration_us_list.append([int(t.item()) for t in disp_duration_output])
            disp_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_bandwidth_output]
            )
            comb_duration_us_list.append([int(t.item()) for t in comb_duration_output])
            comb_bandwidth_GB_list.append(
                [int(t.item()) for t in comb_bandwidth_output]
            )

        if self.rank == 0:
            for i, duration_us in enumerate(disp_duration_us_list):
                print(
                    f"Round {i} dispatch duration {duration_us} bandwidth {disp_bandwidth_GB_list[i]} avg {sum(disp_bandwidth_GB_list[i]) / self.config.world_size}"
                )
            for i, duration_us in enumerate(comb_duration_us_list):
                print(
                    f"Round {i} combine duration {duration_us} bandwidth {comb_bandwidth_GB_list[i]} avg {sum(comb_bandwidth_GB_list[i]) / self.config.world_size}"
                )

        disp_bandwidth_GB_list = disp_bandwidth_GB_list[10:]
        avg_disp_bw_per_round = [
            (sum(round_bw) / len(round_bw)) for round_bw in disp_bandwidth_GB_list
        ]
        avg_disp_bw = sum(avg_disp_bw_per_round) / len(avg_disp_bw_per_round)

        comb_bandwidth_GB_list = comb_bandwidth_GB_list[10:]
        avg_comb_bw_per_round = [
            (sum(round_bw) / len(round_bw)) for round_bw in comb_bandwidth_GB_list
        ]
        avg_comb_bw = sum(avg_comb_bw_per_round) / len(avg_comb_bw_per_round)

        if self.rank == 0:
            print(
                f"dispatch avg bandwidth {avg_disp_bw} combine avg bandwidth {avg_comb_bw}"
            )
        del op


def test_dispatch_combine(local_rank, num_node, gpu_per_node, is_bench=False):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    test_case = EpDispatchCombineTestCase(
        global_rank,
        gpu_per_node,
        world_size,
        torch.bfloat16,  # torch.float8_e4m3fnuz
    )
    test_case.setup()
    if is_bench:
        test_case.bench_dispatch_combine()
    else:
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
            True,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
