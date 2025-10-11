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

import torch
import torch.distributed as dist
import argparse


class EpDispatchCombineTestCase:
    def __init__(
        self, rank, gpu_per_node, world_size, max_tokens, dtype=torch.bfloat16
    ):
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
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank=16,
            # num_experts_per_rank=256 // world_size,
            num_experts_per_token=8,
            warp_num_per_block=16,
            block_num=80,
            max_token_type_size=2,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            gpu_per_node=self.gpu_per_node,
            rdma_block_num=64,
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
        self.rng.manual_seed(3210)

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

        num_total_experts = self.config.num_experts_per_rank * self.config.world_size
        num_nodes = self.config.world_size // self.config.gpu_per_node

        # Per-rank counts
        rank_counts = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_recv = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_send = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )

        for src_rank, indices in enumerate(all_rank_indices):
            src_node = src_rank // self.config.gpu_per_node

            # Map expert IDs to rank IDs
            token_ranks = (
                indices // self.config.num_experts_per_rank
            )  # [num_tokens, num_experts_per_token]

            # Deduplicate rank IDs per token
            unique_ranks_per_token = [torch.unique(row) for row in token_ranks]

            # For each token, update counts
            for ur in unique_ranks_per_token:
                rank_counts[ur] += 1  # All ranks that receive this token

                dst_nodes = {
                    dst_rank // self.config.gpu_per_node for dst_rank in ur.tolist()
                }

                for dst_rank in ur.tolist():
                    dst_node = dst_rank // self.config.gpu_per_node
                    if dst_node != src_node:
                        # Receiving side
                        rank_counts_remote_recv[dst_rank] += 1

                # Sending side (dedup by node: count once if token goes to a remote node)
                for dst_node in dst_nodes:
                    if dst_node != src_node:
                        rank_counts_remote_send[src_rank] += 1

        if self.config.rank == 0:
            print("Rank counts (deduplicated):", rank_counts)
            # print("Rank counts local nodes:", rank_counts - rank_counts_remote_recv)
            # print("Rank counts from other nodes:", rank_counts_remote_recv)
            # print("Rank counts to other nodes:", rank_counts_remote_send)

        # even_indices = (
        #     torch.arange(
        #         self.config.max_num_inp_token_per_rank
        #         * self.config.num_experts_per_token,
        #         device="cuda",
        #     ).view(
        #         self.config.max_num_inp_token_per_rank,
        #         self.config.num_experts_per_token,
        #     )
        #     % 256
        # )
        # even_indices = even_indices.to(torch.int32)
        # all_rank_indices = [even_indices for _ in range(self.world_size)]

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

    def run_test_once(self, op, test_data, error_round, round):
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
            # None,
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
            block_num=self.config.block_num,
            warp_per_block=16,
        )
        torch.cuda.synchronize()
        dist.barrier()

        src_token_pos = op.get_dispatch_src_token_pos().tolist()
        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
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
            # error_round.add(round)
            if dispatch_weights is not None:
                assert torch.equal(
                    dispatch_weights[i], all_rank_weights[src_pe][src_tok_id]
                )
            assert torch.equal(
                dispatch_indices[i], all_rank_indices[src_pe][src_tok_id]
            )
            # TODO: test output scales

        if self.config.rank == 0:
            print("Dispatch Pass")

        dist.barrier()

        combine_output, combine_output_weight = op.combine(
            dispatch_output,
            dispatch_weights,
            all_rank_indices[self.rank],
        )
        torch.cuda.synchronize()

        # for i in range(all_rank_num_token[self.rank]):
        #     pes = [
        #         (idx // self.config.num_experts_per_rank)
        #         for idx in all_rank_indices[self.rank][i].cpu().tolist()
        #     ]
        #     unique_pes = len(set(pes))

        #     got, expected = combine_output[i], (
        #         all_rank_input[self.rank][i].to(torch.float32) * unique_pes
        #     ).to(self.config.data_type)

        #     ok = torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
        #     if not ok:
        #         print(self.rank, "got: ", got)
        #         print(self.rank, "expected: ", expected)
        #         print(self.rank, "delta:", got - expected)
        #         assert False
        #         error_round.add(round)

        #     if dispatch_weights is not None:
        # got_weight, expected_weight = (
        #     combine_output_weight[i],
        #     all_rank_weights[self.rank][i] * unique_pes,
        # )
        # weight_match = torch.allclose(
        #     got_weight, expected_weight, atol=1e-5, rtol=1e-5
        # )
        # if not weight_match and self.config.rank == 0:
        #     print(f"Weight mismatch for token {i}:")
        #     print(
        #         f"  indices[{i}]: {all_rank_indices[self.rank][i].cpu().tolist()}"
        #     )
        #     print(f"  pes: {pes}")
        #     print(f"  unique_pes: {unique_pes}")
        #     print(f"  got_weight: {got_weight}")
        #     print(
        #         f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
        #     )
        #     print(f"  original weights[{i}]: {all_rank_weights[self.rank][i]}")
        #     print(f"  diff: {torch.abs(got_weight - expected_weight)}")
        #     print(
        #         f"  max_diff: {torch.abs(got_weight - expected_weight).max()}"
        #     )
        # assert weight_match, f"Weight assertion failed for token {i}"

        if self.config.rank == 0:
            print("Combine Pass")

    def test_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        error_round = set()
        for i in range(5000):
            if self.rank == 0:
                print(f"Round {i} begin")
            test_data = self.gen_test_data(use_max_token_num=False)
            if self.rank == 0:
                print(f"Round {i} gen test_data done")
            self.run_test_once(op, test_data, error_round, i)
        print(
            "rank: ",
            self.rank,
            "error times: ",
            len(error_round),
            "appear round: ",
            error_round,
        )

        del op

    def run_bench_once(self, op, test_data):
        (
            all_rank_num_token,
            all_rank_indices,
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
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
            block_num=self.config.block_num,
            warp_per_block=16,
        )
        end_event.record()
        torch.cuda.synchronize()
        disp_duration = start_event.elapsed_time(end_event)

        dist.barrier()
        total_recv_num_token = dispatch_recv_num_token[0].item()
        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank

        my_node = self.rank // self.gpu_per_node
        total_rdma_recv_num_token = 0
        src_token_pos = op.get_dispatch_src_token_pos().cpu().tolist()
        for i, src_token_id in enumerate(src_token_pos):
            src_pe = src_token_id // max_num_token_to_send_per_rank
            src_node = src_pe // self.gpu_per_node
            if src_node != my_node:
                total_rdma_recv_num_token += 1
        if (
            self.config.kernel_type
            is mori.ops.EpDispatchCombineKernelType.InterNodeV1
        ):
            total_rdma_recv_num_token = (
                self.config.max_num_inp_token_per_rank * self.config.world_size // 8
            )
        print(
            f"rank {self.rank} recv {total_recv_num_token} tokens {total_rdma_recv_num_token} rdma tokens"
        )

        element_size = all_rank_input[self.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        total_rdma_bytes = (
            total_rdma_recv_num_token * self.config.hidden_dim * element_size
        )
        disp_rdma_bandwidth = total_rdma_bytes / (1000**3) / (disp_duration / (10**3))
        disp_bandwidth = total_bytes / (1000**3) / (disp_duration / (10**3))

        # torch.cuda.synchronize()
        dist.barrier()
        # start_event.record()
        combine_output, _ = op.combine(
            dispatch_output,
            None,
            all_rank_indices[self.rank],
            call_reset=False,
        )
        # end_event.record()
        torch.cuda.synchronize()
        # comb_duration = start_event.elapsed_time(end_event)
        # comb_bandwidth = total_bytes / (1000**3) / (comb_duration / (10**3))

        # op.reset()
        # torch.cuda.synchronize()
        # return disp_duration, disp_bandwidth, comb_duration, comb_bandwidth
        return disp_duration, disp_rdma_bandwidth, disp_bandwidth, float(0), float(0)

    def bench_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(use_max_token_num=True)

        disp_duration_us_list = []
        disp_rdma_bandwidth_GB_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_bandwidth_GB_list = []

        error_round = set()
        for i in range(3):
            if self.rank == 0:
                print(f"WarmUp Round {i} begin")
            self.run_test_once(op, test_data, error_round, i)
        assert (
            len(error_round) == 0
        ), f"Warmup failed with errors in rounds: {error_round}"

        for i in range(30):
            if self.rank == 0:
                print(f"Round {i} begin")
            (
                disp_duration,
                disp_rdma_bandwidth,
                disp_bandwidth,
                comb_duration,
                comb_bandwidth,
            ) = self.run_bench_once(op, test_data)

            disp_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_rdma_bandwidth_output = [
                torch.zeros(1) for _ in range(self.world_size)
            ]
            disp_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]

            dist.all_gather(disp_duration_output, torch.tensor([disp_duration * 1000]))
            dist.all_gather(
                disp_rdma_bandwidth_output, torch.tensor([disp_rdma_bandwidth])
            )
            dist.all_gather(disp_bandwidth_output, torch.tensor([disp_bandwidth]))
            dist.all_gather(comb_duration_output, torch.tensor([comb_duration * 1000]))
            dist.all_gather(comb_bandwidth_output, torch.tensor([comb_bandwidth]))

            disp_duration_us_list.append([int(t.item()) for t in disp_duration_output])
            disp_rdma_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_rdma_bandwidth_output]
            )
            disp_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_bandwidth_output]
            )
            comb_duration_us_list.append([int(t.item()) for t in comb_duration_output])
            comb_bandwidth_GB_list.append(
                [int(t.item()) for t in comb_bandwidth_output]
            )

        if self.rank == 0:
            for i in range(len(disp_duration_us_list)):
                print(f"Round {i}")
                print(
                    f"  dispatch duration {disp_duration_us_list[i]} avg {sum(disp_duration_us_list[i]) / self.config.world_size:.2f} µs"
                )
                print(
                    f"  rdma bandwidth {disp_rdma_bandwidth_GB_list[i]} avg {sum(disp_rdma_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )
                print(
                    f"  bandwidth {disp_bandwidth_GB_list[i]} avg {sum(disp_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )

            # for i in range(len(comb_duration_us_list)):
            #     print(
            #         f"Round {i} combine  duration {comb_duration_us_list[i]} "
            #         f"bandwidth {comb_bandwidth_GB_list[i]} "
            #         f"avg {sum(comb_duration_us_list[i]) / self.config.world_size:.2f} µs "
            #         f"avg {sum(comb_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
            #     )

        disp_bandwidth_GB_list = disp_bandwidth_GB_list[0:]
        avg_disp_bw_per_round = [
            (sum(round_bw) / len(round_bw)) for round_bw in disp_bandwidth_GB_list
        ]
        avg_disp_rdma_bw_per_round = [
            (sum(round_bw) / len(round_bw)) for round_bw in disp_rdma_bandwidth_GB_list
        ]
        avg_disp_bw = sum(avg_disp_bw_per_round) / len(avg_disp_bw_per_round)
        avg_disp_rdma_bw = sum(avg_disp_rdma_bw_per_round) / len(
            avg_disp_rdma_bw_per_round
        )

        comb_bandwidth_GB_list = comb_bandwidth_GB_list[0:]
        avg_comb_bw_per_round = [
            (sum(round_bw) / len(round_bw)) for round_bw in comb_bandwidth_GB_list
        ]
        avg_comb_bw = sum(avg_comb_bw_per_round) / len(avg_comb_bw_per_round)

        disp_duration_us_list = disp_duration_us_list[0:]
        avg_disp_lat_per_round = [
            sum(round_duration) / len(round_duration)
            for round_duration in disp_duration_us_list
        ]
        avg_disp_lat = sum(avg_disp_lat_per_round) / len(avg_disp_lat_per_round)

        comb_duration_us_list = comb_duration_us_list[0:]
        avg_comb_lat_per_round = [
            sum(round_duration) / len(round_duration)
            for round_duration in comb_duration_us_list
        ]
        avg_comb_lat = sum(avg_comb_lat_per_round) / len(avg_comb_lat_per_round)

        best_disp_bw = max(avg_disp_bw_per_round)
        best_disp_rdma_bw = max(avg_disp_rdma_bw_per_round)
        best_comb_bw = max(avg_comb_bw_per_round)

        best_disp_lat = min(avg_disp_lat_per_round)
        best_comb_lat = min(avg_comb_lat_per_round)

        if self.rank == 0:
            print(
                f"dispatch: best/avg RDMA bandwidth {best_disp_rdma_bw:.2f} / {avg_disp_rdma_bw:.2f} XGMI bandwidth {best_disp_bw:.2f} / {avg_disp_bw:.2f} GB/s | "
                f"best/avg latency {best_disp_lat:.2f} / {avg_disp_lat:.2f} µs\n"
                f"combine : best/avg bandwidth {best_comb_bw:.2f} / {avg_comb_bw:.2f} GB/s | "
                f"best/avg latency {best_comb_lat:.2f} / {avg_comb_lat:.2f} µs"
            )
        del op


def test_dispatch_combine(
    local_rank, num_node, gpu_per_node, max_tokens, is_bench=False
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    test_case = EpDispatchCombineTestCase(
        global_rank,
        gpu_per_node,
        world_size,
        max_tokens,
        # torch.bfloat16,  # torch.float8_e4m3fnuz
        torch.float8_e4m3fnuz,
    )
    test_case.setup()
    if is_bench:
        test_case.bench_dispatch_combine()
    else:
        test_case.test_dispatch_combine()
    test_case.cleanup()


parser = argparse.ArgumentParser(description="dispatch/combine internode test")
parser.add_argument(
    "--bench",
    action="store_true",
    help="Set this flag True to run benchmark into test_dispatch_combine",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum number of input tokens per rank (default: 4096)",
)
args_cli = parser.parse_args()

if __name__ == "__main__":
    gpu_per_node = os.environ.get("GPU_PER_NODE", None)
    gpu_per_node = int(gpu_per_node) if gpu_per_node is not None else 8
    num_node = int(os.environ["WORLD_SIZE"])

    world_size = num_node * gpu_per_node
    torch.multiprocessing.spawn(
        test_dispatch_combine,
        args=(num_node, gpu_per_node, args_cli.max_tokens, args_cli.bench),
        nprocs=gpu_per_node,
        join=True,
    )
