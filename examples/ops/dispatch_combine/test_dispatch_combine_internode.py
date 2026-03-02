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
import time
from tqdm import tqdm

os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

kernel_type_map = {
    "v0": mori.ops.EpDispatchCombineKernelType.InterNode,
    "v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    "v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
    "async_ll": mori.ops.EpDispatchCombineKernelType.AsyncLL,
}


class EpDispatchCombineTestCase:
    def __init__(
        self,
        rank,
        gpu_per_node,
        world_size,
        max_tokens,
        kernel_type,
        num_qp,
        quant_type="none",
        dtype=torch.bfloat16,
        hidden_dim=7168,
    ):
        self.rank = rank
        self.gpu_per_node = gpu_per_node
        self.world_size = world_size
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=(
                hidden_dim // 2 if dtype is torch.float4_e2m1fn_x2 else hidden_dim
            ),
            scale_dim=32,
            scale_type_size=4,
            max_num_inp_token_per_rank=(max_tokens + 63) // 64 * 64,
            num_experts_per_rank=16,
            num_experts_per_token=8,
            warp_num_per_block=8,
            block_num=96,
            max_token_type_size=2,
            kernel_type=kernel_type_map[kernel_type],
            gpu_per_node=self.gpu_per_node,
            rdma_block_num=64,
            num_qp_per_pe=num_qp,
            quant_type=quant_type,
        )

    def setup(self):
        local_rank = self.rank % self.gpu_per_node
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        dist.init_process_group(
            backend="cpu:gloo",
            rank=self.rank,
            world_size=self.world_size,
        )

        print("init process group done")
        world_group = torch.distributed.group.WORLD
        assert world_group is not None

        print("process group ok")
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(999)

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

    def gen_test_data(self, max_num_token, use_max_token_num=False):
        # gen num_tokens
        if use_max_token_num:
            num_token = torch.tensor(
                [max_num_token for i in range(self.world_size)]
            ).to(self.device)
        else:
            num_token = torch.randint(
                1,
                max_num_token + 1,
                [self.world_size],
                generator=self.rng,
                device=self.device,
            )

        # gen indices - vectorized version for speed
        num_total_experts = self.config.num_experts_per_rank * self.config.world_size
        all_rank_indices = []
        for r in range(self.world_size):
            num_tok = num_token[r].item()
            # Generate random floats and use argsort to get permutations
            # This is much faster than calling randperm in a loop
            random_vals = torch.rand(
                num_tok,
                num_total_experts,
                generator=self.rng,
                device=self.device,
            )
            # argsort gives us a random permutation, take first K columns
            indices = torch.argsort(random_vals, dim=1)[
                :, : self.config.num_experts_per_token
            ]
            all_rank_indices.append(indices.to(torch.int32))

        # num_total_experts = self.config.num_experts_per_rank * self.config.world_size
        # num_nodes = self.config.world_size // self.config.gpu_per_node

        # even_indices = (
        #     torch.arange(
        #         max_num_token
        #         * self.config.num_experts_per_token,
        #         device="cuda",
        #     ).view(
        #         max_num_token,
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

        # gen scales
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
            data_fp32 = torch.randn(
                num_token[r],
                self.config.hidden_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            if self.config.data_type is torch.float4_e2m1fn_x2:
                data = torch.randint(
                    0,
                    256,
                    (num_token[r], self.config.hidden_dim),
                    dtype=torch.uint8,
                    generator=self.rng,
                    device=self.device,
                )
                data = data.view(torch.float4_e2m1fn_x2)
            else:
                data = data_fp32.to(self.config.data_type)
            all_rank_input.append(data)

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def count_token_num(self, all_rank_indices):
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
        return rank_counts, rank_counts_remote_recv, rank_counts_remote_send

    def run_dispatch(self, op, token, weights, scales, indices):
        if op.config.kernel_type is mori.ops.EpDispatchCombineKernelType.AsyncLL:
            ret = op.dispatch_send(token, weights, scales, indices)
            op.dispatch_recv()
        else:
            ret = op.dispatch(token, weights, scales, indices)
        return ret

    def run_combine(self, op, token, weights, indices):
        if op.config.kernel_type is mori.ops.EpDispatchCombineKernelType.AsyncLL:
            ret = op.combine_send(token, weights, indices)
            op.combine_recv()
        else:
            ret = op.combine(token, weights, indices)
        return ret

    def run_test_once(self, op, test_data, error_round, round):
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = self.run_dispatch(
            op,
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
        )
        torch.cuda.synchronize()

        rank_counts, _, _ = self.count_token_num(all_rank_indices)

        src_token_pos = op.get_dispatch_src_token_pos().tolist()
        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        recv_token_num = len(src_token_pos)

        # Check recv token num
        print(f"rank {self.rank} recv {recv_token_num} tokens")
        token_num_pass = rank_counts[self.rank] == recv_token_num
        if not token_num_pass:
            print(
                f"rank {self.rank} expected token num {rank_counts[self.rank]} got {recv_token_num}"
            )
            assert False

        # Check token equality
        for i, src_token_id in enumerate(src_token_pos):
            src_pe = src_token_id // max_num_token_to_send_per_rank
            src_tok_id = src_token_id % max_num_token_to_send_per_rank
            if self.config.data_type is torch.float4_e2m1fn_x2:
                is_pass = torch.equal(
                    dispatch_output[i].view(torch.uint8),
                    all_rank_input[src_pe][src_tok_id].view(torch.uint8),
                )
            else:
                is_pass = torch.equal(
                    dispatch_output[i], all_rank_input[src_pe][src_tok_id]
                )
            if not is_pass:
                print(
                    f"rank {self.rank} token {i} assert {is_pass} expected {all_rank_input[src_pe][src_tok_id].view(torch.uint8)} got {dispatch_output[i].view(torch.uint8)}"
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
            assert torch.equal(dispatch_scales[i], all_rank_scales[src_pe][src_tok_id])

        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Dispatch Pass")

        # NOTE: weight combine not implemented yet
        if op.config.kernel_type is mori.ops.EpDispatchCombineKernelType.AsyncLL:
            dispatch_weights = None
        combine_output, combine_output_weight = self.run_combine(
            op, dispatch_output, dispatch_weights, all_rank_indices[self.rank]
        )

        torch.cuda.synchronize()
        for i in range(all_rank_num_token[self.rank]):
            if self.config.data_type is torch.float4_e2m1fn_x2:
                continue
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))
            unique_innode_pes = len(
                [
                    pe
                    for pe in set(pes)
                    if (pe // self.gpu_per_node == self.rank // self.gpu_per_node)
                ]
            )
            final_unique_pes = unique_pes
            if final_unique_pes == 0:
                continue

            got, expected = combine_output[i], (
                all_rank_input[self.rank][i].to(torch.float32) * final_unique_pes
            ).to(self.config.data_type)

            atol, rtol = 1e-2, 1e-2
            if getattr(self.config, "quant_type", "none") == "fp8_direct_cast":
                atol, rtol = 1e-1, 1e-1
            ok = torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol)
            if not ok:
                print(
                    self.rank,
                    f"token {i} pes {pes} unique pes {unique_pes} unique innode pes {unique_innode_pes}",
                )
                print(
                    f"{self.rank} got: ",
                    got,
                    f"{self.rank} expected: ",
                    expected,
                    all_rank_input[self.rank][i],
                )
                # delta = got.float() - expected.float()
                # print(self.rank, "delta:", delta)
                # error_round.add(round)
                assert False
                # pass

            if dispatch_weights is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.rank][i] * final_unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match and self.config.rank == 0:
                    print(f"Weight mismatch for token {i}:")
                    print(
                        f"  indices[{i}]: {all_rank_indices[self.rank][i].cpu().tolist()}"
                    )
                    print(f"  pes: {pes}")
                    print(f"  unique_pes: {unique_pes}")
                    print(f"  got_weight: {got_weight}")
                    print(
                        f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                    )
                    print(f"  original weights[{i}]: {all_rank_weights[self.rank][i]}")
                    print(f"  diff: {torch.abs(got_weight - expected_weight)}")
                    print(
                        f"  max_diff: {torch.abs(got_weight - expected_weight).max()}"
                    )
                assert weight_match, f"Weight assertion failed for token {i}"
        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Combine Pass")

    def test_dispatch_combine(self):
        error_round = set()
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(500):
            if self.rank == 0:
                print(f"Round {i} begin")
            test_data = self.gen_test_data(
                max_num_token=self.config.max_num_inp_token_per_rank,
                use_max_token_num=False,
            )
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

    def stress_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        num_test_data = 128
        sync_interval = 128

        if self.rank == 0:
            print("Stress Test")
        test_data_list = [
            self.gen_test_data(
                max_num_token=self.config.max_num_inp_token_per_rank,
                use_max_token_num=False,
            )
            for i in range(num_test_data)
        ]
        for i in tqdm(range(5000)):
            (
                all_rank_num_token,
                all_rank_indices,
                all_rank_input,
                all_rank_weights,
                all_rank_scales,
            ) = test_data_list[i % num_test_data]
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
            )
            combine_output, combine_output_weight = self.run_combine(
                op, dispatch_output, None, all_rank_indices[self.rank]
            )
            if i % sync_interval == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()

        if self.rank == 0:
            print("Stress Test with CUDA Graph")
        test_data = self.gen_test_data(
            max_num_token=self.config.max_num_inp_token_per_rank,
            use_max_token_num=False,
        )
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
            )
            combine_output, combine_output_weight = self.run_combine(
                op, dispatch_output, None, all_rank_indices[self.rank]
            )
        torch.cuda.synchronize()

        for i in tqdm(range(5000)):
            g.replay()
            torch.cuda.synchronize()
            time.sleep(0.0001)

        del op

    def run_bench_once(self, max_num_token, op, test_data, repeat=10):
        num_events = 2 * repeat + 1
        events = [torch.cuda.Event(enable_timing=True) for i in range(num_events)]

        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        warmup_rounds = 3
        for i in range(warmup_rounds):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
            )
            if i == warmup_rounds - 1:
                # Read totalRecvTokenNum after dispatch but before combine resets it
                torch.cuda.synchronize()
                total_recv_num_token = dispatch_recv_num_token[0].item()
            combine_output, combine_output_weight = self.run_combine(
                op, dispatch_output, None, all_rank_indices[self.rank]
            )
            torch.cuda.synchronize()
        total_rdma_recv_num_token = (
            self.config.max_num_inp_token_per_rank * self.config.world_size // 8
        )
        print(
            f"rank {self.rank} recv {total_recv_num_token} tokens {total_rdma_recv_num_token} rdma tokens"
        )

        if hasattr(mori.cpp, "get_debug_time_buf"):
            my_times = op.get_debug_time_buf()
            my_times.zero_()
            if hasattr(mori.cpp, "get_debug_time_offset"):
                my_offsets = op.get_debug_time_offset()
                my_offsets.zero_()

        torch.cuda.synchronize()
        dist.barrier()
        events[0].record()
        for i in range(repeat):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
            )
            events[2 * i + 1].record()
            combine_output, combine_output_weight = self.run_combine(
                op, dispatch_output, None, all_rank_indices[self.rank]
            )
            events[2 * i + 2].record()
        torch.cuda.synchronize()

        element_size = all_rank_input[self.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        ll_mode_scale = (
            max_num_token
            * self.config.num_experts_per_token
            / (total_recv_num_token + 1)  # avoid division by zero
        )
        total_rdma_bytes = (
            total_rdma_recv_num_token * self.config.hidden_dim * element_size
        )

        disp_duration_list = []
        comb_duration_list = []
        for i in range(1, num_events, 2):
            disp_duration_list.append(events[i - 1].elapsed_time(events[i]))
            comb_duration_list.append(events[i].elapsed_time(events[i + 1]))

        disp_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in disp_duration_list
        ]
        disp_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in disp_duration_list
        ]

        comb_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in comb_duration_list
        ]
        comb_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in comb_duration_list
        ]

        if hasattr(mori.cpp, "get_debug_time_buf"):
            output_filename = (
                f"trace_rank_{self.rank}_{time.strftime('%m%d_%H%M%S')}.json"
            )
            mori.kernel_profiler.export_to_perfetto(my_times, output_filename)
            if self.rank == 0:
                print(f"Profiling data exported to {output_filename}")

        return (
            disp_duration_list,
            disp_rdma_bandwidth_list,
            disp_bandwidth_list,
            comb_duration_list,
            comb_rdma_bandwidth_list,
            comb_bandwidth_list,
            ll_mode_scale,
        )

    def bench_dispatch_combine(self, max_num_token):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(
            max_num_token=max_num_token, use_max_token_num=True
        )

        repeat = 10
        disp_duration_us_list = []
        disp_rdma_bandwidth_GB_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_rdma_bandwidth_GB_list = []
        comb_bandwidth_GB_list = []

        error_round = set()
        for i in range(1):
            if self.rank == 0:
                print(f"WarmUp Round {i} begin")
            self.run_test_once(op, test_data, error_round, i)
        assert (
            len(error_round) == 0
        ), f"Warmup failed with errors in rounds: {error_round}"

        (
            disp_duration,
            disp_rdma_bandwidth,
            disp_bandwidth,
            comb_duration,
            comb_rdma_bandwidth,
            comb_bandwidth,
            ll_mode_scale,
        ) = self.run_bench_once(max_num_token, op, test_data, repeat)

        for i in range(repeat):
            disp_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_rdma_bandwidth_output = [
                torch.zeros(1) for _ in range(self.world_size)
            ]
            disp_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_rdma_bandwidth_output = [
                torch.zeros(1) for _ in range(self.world_size)
            ]
            comb_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]

            dist.all_gather(
                disp_duration_output, torch.tensor([disp_duration[i] * 1000])
            )
            dist.all_gather(
                disp_rdma_bandwidth_output, torch.tensor([disp_rdma_bandwidth[i]])
            )
            dist.all_gather(disp_bandwidth_output, torch.tensor([disp_bandwidth[i]]))
            dist.all_gather(
                comb_duration_output, torch.tensor([comb_duration[i] * 1000])
            )
            dist.all_gather(
                comb_rdma_bandwidth_output, torch.tensor([comb_rdma_bandwidth[i]])
            )
            dist.all_gather(comb_bandwidth_output, torch.tensor([comb_bandwidth[i]]))

            disp_duration_us_list.append([int(t.item()) for t in disp_duration_output])
            disp_rdma_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_rdma_bandwidth_output]
            )
            disp_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_bandwidth_output]
            )
            comb_duration_us_list.append([int(t.item()) for t in comb_duration_output])
            comb_rdma_bandwidth_GB_list.append(
                [int(t.item()) for t in comb_rdma_bandwidth_output]
            )
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

            for i in range(len(comb_duration_us_list)):
                print(f"Round {i}")
                print(
                    f"  combine duration {comb_duration_us_list[i]} avg {sum(comb_duration_us_list[i]) / self.config.world_size:.2f} µs"
                )
                print(
                    f"  rdma bandwidth {comb_rdma_bandwidth_GB_list[i]} avg {sum(comb_rdma_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )
                print(
                    f"  bandwidth {comb_bandwidth_GB_list[i]} avg {sum(comb_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )

        if repeat == 1:
            return

        def collect_metrics(per_round_data):
            minv = min([min(data) for data in per_round_data])
            maxv = max([max(data) for data in per_round_data])
            avgl = [(sum(data) / len(data)) for data in per_round_data]
            avgv = sum(avgl) / len(avgl)
            return int(minv), int(maxv), int(avgv)

        disp_bw = collect_metrics(disp_bandwidth_GB_list[1:])
        disp_rdma_bw = collect_metrics(disp_rdma_bandwidth_GB_list[1:])
        disp_ll_bw = [int(e * ll_mode_scale) for e in disp_bw]
        disp_lat = collect_metrics(disp_duration_us_list[1:])

        comb_bw = collect_metrics(comb_bandwidth_GB_list[1:])
        comb_rdma_bw = collect_metrics(comb_rdma_bandwidth_GB_list[1:])
        comb_ll_bw = [int(e * ll_mode_scale) for e in comb_bw]
        comb_lat = collect_metrics(comb_duration_us_list[1:])

        from prettytable import PrettyTable

        disp_table = PrettyTable()
        comb_table = PrettyTable()
        field_names = [
            "Metrics",
            "RDMA Bandwidth (GB/s)",
            "XGMI Bandwidth (GB/s)",
            "LL Bandwidth (GB/s)",
            "Latency (us)",
        ]
        disp_table.title = "Dispatch Performance"
        disp_table.field_names = field_names
        disp_table.add_rows(
            [
                [
                    "Best",
                    disp_rdma_bw[1],
                    disp_bw[1],
                    disp_ll_bw[1],
                    disp_lat[0],
                ],
                [
                    "Worst",
                    disp_rdma_bw[0],
                    disp_bw[0],
                    disp_ll_bw[0],
                    disp_lat[1],
                ],
                [
                    "Average",
                    disp_rdma_bw[2],
                    disp_bw[2],
                    disp_ll_bw[2],
                    disp_lat[2],
                ],
            ]
        )
        comb_table.field_names = field_names
        comb_table.title = "Combine Performance"
        comb_table.add_rows(
            [
                [
                    "Best",
                    comb_rdma_bw[1],
                    comb_bw[1],
                    comb_ll_bw[1],
                    comb_lat[0],
                ],
                [
                    "Worst",
                    comb_rdma_bw[0],
                    comb_bw[0],
                    comb_ll_bw[0],
                    comb_lat[1],
                ],
                [
                    "Average",
                    comb_rdma_bw[2],
                    comb_bw[2],
                    comb_ll_bw[2],
                    comb_lat[2],
                ],
            ]
        )
        if self.rank == 0:
            print(disp_table)
            print(comb_table)

        del op

        return (disp_bw, disp_rdma_bw, disp_ll_bw, disp_lat), (
            comb_bw,
            comb_rdma_bw,
            comb_ll_bw,
            comb_lat,
        )

    def profile_dispatch_combine(self, max_num_token):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(
            max_num_token=max_num_token, use_max_token_num=True
        )

        repeat = 3
        if not hasattr(mori.cpp, "get_debug_time_buf"):
            raise RuntimeError(
                "to use profiling command, re-compile MORI with ENABLE_PROFILER=ON"
            )

        self.run_bench_once(max_num_token, op, test_data, repeat)


def sweep_bench_dispatch_combine(
    local_rank,
    num_node,
    gpu_per_node,
    dtype,
    max_tokens,
    kernel_type,
    num_qp,
    sweep_token_interval,
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank
    sweep_token_interval = int(sweep_token_interval)
    if sweep_token_interval <= 0:
        raise ValueError(f"sweep_token_interval must >= 1, got {sweep_token_interval}")
    test_case = EpDispatchCombineTestCase(
        global_rank, gpu_per_node, world_size, max_tokens, kernel_type, num_qp, dtype
    )
    test_case.setup()

    num_iters = (max_tokens + sweep_token_interval - 1) // sweep_token_interval
    max_token_list = [i * sweep_token_interval for i in range(num_iters)]

    disp_lat_min_list = []
    disp_lat_max_list = []
    disp_lat_avg_list = []
    comb_lat_min_list = []
    comb_lat_max_list = []
    comb_lat_avg_list = []
    for max_token in max_token_list:
        if max_token == 0:
            max_token = 1
        disp_stats, comb_stats = test_case.bench_dispatch_combine(max_token)
        disp_bw, disp_rdma_bw, disp_ll_bw, disp_lat = disp_stats
        comb_bw, comb_rdma_bw, comb_ll_bw, comb_lat = comb_stats

        disp_lat_min_list.append(disp_lat[0])
        comb_lat_min_list.append(comb_lat[0])
        disp_lat_max_list.append(disp_lat[1])
        comb_lat_max_list.append(comb_lat[1])
        disp_lat_avg_list.append(disp_lat[2])
        comb_lat_avg_list.append(comb_lat[2])

    if local_rank == 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(max_token_list, disp_lat_min_list, label="Dispatch Min")
        plt.plot(max_token_list, comb_lat_min_list, label="Combine Min")
        plt.plot(max_token_list, disp_lat_max_list, label="Dispatch Max")
        plt.plot(max_token_list, comb_lat_max_list, label="Combine Max")
        plt.plot(max_token_list, disp_lat_avg_list, label="Dispatch Avg")
        plt.plot(max_token_list, comb_lat_avg_list, label="Combine Avg")

        plt.xticks([i * 16 for i in range(max_tokens // 16)])
        plt.title("Dispatch / Combine Latency (us)")
        plt.xlabel("# of Tokens")
        plt.ylabel("Latency (us)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dispatch_combine_perf.png", dpi=300, bbox_inches="tight")
        test_case.cleanup()


def test_dispatch_combine(
    local_rank,
    num_node,
    gpu_per_node,
    dtype,
    max_tokens,
    kernel_type,
    num_qp,
    quant_type="none",
    cmd="test",
    sweep_token_interval=64,
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    if cmd in ("test", "bench", "stress", "profile"):
        test_case = EpDispatchCombineTestCase(
            global_rank,
            gpu_per_node,
            world_size,
            max_tokens,
            kernel_type,
            num_qp,
            quant_type,
            dtype,
        )
        test_case.setup()
        if cmd == "test":
            test_case.test_dispatch_combine()
        elif cmd == "bench":
            test_case.bench_dispatch_combine(max_tokens)
        elif cmd == "stress":
            test_case.stress_dispatch_combine()
        elif cmd == "profile":
            test_case.profile_dispatch_combine(max_tokens)
        test_case.cleanup()
    elif cmd == "sweep_bench":
        sweep_bench_dispatch_combine(
            local_rank,
            num_node,
            gpu_per_node,
            max_tokens,
            kernel_type,
            num_qp,
            sweep_token_interval,
        )
    else:
        raise ValueError(f"unsupported command: {cmd}")


_DATA_TYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp4": torch.float4_e2m1fn_x2,
}


parser = argparse.ArgumentParser(description="dispatch/combine internode test")
parser.add_argument(
    "--cmd",
    type=str,
    default="test",
    choices=["test", "bench", "stress", "sweep_bench", "profile"],
    help="Available subcommands: test, bench, stress, sweep_bench",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"],
    help="Data type of dispatch / combine",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum number of input tokens per rank (default: 4096)",
)
parser.add_argument(
    "--sweep-token-interval",
    type=int,
    default=2,
    help="Number of token interval when sweep bench",
)
parser.add_argument(
    "--kernel-type",
    type=str,
    default="v1",
    help="Type of kernel to test",
    choices=["v0", "v1", "v1_ll", "async_ll"],
)
parser.add_argument(
    "--num-qp",
    type=int,
    default=1,
    help="Number of qp per processing endpoint",
)
parser.add_argument(
    "--quant-type",
    type=str,
    default="none",
    choices=["none", "fp8_direct_cast"],
    help=(
        "Quantization method used inside Combine. "
        "'fp8_direct_cast' is the current BF16<->FP8 direct cast path."
    ),
)
args_cli = parser.parse_args()

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
            _DATA_TYPE_MAP[args_cli.dtype],
            args_cli.max_tokens,
            args_cli.kernel_type,
            args_cli.num_qp,
            args_cli.quant_type,
            args_cli.cmd,
            args_cli.sweep_token_interval,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
