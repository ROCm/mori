# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# This file is adapted from:
#   https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py
#   https://github.com/deepseek-ai/DeepEP/blob/main/tests/utils.py
# Original project license: MIT
# Modifications have been made to integrate with the MORI project.
# - Major refactor and adaptation for MORI-EP interface
# - Import paths, configuration, and some logic aligned with MORI code style
# - Utility functions (init_dist, bench, bench_kineto, calc_diff, hash_tensor, etc.) inlined from DeepEP's utils.py
# - See project commit history for detailed changes
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
import os
import sys
import numpy as np
import random
import torch
import torch.distributed as dist
from functools import partial
from typing import Optional, Union

import mori
from tests.python.ops.test_dispatch_combine import EpDispatchCombineTestCase


def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        # backend='nccl',
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    world_group = dist.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
        num_nodes,
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
        if False:
            torch.cuda.synchronize()
            if dist.is_initialized():
                try:
                    device_ids = (
                        [torch.cuda.current_device()]
                        if torch.cuda.is_available()
                        else None
                    )
                    dist.barrier(device_ids=device_ids)
                except Exception as e:
                    try:
                        rank = dist.get_rank()
                    except Exception:
                        rank = -1
                    if rank == 0:
                        print(
                            f"[Rank {rank}] Warning: per-iteration barrier failed - {e}",
                            flush=True,
                        )
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)

class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

class suppress_stdout_stderr:

    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for _ in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
                for _ in range(num_tests):
                    fn()
                    if True:
                        torch.cuda.synchronize()
                        if dist.is_initialized():
                            try:
                                device_ids = (
                                    [torch.cuda.current_device()]
                                    if torch.cuda.is_available()
                                    else None
                                )
                                dist.barrier(device_ids=device_ids)
                            except Exception as e:
                                try:
                                    rank = dist.get_rank()
                                except Exception:
                                    rank = -1
                                if rank == 0:
                                    print(
                                        f"[Rank {rank}] Warning: per-iteration barrier failed - {e}",
                                        flush=True,
                                    )
                torch.cuda.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        prof.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) == 1
        ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, "")) / scale
                        )
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [
                event
                for event in profile_data["traceEvents"]
                if f"::{kernel_name}" in event["name"]
            ]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    num_nodes: int,
    group: dist.ProcessGroup,
    seed: int = 0,
    enable_dedup: bool = True,
    fused_moe_adaption: bool = True,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(torch.int32)
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    # Randomly mask some positions
    # for i in range(10):
    #     topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    dedup_topk_idx = topk_idx.clone()
    for token_id in range(num_tokens):
        dst_ranks = (topk_idx[token_id] // num_local_experts).cpu().numpy()
        seen = set()
        for k in range(num_topk):
            if topk_idx[token_id, k] == -1 or dst_ranks[k] in seen:
                dedup_topk_idx[token_id, k] = -1
            else:
                seen.add(dst_ranks[k])

    validation_topk_idx = dedup_topk_idx if enable_dedup else topk_idx

    multi_node = num_nodes > 1
    if multi_node:
        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        block_num, warp_num_per_block = 64, 8
        dispatch_block_num, dispatch_warp_per_block = block_num, warp_num_per_block
        combine_block_num, combine_warp_per_block = block_num, warp_num_per_block
        rdma_block_num = 32
    else:
        kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        block_num, warp_num_per_block = 64, 16
        dispatch_block_num, dispatch_warp_per_block = block_num, warp_num_per_block
        combine_block_num, combine_warp_per_block = block_num, 4
        rdma_block_num = 0

    mori.shmem.shmem_torch_process_group_init("default")

    zero_copy_config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=num_ranks,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        use_external_inp_buf=False,
        kernel_type=kernel_type,
        gpu_per_node=num_ranks // num_nodes,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=4 if multi_node else 0,
    )
    zero_copy_op = mori.ops.EpDispatchCombineOp(zero_copy_config)

    non_zero_copy_config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=num_ranks,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        use_external_inp_buf=True,
        kernel_type=kernel_type,
        gpu_per_node=num_ranks // num_nodes,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=4 if multi_node else 0,
    )
    non_zero_copy_op = mori.ops.EpDispatchCombineOp(non_zero_copy_config)

    # Check dispatch correctness
    do_check = True
    for zero_copy in (False, True):
        # multiple node does not support zero_copy
        if multi_node and zero_copy:
            continue
        test_op = zero_copy_op if zero_copy else non_zero_copy_op
        test_config = zero_copy_config if zero_copy else non_zero_copy_config
        hash_value, num_times = 0, 0
        if fused_moe_adaption:
            (
                packed_recv_x,
                packed_recv_topk_weights,
                _,
                packed_recv_topk_idx,
                packed_recv_count,
            ) = test_op.dispatch(
                x,
                topk_weights,
                None,
                topk_idx,
                block_num=dispatch_block_num,
                warp_per_block=dispatch_warp_per_block,
            )
        else:
            assert False, "fused_moe_adaption=False is not implemented"
        torch.cuda.synchronize()

        assert fused_moe_adaption == (
            (packed_recv_x.ndim != 3)
        ), f"{fused_moe_adaption} != {(packed_recv_x.ndim != 3)}"
        simulated_gemm_x = packed_recv_x.clone()
        all_topk_idx = torch.empty(
            (num_ranks, num_tokens, num_topk),
            dtype=validation_topk_idx.dtype,
            device="cuda",
        )
        dist.all_gather_into_tensor(all_topk_idx, validation_topk_idx, group=group)
        # for checking topk index correctness
        all_origin_topk_idx = torch.empty(
            (num_ranks, num_tokens, num_topk),
            dtype=topk_idx.dtype,
            device="cuda",
        )
        dist.all_gather_into_tensor(all_origin_topk_idx, topk_idx, group=group)

        all_topk_weights = torch.empty(
            (num_ranks, num_tokens, num_topk),
            dtype=topk_weights.dtype,
            device="cuda",
        )
        dist.all_gather_into_tensor(all_topk_weights, topk_weights, group=group)

        if do_check and fused_moe_adaption:
            global_recv_x = packed_recv_x

            # Check expert indices
            num_total_valid_tokens = packed_recv_count.item()
            assert (
                num_total_valid_tokens
                == (all_topk_idx // num_local_experts == rank).sum().item()
            ), f"{num_total_valid_tokens} != {(all_topk_idx // num_local_experts == rank).sum().item()}"

            # Check received data
            src_token_pos = test_op.get_dispatch_src_token_pos()

            rank_token_counts = {}
            for i, pos in enumerate(src_token_pos[:num_total_valid_tokens]):
                src_rank = int(pos) // test_config.max_num_inp_token_per_rank
                src_id = int(pos) % test_config.max_num_inp_token_per_rank
                recv_token = global_recv_x[i]

                # Check that token values are consistent (amin == amax for first hidden-128 dims)
                recv_token_amin = recv_token[:-128].amin()
                recv_token_amax = recv_token[:-128].amax()
                assert torch.equal(
                    recv_token_amin, recv_token_amax
                ), f"Token {i} values inconsistent: amin={recv_token_amin}, amax={recv_token_amax}"

                # Check that all values in first hidden-128 dims equal src_rank - rank_offset
                expected_value = src_rank - rank_offset
                assert (
                    recv_token[:-128] - expected_value
                ).sum().item() == 0, f"Token {i} from rank {src_rank} has incorrect values, expected all {expected_value}"

                # Check that last 128 dims contain source token ID
                assert (
                    recv_token[-128:] - src_id
                ).sum().item() == 0, f"Token {i} last 128 dims should be {src_id}"

                # Verify topk_idx and topk_weights
                assert torch.equal(
                    packed_recv_topk_idx[i], all_origin_topk_idx[src_rank, src_id]
                )
                assert torch.equal(
                    packed_recv_topk_weights[i], all_topk_weights[src_rank, src_id]
                )

                # Count tokens from each rank
                rank_token_counts[src_rank] = rank_token_counts.get(src_rank, 0) + 1

                hash_value ^= hash_tensor(recv_token.unsqueeze(0))
            # Verify token counts from each rank match expectations
            for src_rank in range(num_ranks):
                expected_count = (
                    (all_topk_idx[src_rank] // num_local_experts == rank).sum().item()
                )
                actual_count = rank_token_counts.get(src_rank, 0)
                assert (
                    actual_count == expected_count
                ), f"Received {actual_count} tokens from rank {src_rank}, expected {expected_count}"

        # Check combine correctness
        if zero_copy:
            combine_input = test_op.get_registered_combine_input_buffer(
                test_config.data_type
            )
            combine_input[:, :].copy_(simulated_gemm_x)
        out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        combined_x, _ = test_op.combine(
            simulated_gemm_x,
            None,
            topk_idx,
            block_num=combine_block_num,
            warp_per_block=combine_warp_per_block,
        )
        torch.cuda.synchronize()
        if do_check:
            diff = calc_diff(
                x * (validation_topk_idx != -1).sum(dim=-1).view(-1, 1),
                combined_x,
            )
            assert torch.isnan(combined_x).sum().item() == 0
            assert diff < 1e-5, f"Error: {diff=}, {zero_copy=}"
            hash_value ^= hash_tensor(combined_x)

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, use_fp8: bool):
        bench_op = zero_copy_op if zero_copy else non_zero_copy_op
        (
            recv_x,
            recv_topk_weights,
            _,
            recv_topk_idx,
            recv_count,
        ) = bench_op.dispatch(
            x,
            topk_weights,
            None,
            topk_idx,
            block_num=dispatch_block_num,
            warp_per_block=dispatch_warp_per_block,
        )

        simulated_gemm_x = recv_x.clone()
        if zero_copy:
            combine_input = bench_op.get_registered_combine_input_buffer(
                zero_copy_config.data_type
            )
            combine_input[:, :].copy_(simulated_gemm_x)
        combined_x, _ = bench_op.combine(
            simulated_gemm_x,
            None,
            topk_idx,
            block_num=combine_block_num,
            warp_per_block=combine_warp_per_block,
        )

    # Calculate bandwidth
    bench_use_fp8 = False
    num_bf16_bytes = hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (validation_topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_bf16_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections
    print(
        f"[rank {rank}] num_dispatch_comm_bytes {num_dispatch_comm_bytes} num_combine_comm_bytes {num_combine_comm_bytes}"
    )

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(
        partial(test_func, zero_copy=False, use_fp8=bench_use_fp8)
    )
    print(
        f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
        flush=True,
    )

    # Separate profiling
    group.barrier()
    if multi_node:
        dispatch_t, combine_t, dispatch_copy_t, combine_all_t = bench_kineto(
            partial(
                test_func,
                zero_copy=False,
                use_fp8=bench_use_fp8,
            ),
            kernel_names=(
                (
                    "EpDispatchInterNodeV1Kernel",
                    "EpCombineInterNodeV1Kernel",
                    "EpDispatchCopyToStaging",
                    "EpCombineAll",
                )
            ),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
        )
        print(f'[rank {rank}] EpDispatchCopyToStaging avg_t={dispatch_copy_t * 1e6:.2f} us | '
                  f'EpCombineAll avg_t={combine_all_t * 1e6:.2f} us', flush=True)
        dispatch_t += dispatch_copy_t
        combine_t += combine_all_t
        print(
            f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
            f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
            flush=True,
        )
    else:
        dispatch_t, combine_t = bench_kineto(
            partial(
                test_func,
                zero_copy=True,
                use_fp8=bench_use_fp8,
            ),
            kernel_names=(
                ("EpDispatchIntraNodeKernel", "EpCombineIntraNodeKernel")
            ),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
        )
        print(
            f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
            f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
            flush=True,
        )

    return hash_value


# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group, num_nodes = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 288

    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        num_nodes,
        group,
        seed=1,
    )

    do_pressure_test = False
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        ref_hash = test_main(
            num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            num_nodes,
            group,
            seed=seed,
        )
        for i in range(20):
            assert (
                test_main(
                    num_tokens,
                    hidden,
                    num_experts,
                    num_topk,
                    rank,
                    num_ranks,
                    num_nodes,
                    group,
                    seed=seed,
                )
                == ref_hash
            ), f"Error: seed={seed}"

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
