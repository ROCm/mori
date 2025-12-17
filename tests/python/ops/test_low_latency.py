"""
Low Latency Dispatch/Combine Test - Auto-detects single/multi-node

Auto selection:
  - Single-node: IntraNode kernel
  - Multi-node: InterNodeV1LL kernel

Usage (默认每节点8个GPU):
  Single-node (using spawn):
    python test_low_latency.py
  
  Multi-node (run on each node separately, WORLD_SIZE=节点数, RANK=节点编号):
    Node 0: RANK=0 WORLD_SIZE=2 MASTER_ADDR=node0_ip MASTER_PORT=29500 python test_low_latency.py
    Node 1: RANK=1 WORLD_SIZE=2 MASTER_ADDR=node0_ip MASTER_PORT=29500 python test_low_latency.py
"""
import random
import numpy as np
import torch
import torch.distributed as dist
from functools import partial

# import deep_ep
import mori
from tests.python.ops.test_dispatch_combine import EpDispatchCombineTestCase
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: int = None, seed: int = 0,
              enable_dedup: bool = True, fused_moe_adaption: bool = True, gpu_per_node: int = 8):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(torch.int32)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions diff:mori不支持-1
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

    # Auto-detect: multi-node uses InterNodeV1LL, single-node uses IntraNode
    num_nodes = (num_ranks + gpu_per_node - 1) // gpu_per_node
    is_multi_node = num_nodes > 1
    
    if is_multi_node:
        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        scale_dim, scale_type_size, block_num, rdma_block_num, warp_num_per_block = 32, 4, 64, 32, 8
    else:
        kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        scale_dim, scale_type_size, block_num, rdma_block_num, warp_num_per_block = 0, 0, 80, 0, 16

    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=num_ranks,
        hidden_dim=hidden,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=2,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        use_external_inp_buf=False,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=4 if is_multi_node else 0,
    )
    mori.shmem.shmem_torch_process_group_init("default")
    op = mori.ops.EpDispatchCombineOp(config)

    no_zero_copy_config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=num_ranks,
        hidden_dim=hidden,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=2,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        use_external_inp_buf=True,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=4 if is_multi_node else 0,
    )
    no_zero_copy_op = mori.ops.EpDispatchCombineOp(no_zero_copy_config)

    # Generate scales only for multi-node
    scales = torch.rand((num_tokens, scale_dim), dtype=torch.float32, device='cuda') if is_multi_node else None

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for return_recv_hook in (False, ):
        for dispatch_use_fp8 in (False, ):
            # num_times += 1 diff: mori dispatch-combine成对调用，多次调dispatch会累加recv_count导致出错
            for i in range((num_times % 2) + 1):
                if fused_moe_adaption:
                    # packed_recv_x, packed_recv_topk_idx, packed_recv_topk_weights, packed_recv_count, handle, event, hook = \
                    #     buffer.low_latency_dispatch_rocm(x, topk_idx, num_tokens, num_experts, use_fp8=dispatch_use_fp8,
                    #                                      async_finish=not return_recv_hook, return_recv_hook=return_recv_hook,
                    #                                      topk_weights=topk_weights)
                    (
                        packed_recv_x,
                        packed_recv_topk_weights,
                        packed_recv_scales,
                        packed_recv_topk_idx,
                        packed_recv_count,
                    ) = no_zero_copy_op.dispatch(
                        x,
                        topk_weights,
                        scales,
                        topk_idx,
                        block_num=block_num,
                        warp_per_block=16,
                    )
                else:
                    packed_recv_x, packed_recv_count, handle, event, hook = \
                        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts, use_fp8=dispatch_use_fp8,
                                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                # hook() if return_recv_hook else event.current_stream_wait()
                torch.cuda.synchronize()
            packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
            assert fused_moe_adaption == ((packed_recv_x[0].ndim != 3) if dispatch_use_fp8 else (packed_recv_x.ndim != 3)), f'{fused_moe_adaption} != {(packed_recv_x[0].ndim != 3) if dispatch_use_fp8 else (packed_recv_x.ndim != 3)}'
            if not fused_moe_adaption:
                simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                    if dispatch_use_fp8 else packed_recv_x.clone()
            else:
                simulated_gemm_x = per_token_cast_back(packed_recv_x[0], packed_recv_x[1]) \
                    if dispatch_use_fp8 else packed_recv_x.clone()
            all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=validation_topk_idx.dtype, device='cuda')
            dist.all_gather_into_tensor(all_topk_idx, validation_topk_idx, group=group)
            # for i in range(num_local_experts if do_check and (not fused_moe_adaption) else 0):
            #     expert_id = rank * num_local_experts + i
            #     recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
            #     recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

            #     # Check expert indices
            #     int_mask = (2 ** 32) - 1
            #     num_valid_tokens = recv_count.item()
            #     assert num_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
            #     assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

            #     # Check received data
            #     recv_x = recv_x[:num_valid_tokens]
            #     recv_x_amin = recv_x[:, :-128].amin(dim=-1)
            #     recv_src_info = recv_src_info[:num_valid_tokens]
            #     assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
            #     assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
            #     for j in range(num_ranks):
            #         begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
            #         assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
            #         assert (recv_x[begin_idx:begin_idx + count][:-128] - j).sum().item() == 0
            #         assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
            #     if dispatch_use_fp8:
            #         hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
            #         hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
            #     else:
            #         hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

            if do_check and fused_moe_adaption:
                global_recv_x = per_token_cast_back(packed_recv_x[0], packed_recv_x[1]) if dispatch_use_fp8 else packed_recv_x
                # global_recv_src_info = handle[0]
                # Check expert indices
                # int_mask = (2 ** 32) - 1
                num_total_valid_tokens = packed_recv_count.item()
                # recv_layout_range = handle[1]
                # assert num_total_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_total_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                # if rank == 0:
                #     print(
                #         f"[rank {rank}] num_total_valid_tokens {num_total_valid_tokens} expected_recv_tokens {(all_topk_idx // num_local_experts == rank).sum().item()}"
                #     )
                #     print(
                #         f"[rank {rank}] no-dedup expected_recv_tokens {(topk_idx // num_local_experts == rank).sum().item()}"
                #     )
                #     for peer in range(num_ranks):
                #         print(f"[rank {rank}] peer {peer} expected_recv_tokens {(all_topk_idx[peer] // num_local_experts == rank).sum().item()}")
                assert num_total_valid_tokens == (all_topk_idx // num_local_experts == rank).sum().item(), f'{num_total_valid_tokens} != {(all_topk_idx // num_local_experts == rank).sum().item()}'
                # Check received data
                # for i in range(num_local_experts):
                #     for r in range(num_ranks):
                #         expert_id = rank * num_local_experts + i
                #         count = (recv_layout_range[i][r] & int_mask).item()
                #         begin_idx = (recv_layout_range[i][r] >> 32).item()
                #         recv_x = global_recv_x[begin_idx:begin_idx + count]
                #         recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                #         recv_src_info = global_recv_src_info[begin_idx:begin_idx + count]
                #         assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                #         assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                #         assert (recv_x_amin == r - rank_offset).sum().item() == (all_topk_idx[r] == expert_id).sum().item()
                #         assert (recv_x[:, :-128] - r + rank_offset).sum().item() == 0
                #         if dispatch_use_fp8:
                #             hash_value ^= hash_tensor(packed_recv_x[0][begin_idx:begin_idx + count])
                #             hash_value ^= hash_tensor(packed_recv_x[1][begin_idx:begin_idx + count])
                #         else:
                #             hash_value ^= hash_tensor(packed_recv_x[begin_idx:begin_idx + count])
                #         # TODO check recv topk index and weights

            # Check combine correctness diff:mori只支持zero_copy为true
            for zero_copy in (False, ):
                if zero_copy:
                    if fused_moe_adaption:
                        # buffer.get_next_low_latency_combine_buffer(handle)[:, :] = simulated_gemm_x
                        combine_input = op.get_registered_combine_input_buffer(config.data_type)
                        combine_input[:, :].copy_(
                            simulated_gemm_x
                        )
                    else:
                        buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    # combined_x, event, hook = buffer.low_latency_combine_rocm(simulated_gemm_x, topk_idx, None, handle,
                    #                                                         async_finish=not return_recv_hook, zero_copy=zero_copy,
                    #                                                         return_recv_hook=return_recv_hook, out=out)
                    combined_x, _ = op.combine(
                        simulated_gemm_x,
                        None,
                        topk_idx,
                        block_num=block_num,
                        warp_per_block=16,
                    )
                else:
                    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    # combined_x, event, hook = buffer.low_latency_combine_rocm(simulated_gemm_x, topk_idx, None, handle,
                    #                                                         async_finish=not return_recv_hook, zero_copy=zero_copy,
                    #                                                         return_recv_hook=return_recv_hook, out=out)
                    # print(f'simulated_gemm_x[:5, :8]:\n{simulated_gemm_x[:5, :8]}')
                    combined_x, _ = no_zero_copy_op.combine(
                        simulated_gemm_x,
                        None,
                        topk_idx,
                        block_num=block_num,
                        warp_per_block=16,
                    )
                # hook() if return_recv_hook else event.current_stream_wait()
                torch.cuda.synchronize()
                if do_check:
                    if fused_moe_adaption:
                        diff = calc_diff(x * (validation_topk_idx != -1).sum(dim=-1, dtype=torch.float32).view(-1, 1), combined_x)
                    else:
                        diff = calc_diff(x * topk_weights.masked_fill(validation_topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                    assert torch.isnan(combined_x).sum().item() == 0
                    assert diff < 1e-5, f'Error: {diff=}, {zero_copy=}'
                    hash_value ^= hash_tensor(combined_x)

    def create_test_cast_with_outliers(num_outliers):
        tmp = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        tmp /= tmp.abs().amax(dim=1).view(-1, 1)
        assert tmp.abs().amax().item() <= 1

        # Create some amax outliers
        for i in range(num_outliers):
            tmp[random.randint(0, num_tokens - 1)] *= 1e3
        return tmp

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, use_fp8: bool, return_recv_hook: bool):
        if fused_moe_adaption:
            if zero_copy:
                # recv_x, recv_topk_idx, recv_topk_weights, recv_count, handle, event, hook = \
                #     buffer.low_latency_dispatch_rocm(x, topk_idx, num_tokens, num_experts,
                #                                 use_fp8=use_fp8, async_finish=False, return_recv_hook=return_recv_hook,
                #                                 topk_weights=topk_weights)
                (
                    recv_x,
                    recv_topk_weights,
                    recv_scales,
                    recv_topk_idx,
                    recv_count,
                ) = op.dispatch(
                    x,
                    topk_weights,
                    scales,
                    topk_idx,
                    block_num=block_num,
                    warp_per_block=16,
                )
                # large_gemm_with_hook(hook) if return_recv_hook else None
                if zero_copy:
                    # buffer.get_next_low_latency_combine_buffer(handle)[:, :] = simulated_gemm_x
                    combine_input = op.get_registered_combine_input_buffer(config.data_type)
                    combine_input[:, :].copy_(
                        simulated_gemm_x
                    )
                # combined_x, event, hook = buffer.low_latency_combine_rocm(simulated_gemm_x, topk_idx, None, handle,
                #                                                           zero_copy=zero_copy, return_recv_hook=return_recv_hook)
                combined_x, _ = op.combine(
                    simulated_gemm_x,
                    None,
                    topk_idx,
                    block_num=block_num,
                    warp_per_block=16,
                )
                # large_gemm_with_hook(hook) if return_recv_hook else None
            else:
                (
                    recv_x,
                    recv_topk_weights,
                    recv_scales,
                    recv_topk_idx,
                    recv_count,
                ) = no_zero_copy_op.dispatch(
                    x,
                    topk_weights,
                    scales,
                    topk_idx,
                    block_num=block_num,
                )
                # large_gemm_with_hook(hook) if return_recv_hook else None
                combined_x, _ = no_zero_copy_op.combine(
                    simulated_gemm_x,
                    None,
                    topk_idx,
                    block_num=block_num,
                )
                # large_gemm_with_hook(hook) if return_recv_hook else None
        else:
            recv_x, recv_count, handle, event, hook = \
                buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                            use_fp8=use_fp8, async_finish=False, return_recv_hook=return_recv_hook)
            large_gemm_with_hook(hook) if return_recv_hook else None
            if zero_copy:
                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                zero_copy=zero_copy, return_recv_hook=return_recv_hook)
            large_gemm_with_hook(hook) if return_recv_hook else None

    # Unlike the original case, FP8 quantization is not yet supported.
    bench_use_fp8 = False

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (validation_topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += (num_fp8_bytes if bench_use_fp8 else num_bf16_bytes) * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections
    print(f'[rank {rank}] num_dispatch_comm_bytes {num_dispatch_comm_bytes} num_combine_comm_bytes {num_combine_comm_bytes}')

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, zero_copy=False, use_fp8=bench_use_fp8, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    # Separate profiling: measure dispatch and combine in one pass with barrier between them
    group.barrier()
    
    use_zero_copy = not is_multi_node
    current_op = op if use_zero_copy else no_zero_copy_op
    
    # Warmup
    for _ in range(10):
        (recv_x, recv_topk_weights, recv_scales, recv_topk_idx, recv_count) = current_op.dispatch(
            x, topk_weights, scales, topk_idx, block_num=block_num
        )
        group.barrier()
        if use_zero_copy:
            combine_input = current_op.get_registered_combine_input_buffer(config.data_type)
            combine_input[:, :].copy_(simulated_gemm_x)
        combined_x, _ = current_op.combine(
            simulated_gemm_x, None, topk_idx, block_num=block_num
        )
        torch.cuda.synchronize()
    
    # Benchmark: measure both dispatch and combine in one iteration
    dispatch_times = []
    combine_times = []
    for _ in range(30):
        group.barrier()
        
        # Time dispatch
        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        dispatch_start.record()
        (recv_x, recv_topk_weights, recv_scales, recv_topk_idx, recv_count) = current_op.dispatch(
            x, topk_weights, scales, topk_idx, block_num=block_num
        )
        dispatch_end.record()
        
        # Barrier to separate dispatch and combine (before sync to avoid timing barrier wait)
        group.barrier()
        torch.cuda.synchronize()
        # Time combine
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)
        if use_zero_copy:
            combine_input = current_op.get_registered_combine_input_buffer(config.data_type)
            combine_input[:, :].copy_(simulated_gemm_x)
        combine_start.record()
        combined_x, _ = current_op.combine(
            simulated_gemm_x, None, topk_idx, block_num=block_num
        )
        combine_end.record()
        
        # Synchronize once at the end to get both timings
        torch.cuda.synchronize()
        dispatch_times.append(dispatch_start.elapsed_time(dispatch_end) / 1000.0)  # Convert to seconds
        combine_times.append(combine_start.elapsed_time(combine_end) / 1000.0)
    
    dispatch_avg = np.mean(dispatch_times)
    dispatch_min = np.min(dispatch_times)
    dispatch_max = np.max(dispatch_times)
    combine_avg = np.mean(combine_times)
    combine_min = np.min(combine_times)
    combine_max = np.max(combine_times)
    
    print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_avg:.2f} GB/s, '
          f'avg_t={dispatch_avg * 1e6:.2f} us (min={dispatch_min * 1e6:.2f}, max={dispatch_max * 1e6:.2f}) | '
          f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_avg:.2f} GB/s, '
          f'avg_t={combine_avg * 1e6:.2f} us (min={combine_min * 1e6:.2f}, max={combine_max * 1e6:.2f})', flush=True)

    return hash_value


# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int, gpu_per_node: int = 8):
    from utils import cleanup_dist
    
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 288

    try:
        # num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
        # if local_rank == 0:
        #     print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
        # buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
        #                         num_qps_per_rank=num_experts // num_ranks)
        test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, None, seed=1, gpu_per_node=gpu_per_node)

        do_pressure_test = False
        for seed in range(int(1e9) if do_pressure_test else 0):
            if rank == 0:
                print(f'Testing with seed {seed} ...', flush=True)
            ref_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, None, seed=seed, gpu_per_node=gpu_per_node)
            for i in range(20):
                assert test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, None, seed=seed, gpu_per_node=gpu_per_node) == ref_hash, f'Error: seed={seed}'
        
        
    except Exception as e:
        print(f'[Rank {rank}] Error during test: {e}', flush=True)
        raise
    finally:
        cleanup_dist(rank)

if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)