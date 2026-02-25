#!/usr/bin/env python3
"""
AllReduce SDMA Test using torch.distributed and multiprocessing.

Tests both out-of-place (copy_output_to_user=False, read from transit buffer)
and in-place (allreduce_inplace) modes, verifying correctness and measuring bandwidth.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _verify_allreduce_result(data_cpu, elems, my_pe, npes, label="", dtype=torch.uint32):
    """Verify allreduce result: each element should equal sum of (pe+1)*1000 across all PEs."""
    expected_value = sum((pe + 1) * 1000 for pe in range(npes))
    first_n = min(elems, len(data_cpu))
    chunk = data_cpu[:first_n]

    if dtype in (torch.float16, torch.bfloat16):
        expected_arr = np.full(first_n, expected_value, dtype=np.float32)
        chunk_f = chunk.astype(np.float32)
        if np.allclose(chunk_f, expected_arr, rtol=1e-2, atol=1.0):
            return True
        else:
            mismatches = np.where(~np.isclose(chunk_f, expected_arr, rtol=1e-2, atol=1.0))[0]
            print(f"  PE {my_pe} [{label}]: FAILED! Expected ~{expected_value}, "
                  f"got {len(mismatches)} mismatches in first {first_n} elements. "
                  f"First mismatch at idx {mismatches[0]}: {chunk[mismatches[0]]}")
            return False
    else:
        if np.all(chunk == expected_value):
            return True
        else:
            mismatches = np.where(chunk != expected_value)[0]
            print(f"  PE {my_pe} [{label}]: FAILED! Expected {expected_value}, "
                  f"got {len(mismatches)} mismatches in first {first_n} elements. "
                  f"First mismatch at idx {mismatches[0]}: {chunk[mismatches[0]]}")
            return False


def _run_benchmark(allreduce_fn, iterations, warmup, stream, rank):
    """Run warmup + measurement iterations, return list of measured times."""
    exec_times = []
    total_iters = warmup + iterations

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(total_iters):
        ev_start.record(stream)
        success = allreduce_fn()
        ev_end.record(stream)
        stream.synchronize()

        if not success:
            print(f"PE {rank}: AllReduce failed at iteration {i}")
            break

        elapsed = ev_start.elapsed_time(ev_end) / 1000.0
        if i >= warmup:
            exec_times.append(elapsed)
        elif rank == 0:
            print(f"  Warmup {i + 1}/{warmup}: {elapsed * 1000:.3f} ms")

    return exec_times


def _print_stats(exec_times, data_bytes, npes, rank, label):
    """Aggregate and print performance statistics."""
    if len(exec_times) > 0:
        avg_time = np.mean(exec_times)
        min_time = np.min(exec_times)
        max_time = np.max(exec_times)
    else:
        avg_time = min_time = max_time = 0.0

    min_t = torch.tensor([min_time], dtype=torch.float64)
    max_t = torch.tensor([max_time], dtype=torch.float64)
    avg_t = torch.tensor([avg_time], dtype=torch.float64)

    dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)

    if rank == 0:
        g_min = min_t.item()
        g_max = max_t.item()
        g_avg = avg_t.item() / npes
        # AllReduce bandwidth: each rank has data_bytes, the algorithm factor is 2*(N-1)/N
        algo_bw = data_bytes / g_avg / (1024.0 ** 3) if g_avg > 0 else 0
        bus_bw = algo_bw * 2 * (npes - 1) / npes if npes > 1 else algo_bw

        print(f"\n--- {label} Performance ---")
        print(f"  Min time : {g_min * 1000:.3f} ms")
        print(f"  Max time : {g_max * 1000:.3f} ms")
        print(f"  Avg time : {g_avg * 1000:.3f} ms")
        print(f"  Algo BW  : {algo_bw:.2f} GB/s")
        print(f"  Bus BW   : {bus_bw:.2f} GB/s")
        print(f"  Data size: {data_bytes / (1024.0 ** 2):.2f} MB per rank")

    return avg_time


def _test_allreduce(rank, world_size, port, elems, iterations, warmup, dtype=torch.uint32):
    """Worker function for each process."""

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()

        assert my_pe == rank
        assert npes == world_size

        elem_size = torch.tensor([], dtype=dtype).element_size()
        data_bytes = elems * elem_size
        output_buf_size = npes * (elems // npes + 64) * elem_size
        dtype_name = str(dtype).split('.')[-1]

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"AllReduce SDMA Test (dtype={dtype_name})")
            print(f"  World size      : {world_size}")
            print(f"  Elements per PE : {elems:,}")
            print(f"  Data size       : {data_bytes / (1024 ** 2):.2f} MB per rank")
            print(f"  Iterations      : {iterations} (warmup: {warmup})")
            print(f"{'=' * 70}")

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)

        fill_value = (my_pe + 1) * 1000

        # ==================================================================
        # Test 1: Out-of-place with copy_output_to_user=False
        #         Read result from output_transit_buffer
        # ==================================================================
        if rank == 0:
            print(f"\n>>> Test 1: Out-of-place (copy_output_to_user=False, {dtype_name})")

        ar_nocp = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=data_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=False,
            dtype=dtype,
        )

        input_tensor = torch.full((elems,), fill_value,
                                  dtype=dtype, device=device)
        output_tensor = torch.zeros(elems, dtype=dtype, device=device)

        torch.cuda.synchronize()
        dist.barrier()

        times_nocp = _run_benchmark(
            lambda: ar_nocp(input_tensor, output_tensor, elems, stream),
            iterations, warmup, stream, rank,
        )

        transit_buf = ar_nocp.get_output_transit_buffer(device=input_tensor)
        transit_cpu = transit_buf.cpu().numpy()

        nocp_ok = _verify_allreduce_result(transit_cpu, elems, my_pe, npes,
                                           f"out-of-place/transit/{dtype_name}",
                                           dtype=dtype)

        out_cpu = output_tensor.cpu().numpy()
        zero_check = np.all(out_cpu == 0) if dtype in (torch.uint32, torch.int32) else np.allclose(out_cpu, 0)
        if zero_check:
            if rank == 0:
                print(f"  PE {rank}: output_tensor correctly untouched (all zeros)")
        else:
            if rank == 0:
                print(f"  PE {rank}: WARNING — output_tensor was modified despite copy_output_to_user=False")

        dist.barrier()
        _print_stats(times_nocp, data_bytes, npes, rank,
                     f"Out-of-place (transit buf, {dtype_name})")

        # ==================================================================
        # Test 2: allreduce_inplace
        # ==================================================================
        if rank == 0:
            print(f"\n>>> Test 2: In-place (allreduce_inplace, {dtype_name})")

        ar_inp = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=data_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=False,
            dtype=dtype,
        )

        inplace_tensor = torch.full((elems,), fill_value,
                                    dtype=dtype, device=device)

        torch.cuda.synchronize()
        dist.barrier()

        times_inp = _run_benchmark(
            lambda: ar_inp.allreduce_inplace(inplace_tensor, elems, stream),
            iterations, warmup, stream, rank,
        )

        inp_cpu = inplace_tensor.cpu().numpy()
        inp_ok = _verify_allreduce_result(inp_cpu, elems, my_pe, npes,
                                          f"inplace/{dtype_name}", dtype=dtype)

        if rank == 0 and inp_ok:
            expected = sum((pe + 1) * 1000 for pe in range(npes))
            print(f"  PE {rank}: inplace result verified (all values ~ {expected})")

        dist.barrier()
        _print_stats(times_inp, data_bytes, npes, rank, f"In-place ({dtype_name})")

        # ==================================================================
        # Final summary
        # ==================================================================
        all_ok = nocp_ok and inp_ok
        ok_tensor = torch.tensor([1 if all_ok else 0], dtype=torch.int32)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            passed = ok_tensor.item()
            print(f"\n{'=' * 70}")
            print(f"PEs passed ({dtype_name}): {passed}/{npes}")
            if passed == npes:
                print(f"=== All Tests PASSED ({dtype_name}) ===")
            else:
                print(f"=== SOME Tests FAILED ({dtype_name}) ===")
            print(f"{'=' * 70}\n")

        # Cleanup
        torch.cuda.synchronize()
        dist.barrier()
        del ar_nocp, ar_inp
        dist.barrier()
        shmem.shmem_finalize()

        if not all_ok:
            raise AssertionError(f"PE {rank}: AllReduce verification failed ({dtype_name})")


_DTYPE_MAP = {
    "uint32": torch.uint32,
    "int32": torch.int32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def test_allreduce(elems=67108864, world_size=8, iterations=10, warmup=10,
                   dtype=torch.uint32):
    """Run AllReduce SDMA test."""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allreduce,
        args=(world_size, port, elems, iterations, warmup, dtype),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test AllReduce SDMA (out-of-place + inplace, correctness + bandwidth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=10, help="Measurement iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA")
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=list(_DTYPE_MAP.keys()),
                        help="Data type (uint32, fp16, bf16)")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)

    dtype = _DTYPE_MAP[args.dtype]
    dtype_name = str(dtype).split('.')[-1]

    print(f"AllReduce SDMA Test")
    print(f"  Elements per PE : {args.elems:,}")
    print(f"  World size      : {args.world_size}")
    print(f"  Iterations      : {args.iterations}")
    print(f"  Warmup          : {args.warmup}")
    print(f"  Dtype           : {dtype_name}")
    print("-" * 60)

    test_allreduce(args.elems, args.world_size, args.iterations, args.warmup, dtype)
