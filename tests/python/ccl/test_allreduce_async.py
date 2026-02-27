#!/usr/bin/env python3
"""
AllReduce Async SDMA Test using torch.distributed and multiprocessing.

Uses the async API (start_async + wait_async) which internally performs:
  - ReduceScatter (SDMA or P2P depending on data size)
  - AllGather via SDMA
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _test_allreduce_async(rank, world_size, port, elems, iterations, warmup):
    """Worker function for each process."""

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()

        assert my_pe == rank
        assert npes == world_size

        bytes_per_pe = elems * 4  # uint32 = 4 bytes
        total_bytes = bytes_per_pe * npes
        output_buf_size = npes * (elems // npes + 64) * 4

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"AllReduce Async SDMA Test")
            print(f"  World size      : {world_size}")
            print(f"  Elements per PE : {elems:,}")
            print(f"  Data size       : {bytes_per_pe / (1024**2):.2f} MB per PE")
            print(f"  Iterations      : {iterations} (warmup: {warmup})")
            print(f"{'='*60}\n")

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)

        allreduce = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=bytes_per_pe,
            output_buffer_size=output_buf_size,
            copy_output_to_user=True,
        )
        print(f"PE {rank}: Created AllreduceSdma object")

        # Data init: each PE fills all elements with (myPe + 1)
        input_tensor = torch.full((elems,), my_pe + 1,
                                  dtype=torch.uint32, device=device)
        output_tensor = torch.zeros(elems, dtype=torch.uint32, device=device)

        if rank == 0:
            print(f"PE {rank}: Input data = all {my_pe + 1}")

        torch.cuda.synchronize()
        dist.barrier()

        # Run warmup + measurement iterations using async API
        exec_times = []
        total_iters = warmup + iterations

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        if rank == 0:
            print(f"\nUsing ASYNC mode (start_async + wait_async)")
            if warmup > 0:
                print(f"Warmup iterations: {warmup}, Measurement iterations: {iterations}\n")

        for iter_idx in range(total_iters):
            if rank == 0 and (iter_idx == 0 or iter_idx == warmup):
                stage = "Warmup" if iter_idx < warmup else "Measurement"
                print(f"\n--- {stage} Iteration {iter_idx + 1} ---")

            dist.barrier()

            ev_start.record(stream)
            allreduce.start_async(input_tensor, output_tensor, elems, stream)
            allreduce.wait_async(stream)
            ev_end.record(stream)
            torch.cuda.synchronize()
            exec_time = ev_start.elapsed_time(ev_end) / 1000.0

            if iter_idx >= warmup:
                exec_times.append(exec_time)
                if rank == 0 and len(exec_times) == 1:
                    print(f"PE {rank}: First measurement iteration: {exec_time:.6f}s")

            dist.barrier()

        # Local statistics
        if len(exec_times) > 0:
            avg_time = np.mean(exec_times)
            min_time = np.min(exec_times)
            max_time = np.max(exec_times)
        else:
            avg_time = min_time = max_time = 0.0

        if rank == 0:
            print(f"\nPE {rank} local statistics:")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Avg time: {avg_time:.6f}s")

        # Verify: expected value = npes * (npes + 1) / 2
        torch.cuda.synchronize()
        dist.barrier()

        output_cpu = output_tensor.cpu().numpy()
        expected_value = npes * (npes + 1) // 2

        success = True
        if np.all(output_cpu == expected_value):
            print(f"PE {rank}: AllReduce verification PASSED! All {elems} elements = {expected_value}")
        else:
            mismatches = np.where(output_cpu != expected_value)[0]
            print(f"PE {rank}: AllReduce verification FAILED! "
                  f"{len(mismatches)} mismatches, first at [{mismatches[0]}]={output_cpu[mismatches[0]]}, "
                  f"expected {expected_value}")
            success = False

        # Global statistics
        torch.cuda.synchronize()
        dist.barrier()

        min_t = torch.tensor([min_time], dtype=torch.float64)
        max_t = torch.tensor([max_time], dtype=torch.float64)
        avg_t = torch.tensor([avg_time], dtype=torch.float64)
        ok_t = torch.tensor([1 if success else 0], dtype=torch.int32)

        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(ok_t, op=dist.ReduceOp.SUM)

        g_min = min_t.item()
        g_max = max_t.item()
        g_avg = avg_t.item() / npes
        passed = ok_t.item()

        if rank == 0:
            algo_bw = bytes_per_pe / g_avg / (1024.0**3) if g_avg > 0 else 0
            bus_bw = algo_bw * 2 * (npes - 1) / npes if npes > 1 else algo_bw

            print(f"\n=== Performance Statistics ===")
            print(f"Min time: {g_min:.6f}s")
            print(f"Max time: {g_max:.6f}s")
            print(f"Avg time: {g_avg:.6f}s")
            print(f"Algo bandwidth: {algo_bw:.2f} GB/s (data size: {bytes_per_pe / (1024.0**3):.3f} GB)")
            print(f"Bus  bandwidth: {bus_bw:.2f} GB/s (factor: 2*(N-1)/N = {2.0*(npes-1)/npes:.2f})")
            print(f"\nPEs passed: {passed}/{npes}")

            if passed == npes:
                print(f"\n=== AllReduce Async Test PASSED ===\n")
            else:
                print(f"\n=== AllReduce Async Test FAILED ===\n")

        # Cleanup
        torch.cuda.synchronize()
        dist.barrier()
        del allreduce
        dist.barrier()
        shmem.shmem_finalize()

        if not success:
            raise AssertionError(f"PE {rank}: AllReduce verification failed")


def test_allreduce_async(elems=67108864, world_size=8, iterations=10, warmup=10):
    """Run AllReduce Async SDMA test."""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allreduce_async,
        args=(world_size, port, elems, iterations, warmup),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test AllReduce Async SDMA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=10, help="Measurement iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)

    print(f"AllReduce Async SDMA Test")
    print(f"  Elements per PE : {args.elems:,}")
    print(f"  World size      : {args.world_size}")
    print(f"  Iterations      : {args.iterations}")
    print(f"  Warmup          : {args.warmup}")
    print("-" * 60)

    test_allreduce_async(args.elems, args.world_size, args.iterations, args.warmup)
