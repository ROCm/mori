#!/usr/bin/env python3
"""
AllReduce Async SDMA Test using torch.distributed and multiprocessing.
Supports multiple data types (uint32, fp16, bf16) matching the sync test.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _to_numpy(tensor):
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().numpy()
    return tensor.numpy()


def _test_allreduce_async(rank, world_size, port, elems, iterations, warmup,
                          copy_output=True, dtype=torch.uint32):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank
        assert npes == world_size

        elem_size = torch.tensor([], dtype=dtype).element_size()
        bytes_per_pe = elems * elem_size
        output_buf_size = npes * (elems // npes + 64) * elem_size
        dtype_name = str(dtype).split('.')[-1]

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"AllReduce Async SDMA Test (dtype={dtype_name})")
            print(f"  World size      : {world_size}")
            print(f"  Elements per PE : {elems:,}")
            print(f"  Data size       : {bytes_per_pe / (1024**2):.2f} MB per PE")
            print(f"  Iterations      : {iterations} (warmup: {warmup})")
            print(f"  copy_output     : {copy_output}")
            print(f"{'='*60}\n")

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)

        allreduce = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=bytes_per_pe,
            output_buffer_size=output_buf_size,
            copy_output_to_user=copy_output,
            dtype=dtype,
        )

        fill_value = (my_pe + 1) * 1000
        if dtype in (torch.float16, torch.bfloat16):
            fill_value = float(fill_value)
        input_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
        output_tensor = torch.zeros(elems, dtype=dtype, device=device)

        torch.cuda.synchronize()
        dist.barrier()

        exec_times = []
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        for iter_idx in range(warmup + iterations):
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
                    print(f"PE {rank}: First measurement: {exec_time*1000:.3f} ms")

            dist.barrier()

        if len(exec_times) > 0:
            avg_time = np.mean(exec_times)
            min_time = np.min(exec_times)
            max_time = np.max(exec_times)
        else:
            avg_time = min_time = max_time = 0.0

        # Verify
        torch.cuda.synchronize()
        dist.barrier()

        expected_value = sum((pe + 1) * 1000 for pe in range(npes))

        if copy_output:
            verify_tensor = output_tensor.cpu()
        else:
            transit_buf = allreduce.get_output_transit_buffer(device=input_tensor)
            verify_tensor = transit_buf.cpu()[:elems]

        if dtype in (torch.float16, torch.bfloat16):
            verify_cpu = verify_tensor.float().numpy()
            match = np.allclose(verify_cpu, expected_value, rtol=1e-2, atol=1.0)
        else:
            verify_cpu = verify_tensor.numpy()
            match = np.all(verify_cpu == expected_value)

        success = True
        src = "output_tensor" if copy_output else "transit_buffer"
        if match:
            print(f"PE {rank}: PASSED ({src}, {dtype_name}), all {elems} elements = {expected_value}")
        else:
            if dtype in (torch.float16, torch.bfloat16):
                bad = np.where(~np.isclose(verify_cpu, expected_value, rtol=1e-2, atol=1.0))[0]
            else:
                bad = np.where(verify_cpu != expected_value)[0]
            print(f"PE {rank}: FAILED ({src}, {dtype_name}), {len(bad)} mismatches, "
                  f"first [{bad[0]}]={verify_cpu[bad[0]]}, expected {expected_value}")
            success = False

        # Global stats
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

        if rank == 0:
            g_avg = avg_t.item() / npes
            algo_bw = bytes_per_pe / g_avg / (1024.0**3) if g_avg > 0 else 0
            bus_bw = algo_bw * 2 * (npes - 1) / npes if npes > 1 else algo_bw

            print(f"\n=== Performance ({dtype_name}) ===")
            print(f"  Min: {min_t.item()*1000:.3f} ms  Max: {max_t.item()*1000:.3f} ms  Avg: {g_avg*1000:.3f} ms")
            print(f"  Algo BW: {algo_bw:.2f} GB/s  Bus BW: {bus_bw:.2f} GB/s")
            print(f"  Data: {bytes_per_pe/(1024**2):.2f} MB/rank")
            passed = ok_t.item()
            print(f"  PEs passed: {passed}/{npes}")
            print(f"  {'=== PASSED ===' if passed == npes else '=== FAILED ==='}\n")

        torch.cuda.synchronize()
        dist.barrier()
        del allreduce
        dist.barrier()
        shmem.shmem_finalize()

        if not success:
            raise AssertionError(f"PE {rank}: AllReduce async verification failed ({dtype_name})")


_DTYPE_MAP = {
    "uint32": torch.uint32, "int32": torch.int32,
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def test_allreduce_async(elems=67108864, world_size=8, iterations=10, warmup=10,
                         copy_output=True, dtype=torch.uint32):
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allreduce_async,
        args=(world_size, port, elems, iterations, warmup, copy_output, dtype),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test AllReduce Async SDMA")
    parser.add_argument("--elems", type=int, default=67108864)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1])
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=list(_DTYPE_MAP.keys()))
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)

    dtype = _DTYPE_MAP[args.dtype]
    dtype_name = str(dtype).split('.')[-1]
    copy_output = not args.no_copy

    print(f"AllReduce Async SDMA Test")
    print(f"  Elements: {args.elems:,}  World: {args.world_size}  "
          f"Iters: {args.iterations}  Warmup: {args.warmup}  "
          f"Dtype: {dtype_name}  Copy: {copy_output}")
    print("-" * 60)

    test_allreduce_async(args.elems, args.world_size, args.iterations,
                         args.warmup, copy_output, dtype)
