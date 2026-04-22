#!/usr/bin/env python3
"""
Minimal test: measure ShmemSymmetricRegister host us cost.

Runs 8-rank spawn + for each of several (ptr, size) configurations calls
ar._handle.register_user_output(ptr, size) and prints last_register_us.

Goal: determine whether D' fast path is viable. If register is < 500 us
on a typical 256 MB buffer, it's cheap enough to cache once. If it is
in the ms range on every miss, D' loses more than it gains on any
fresh-tensor workload.
"""
import os
import sys
import time

import torch
import torch.distributed as dist


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Import after CUDA ctx is ready so mori inits on the right device.
    from mori import shmem
    from mori.ccl import AllreduceSdma

    shmem.shmem_torch_process_group_init("default")

    my_pe = shmem.shmem_mype()
    npes = shmem.shmem_npes()
    assert my_pe == rank and npes == world_size

    elem_size = 4  # uint32
    # Create AR with enough transit capacity for max size we will test.
    max_mb = 256
    max_bytes = max_mb * 1024 * 1024
    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=max_bytes,
        output_buffer_size=npes * (max_bytes // npes + 64) * elem_size,
        copy_output_to_user=True,
        dtype=torch.uint32,
    )

    device = torch.device(f"cuda:{rank}")

    for size_mb in [1, 16, 64, 128, 256]:
        n_elems = size_mb * 1024 * 1024 // elem_size
        bytes_sz = n_elems * elem_size

        # Use a fresh tensor every round to force cache miss.
        buf = torch.zeros(n_elems, dtype=torch.uint32, device=device)
        ptr = buf.data_ptr()

        # All ranks must call collectively, same size.
        dist.barrier()
        t0 = time.perf_counter()
        ok = ar._handle.register_user_output(ptr, bytes_sz)
        t1 = time.perf_counter()
        dist.barrier()

        host_us = (t1 - t0) * 1e6
        c_us = ar._handle.last_register_us()
        if rank == 0:
            print(f"[size={size_mb:3d} MB] ok={ok}  "
                  f"python host wall={host_us:8.1f} us  "
                  f"C++ chrono={c_us:8.1f} us",
                  flush=True)

        # Now call it again to verify cache hit → should be ~0
        dist.barrier()
        t0 = time.perf_counter()
        ok = ar._handle.register_user_output(ptr, bytes_sz)
        t1 = time.perf_counter()
        host_us2 = (t1 - t0) * 1e6
        c_us2 = ar._handle.last_register_us()
        if rank == 0:
            print(f"[size={size_mb:3d} MB] repeat (cache hit)  "
                  f"python host wall={host_us2:8.1f} us  "
                  f"C++ chrono={c_us2:8.1f} us",
                  flush=True)

        del buf

    if rank == 0:
        print(f"cache_hits={ar._handle.cache_hits()}, "
              f"cache_misses={ar._handle.cache_misses()}",
              flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    world_size = 8
    port = 29500 + int(os.getpid() % 1000)
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
