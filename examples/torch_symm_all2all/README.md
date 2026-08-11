# All-to-all over the mori torch SymmetricMemory backend

A custom HIP kernel doing one-shot all-to-all against a flat symmetric window. **Pure
HIP** — it does not use mori's shmem or cco. The only thing it needs from the backend is
`(flat_base, stride)`, after which every peer is addressed arithmetically:

```
recv slot of rank p, chunk from rank r  ==  flat_base + p*stride + r*chunk_bytes
```

so the kernel takes two pointers and three integers rather than an N-entry pointer array,
and the destination rank can be computed at run time.

```bash
python3 setup.py build_ext --inplace
torchrun --nnodes=1 --nproc_per_node=8 all2all.py --chunk-kib 256
```

The extension is built here, not by mori's CMake — the example is self-contained.

## What it does

`recv` is an ordinary `symm_mem.empty()` tensor, made symmetric by
`symm_mem.rendezvous()`. `send` is plain local memory. The kernel **pushes**: each rank
writes its own chunk into every peer's receive window, so there are no remote reads and no
per-peer handshake — just one barrier afterwards, before anyone reads what landed.

```python
register_symm_backend()
symm_mem.set_backend("MORI")

recv = symm_mem.empty(world * elems, dtype=torch.int32, device=device)
hdl  = symm_mem.rendezvous(recv, group_name)
base, stride = flat_layout(recv)

all2all_kernel.all2all_push(send, base, stride, chunk_bytes, rank, world)
```

## Measured — 8x MI355X (gfx950), ROCm 7.2.4, 256 KiB per peer

```
world=8  handle=posix_fd  chunk=256 KiB
peer(r) == base + r*stride: True
correctness: OK (8x8 chunks)
push all-to-all: 17.0 us/iter, 795.1 GB/s aggregate
```

| ranks | aggregate |
|---|---|
| 2 | 20.1 GB/s |
| 4 | 153.4 GB/s |
| 8 | 795.1 GB/s |

Aggregate counts only the `(world-1)` chunks that leave the device; the self chunk stays
local. gfx9 has no fabric support, so the backend falls back to POSIX fd here — the kernel
neither knows nor cares.

## Notes

`dist.barrier()` is used between the kernel and the reads because the backend has no
device-side barrier yet (`put_signal`/`wait_signal` raise). A real workload would want
signal-pad synchronisation instead, which is why the timed loop measures the kernel alone.
