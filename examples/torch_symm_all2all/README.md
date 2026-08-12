# All-to-all over the mori torch SymmetricMemory backend

A custom HIP kernel doing one-shot all-to-all against a flat symmetric window. **Pure
HIP** — it does not use mori's shmem or cco. The only thing it needs from the backend is
`(flat_base, stride)`, after which every peer is addressed arithmetically:

```
recv slot of rank p, chunk from rank r  ==  flat_base + p*stride + r*chunk_bytes
```

so the kernel takes two pointers and a few integers rather than an N-entry pointer array,
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

## Measured

Aggregate counts only the `(world-1)` chunks that leave the device; the self chunk stays
local. Correctness and `peer(r) == base + r*stride` hold on every configuration below.

4 MiB per peer:

| ranks | MI355X / gfx950 | MI355X-class / gfx1250 | MI308X / gfx942 |
|---|---|---|---|
| 2 | 103.8 GB/s | 435.2 GB/s | 54.1 GB/s |
| 4 | 517.8 GB/s | 1499.1 GB/s | 112.2 GB/s |
| 8 | 1858.5 GB/s | — | 184.8 GB/s |

256 KiB per peer, where launch and barrier cost still shows:

| ranks | gfx950 | gfx1250 | gfx942 |
|---|---|---|---|
| 2 | 67.7 GB/s | 75.8 GB/s | 40.3 GB/s |
| 4 | 428.0 GB/s | 289.8 GB/s | 139.4 GB/s |
| 8 | 1459.0 GB/s | — | 250.0 GB/s |

gfx1250 exports **fabric** handles; gfx950 and gfx942 have no fabric support at
`hipMemCreate` and fall back to POSIX fd. The kernel neither knows nor cares — it sees the
same flat window either way. The gfx1250 box has 4 GPUs, hence no 8-rank column.

Grid shape dominates these numbers far more than the handle type does. An earlier version
launched one block per destination rank, which left all but `world` CUs idle and could not
keep enough writes in flight to cover interconnect latency; it measured 15.8 GB/s on
gfx1250 and 745 GB/s on gfx950 at the same 4 MiB payload. Splitting each chunk across
`blocks_per_peer` blocks is worth ~2.5x on gfx950 and ~95x on gfx1250.

The gfx942 column is limited by the allocator, not the fabric. `ubench/06` measures that
box's XGMI at 48.4 GB/s per link unidirectional (76% of theoretical) and 2637 GB/s
aggregate all-to-all via `hipMemcpyPeer`, so 185 GB/s is ~7% of what the interconnect can
carry. The cause is local: writing the VMM window runs at ~50 GB/s against ~880 GB/s for a
plain `hipMalloc` tensor. A GPU's *own* slot is as slow as a peer's, torch's `data_ptr`
mapping is as slow as the flat alias, `copy_()` is as slow as our kernel, and Uncached
matches Pinned — so it is the `hipMemCreate` memory itself on this ROCm build, not the
aliasing, the handle type, or anything crossing XGMI. gfx1250 shows no such penalty
(window and plain tensor both 9.7 GB/s under an identical probe). This is also why mori's
shmem uses `hipMalloc` + hipIpc on gfx9, at the cost of scattered rather than flat peer
pointers.

Allocating the window as uncached/fine-grained (as mori's cco windows are) was measured
and rejected: on gfx1250 it costs about half the bandwidth (712 vs 1499 GB/s at 4 ranks),
and it changes nothing on gfx950. The backend uses ordinary coarse-grained pinned memory.

## Notes

`dist.barrier()` is used between the kernel and the reads because the backend has no
device-side barrier yet (`put_signal`/`wait_signal` raise). A real workload would want
signal-pad synchronisation instead, which is why the timed loop measures the kernel alone.
