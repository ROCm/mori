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

recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
hdl  = symm_mem.rendezvous(recv, group_name)
base, stride = flat_layout(recv)

all2all_kernel.all2all_push(send, base, stride, chunk_bytes, rank_id, world_size)
```

`flat_layout()` is a convenience. torch already publishes the peer pointers, so the same
two numbers are just `hdl.buffer_ptrs[0]` and `hdl.buffer_ptrs[1] - hdl.buffer_ptrs[0]`;
the helper only adds a check that the stride is uniform, which torch's API does not
promise and which backends handing back scattered per-rank pointers would fail.

## Measured

Aggregate counts only the `(world_size-1)` chunks that leave the device; the self chunk stays
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

gfx1250 exports **fabric** handles; gfx950 and gfx942 fall back to POSIX fd, having no
fabric support at `hipMemCreate`. The kernel sees the same flat window either way. The
gfx1250 box has 4 GPUs, hence no 8-rank column.

Grid shape matters more than the handle type. An earlier version launched one block per
destination rank, leaving all but `world_size` CUs idle and unable to keep enough writes in
flight to cover interconnect latency: 15.8 GB/s on gfx1250 and 745 GB/s on gfx950 at the
same 4 MiB payload. Splitting each chunk across `blocks_per_peer` blocks is worth ~2.5x on
gfx950 and ~95x on gfx1250.

Uncached/fine-grained windows, as mori's cco windows are, were measured and rejected: half
the bandwidth on gfx1250 (712 vs 1499 GB/s at 4 ranks) and no change on gfx950. The backend
uses coarse-grained pinned memory.

### Why gfx942 is slow

A driver limitation, not a fabric or allocator one. That box's XGMI is healthy: `ubench/06`
measures 48.4 GB/s per link and 2637 GB/s aggregate all-to-all via `hipMemcpyPeer`.

Granting one peer re-maps the buffer, **in the owner's own page tables**, as
`AMDGPU_PTE_SYSTEM | MTYPE_UC` — uncached, addressed as bus memory rather than local VRAM.
The owner's access to its own HBM then leaves the chip, and 55.8 GB/s is PCIe 5 x16, which
this box measures at 54-56 GB/s h2d/d2h. A standalone HIP program (no torch, no mori)
writing 256 MiB:

| `hipMemSetAccess` grants | gfx942 write | gfx942 read | gfx950 write |
|---|---|---|---|
| self only | 2671.5 GB/s | 1987.4 GB/s | 6515.5 GB/s |
| self + 1 peer | **55.8 GB/s** | **55.2 GB/s** | 6541.4 GB/s |
| self + 4 peers | 54.8 GB/s | 55.2 GB/s | 6543.6 GB/s |

One grant is enough; more cost nothing further. Confirmed by tracing
`amdgpu:amdgpu_vm_set_ptes` against a size-fingerprinted buffer: a `SYSTEM|MTYPE_UC` group
tracking the allocation exactly (100 pages at 200 MiB, 156 at 314 MiB) appears only once a
peer is granted. The pages never move — `hipMemGetInfo` is flat across the grant — so it is
the mapping, not migration.

It is specific to the VMM path. `hipMalloc` with `hipDeviceEnablePeerAccess` for all 7
peers keeps full bandwidth on the same box (2668.5 -> 2665.2 GB/s), because the two paths
use different kernel interfaces: `hipMemSetAccess` reaches libdrm `amdgpu_bo_va_op`, the
DRM path, while ordinary allocations go through `hsaKmtMapMemoryToGPUNodes`, the KFD one.
Plain VMM, flat multi-slot reservations, self-aliasing and Pinned-vs-Uncached are all free,
so the flat window design is not what costs gfx942 its bandwidth.

Possibly a missing kernel option rather than silicon: this box runs a 5.10 kernel with a
DKMS backport and has `CONFIG_PCI_P2PDMA` and `CONFIG_DMABUF_MOVE_NOTIFY` unset, both of
which the DRM cross-device path wants, while the unaffected gfx950 box (6.8) has both.
Untested — it needs a rebuilt kernel or a modern-kernel MI300.

The escape hatch is what aiter's custom allreduce does: `hipMalloc` + hipIpc stays on the
KFD path and keeps full bandwidth, at the cost of scattered peer pointers and an N-entry
pointer array instead of a flat window.

## Notes

`dist.barrier()` is used between the kernel and the reads because the backend has no
device-side barrier yet (`barrier`/`put_signal`/`wait_signal` raise). Since none of them
are implemented, the signal pad is not reserved either — appending torch's 9216-byte pad
to a page-aligned window would cost a whole extra 2 MiB page, physical backing being
2 MiB-paged. Build with `MORI_SYMM_SIGNAL_PAD=ON` to reserve it. A real workload would want
signal-pad synchronisation instead, which is why the timed loop measures the kernel alone.
