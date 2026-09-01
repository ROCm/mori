# All-to-all over the mori torch SymmetricMemory backend

One-shot all-to-all against a symmetric window, written three ways — `all2all_hip.py`,
`all2all_triton.py` and `all2all_lsa.py`. Each is self-contained and runs on its own. The
first two need nothing from mori but the peer pointers torch publishes:

```
recv slot of rank p, chunk from rank r  ==  peers[p] + r*chunk_bytes
```

```bash
torchrun --nnodes=1 --nproc_per_node=8 all2all_hip.py --chunk-kib 256
torchrun --nnodes=1 --nproc_per_node=8 all2all_triton.py --chunk-kib 256
torchrun --nnodes=1 --nproc_per_node=8 all2all_lsa.py --chunk-kib 256
```

Needs **torch >= 2.9**: `symm_mem.set_backend()`, and the SymmetricMemory interface the
backend implements, do not exist before that.

There is no build step. The HIP kernel is a string in `all2all_hip.py`, JIT-built by
torch's `cpp_extension` on first run (concurrent ranks share one build — torch holds a file
lock), and the Triton kernel needs no C++ at all.

## What it does

`recv` is an ordinary `symm_mem.empty()` tensor made symmetric by `symm_mem.rendezvous()`.
`send` is plain local memory. The kernel **pushes**: each rank writes its own chunk into
every peer's receive window, so there are no remote reads and no per-peer handshake — just
one barrier afterwards, before anyone reads what landed.

```python
import mori.allocator          # importing registers the "MORI" backend
symm_mem.set_backend("MORI")

recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
hdl  = symm_mem.rendezvous(recv, group_name)

all2all_push_ptrs(send, hdl.buffer_ptrs_dev, chunk_bytes, rank_id, world_size)
```

`hdl.buffer_ptrs_dev` is the N-entry device array, which is what torch's model provides and
what its own backends force — they map each peer at an unrelated address. This backend does
map every rank into one evenly-strided span, but `base + rank*stride` is deliberately not
exposed: [ROCm/mori#557](https://github.com/ROCm/mori/issues/557) is where the flat window
arrives, as cco's `ccoWindowDevice` rather than as a third scheme invented here. An earlier
version of this example implemented both forms to compare them; they measured within a
couple of percent of each other, so nothing is given up by shipping only the array.

## The Triton version

Same push through the same array, in the idiom torch's own symmetric-memory Triton kernels
use: `buffer_ptrs_dev` goes in as a plain integer and becomes a pointer inside the kernel.

```python
peers = peer_ptrs.to(tl.pointer_type(tl.uint64))
dst = tl.load(peers + peer).to(tl.pointer_type(tl.int32))
```

The grid is the same `(world_size * blocks_per_peer,)`, and `BLOCK=1024` int32 over 4 warps
is the same dwordx4-per-lane store the HIP kernel does. One Triton detail worth knowing:
the kernel is declared `@triton.jit(do_not_specialize=["chunk_elems", "rank_id",
"blocks_per_peer"])`, because Triton turns an `int` argument that happens to equal `1` into
a `constexpr` — without it, **rank 1 alone** fails to compile.

## The LSA version

`all2all_lsa.py` is the same push again, with the peer address *computed* instead of
loaded. It is the one file here that does use cco: `register_external_window()` aliases the
tensor's VMM handle into cco's flat LSA space, and the kernel asks the window for the
address.

```python
hdl      = symm_mem.rendezvous(recv, group_name)          # registers the cco window
window   = mori.allocator.window_handle(recv)             # where to write
dev_comm = mori.allocator.dev_comm(group_name, key="all2all_lsa")   # who I am
...
my_lsa = cco.DevComm.lsa_rank(dev_comm)                   # inside the kernel
addr   = cco.Window.lsa_ptr(window, peer, my_lsa * chunk_bytes)
```

Both indices `lsa_ptr` takes are **LSA** ranks. On one node they equal world ranks, which
is why passing `rank_id` in from Python also works today and would quietly stop working
with a second node. The DevComm is what answers it properly, and it is cached per
`(group, key)` because the resources a collective needs — signal counts, QP counts,
connection type — belong to that collective, not to the group.

No `Communicator` is built: `rendezvous()` *is* the cco window registration, and the
backend keeps one communicator per process group. `window_handle()` is the backend's own
accessor -- torch's handle has nowhere to carry a window -- and it mirrors
`NCCLSymmetricMemory::get_window()`. The example also uses `hdl.barrier(0)`, the backend's
barrier over cco's, in place of `dist.barrier()`.

It measures the same as the pointer array — 8×gfx950, 4 MiB per peer, three repeats each:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| `all2all_lsa.py` | 1924.7 | 1964.3 | 1862.9 GB/s |
| `all2all_triton.py` | 1944.5 | 1922.3 | 1914.4 GB/s |

which is the point: the reason to write against the window is not throughput. An address
array can only describe peers that are directly load/store-able, while `ccoWindowDevice`
also carries the RDMA MR for peers that are not, so this is the addressing mode a
scale-out version has to be built on. See [ROCm/mori#557](https://github.com/ROCm/mori/issues/557).

Use the same `blocks_per_peer` heuristic as its siblings. An earlier draft capped it at 16
and lost ~6% at 8 ranks, which reads exactly like an addressing-mode difference and is not
one.

## Measured

Aggregate counts only the `(world_size-1)` chunks that leave the device; the self chunk
stays local. Every run is correctness-checked first.

4 MiB per peer:

| ranks | MI355X / gfx950 | MI355X-class / gfx1250 | MI308X / gfx942 |
|---|---|---|---|
| 2 | 107.8 GB/s | 514.6 GB/s | 54.1 GB/s |
| 4 | 500-567 GB/s | 1705.1 GB/s | 112.2 GB/s |
| 8 | 1746-1874 GB/s | — | 184.8 GB/s |

256 KiB per peer, where launch and barrier cost still shows:

| ranks | gfx950 | gfx1250 | gfx942 |
|---|---|---|---|
| 2 | 76-84 GB/s | 92.9 GB/s | 40.3 GB/s |
| 4 | 461-485 GB/s | 516.8 GB/s | 139.4 GB/s |
| 8 | 1522-1829 GB/s | — | 250.0 GB/s |

Ranges are across repeats: anything above 2 ranks moves 6-13% run to run, so read the
single-value columns as ±10% too. The gfx950 column is from the code as it stands; gfx1250
and gfx942 were measured before the example was split in two, with the same kernel and grid.
gfx1250 exports **fabric** handles and its column is from an idle box; gfx950 and gfx942
fall back to POSIX fd, having no fabric support at `hipMemCreate`. The kernel sees the same
window either way. The gfx1250 box has 4 GPUs, hence no 8-rank column.

Grid shape matters more than the handle type. An earlier version launched one block per
destination rank, leaving all but `world_size` CUs idle and unable to keep enough writes in
flight to cover interconnect latency: 15.8 GB/s on gfx1250 and 745 GB/s on gfx950 at the
same 4 MiB payload. Splitting each chunk across `blocks_per_peer` blocks is worth ~2.5x on
gfx950 and ~95x on gfx1250.

Uncached/fine-grained windows, as mori's cco windows are, were measured and rejected: half
the bandwidth on gfx1250 (712 vs 1499 GB/s at 4 ranks) and no change on gfx950. The backend
uses coarse-grained pinned memory.

### HIP vs Triton

Same grid, same block size, same pointer array. us/iter, range over repeats.

8×gfx950, torch 2.10 / Triton 3.7, POSIX-fd handles:

| chunk per peer | HIP | Triton |
|---|---|---|
| 4 MiB | 128.9-140.6 us | 118.6-139.7 us |
| 1 MiB | 29.7-30.3 us | 30.1-33.4 us |
| 256 KiB | 7.9-9.6 us | 9.1-10.0 us |

4×gfx1250, torch 2.11 / Triton 3.8, fabric handles, idle box:

| chunk per peer | HIP | Triton |
|---|---|---|
| 4 MiB | 29.6-30.5 us | 32.2-32.5 us |
| 1 MiB | 10.0-10.1 us | 11.9-12.6 us |
| 256 KiB | 5.9-6.2 us | 12.0-12.1 us |

The Triton kernel needs no change between the two — wave64 gfx950 and wave32 gfx1250 both
just work, and forcing 256 lanes on gfx1250 with `num_warps=8` moves nothing.

**Where Triton loses, it loses on the host, not on the device.** Launches are asynchronous,
so a steady-state iteration costs `max(kernel, launch)`, and the Python launcher is the
larger of the two for a long way up. Timing the launch call alone, 500 calls with no sync:

| | HIP, pybind | Triton |
|---|---|---|
| gfx950, torch 2.10 / Triton 3.7 | 2.90 us | 9.01 us |
| gfx1250, torch 2.11 / Triton 3.8 | 2.44-3.03 us | 10.84-10.94 us |

That predicts both tables. On gfx1250 the Triton row is pinned near 12 us at 256 KiB *and*
at 1 MiB despite 4x the data — it is not moving data, it is waiting on the launcher — and
only at 4 MiB, where the kernel runs ~30 us, does it come within 10%. Adding a
`torch.cuda.synchronize()` to the launch benchmark costs Triton 0.03 us and HIP 1.7-1.9 us:
the device is already idle when the Triton loop ends, because the host cannot feed it fast
enough to build a queue. On gfx950 the 9 us launcher sits just under the 8-10 us kernel, so
the two are close at every size measured.

Worth remembering before reading anything into a small-message Triton number on a
symmetric-memory benchmark: it may be measuring `triton.JITFunction.run`. Real uses amortise
that — CUDA graphs, or a persistent kernel that does many transfers per launch.

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

Possibly a missing kernel option rather than silicon: this box runs a 5.10 kernel with a
DKMS backport and has `CONFIG_PCI_P2PDMA` and `CONFIG_DMABUF_MOVE_NOTIFY` unset, both of
which the DRM cross-device path wants, while the unaffected gfx950 box (6.8) has both.
Untested — it needs a rebuilt kernel or a modern-kernel MI300.

The escape hatch is what aiter's custom allreduce does: `hipMalloc` + hipIpc stays on the
KFD path and keeps full bandwidth, at the cost of scattered peer pointers — which, since
this backend exposes the pointer array anyway, kernels would not notice.

## Notes

`dist.barrier()` is used between the kernel and the reads because the examples predate the
backend's own `barrier()`, which now runs over cco's host barrier. The signal pad is always
present -- its own cco window rather than a tail on the buffer -- so torch's `symm_mem`
collectives work without a build flag. `put_signal`/`wait_signal` still raise: they need a
`ccoDevComm`, which the backend does not create yet.
