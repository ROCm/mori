# MORI CCO Guide

**CCO** (Collective Communication Object) is MORI's GPU communication layer
built around an explicit communicator handle. Unlike a process-global singleton,
every `ccoComm` is independently allocated, so a single process can hold multiple
independent communicators and drive them concurrently from multiple threads.

CCO exposes GPU-initiated one-sided communication over three transports:

- **LSA** (Local Symmetric Access) — intra-node peer-to-peer over a flat
  symmetric virtual address space (XGMI). The kernel gets a peer's
  load/store-addressable pointer and writes it directly.
- **GDA** (GPU-Direct Async) — cross-node one-sided RDMA (put / get / signal /
  counter) issued from the device.
- **SDMA** — copy-engine transfers with a device-visible signal pool.
- **ccoProxy** — cross-node RDMA with the GDA window API, where the kernel
  queues each op and a host thread posts it
  (see [ccoProxy](#10-ccoproxy-host-proxy-rdma)).

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Concepts](#1-concepts)
3. [Initialization](#2-initialization)
4. [Memory and Windows](#3-memory-and-windows)
5. [Device Communicator](#4-device-communicator)
6. [Host Barrier](#5-host-barrier)
7. [Device-Side Programming](#6-device-side-programming)
8. [Examples](#7-examples)
9. [C++ API](#8-c-api)
10. [Environment Variables](#9-environment-variables)
11. [ccoProxy (host-proxy RDMA)](#10-ccoproxy-host-proxy-rdma)

## Quick Reference

```python
from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE

# Bootstrap: rank 0 mints a unique id, broadcast it out-of-band (MPI / torch.dist)
uid = Communicator.get_unique_id() if rank == 0 else None
uid = comm_mpi.bcast(uid, root=0)

# Create the communicator (reserves per-rank flat VMM)
with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
    # Allocate symmetric memory and register a P2P/RDMA window over it
    mem = comm.alloc_mem(4096)
    win = comm.register_window(mem.ptr, mem.size)

    # Create a device communicator (pass into kernels)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE
    reqs.lsa_barrier_count = 1
    dc = comm.create_dev_comm(reqs)

    comm.barrier()   # collective host barrier
# resources are released on context exit
```

Run with any launcher that can broadcast the unique id:

```bash
mpirun -np 8 python main.py         # mpi4py bootstrap
# or torchrun --standalone --nproc_per_node=8 main.py  (torch.distributed gloo)
```

## 1. Concepts

### Communicator (`ccoComm`)
An explicit handle created collectively by all ranks. Holds the flat symmetric
VA reservation, the per-rank slot allocator, intra-node topology, and transport
resources. Multiple communicators can coexist in one process.

### Symmetric memory and windows
`alloc_mem(size)` reserves GPU memory inside the communicator's flat VA at the
same offset on every rank. `register_window(ptr, size)` makes that region
peer-reachable: it P2P-maps the region to intra-node peers and registers an RDMA
memory region for cross-node peers. A registered window is therefore reachable
by **both** LSA (intra-node) and GDA (cross-node) with no extra setup.

### Device communicator (`ccoDevComm`)
A trivially-copyable host struct filled by `create_dev_comm`. It carries device
pointers plus topology and per-session resources (signal/counter pools,
barriers). Pass it **by value** into kernels — it lands in kernel-argument space,
so kernels read it without a GPU-memory dereference.

### Teams
Logical rank-subset descriptors (`world`, `lsa`, `cross-node`, `rail`) that let
device code address peers without hard-coding topology.

## 2. Initialization

CCO uses a self-contained socket bootstrap that needs only a 128-byte unique id.
Rank 0 generates it; you broadcast it to all ranks with any out-of-band channel
(MPI, `torch.distributed`, a file), then every rank calls `init`.

```python
from mpi4py import MPI
from mori.cco import Communicator

comm_mpi = MPI.COMM_WORLD
rank, nranks = comm_mpi.Get_rank(), comm_mpi.Get_size()

uid = Communicator.get_unique_id() if rank == 0 else None
uid = comm_mpi.bcast(uid, root=0)

comm = Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024)
# ... use comm ...
comm.destroy()   # or use `with Communicator.init(...) as comm:`
```

The rendezvous network interface is selected via `MORI_SOCKET_IFNAME`
(see [Environment Variables](#9-environment-variables)).

## 3. Memory and Windows

```python
mem = comm.alloc_mem(size)          # AllocatedMemory: .ptr, .size
win = comm.register_window(mem.ptr, mem.size)   # RegisteredWindow: .handle, .local_ptr
```

- `alloc_mem` / `register_window` are the two-step form; both track the resource
  on the communicator, so they are freed automatically on `comm.destroy()` (or
  context exit).
- Window registration is **collective**: all ranks must call it in the same order
  with the same size.
- `win.handle` is the device-side window handle to hand to kernels;
  `win.local_ptr` is this rank's local pointer into the window.

## 4. Device Communicator

`create_dev_comm` allocates the per-session device resources described by a
`CCODevCommRequirements`. Common fields:

| Field | Meaning |
|---|---|
| `gda_connection_type` | `GDA_CONNECTION_NONE` / `CROSSNODE` / `FULL` / `RAIL` — which peers get RDMA QPs |
| `gda_signal_count` | # of GDA signal slots (completion signalling) |
| `gda_counter_count` | # of GDA counter slots |
| `lsa_barrier_count` | # of intra-node (LSA) barriers |
| `sdma_queue_count` | # of SDMA queues (0 = default) |

```python
reqs = CCODevCommRequirements()          # safe defaults
reqs.gda_connection_type = GDA_CONNECTION_CROSSNODE
reqs.gda_signal_count = 128
dc = comm.create_dev_comm(reqs)
# dc.rank / dc.world_size / dc.lsa_size / dc.lsa_rank query topology
```

## 5. Host Barrier

```python
comm.barrier()   # collective across all ranks in the communicator
```

## 6. Device-Side Programming

Device kernels include the CCO header directly and take the `ccoDevComm` by value.

- **LSA (intra-node):** get a peer's directly-addressable pointer and
  load/store it in the kernel.

  ```cpp
  #include "mori/cco/cco.hpp"
  // win: ccoWindow_t obtained on the host
  void* peer = ccoGetLsaPeerPtr(win, peerLsaRank, offset);
  // *peer is a normal load/store target
  ```

- **GDA (cross-node):** GPU-initiated RDMA put/get with signalling, via the
  scale-out header. Provider dispatch is compile-time (per NIC). A handle's
  doorbell mode is compile-time too, `ccoGda<P, ccoGdaDoorbellMode>`:
  `Immediate` (default) posts ring for themselves, `Deferred` posts are released
  by the next ring that reaches them, and `CallerOrdered` posts are released
  only by that mode's flush, for callers that order every post before it.
  Handles on one context share its QPs whatever their mode; see
  `ccoGdaDoorbellMode` for the rules.

  ```cpp
  #include "mori/cco/cco_scale_out.hpp"   // pulls in cco.hpp + RDMA core
  ```

- **Barriers:** the LSA barrier session (`ccoLsaBarrierSession`) provides
  `arrive` / `wait` / `sync` at thread / warp / block granularity
  (`ccoCoopThread` / `ccoCoopWarp` / `ccoCoopBlock`); the GDA barrier session
  (`ccoGdaBarrierSession`) provides `sync` at warp or block granularity on an
  Immediate `ccoGda<P>` handle.

FlyDSL kernels can drive the same device API — see the Python examples below.

## 7. Examples

Runnable examples live in [`examples/cco/`](https://github.com/ROCm/mori/blob/main/examples/cco/):

| Example | Lang | Shows |
|---|---|---|
| `python/01_barrier` | py | host barrier across ranks (full lifecycle, no kernel) |
| `python/02_lsa_put` | py | intra-node LSA put via a hand-written `.hip` kernel |
| `python/03_flydsl_put` | py | FlyDSL GDA put + signal/wait |
| `python/04_flydsl_lsa_put` | py | FlyDSL LSA direct peer-pointer store |
| `python/05_flydsl_lsa_allreduce` | py | FlyDSL LSA custom all-reduce |
| `python/06_flydsl_gda_modes` | py | FlyDSL GDA (thread_mode, coop) × signal matrix, plus DEFERRED / CALLER_ORDERED handles |
| `cpp/01_lsa_put.cpp` | c++ | intra-node LSA put (includes only `cco.hpp`) |
| `cpp/02_gda_put.cpp` | c++ | GPU-initiated RDMA put + signal/wait (`cco_scale_out.hpp`) |

## 8. C++ API

The host control plane and device layer are a self-contained header pair;
include `cco.hpp` for host + LSA, or `cco_scale_out.hpp` for GDA.

```cpp
#include "mori/cco/cco.hpp"

ccoUniqueId id;
if (rank == 0) ccoGetUniqueId(&id);
/* broadcast id to all ranks */
ccoComm* comm;
ccoCommCreate(id, nRanks, rank, perRankVmmSize, &comm);

void* ptr;
ccoMemAlloc(comm, size, &ptr);
ccoWindow_t win;
ccoWindowRegister(comm, ptr, size, &win);   // or the size-only overload (alloc+register)

ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
ccoDevComm devComm;
ccoDevCommCreate(comm, &reqs, &devComm);     // pass devComm by value into kernels

ccoBarrierAll(comm);

ccoDevCommDestroy(comm, &devComm);
ccoWindowDeregister(comm, win);
ccoMemFree(comm, ptr);
ccoCommDestroy(comm);
```

## 9. Environment Variables

| Variable | Purpose |
|---|---|
| `MORI_SOCKET_IFNAME` | Network interface for the unique-id socket rendezvous |
| `MORI_RDMA_DEVICES` | Comma-separated RDMA devices to use (GDA) |
| `MORI_RDMA_TC` | RDMA traffic class (GDA) |
| `MORI_RDMA_SL` | RDMA service level (GDA) |
| `MORI_IGNORE_CPU_AFFINITY` | `1` skips the CPU bind done at communicator creation, leaving placement to an outer `numactl`/`torchrun` |
| `MORI_CPU_AFFINITY_NO_SPLIT` | `1` binds to the whole NUMA node instead of this rank's own slice of it |

Communicator creation (`ccoCommCreate`) pins the calling thread to the CPUs
local to its current HIP device before the bootstrap and any worker thread, so
new threads inherit the affinity. The GPU-local CPU list (sysfs
`local_cpulist`) is intersected with the existing cpuset, so cgroup limits and
an outer binding are respected. Within that set the CPUs are grouped into
physical cores by `thread_siblings_list` and partitioned into one disjoint
slice per GPU on the NUMA node, so two ranks cannot end up on the two SMT
siblings of one core. The split applies only when the process sees more than
one GPU on that NUMA node and there is at least one physical core per GPU;
otherwise it falls back to binding the whole node.

Supported GPUs: MI308X / MI300X / MI325X / MI355X. Supported NICs: AMD Pollara
(AINIC), Mellanox ConnectX-7, Broadcom Thor2.

## 10. ccoProxy (host-proxy RDMA)

`ccoProxy` (in `mori/cco/cco_scale_out.hpp`) is a cross-node transport with the same
window, signal and counter model as GDA, but the GPU does not drive the NIC.
A kernel pushes each op as a small descriptor into a device-to-host queue
(KIWI); host proxy threads pop it and post the RDMA work request on
host-created RC QPs. Keys and remote offsets are resolved on the GPU exactly as
for GDA, so the same windows and signal slots work for both.

Use it where GPU-initiated RDMA is unavailable or slow, or where NIC atomics
into GPU memory fail, e.g. ionic VFs in VMs. Its signal-set ops
(`ccoGda_SignalSet` / `ccoGda_WindowSignalSet`) are plain RDMA writes, so a
proxy DevComm can run with no NIC atomics at all. Its cost is host CPU: each
DevComm with proxy queues runs busy-polling proxy threads.

### Enabling it

- **Build:** `BUILD_CCO_PROXY` (CMake option and `setup.py` env var) is `ON` by
  default. The host engine is built when `GPU_TARGETS` contains gfx942 or
  gfx950; at run time `ccoDevCommCreate` accepts proxy queues only on those
  GPUs. `MORI_CCO_PROXY` is defined when the engine is built in.
- **Requirements:** proxy queues are requested per DevComm. All ranks must pass
  the same queue, thread and QP counts and `proxyOnly`; creation fails on every
  rank otherwise.

| Field (C++ / Python) | Meaning |
|---|---|
| `proxySpscQueueCount` / `proxy_spsc_queue_count` | SPSC queues (one wavefront at a time per queue) |
| `proxyMpscQueueCount` / `proxy_mpsc_queue_count` | MPSC queues (any number of wavefronts) |
| `proxyThreadCount` / `proxy_thread_count` | host proxy threads (0 = 4), capped at the queue count |
| `proxyQueueCapacity` / `proxy_queue_capacity` | ring size in 64-byte lines, power of two (0 = 4096) |
| `proxyQpsPerPeer` / `proxy_qps_per_peer` | RC QPs per (proxy thread, peer); 0 = `MORI_CCO_PROXY_QPS_PER_PEER`, else 4 |
| `proxyOnly` / `proxy_only` | `1`: create no GDA QPs; `ccoGda` on this DevComm traps |

`gdaConnectionType` must resolve to a non-NONE peer set (the proxy reaches
those peers and uses the GDA signal slots). A proxy-only DevComm cannot have
GDA barriers (`railGdaBarrierCount` / `barrierCount`) when there are cross-node
peers. The GPU's VRAM must be host-mapped (large BAR). Each DevComm with proxy
queues starts its own busy-polling threads, so prefer one per communicator.
Several QPs per peer let NICs that serve each QP serially run a thread's queues
in parallel.

| Variable | Purpose |
|---|---|
| `MORI_CCO_DISABLE_PROXY` | `1` makes `ccoDevCommCreate` reject proxy queues |
| `MORI_CCO_PROXY_CPUS` | cpulist for the proxy threads (thread t on entry t mod size); default: GPU-local cores in the affinity mask, one physical core per thread, from the top, skipping the core of CPU 0 |
| `MORI_CCO_PROXY_QPS_PER_PEER` | QPs per (proxy thread, peer) when `proxyQpsPerPeer` is 0 |
| `MORI_CCO_PROXY_ABORT_ON_ERROR` | `1` aborts the process at the first proxy error |

### Device API

`ccoProxy<Queue = SPSCD2HQueue, Fence = ccoProxyFenceEachOp>` is constructed in
the kernel from the DevComm and a queue: an index, or a selector.

| Call | Does |
|---|---|
| `ccoProxy<>(dc, i)` / `ccoProxy<MPSCD2HQueue>(dc, i)` | SPSC queue `i` (out of range traps); MPSC queue `i` mod the pool size |
| `ccoProxy<>(dc, ccoProxyQueuePerWave)` / `ccoProxyQueuePerBlock` | the wavefront's / block's linear id as the SPSC index |
| `put(peer, dstWin, dstOff, srcWin, srcOff, bytes, action, coop)` | RDMA write, optionally followed by a signal on the same QP; split above the port's max message size |
| `putValue(peer, dstWin, dstOff, value, action, coop)` | inline write of a 1–8 byte value, optionally followed by a signal |
| `signal(peer, action, coop)` | signal only |
| action | `ccoGda_NoSignal`, `ccoGda_SignalInc` / `SignalAdd` / `WindowSignalAdd` (NIC atomics), `ccoGda_SignalSet` / `WindowSignalSet` (8-byte write, no atomic) |
| `flush(coop)` | wait until every op this queue pushed before it completed remotely; sources are reusable |
| `int flush(coop, timeoutCycles)` | never traps: 0 done, 1 timed out, 2 DevComm failed |
| `flushAsync(peer, &req, coop)` / `wait(req, coop[, timeoutCycles])` | push the flush now, wait later; requests can be waited in any order |
| `counter(ccoGda_CounterInc{id}, coop)` | `flush`, then add 1 to a local counter; `readCounter` / `waitCounter` / `resetCounter` |
| `readSignal` / `waitSignal` / `resetSignal` | local signal slots, shared with GDA |
| `fence(coop)` | release fence for `ccoProxyFenceCaller` handles |
| `failed()` | whether the DevComm has failed |

- **Coops:** `ccoCoopWarp` and `ccoCoopBlock` issue one op per group (all lanes
  pass the same arguments); `ccoCoopLaneMasked{active}` issues one op per
  active lane (`put` / `putValue` / `signal` only). The ops reject
  `ccoCoopThread` at compile time. The issuing wavefront (wavefront 0 for `ccoCoopBlock`) must call
  with all 64 lanes active from converged code; a partial wavefront traps.
  `flush(ccoCoopBlock)` also flushes every other wavefront of the block that
  pushed through its own handle.
- **Fence modes:** `ccoProxyFenceEachOp` (default) fences before every push, so
  earlier stores are visible to the NIC. `ccoProxyFenceCaller` skips that; call
  `fence()` after writing data the next puts read or signals publish.
  `waitSignal`, `waitCounter`, `flush` and `wait` end with an acquire; after
  `readSignal` or polling a window, issue it yourself.
- **SPSC queues:** one wavefront owns a queue at a time. The handle's
  destructor (or `release()`) saves its state; to hand the queue to another
  wavefront in the same kernel, order the old owner's release before the new
  owner's constructor (`__syncthreads()` within a block). Across launches,
  stream order is enough.

### Ordering and failure

- Ops one wavefront pushes to one queue reach each peer in program order on one
  QP, so a put followed by a signal (or a put carrying a signal) to the same
  peer is delivered data first. Different queues are unordered, as different
  GDA contexts are.
- `SignalSet` / `WindowSignalSet`: every set of a slot goes through one queue,
  values only increase, and the slot gets no `SignalInc` / `SignalAdd`. The
  window MRs must not use relaxed ordering; `ccoDevCommCreate` warns when
  `MORI_IB_ENABLE_RELAXED_ORDERING=1` applies it.
- The first proxy error fails the DevComm for good: later ops are dropped,
  `flush`, `wait`, `counter`, `waitSignal` and `waitCounter` trap, the timeout
  forms return 2, and `failed()` returns true. On the host,
  `ccoDevCommGetError(comm, &dc, &err)` returns 1 and the first error (code,
  peer, queue, op, completion status, message); it is safe to call while
  kernels run.

### Example

```cpp
#include "mori/cco/cco_scale_out.hpp"

// Host: reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
//       reqs.proxySpscQueueCount = nBlocks; reqs.gdaSignalCount >= nBlocks;
//       optionally reqs.proxyOnly = 1.
__global__ void send(ccoDevComm dc, ccoWindow_t dst, ccoWindow_t src, size_t chunk, int peer) {
  ccoProxy<> proxy{dc, ccoProxyQueuePerBlock};
  const size_t off = blockIdx.x * chunk;
  proxy.put(peer, dst, off, src, off, chunk, ccoGda_SignalInc{blockIdx.x}, ccoCoopBlock{});
  proxy.flush(ccoCoopBlock{});  // src may be overwritten now
}

__global__ void recv(ccoDevComm dc, uint64_t round) {
  ccoProxy<> proxy{dc, ccoProxyQueuePerBlock};
  proxy.waitSignal(blockIdx.x, round, ccoCoopBlock{});  // chunk blockIdx.x has landed
}
```
