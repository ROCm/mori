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
  scale-out header. Provider dispatch is compile-time (per NIC).

  ```cpp
  #include "mori/cco/cco_scale_out.hpp"   // pulls in cco.hpp + RDMA core
  ```

- **Barriers:** the LSA barrier session (`ccoLsaBarrierSession`) and the GDA
  barrier session provide `arrive` / `wait` / `sync` at thread / warp / block
  granularity (`ccoCoopThread` / `ccoCoopWarp` / `ccoCoopBlock`).

FlyDSL kernels can drive the same device API — see the Python examples below.

## 7. Examples

Runnable examples live in [`examples/cco/`](../examples/cco/):

| Example | Lang | Shows |
|---|---|---|
| `python/01_barrier` | py | host barrier across ranks (full lifecycle, no kernel) |
| `python/02_lsa_put` | py | intra-node LSA put via a hand-written `.hip` kernel |
| `python/03_flydsl_put` | py | FlyDSL GDA put + signal/wait |
| `python/04_flydsl_lsa_put` | py | FlyDSL LSA direct peer-pointer store |
| `python/05_flydsl_lsa_allreduce` | py | FlyDSL LSA custom all-reduce |
| `python/06_flydsl_gda_modes` | py | FlyDSL GDA (thread_mode, coop) × signal matrix |
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

Supported GPUs: MI308X / MI300X / MI325X / MI355X. Supported NICs: AMD Pollara
(AINIC), Mellanox ConnectX-7, Broadcom Thor2.
