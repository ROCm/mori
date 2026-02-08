# MORI Shmem Guide

MORI Shmem provides OpenSHMEM-style symmetric memory APIs for GPU memory management and RDMA communication. It is the foundation layer that MORI-EP and MORI-IO build upon — shmem must be initialized before using any other MORI component.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. Concepts](#1-concepts)
- [2. Initialization](#2-initialization)
- [3. Query APIs](#3-query-apis)
- [4. Memory Management](#4-memory-management)
- [5. P2P Address Translation](#5-p2p-address-translation)
- [6. Synchronization](#6-synchronization)
- [7. HIP Module Init (Triton Integration)](#7-hip-module-init-triton-integration)
- [8. Initialization Flags](#8-initialization-flags)
- [Environment Variables](#environment-variables)
- [Source Files](#source-files)

## Quick Reference

```python
import torch
import mori

# Prerequisite: torch.distributed must be initialized and a process group
# registered under "default". See section 2. Initialization for details.
mori.shmem.shmem_torch_process_group_init("default")

# Query
my_rank = mori.shmem.shmem_mype()
num_ranks = mori.shmem.shmem_npes()

# Allocate symmetric memory
ptr = mori.shmem.shmem_malloc(size_in_bytes)

# Register existing buffer for RDMA
mori.shmem.shmem_buffer_register(tensor.data_ptr(), tensor.nbytes)

# P2P address translation (intra-node)
remote_ptr = mori.shmem.shmem_ptr_p2p(ptr, my_pe, dest_pe)

# Synchronize
mori.shmem.shmem_barrier_all()

# Cleanup
mori.shmem.shmem_free(ptr)
mori.shmem.shmem_finalize()
```

**Imports:**

| What | Import |
|------|--------|
| All shmem APIs | `import mori.shmem` or `from mori import shmem` |
| Init flags | `mori.shmem.MORI_SHMEM_INIT_WITH_MPI_COMM`, `mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID` |

---

## 1. Concepts

### Symmetric Memory

Symmetric memory is GPU memory that is allocated at the same virtual offset across all participating PEs (Processing Elements / ranks). This enables RDMA hardware to directly access remote GPU memory using simple address arithmetic — no address translation tables needed at runtime.

### Processing Element (PE)

A PE is a participant in the symmetric memory domain. Each GPU rank maps to one PE. PEs are numbered 0 to N-1.

### Symmetric Heap

The symmetric heap is a pre-allocated region of GPU memory from which `shmem_malloc` allocates. Its size is controlled by the `MORI_SHMEM_HEAP_SIZE` environment variable:

```bash
export MORI_SHMEM_HEAP_SIZE=6G  # Must be set before shmem init
```

---

## 2. Initialization

Shmem must be initialized exactly once per process. There are two methods:

### Method 1: PyTorch Process Group (Recommended)

Use this when PyTorch distributed is already initialized:

```python
import torch
import torch.distributed as dist
import mori

# Standard PyTorch distributed init
dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)

# Register the process group with a name
world_group = dist.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)

# Initialize shmem from the registered process group
mori.shmem.shmem_torch_process_group_init("default")
```

### Method 2: Unique ID (Standalone)

Use this when PyTorch distributed is not available (e.g., MPI-only environments, C++ applications):

```python
import mori

# Rank 0 generates a unique ID
if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()  # Returns 128 bytes
    # Broadcast unique_id to all ranks via your transport (MPI, TCP, file, etc.)

# All ranks initialize with the same unique ID
mori.shmem.shmem_init_attr(
    mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
    rank,        # My rank
    world_size,  # Total ranks
    unique_id,   # Shared unique ID (bytes)
)
```

### Finalization

Always finalize shmem before process exit:

```python
mori.shmem.shmem_finalize()
# Then destroy PyTorch process group if applicable
dist.destroy_process_group()
```

---

## 3. Query APIs

```python
# Get my PE (rank) ID — 0 to npes-1
my_pe = mori.shmem.shmem_mype()

# Get total number of PEs
total_pes = mori.shmem.shmem_npes()

# Get number of RDMA queue pairs per PE
num_qp = mori.shmem.shmem_num_qp_per_pe()
```

---

## 4. Memory Management

### Allocating Symmetric Memory

```python
# Basic allocation
ptr = mori.shmem.shmem_malloc(size_in_bytes)

# Aligned allocation (alignment must be power of 2)
ptr = mori.shmem.shmem_malloc_align(alignment=256, size=size_in_bytes)

# Allocation with flags
ptr = mori.shmem.shmem_ext_malloc_with_flags(size_in_bytes, flags)
```

All allocation functions return an integer address (`int`). The allocated memory is symmetric — the same offset is reserved on every PE.

### Freeing Symmetric Memory

```python
mori.shmem.shmem_free(ptr)
```

### Registering Existing Buffers

Register a PyTorch tensor or other existing GPU memory for RDMA operations without allocating new symmetric memory:

```python
# Register
tensor = torch.zeros(1024, 7168, dtype=torch.bfloat16, device="cuda")
mori.shmem.shmem_buffer_register(tensor.data_ptr(), tensor.nbytes)

# ... use tensor in MORI operations ...

# Deregister when done
mori.shmem.shmem_buffer_deregister(tensor.data_ptr(), tensor.nbytes)
```

Both functions return a status code (0 for success).

---

## 5. P2P Address Translation

For intra-node GPU-to-GPU access, translate a local symmetric pointer to its P2P-accessible address on a remote PE:

```python
remote_ptr = mori.shmem.shmem_ptr_p2p(local_ptr, my_pe, dest_pe)
```

| Return Value | Meaning |
|-------------|---------|
| Non-zero | P2P address on `dest_pe` (same-node, XGMI-connected GPUs) |
| 0 | Connection uses RDMA transport (different nodes) or pointer is invalid |

---

## 6. Synchronization

Global barrier — blocks until all PEs reach the barrier:

```python
mori.shmem.shmem_barrier_all()
```

---

## 7. HIP Module Init (Triton Integration)

When using Triton-compiled kernels that access shmem device symbols, initialize the HIP module:

```python
mori.shmem.shmem_module_init(hip_module_handle)
```

This copies the current GPU states to the `globalGpuStates` symbol in the dynamically compiled Triton kernel module.

---

## 8. Initialization Flags

| Flag | Value | Description |
|------|-------|-------------|
| `MORI_SHMEM_INIT_WITH_MPI_COMM` | 0 | Initialize using MPI communicator |
| `MORI_SHMEM_INIT_WITH_UNIQUEID` | 1 | Initialize using broadcast unique ID |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MORI_SHMEM_HEAP_SIZE` | Symmetric heap size (e.g., `"6G"`, `"2G"`, `"512M"`). Must be set before initialization. |
| `MORI_RDMA_DEVICES` | RDMA NIC selection. Include: `mlx5_0,mlx5_1`. Exclude: `^mlx5_2,mlx5_3` |

---

## Source Files

| File | Description |
|------|-------------|
| `python/mori/shmem/api.py` | Python shmem API |
| `python/mori/shmem/__init__.py` | Public exports |
| `include/mori/shmem/` | C++ shmem headers |
| `src/pybind/mori.cpp` | Python bindings (shmem section) |
