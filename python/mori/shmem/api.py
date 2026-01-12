# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from mori import cpp as mori_cpp

# Initialization flags
MORI_SHMEM_INIT_WITH_MPI_COMM = mori_cpp.MORI_SHMEM_INIT_WITH_MPI_COMM
MORI_SHMEM_INIT_WITH_UNIQUEID = mori_cpp.MORI_SHMEM_INIT_WITH_UNIQUEID

# Traditional initialization (requires PyTorch distributed)
def shmem_torch_process_group_init(group_name: str):
    """Initialize shmem from PyTorch process group.
    
    Args:
        group_name: Name of the PyTorch process group
        
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_torch_process_group_init(group_name)


# UniqueId-based initialization (nvshmem/rocshmem compatible)
def shmem_get_unique_id() -> bytes:
    """Get a unique ID for shmem initialization.
    
    This should be called by rank 0 and broadcast to all ranks.
    
    Returns:
        Unique ID as bytes (128 bytes)
    """
    return mori_cpp.shmem_get_unique_id()


def shmem_init_attr(flags: int, rank: int, nranks: int, unique_id: bytes):
    """Initialize shmem with attributes using unique ID.
    
    This allows initialization without PyTorch distributed or MPI.
    
    Args:
        flags: Initialization flags (use MORI_SHMEM_INIT_WITH_UNIQUEID)
        rank: My rank/PE ID
        nranks: Total number of ranks/PEs
        unique_id: Unique ID from shmem_get_unique_id()
        
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_init_attr(flags, rank, nranks, unique_id)


def shmem_finalize():
    """Finalize shmem and cleanup resources.
    
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_finalize()


def shmem_module_init(hip_module: int):
    """Initialize globalGpuStates in a specific HIP module.
    
    This is used by Triton to initialize device symbols in dynamically
    compiled kernel modules. It copies the current GpuStates values
    to the module's globalGpuStates symbol.
    
    Args:
        hip_module: HIP module handle (from Triton kernel compilation)
        
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_module_init(hip_module)


# Query APIs
def shmem_mype() -> int:
    """Get my PE (process element) ID.
    
    Returns:
        My PE ID (0 to npes-1)
    """
    return mori_cpp.shmem_mype()


def shmem_npes() -> int:
    """Get total number of PEs.
    
    Returns:
        Total number of PEs
    """
    return mori_cpp.shmem_npes()


# Collective operations
def shmem_barrier_all():
    """Global barrier synchronization.
    
    All PEs must call this function. It blocks until all PEs reach the barrier.
    """
    return mori_cpp.shmem_barrier_all()


# Symmetric memory management
def shmem_malloc(size: int) -> int:
    """Allocate symmetric memory.
    
    This allocates memory that is symmetric across all PEs and can be
    accessed remotely via RDMA operations.
    
    Args:
        size: Size in bytes to allocate
        
    Returns:
        Address of allocated memory as int (use ctypes or data_ptr())
    """
    return mori_cpp.shmem_malloc(size)


def shmem_malloc_align(alignment: int, size: int) -> int:
    """Allocate aligned symmetric memory.
    
    Args:
        alignment: Alignment requirement in bytes (must be power of 2)
        size: Size in bytes to allocate
        
    Returns:
        Address of allocated memory as int
    """
    return mori_cpp.shmem_malloc_align(alignment, size)


def shmem_ext_malloc_with_flags(size: int, flags: int) -> int:
    """Allocate symmetric memory with specific flags.
    
    Args:
        size: Size in bytes to allocate
        flags: Allocation flags
        
    Returns:
        Address of allocated memory as int
    """
    return mori_cpp.shmem_ext_malloc_with_flags(size, flags)


def shmem_free(ptr: int):
    """Free symmetric memory.
    
    Args:
        ptr: Address of memory to free (as returned by shmem_malloc)
    """
    return mori_cpp.shmem_free(ptr)


# Buffer registration
def shmem_buffer_register(ptr: int, size: int) -> int:
    """Register an existing buffer for RDMA operations.
    
    This allows using existing memory (e.g., PyTorch tensors) for
    RDMA operations without allocating new symmetric memory.
    
    Args:
        ptr: Address of buffer to register
        size: Size of buffer in bytes
        
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_buffer_register(ptr, size)


def shmem_buffer_deregister(ptr: int, size: int) -> int:
    """Deregister a buffer from RDMA.
    
    Args:
        ptr: Address of buffer to deregister
        size: Size of buffer in bytes
        
    Returns:
        Status code (0 for success)
    """
    return mori_cpp.shmem_buffer_deregister(ptr, size)


def shmem_ptr_p2p(dest_ptr: int, my_pe: int, dest_pe: int) -> int:
    """Convert local symmetric memory pointer to remote P2P address.
    
    This function translates a local symmetric memory pointer to the corresponding
    P2P (Peer-to-Peer) accessible address on a remote PE. This is useful for
    direct GPU-to-GPU memory access within a node.
    
    Args:
        dest_ptr: Local symmetric memory pointer (as int/uint64)
        my_pe: My PE (process element) ID
        dest_pe: Target PE ID to get P2P address for
        
    Returns:
        - Non-zero P2P address: If connection uses P2P transport (same node GPUs)
        - 0: If connection uses RDMA transport (different nodes) or if pointer is invalid
    """
    return mori_cpp.shmem_ptr_p2p(dest_ptr, my_pe, dest_pe)

def shmem_num_qp_per_pe():
    """Get number of QPs per PE.
    
    Returns:
        Number of QPs per PE
    """
    return mori_cpp.shmem_num_qp_per_pe()
