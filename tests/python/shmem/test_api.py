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
import pytest
import mori
import mori.shmem as shmem
from tests.python.utils import TorchDistContext, get_free_port
import torch


def _test_torch_init(rank, world_size, port):
    """Test basic PyTorch-based initialization"""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        assert rank == shmem.shmem_mype()
        assert world_size == shmem.shmem_npes()
        shmem.shmem_finalize()


def _test_shmem_malloc(rank, world_size, port):
    """Test symmetric memory allocation and deallocation"""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        # Test basic malloc
        size = 4096
        ptr = shmem.shmem_malloc(size)
        assert ptr != 0, "shmem_malloc returned NULL"
        
        # Test aligned malloc
        aligned_ptr = shmem.shmem_malloc_align(256, size)
        assert aligned_ptr != 0, "shmem_malloc_align returned NULL"
        assert aligned_ptr % 256 == 0, f"Memory not aligned: 0x{aligned_ptr:x}"
        
        # Barrier before freeing
        shmem.shmem_barrier_all()
        
        # Free memory
        shmem.shmem_free(ptr)
        shmem.shmem_free(aligned_ptr)
        
        shmem.shmem_finalize()


def _test_shmem_barrier(rank, world_size, port):
    """Test global barrier synchronization"""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        # All ranks should pass barrier
        shmem.shmem_barrier_all()
        
        # Allocate memory
        ptr = shmem.shmem_malloc(1024)
        
        # Another barrier
        shmem.shmem_barrier_all()
        
        shmem.shmem_free(ptr)
        shmem.shmem_finalize()


def _test_shmem_barrier_on_stream(rank, world_size, port):
    """Test device barrier on a HIP stream"""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        # Test with a PyTorch stream
        stream = torch.cuda.Stream()
        shmem.shmem_barrier_on_stream(stream)
        stream.synchronize()

        # Test with default (null) stream
        shmem.shmem_barrier_on_stream()
        torch.cuda.synchronize()

        # Host barrier to confirm all PEs completed
        shmem.shmem_barrier_all()

        shmem.shmem_finalize()


def _test_buffer_register(rank, world_size, port):
    """Test buffer registration with PyTorch tensors"""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        # Create a GPU tensor
        tensor = torch.ones(1024, device='cuda', dtype=torch.float32)
        tensor_ptr = tensor.data_ptr()
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Register the tensor buffer
        ret = shmem.shmem_buffer_register(tensor_ptr, tensor_size)
        assert ret == 0, f"shmem_buffer_register failed with code {ret}"
        
        shmem.shmem_barrier_all()
        
        # Deregister the buffer
        ret = shmem.shmem_buffer_deregister(tensor_ptr, tensor_size)
        assert ret == 0, f"shmem_buffer_deregister failed with code {ret}"
        
        shmem.shmem_finalize()


def _test_uniqueid_init(rank, world_size, port):
    """Test UniqueID-based initialization"""
    import os
    import time
    import shutil
    
    # For single-node testing, use loopback interface
    os.environ['MORI_SOCKET_IFNAME'] = 'lo'
    
    uid_dir = f"/tmp/mori_shmem_test_{port}"
    uid_file = os.path.join(uid_dir, "uniqueid")
    ready_file = os.path.join(uid_dir, f"ready_{rank}")
    
    try:
        # Create directory (rank 0 only)
        if rank == 0:
            os.makedirs(uid_dir, exist_ok=True)
            
            # Rank 0 generates unique ID
            unique_id = shmem.shmem_get_unique_id()
            assert len(unique_id) == 128, f"Unique ID should be 128 bytes, got {len(unique_id)}"
            
            # Write to file for other ranks
            with open(uid_file, 'wb') as f:
                f.write(unique_id)
            
            # Signal that file is ready
            with open(ready_file, 'w') as f:
                f.write('ready')
        else:
            # Other ranks wait for directory
            max_wait = 30
            for i in range(max_wait * 10):
                if os.path.exists(uid_dir):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Rank {rank}: Timeout waiting for directory")
            
            # Wait for rank 0's ready signal
            rank0_ready = os.path.join(uid_dir, "ready_0")
            for i in range(max_wait * 10):
                if os.path.exists(rank0_ready) and os.path.exists(uid_file):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Rank {rank}: Timeout waiting for unique ID file")
            
            # Read unique ID
            with open(uid_file, 'rb') as f:
                unique_id = f.read()
        
        # Verify we have the unique_id
        assert len(unique_id) == 128, f"Rank {rank}: Invalid unique ID length: {len(unique_id)}"
        
        # Small delay to avoid thundering herd
        time.sleep(0.01 * rank)
        
        # Initialize with unique ID (no need to wait for other ranks)
        print(f"Rank {rank}: Starting shmem_init_attr...", flush=True)
        ret = shmem.shmem_init_attr(
            shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank,
            world_size,
            unique_id
        )
        print(f"Rank {rank}: shmem_init_attr returned {ret}")
        assert ret == 0, f"shmem_init_attr failed with code {ret}"
        
        # Verify
        my_rank = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_rank == rank, f"Rank mismatch: expected {rank}, got {my_rank}"
        assert npes == world_size, f"World size mismatch: expected {world_size}, got {npes}"
        
        # Test barrier
        print(f"Rank {rank}: Calling shmem_barrier_all...")
        shmem.shmem_barrier_all()
        print(f"Rank {rank}: Barrier passed")
        
        # Test shmem_malloc APIs
        print(f"Rank {rank}: Testing shmem_malloc...")
        ptr1 = shmem.shmem_malloc(4096)
        assert ptr1 != 0, f"Rank {rank}: shmem_malloc returned NULL"
        
        ptr2 = shmem.shmem_malloc_align(256, 8192)
        assert ptr2 != 0, f"Rank {rank}: shmem_malloc_align returned NULL"
        assert ptr2 % 256 == 0, f"Rank {rank}: Memory not aligned: 0x{ptr2:x}"
        
        # Barrier before freeing
        shmem.shmem_barrier_all()
        
        # Free memory
        shmem.shmem_free(ptr1)
        shmem.shmem_free(ptr2)
        print(f"Rank {rank}: shmem_malloc tests passed")
        
        shmem.shmem_finalize()
        print(f"Rank {rank}: Finalized")
        
    finally:
        # Cleanup - rank 0 waits for all others to finish
        if rank == 0:
            # Wait for all ranks to signal ready
            time.sleep(2.0)
            # Clean up directory
            if os.path.exists(uid_dir):
                shutil.rmtree(uid_dir, ignore_errors=True)


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_torch_init(world_size):
    """Test basic initialization"""
    torch.multiprocessing.spawn(
        _test_torch_init,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_shmem_malloc(world_size):
    """Test symmetric memory allocation"""
    torch.multiprocessing.spawn(
        _test_shmem_malloc,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_shmem_barrier(world_size):
    """Test barrier synchronization"""
    torch.multiprocessing.spawn(
        _test_shmem_barrier,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_shmem_barrier_on_stream(world_size):
    """Test device barrier on stream"""
    torch.multiprocessing.spawn(
        _test_shmem_barrier_on_stream,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_buffer_register(world_size):
    """Test buffer registration with PyTorch tensors"""
    torch.multiprocessing.spawn(
        _test_buffer_register,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", (2, 4, 8)) 
def test_uniqueid_init(world_size):
    """Test UniqueID-based initialization (without PyTorch distributed)"""
    torch.multiprocessing.spawn(
        _test_uniqueid_init,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )
