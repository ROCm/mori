import torch
import torch.distributed as dist
from mori import cpp as mori_cpp

def test_all2all_sdma():
    """正确的多进程All2All测试用例"""
    
    # 1. 初始化分布式环境（通常在外部用torchrun或mpirun启动）
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')  # 或'mpi'
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} starting All2All test...")
    
    # 2. 每个进程准备自己的数据
    local_count = 1024 * 1024  # 每个进程1M元素
    input_tensor = torch.full(
        (local_count,), 
        float(rank),  # 每个进程用它的rank值填充
        dtype=torch.float32, 
        device='cuda'
    )
    
    # 3. 准备输出缓冲区
    total_count = local_count * world_size
    output_tensor = torch.empty(total_count, dtype=torch.float32, device='cuda')
    
    # 4. 确保所有进程就绪（重要！）
    dist.barrier()
    
    # 5. 执行All2All
    duration = mori_cpp.all2all_sdma(
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        local_count,  # 每个进程的元素数量
        0  # 默认流
    )
    
    # 6. 同步等待完成
    torch.cuda.synchronize()
    dist.barrier()
    
    print(f"Rank {rank}: All2All completed in {duration:.6f} seconds")
    
    # 7. 验证结果
    # 验证每个块是否包含正确的数据
    output_reshaped = output_tensor.view(world_size, local_count)
    
    for i in range(world_size):
        expected_value = float(i)  # 第i个块应该来自进程i
        chunk = output_reshaped[i]
        
        if not torch.allclose(chunk, torch.tensor(expected_value, device='cuda')):
            print(f"Rank {rank}: Verification failed for chunk {i}")
            return False
    
    if rank == 0:
        print(f"All {world_size} processes passed verification!")
    
    return True


def benchmark_all2all():
    """All2All性能基准测试"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 测试不同大小的数据
    test_sizes = [
        1 * 1024,        # 1K
        32 * 1024,       # 32K
        1024 * 1024,     # 1M
        8 * 1024 * 1024  # 8M
    ]
    
    if rank == 0:
        print("\n=== All2All SDMA Benchmark ===")
        print(f"World Size: {world_size}")
    
    for size in test_sizes:
        local_count = size
        total_count = local_count * world_size
        
        # 准备数据
        input_tensor = torch.full((local_count,), float(rank), 
                                  dtype=torch.float32, device='cuda')
        output_tensor = torch.empty(total_count, dtype=torch.float32, device='cuda')
        
        # 预热
        for _ in range(3):
            mori_cpp.all2all_sdma(
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                local_count,
                0
            )
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # 实际测试
        durations = []
        for _ in range(10):
            duration = mori_cpp.all2all_sdma(
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                local_count,
                0
            )
            durations.append(duration)
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # 计算统计信息
        if rank == 0:
            avg_duration = sum(durations) / len(durations)
            data_size_bytes = total_count * 4  # float32 = 4 bytes
            bandwidth_gbps = data_size_bytes / avg_duration / (1024**3)
            
            print(f"Size: {size:8d} elements/rank | "
                  f"Total: {total_count:10d} elements | "
                  f"Time: {avg_duration:.6f}s | "
                  f"BW: {bandwidth_gbps:.2f} GB/s")


if __name__ == "__main__":
    # 注意：这个脚本应该用分布式方式启动，例如：
    # torchrun --nproc_per_node=4 test_all2all.py
    # 或
    # mpirun -np 4 python test_all2all.py
    
    # 运行测试
    success = test_all2all_sdma()
    
    if success and dist.get_rank() == 0:
        # 运行基准测试
        benchmark_all2all()