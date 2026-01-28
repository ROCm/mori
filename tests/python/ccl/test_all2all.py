# test_detailed.py
from mpi4py import MPI
import torch
import sys
import os
import ctypes

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mori import cpp as mori_cpp

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"PE {rank}: 详细测试")

# 初始化SHMEM
print(f"PE {rank}: 初始化SHMEM...")
status = mori_cpp.shmem_mpi_init()
print(f"PE {rank}: SHMEM初始化状态: {status}")

# 获取PE信息
my_pe = mori_cpp.shmem_my_pe()
n_pes = mori_cpp.shmem_n_pes()
print(f"PE {rank}: SHMEM PE {my_pe} of {n_pes}")

# 测试内存分配
elems_per_pe = 1
input_size = elems_per_pe
output_size = elems_per_pe * n_pes

print(f"PE {rank}: 分配 {input_size} 输入元素, {output_size} 输出元素")

# 分配GPU内存
input_tensor = torch.zeros(input_size, dtype=torch.int32, device='cuda')
output_tensor = torch.zeros(output_size, dtype=torch.int32, device='cuda')

# 获取指针
input_ptr = input_tensor.data_ptr()
output_ptr = output_tensor.data_ptr()

print(f"PE {rank}: PyTorch指针:")
print(f"  输入: {input_ptr} (0x{input_ptr:x})")
print(f"  输出: {output_ptr} (0x{output_ptr:x})")
print(f"  输入类型: {type(input_ptr)}")
print(f"  输出类型: {type(output_ptr)}")

# 将指针转换为uintptr_t（C++绑定期望的类型）
# 在Python中，我们可以使用ctypes
input_ptr_uint = ctypes.c_uint64(input_ptr).value
output_ptr_uint = ctypes.c_uint64(output_ptr).value

print(f"PE {rank}: 转换为uintptr_t:")
print(f"  输入: {input_ptr_uint} (0x{input_ptr_uint:x})")
print(f"  输出: {output_ptr_uint} (0x{output_ptr_uint:x})")

# 初始化数据
input_tensor.fill_(rank * 100 + 1)
print(f"PE {rank}: 输入数据: {input_tensor.cpu().numpy()}")

# 同步
torch.cuda.synchronize()
comm.Barrier()

print(f"PE {rank}: 调用 all2all_sdma_int32...")

try:
    # 尝试不同的调用方式
    
    # 方式1: 直接传递PyTorch指针
    print(f"\nPE {rank}: 方式1 - 直接传递PyTorch指针")
    result1 = mori_cpp.all2all_sdma_int32(
        input_ptr,      # Python整数
        output_ptr,     # Python整数
        elems_per_pe,
        0
    )
    print(f"PE {rank}: 结果1: {result1}")
    
except Exception as e:
    print(f"PE {rank}: 方式1失败: {e}")
    
    # 方式2: 使用uintptr_t转换
    print(f"\nPE {rank}: 方式2 - 使用uintptr_t转换")
    try:
        result2 = mori_cpp.all2all_sdma_int32(
            input_ptr_uint,
            output_ptr_uint,
            elems_per_pe,
            0
        )
        print(f"PE {rank}: 结果2: {result2}")
    except Exception as e2:
        print(f"PE {rank}: 方式2失败: {e2}")
        
        # 方式3: 使用int()显式转换
        print(f"\nPE {rank}: 方式3 - 使用int()显式转换")
        try:
            result3 = mori_cpp.all2all_sdma_int32(
                int(input_ptr),
                int(output_ptr),
                int(elems_per_pe),
                int(0)
            )
            print(f"PE {rank}: 结果3: {result3}")
        except Exception as e3:
            print(f"PE {rank}: 方式3失败: {e3}")

comm.Barrier()
if rank == 0:
    print("\n=== 测试完成 ===")