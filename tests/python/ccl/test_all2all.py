# exact_cpp_match.py
from mpi4py import MPI
import sys
import os
import ctypes
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mori import cpp as mori_cpp

# MPI初始化
MPI_Init = None  # mpi4py已经初始化了

# 初始化SHMEM
status = mori_cpp.shmem_mpi_init()
print(f"SHMEM init status: {status}")

my_pe = mori_cpp.shmem_my_pe()
n_pes = mori_cpp.shmem_n_pes()
print(f"PE {my_pe} of {n_pes} started")

# 配置（和C++示例完全一样）
elems_per_pe = 8 * 1024 * 1024  # 8M个元素
bytes_per_pe = elems_per_pe * 4  # sizeof(uint32_t) = 4
total_bytes = bytes_per_pe * n_pes

print(f"Each PE contributes {elems_per_pe} elements")
print(f"bytesPerPe: {bytes_per_pe}, totalBytes: {total_bytes}")

# 加载HIP库
libhip = ctypes.CDLL("libamdhip64.so")

# 定义函数
hipExtMallocWithFlags = libhip.hipExtMallocWithFlags
hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
hipExtMallocWithFlags.restype = ctypes.c_int

hipFree = libhip.hipFree
hipFree.argtypes = [ctypes.c_void_p]
hipFree.restype = ctypes.c_int

hipMemcpy = libhip.hipMemcpy
hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hipMemcpy.restype = ctypes.c_int

hipStreamCreate = libhip.hipStreamCreate
hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
hipStreamCreate.restype = ctypes.c_int

hipStreamDestroy = libhip.hipStreamDestroy
hipStreamDestroy.argtypes = [ctypes.c_void_p]
hipStreamDestroy.restype = ctypes.c_int

hipDeviceSynchronize = libhip.hipDeviceSynchronize
hipDeviceSynchronize.argtypes = []
hipDeviceSynchronize.restype = ctypes.c_int

# 常量
hipDeviceMallocUncached = 0x1  # 需要确认正确值
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2

# 1. 分配设备内存（和C++完全一样）
print(f"PE {my_pe}: Allocating device memory...")
out_put_buff = ctypes.c_void_p()
in_put_buff = ctypes.c_void_p()

# 输出缓冲区
status1 = hipExtMallocWithFlags(ctypes.byref(out_put_buff), total_bytes, hipDeviceMallocUncached)
if status1 != 0:
    print(f"PE {my_pe}: ERROR: hipExtMallocWithFlags for output failed: {status1}")
    sys.exit(1)

# 输入缓冲区
status2 = hipExtMallocWithFlags(ctypes.byref(in_put_buff), total_bytes, hipDeviceMallocUncached)
if status2 != 0:
    print(f"PE {my_pe}: ERROR: hipExtMallocWithFlags for input failed: {status2}")
    hipFree(out_put_buff)
    sys.exit(1)

print(f"PE {my_pe}: Device memory allocated: input={in_put_buff.value:#x}, output={out_put_buff.value:#x}")

# 2. 初始化数据（和C++完全一样）
print(f"PE {my_pe}: Initializing data...")
host_data = np.zeros(elems_per_pe * n_pes, dtype=np.uint32)

# 每个PE填充所有块的数据
for k in range(n_pes):
    base_value = k + (my_pe + 1) * 100  # k + (myPe + 1) * 100
    host_data[k*elems_per_pe:(k+1)*elems_per_pe] = base_value

# 复制到设备
copy_status = hipMemcpy(in_put_buff, host_data.ctypes.data, total_bytes, hipMemcpyHostToDevice)
if copy_status != 0:
    print(f"PE {my_pe}: ERROR: hipMemcpy failed: {copy_status}")
    hipFree(out_put_buff)
    hipFree(in_put_buff)
    sys.exit(1)

hipDeviceSynchronize()

# 打印初始数据（前4个元素）
print(f"PE {my_pe}: Initial data (first 4 elements of each chunk):")
for pe in range(n_pes):
    start = pe * elems_per_pe
    end = start + min(4, elems_per_pe)
    print(f"  Chunk {pe}: {host_data[start:end]}...")

# 3. 创建stream（和C++完全一样）
print(f"PE {my_pe}: Creating stream...")
stream = ctypes.c_void_p()
stream_status = hipStreamCreate(ctypes.byref(stream))
if stream_status != 0:
    print(f"PE {my_pe}: ERROR: hipStreamCreate failed: {stream_status}")
    hipFree(out_put_buff)
    hipFree(in_put_buff)
    sys.exit(1)

print(f"PE {my_pe}: Stream created: {stream.value:#x}")

# 同步
hipDeviceSynchronize()
MPI.COMM_WORLD.Barrier()

if my_pe == 0:
    print("\n=== Starting All2all Operation ===\n")

# 4. 调用all2all_sdma（需要4个参数的绑定）
print(f"PE {my_pe}: Calling all2all_sdma...")

# 检查是否有uint32_t版本的绑定
if hasattr(mori_cpp, 'all2all_sdma_uint32'):
    func = mori_cpp.all2all_sdma_uint32
    print(f"PE {my_pe}: Using all2all_sdma_uint32")
else:
    func = mori_cpp.all2all_sdma_int32
    print(f"PE {my_pe}: Using all2all_sdma_int32 (may need type conversion)")

# 转换指针类型
input_uintptr = ctypes.c_uint64(in_put_buff.value).value
output_uintptr = ctypes.c_uint64(out_put_buff.value).value
stream_uintptr = ctypes.c_uint64(stream.value).value

try:
    local_duration = func(
        input_uintptr,
        output_uintptr,
        elems_per_pe,
        stream_uintptr
    )
    
    print(f"PE {my_pe}: Function returned: {local_duration}")
    
    if local_duration < 0:
        print(f"PE {my_pe}: ❌ Function returned error: {local_duration}")
    else:
        print(f"PE {my_pe}: ✅ Success! Duration: {local_duration:.9f}s")
        
except Exception as e:
    print(f"PE {my_pe}: ❌ Exception: {e}")
    import traceback
    traceback.print_exc()
    local_duration = -999.0

# 同步
hipDeviceSynchronize()
MPI.COMM_WORLD.Barrier()

# 5. 验证结果（如果成功）
if local_duration > 0:
    print(f"PE {my_pe}: Copying result back to host...")
    
    result_data = np.zeros(elems_per_pe * n_pes, dtype=np.uint32)
    copy_back_status = hipMemcpy(result_data.ctypes.data, out_put_buff, total_bytes, hipMemcpyDeviceToHost)
    
    if copy_back_status != 0:
        print(f"PE {my_pe}: ERROR: hipMemcpy back failed: {copy_back_status}")
    else:
        print(f"PE {my_pe}: All2all result (first 4 elements of each chunk):")
        for pe in range(n_pes):
            start = pe * elems_per_pe
            end = start + min(4, elems_per_pe)
            print(f"  Chunk {pe}: {result_data[start:end]}...")
        
        # 验证（和C++完全一样）
        success = True
        for pe in range(n_pes):
            expected_value = my_pe + (pe + 1) * 100
            for i in range(min(elems_per_pe, 10)):  # 只检查前10个
                if result_data[pe * elems_per_pe + i] != expected_value:
                    print(f"PE {my_pe}: Verification FAILED at chunk {pe}, element {i}: "
                          f"expected {expected_value}, got {result_data[pe * elems_per_pe + i]}")
                    success = False
                    break
            if not success:
                break
        
        if success:
            print(f"PE {my_pe}: Verification PASSED!")
        
        # 计算带宽（和C++完全一样）
        local_bandwidth = total_bytes / local_duration
        local_bandwidth /= (1024.0 * 1024.0 * 1024.0)  # 转换为GB/s
        print(f"PE {my_pe}: totalBytes: {total_bytes/(1024*1024*1024):.9f}GB, "
              f"local time: {local_duration:.9f}s, "
              f"local bandwidth: {local_bandwidth:.9f} GB/s")

# 6. 清理
print(f"PE {my_pe}: Cleaning up...")
hipFree(out_put_buff)
hipFree(in_put_buff)
hipStreamDestroy(stream)

MPI.COMM_WORLD.Barrier()
if my_pe == 0:
    print("\n=== All2all Test Completed ===")