#!/usr/bin/env python3
"""
最简单的All2All测试
"""

import sys
import numpy as np
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    my_pe = comm.Get_rank()
    npes = comm.Get_size()

    print(f"PE {my_pe}/{npes}: Starting test")

    # 导入模块
    try:
        import mori.cpp
        print(f"PE {my_pe}: Imported mori.cpp")
    except ImportError as e:
        print(f"PE {my_pe}: Import error: {e}")
        return 1

    # 初始化SHMEM
    try:
        status = mori.ShmemMpiInit(comm)
        print(f"PE {my_pe}: ShmemMpiInit status: {status}")
        if status != 0:
            return 1
    except Exception as e:
        print(f"PE {my_pe}: SHMEM init error: {e}")
        return 1

    # 获取SHMEM信息
    try:
        my_pe_shmem = mori.ShmemMyPe()
        npes_shmem = mori.ShmemNPes()
        print(f"PE {my_pe}: SHMEM myPe={my_pe_shmem}, npes={npes_shmem}")
    except Exception as e:
        print(f"PE {my_pe}: Failed to get SHMEM info: {e}")
        return 1

    # 创建All2all对象
    try:
        all2all = mori.cpp.All2allSdma(my_pe_shmem, npes_shmem)
        print(f"PE {my_pe}: Created All2allSdma")
    except Exception as e:
        print(f"PE {my_pe}: Failed to create All2allSdma: {e}")
        return 1

    # 测试数据
    elems = 1024
    input_data = np.ones(elems, dtype=np.uint32) * (my_pe_shmem + 100)
    output_data = np.zeros(elems * npes_shmem, dtype=np.uint32)

    # 执行All2All
    try:
        time = all2all(input_data, output_data, elems)
        print(f"PE {my_pe}: All2All completed in {time:.6f}s")
    except Exception as e:
        print(f"PE {my_pe}: All2All error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 验证
    success = True
    for pe in range(npes_shmem):
        chunk = output_data[pe*elems:(pe+1)*elems]
        if not np.all(chunk == pe + 100):
            print(f"PE {my_pe}: Chunk {pe} verification failed")
            success = False

    # 同步结果
    all_success = comm.allgather(success)

    if my_pe == 0:
        print("\n" + "="*50)
        if all(all_success):
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED!")
        print("="*50)

    # 清理
    del all2all
    mori.cpp.ShmemFinalize()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
