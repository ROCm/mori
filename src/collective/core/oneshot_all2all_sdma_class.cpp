// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "mori/collective/core/topology_detector.hpp"
#include <mpi.h>
#include <cassert>
#include "oneshot_all2all_sdma_class.hpp"

namespace mori {
namespace collective {

template <typename T>
All2allSdma<T>::All2allSdma(int myPe, int npes) 
    : myPe_(myPe), npes_(npes), dtype_size_(sizeof(T)) {
    
    // 计算flags大小并分配内存
    int flagsSize = npes_ * sizeof(uint64_t);
    
    // 使用ShmemMalloc分配对称内存
    void* flags = shmem::ShmemMalloc(flagsSize);
    if (flags == nullptr) {
        throw std::runtime_error("Failed to allocate flags memory");
    }
    
    // 使用unique_ptr管理内存，确保异常安全
    flags_.reset(static_cast<uint64_t*>(flags));
    
    // 初始化flags为0
    memset(flags_.get(), 0, flagsSize);
    
    // 获取对称内存对象指针
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_.get());
    
    if (!flagsObj_.IsValid()) {
        throw std::runtime_error("Failed to get valid flags memory object");
    }
    
    printf("All2allSdma initialized: PE %d of %d, flags allocated at %p\n", 
           myPe_, npes_, flags_.get());
}

template <typename T>
All2allSdma<T>::~All2allSdma() {
    // flags_使用unique_ptr自动管理，不需要手动释放
    // 如果需要在析构时清理，可以添加清理代码
    if (flags_) {
        // unique_ptr会自动释放内存
        printf("All2allSdma destroyed: PE %d\n", myPe_);
    }
}

template <typename T>
double All2allSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    // 检查参数有效性
    if (input == nullptr || output == nullptr || total_count == 0) {
        fprintf(stderr, "Invalid parameters: input=%p, output=%p, total_count=%zu\n", 
                input, output, total_count);
        return -1.0;
    }
    
    if (!flagsObj_.IsValid()) {
        fprintf(stderr, "Flags memory object is invalid\n");
        return -1.0;
    }
    
    // 注册输入输出缓冲区
    application::SymmMemObjPtr inPutBuffObj =
        shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size_);
    
    application::SymmMemObjPtr outPutBuffObj =
        shmem::ShmemSymmetricRegister(static_cast<void*>(output), total_count * dtype_size_ * npes_);
    
    if (!inPutBuffObj.IsValid()) {
        fprintf(stderr, "Failed to register input buffer\n");
        return -1.0;
    }
    
    if (!outPutBuffObj.IsValid()) {
        fprintf(stderr, "Failed to register output buffer\n");
        return -1.0;
    }
    
    // 执行All2All操作
    double start = MPI_Wtime();
    
    // 重置标志位
    resetFlags();
    
    // 调用内核
    OneShotAll2allSdmaKernel<T><<<1, 512>>>(myPe_, npes_, inPutBuffObj, 
                                           outPutBuffObj, flagsObj_, total_count);
    
    // 同步GPU
    if (stream != nullptr) {
        HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
    } else {
        HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    }
    
    double end = MPI_Wtime();
    double duration = end - start;
    
    printf("======== PE %d: All2all_sdma time: %.9f seconds ======== \n", myPe_, duration);
    
    return duration;
}

template <typename T>
void All2allSdma<T>::resetFlags() {
    if (flags_) {
        int flagsSize = npes_ * sizeof(uint64_t);
        memset(flags_.get(), 0, flagsSize);
    }
}

// 显式实例化常用类型
template class All2allSdma<uint32_t>;
template class All2allSdma<uint64_t>;
template class All2allSdma<float>;
template class All2allSdma<double>;

} // namespace collective
} // namespace mori
