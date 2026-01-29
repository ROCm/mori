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
#pragma once

#include <hip/hip_runtime.h>
#include "oneshot_sdma_kernel.hpp"
#include <mpi.h>


#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

namespace mori {
namespace collective {

template <typename T>
double AllGather_sdma(T* input, T* output, size_t total_count,
                                          hipStream_t stream_ccl) {

  // 创建hipBLAS句柄和stream
  hipblasHandle_t handle;
  hipblasCreate(&handle);

  hipStream_t gstream;
  hipStream_t cstream;
  hipStreamCreate(&gstream);
  hipStreamCreate(&cstream);
  hipblasSetStream(handle, gstream);

  // 分配设备内存
  __half *dA, *dB, *dC;
  size_t size_A = m * k * sizeof(__half);
  size_t size_B = k * n * sizeof(__half);
  size_t size_C = m * n * sizeof(__half);

  hipMalloc(&dA, size_A);
  hipMalloc(&dB, size_B);
  hipMalloc(&dC, size_C);

  // 初始化主机数据
  __half *hA = new __half[m * k];
  __half *hB = new __half[k * n];
  __half *hC = new __half[m * n];

  __half one_half = __float2half(1.0f);
  __half zero_half = __float2half(0.0f);

  for (int i = 0; i < m * k; i++) hA[i] = one_half;
  for (int i = 0; i < k * n; i++) hB[i] = one_half;
  for (int i = 0; i < m * n; i++) hC[i] = zero_half;


  // 拷贝到设备
  hipMemcpy(dA, hA, size_A, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, size_B, hipMemcpyHostToDevice);
  hipMemcpy(dC, hC, size_C, hipMemcpyHostToDevice);
  
  float total_ms = 0;
  
  int myPe =  shmem::ShmemMyPe();
  int npes =  shmem::ShmemNPes();
  size_t dtype_size = sizeof(T);

  application::SymmMemObjPtr inPutBuffObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size);

  application::SymmMemObjPtr outPutBuffObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(output), total_count * dtype_size * npes);

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);

  assert(inPutBuffObj.IsValid());
  assert(outPutBuffObj.IsValid());
  assert(flagsObj.IsValid());

  hipStreamSynchronize(gstream); //warm up
  for (int i = 0; i < 10; i++) {
      hipblasGemmEx(handle,
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    dA, HIPBLAS_R_16F, m,
                    dB, HIPBLAS_R_16F, k,
                    &beta,                                                                                                                                                                                              
                    dC, HIPBLAS_R_16F, m,                                                                                                                                                                               
                    HIPBLAS_R_32F,  // 计算精度为FP32                                                                                                                                                                   
                    HIPBLAS_GEMM_DEFAULT);                                                                                                                                                                              
  }                                                                                                                                                                                                                    
  hipStreamSynchronize(gstream);  

  hipStreamSynchronize(stream_ccl); //warm up
  for(int i =0 ; i<10;i++)
    OneShotAllGatherSdmaKernel<T><<<1, 512, 0, stream_ccl>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);                                                                                                                                                                                                                  
  hipStreamSynchronize(stream_ccl);  

  //double start = MPI_Wtime();
  for(int i=0;i<10;i++){
    hipEventRecord(start);
    OneShotAllGatherSdmaKernel<T><<<1, 512, 0, stream_ccl>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);
    hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,m, n, k,&alpha,dA, HIPBLAS_R_16F, m,dB, HIPBLAS_R_16F, k,&beta,dC, HIPBLAS_R_16F, m,HIPBLAS_R_32F,  // 计算精度为FP32
                     HIPBLAS_GEMM_DEFAULT);
    hipEventRecord(stop);

    float ms;                                                                                                                                                                                                        
    hipEventElapsedTime(&ms, start, stop);                                                                                                                                                                           
    total_ms += ms;   

  }

  double end = MPI_Wtime();

   
  //shmem::ShmemFree(flags);
  return end-start;
}
}
}
