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
  int m = 8192, n = 8192, k = 8192;

  // 设置标量参数
  float alpha = 1.0f;
  float beta = 0.0f;
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
  
  float total_c = 0;
  float total_g = 0;
  float total = 0;
  // 测量                                                                                                                                                                                                              
  hipEvent_t start_g, stop_g;                                                                                                                                                                                              
  hipEventCreate(&start_g);                                                                                                                                                                                              
  hipEventCreate(&stop_g);   

  hipEvent_t start_c, stop_c;                                                                                                                                                                                              
  hipEventCreate(&start_c);                                                                                                                                                                                              
  hipEventCreate(&stop_c);   

  hipEvent_t start, stop;                                                                                                                                                                                              
  hipEventCreate(&start);                                                                                                                                                                                              
  hipEventCreate(&stop); 


  hipEvent_t start_s, mid_1, mid_2, stop_s;                                                                                                                                                                                              
  hipEventCreate(&start_s);                                                                                                                                                                                              
  hipEventCreate(&stop_s);
  hipEventCreate(&mid_1);
  hipEventCreate(&mid_2); 


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
                    HIPBLAS_COMPUTE_32F,  // 计算精度为FP32                                                                                                                                                                   
                    HIPBLAS_GEMM_DEFAULT);                                                                                                                                                                              
  }                                                                                                                                                                                                                    
  hipStreamSynchronize(gstream);

  MPI_Barrier(MPI_COMM_WORLD);  
  hipStreamSynchronize(stream_ccl); //warm up
  for(int i =0 ; i<10;i++)
    OneShotAllGatherSdmaKernel<T><<<1, 512, 0, stream_ccl>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);                                                                                                                                                                                                                  
  hipStreamSynchronize(stream_ccl);  

  float tg = 0;
  float tc = 0;
  float tt = 0;
  hipblasHandle_t handle_d;
  hipblasCreate(&handle_d); 
  for(int i=0;i<10;i++){
    hipEventRecord(start_s);
    hipblasGemmEx(handle_d,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                m, n, k,
                &alpha,
                dA, HIPBLAS_R_16F, m,
                dB, HIPBLAS_R_16F, k,
                &beta,                                                                                                                                                                                              
                dC, HIPBLAS_R_16F, m,                                                                                                                                                                               
                HIPBLAS_COMPUTE_32F,  // 计算精度为FP32                                                                                                                                                                   
                HIPBLAS_GEMM_DEFAULT);
    hipEventRecord(mid_1);
    MPI_Barrier(MPI_COMM_WORLD); 
    hipEventRecord(mid_2); 
    OneShotAllGatherSdmaKernel<T><<<1, 512>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);
    hipEventRecord(stop_s);
    hipDeviceSynchronize();

    float mssc;
    float mssg;
    float msst;                                                                                                                                                                                                        
    hipEventElapsedTime(&mssg, start_s, mid_1);
    hipEventElapsedTime(&mssc, mid_2,  stop_s);
    hipEventElapsedTime(&msst, start_s, stop_s);
    tg += mssg;
    tc += mssc;
    tt += msst; 
  }
  if(myPe == 0){
    printf("============ avg sequential gemm time  :%0.9f    ms============= \n", tg/10.0);
    printf("============ avg sequential coll time  :%0.9f    ms============= \n", tc/10.0);
    printf("============ avg sequential total time :%0.9f    ms============= \n", tt/10.0);
  }



  MPI_Barrier(MPI_COMM_WORLD);
  //double start = MPI_Wtime();
  for(int i=0;i<10;i++){

    hipEventRecord(start);
    hipEventRecord(start_c,stream_ccl);
    OneShotAllGatherSdmaKernel<T><<<1, 512, 0, stream_ccl>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);
    hipEventRecord(start_g, gstream);
    hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,m, n, k,&alpha,dA, HIPBLAS_R_16F, m,dB, HIPBLAS_R_16F, k,&beta,dC, HIPBLAS_R_16F, m,HIPBLAS_COMPUTE_32F,  // 计算精度为FP32
                     HIPBLAS_GEMM_DEFAULT);
    hipEventRecord(stop_g, gstream);
    hipEventRecord(stop_c,stream_ccl);
    hipEventRecord(stop);
    hipStreamSynchronize(stream_ccl);
    hipStreamSynchronize(gstream);
    hipDeviceSynchronize();


    float msc;                                                                                                                                                                                                        
    hipEventElapsedTime(&msc, start_c, stop_c);                                                                                                                                                                           
    total_c += msc;

    float msg;                                                                                                                                                                                                        
    hipEventElapsedTime(&msg, start_g, stop_g);                                                                                                                                                                           
    total_g += msg; 

    float mst;
    hipEventElapsedTime(&mst, start, stop);                                                                                                                                                                           
    total += mst;
  }


  if(myPe == 0){
    double global_bandwidth = total_count * dtype_size * npes /total_c/10.0;
    global_bandwidth /= (1024.0 * 1024.0 * 1024.0);

    printf("============ avg coll time    :%0.9f    ms============= \n", total_c/10.0);
    printf("============ avg coll bw      :%0.9f    GB/s ============= \n", global_bandwidth*100000.0);
    printf("============ avg gemm time    :%0.9f    ms============= \n", total_g/10.0);
    printf("============ avg overlap time :%0.9f    ms============= \n", total/10.0);
  }

  return total_c;
}
}
}
