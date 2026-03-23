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

#include <memory>
#include <vector>

#include "kernels/kernel_entity.cuh"

/**
 * IntraNode - single node communication operator class implementation
 * */
class IntraNode {
 public:
  IntraNode(int num_ranks, int rank, size_t max_size = 8192 * 1024 * 8, bool fully_connected = true)
      : num_ranks_(num_ranks),
        rank_(rank),
        max_size_(max_size),
        fully_connected_(fully_connected),
        meta_buffer_(nullptr),
        rank_data_buffer_(nullptr),
        pre_registered_buffer_(nullptr),
        kernel_entity_(nullptr) {
    if (num_ranks_ < 2 || num_ranks_ > 8) {
      throw std::runtime_error("IntraNode only supports 2-8 ranks");
    }
    if (num_ranks_ % 2 != 0) {
      throw std::runtime_error("IntraNode requires even number of ranks");
    }
    if (rank_ < 0 || rank_ >= num_ranks_) {
      throw std::runtime_error("Invalid rank: " + std::to_string(rank_));
    }
  }

  ~IntraNode() { dispose(); }

  // initialize kernel entity
  void init_custom_ar(const std::vector<hipIpcMemHandle_t>& handles,
                      const std::vector<int64_t>& offsets) {
    if (handles.size() != static_cast<size_t>(num_ranks_) ||
        offsets.size() != static_cast<size_t>(num_ranks_)) {
      throw std::runtime_error("handles and offsets size must match num_ranks");
    }
    kernel_entity_ = std::make_unique<CommKernelEntity>(
        reinterpret_cast<Signal*>(meta_buffer_), rank_data_buffer_,
        8 * 1024 * 1024,  // rank_data_size
        handles.data(), offsets, rank_, fully_connected_);
  }

  // launch kernel for allreduce
  template <typename T>
  void all_reduce(hipStream_t stream, T* input, T* output, int size, bool use_new = true,
                  bool registered = false) {
    if (!kernel_entity_) {
      throw std::runtime_error("kernel_entity not initialized, call init_custom_ar first");
    }

    if (registered) {
      // CUDA Graph mode
      kernel_entity_->dispatchAllReduce<T>(stream, input, output, size, use_new);
    } else {
      // Eager mode
      auto input_size = size * sizeof(T);
      if (input_size > max_size_) {
        throw std::runtime_error("input size exceeds max_size");
      }

      HIP_CALL(hipMemcpyAsync(pre_registered_buffer_, input, input_size, hipMemcpyDeviceToDevice,
                              stream));
      kernel_entity_->dispatchAllReduce<T>(stream, reinterpret_cast<T*>(pre_registered_buffer_),
                                           output, size, use_new);
    }
  }

  // release all device buffers
  void dispose() {
    kernel_entity_.reset();

    if (meta_buffer_) {
      HIP_CALL(hipFree(meta_buffer_));
      meta_buffer_ = nullptr;
    }

    if (rank_data_buffer_) {
      HIP_CALL(hipFree(rank_data_buffer_));
      rank_data_buffer_ = nullptr;
    }

    if (pre_registered_buffer_) {
      HIP_CALL(hipFree(pre_registered_buffer_));
      pre_registered_buffer_ = nullptr;
    }
  }

  // return Signal size
  static size_t meta_size() { return sizeof(Signal); }

  // register buffer, used for eager mode
  // IPC handles can not bind with dynamic addresses or changeable addresses
  // so buffer were pre-alllocated
  // d2d copy input data to buffer
  // replaced by register_pre_allocated_buffer, but maybe needed for future use
  /*
  void register_buffer(void* buffer,
                      const std::vector<hipIpcMemHandle_t>& handles,
                      const std::vector<int64_t>& offsets) {
    if (!kernel_entity_) {
      throw std::runtime_error("kernel_entity not initialized");
    }
    kernel_entity_->register_buffer(buffer, handles, offsets);
  }
  */

  // register pre-allocated buffer, used for eager mode
  // bind IPC handles with pre-allocated buffer
  void register_pre_allocated_buffer(const std::vector<hipIpcMemHandle_t>& handles,
                                     const std::vector<int64_t>& offsets) {
    if (!kernel_entity_) {
      throw std::runtime_error("kernel_entity not initialized");
    }
    kernel_entity_->register_buffer(pre_registered_buffer_, handles, offsets);
  }

  // get IPC handle of pre-registered buffer
  hipIpcMemHandle_t get_buffer_ipc_handle() const {
    if (!pre_registered_buffer_) {
      throw std::runtime_error("pre_registered_buffer not allocated");
    }

    hipIpcMemHandle_t handle;
    HIP_CALL(hipIpcGetMemHandle(&handle, pre_registered_buffer_));
    return handle;
  }

  // get IPC meta of graph buffer
  std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    if (!kernel_entity_) {
      throw std::runtime_error("kernel_entity not initialized");
    }

    auto [handles_bytes, offsets] = kernel_entity_->get_graph_buffer_ipc_meta();
    return {handles_bytes, offsets};
  }

  // register graph buffers, used for CUDA Graph mode
  void register_graph_buffers(const std::vector<std::vector<hipIpcMemHandle_t>>& handles,
                              const std::vector<std::vector<int64_t>>& offsets) {
    if (!kernel_entity_) {
      throw std::runtime_error("kernel_entity not initialized");
    }
    kernel_entity_->register_graph_buffers(handles, offsets);
  }

  // support function, can be called by executor

  // allocate meta buffer, rank data buffer and pre-registered buffer
  void allocate_buffers() {
    // 1. allocate meta buffer (for signal sync and temp buffer)
    size_t meta_total_size = sizeof(Signal) + max_size_;
    void* buffer;

    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    hipStream_t stream = 0;

    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));
    HIP_CALL(hipExtMallocWithFlags(&buffer, meta_total_size, hipDeviceMallocUncached));
    HIP_CALL(hipMemsetAsync(buffer, 0, meta_total_size, stream));
    HIP_CALL(hipStreamSynchronize(stream));
    HIP_CALL(hipThreadExchangeStreamCaptureMode(&mode));

    meta_buffer_ = buffer;

    // 2. allocate rank data buffer (for RankData array)
    size_t rank_data_size = 8 * 1024 * 1024;
    HIP_CALL(hipMalloc(&rank_data_buffer_, rank_data_size));

    // 3. allocate pre-registered buffer (for eager mode data storage)
    HIP_CALL(hipMalloc(&pre_registered_buffer_, max_size_));
  }

  // 获取本地 meta buffer 的 IPC handle
  hipIpcMemHandle_t get_meta_buffer_ipc_handle() const {
    if (!meta_buffer_) {
      throw std::runtime_error("meta_buffer not allocated");
    }

    hipIpcMemHandle_t handle;
    HIP_CALL(hipIpcGetMemHandle(&handle, meta_buffer_));
    return handle;
  }

  // get member variables, can be accessed by executor
  int get_rank() const { return rank_; }
  int get_num_ranks() const { return num_ranks_; }
  void* get_meta_buffer() const { return meta_buffer_; }
  void* get_rank_data_buffer() const { return rank_data_buffer_; }

  // get pre-registered buffer, can be accessed by executor
  void* get_pre_registered_buffer() const { return pre_registered_buffer_; }

 private:
  int num_ranks_;
  int rank_;
  size_t max_size_;
  bool fully_connected_;

  // device memory
  void* meta_buffer_;            // Signal metadata + temp buffer
  void* rank_data_buffer_;       // Rank data storage
  void* pre_registered_buffer_;  // Pre-registered buffer for eager mode

  // Kernel entity, as a member variable
  std::unique_ptr<CommKernelEntity> kernel_entity_;
};
