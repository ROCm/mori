#pragma once
#include "kernel_impl.cuh"

#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if(err != hipSuccess)                                                \
        {                                                                    \
            printf("\n[AITER] %s:%d fail to call %s ---> [HIP error](%s)\n", \
                   __FILE__,                                                 \
                   __LINE__,                                                 \
                   #call,                                                    \
                   hipGetErrorString(err));                                  \
            exit(0);                                                         \
        }                                                                    \
    } while(0)

/*
 * same as custom_all_reduce.cuh
 * class CustomAllreduce
 * */

// IPC handle key type (same as aiter)
using IPC_KEY = std::array<uint8_t, sizeof(hipIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(hipIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(hipIpcMemHandle_t));

class CommKernelEntity {
public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  // below are device pointers
  RankSignals sg_;
  Signal* self_sg_;
  std::unordered_map<void*, RankData*> buffers_;

  // stores the registered device pointers from all ranks
  RankData* d_rank_data_base_;
  RankData* d_rank_data_end_;

  // used for hipgraph on, in capture stage, save input addresses for bind IPC handles
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  CommKernelEntity(Signal* meta, void* rank_data, size_t rank_data_sz,
                   const hipIpcMemHandle_t* handles,
                   const std::vector<int64_t> &offsets, int rank,
                   bool fully_connected = true) :
    rank_(rank),
    world_size_(offsets.size()),
    full_nvlink_(fully_connected),
    self_sg_(meta),
    d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
    d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; ++i) {
      Signal* rank_sg;
      if (i != rank_) {
        char* handle = open_ipc_handle(&handles[i]);
        handle += offsets[i];
        rank_sg = (Signal*)handle;
      }
      else {
        rank_sg = self_sg_;
      }
      sg_.signals[i] = rank_sg;
    }
    printf("rank %d, CommKernelEntity constructed, meta ipc bind finished\n", rank_);
  }

  // register buffer with IPC handles from all ranks
  void register_buffer(void* self_ptr, 
                       const std::vector<hipIpcMemHandle_t>& handles,
                       const std::vector<int64_t>& offsets) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      if (i != rank_) {
        char* handle = open_ipc_handle(&handles[i]);
        handle += offsets[i];
        data.ptrs[i] = handle;
      }
      else {
        data.ptrs[i] = self_ptr;
      }
    }
    auto d_data = d_rank_data_base_++;
    HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
    buffers_[self_ptr] = d_data;
  }

  // get graph buffer IPC meta (returns handles as bytes and offsets)
  std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    printf("rank %d, get_graph_buffer_ipc_meta called\n", rank_);
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(hipIpcMemHandle_t);
    std::vector<uint8_t> handles(handle_sz * num_buffers, 0);
    std::vector<int64_t> offsets(num_buffers);
    
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      
      // Get base address of allocation
      if (hipPointerGetAttribute(&base_ptr,
                                HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                (hipDeviceptr_t)ptr) != hipSuccess) {
        throw std::runtime_error("failed to get pointer attr");
      }
      
      // Get IPC handle for base address
      HIP_CALL(hipIpcGetMemHandle(
          (hipIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      
      // Calculate offset from base
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    
    return std::make_pair(handles, offsets);
    printf("rank %d, get_graph_buffer_ipc_meta finished\n", rank_);
  }

  // register graph buffers captured during CUDA graph recording
  void register_graph_buffers(
      const std::vector<std::vector<hipIpcMemHandle_t>>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    printf("rank %d, register_graph_buffers called, handles size: %zu, offsets size: %zu\n", rank_, handles.size(), offsets.size());
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        }
        else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    HIP_CALL(hipMemcpy(d_rank_data_base_, rank_data.data(),
                       sizeof(RankData) * num_buffers,
                       hipMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
    printf("rank %d, register_graph_buffers finished\n", rank_);
  }

  // main allreduce dispatch function
  // TODO: add dispatch logic about naive kernel
  template <typename T>
  void dispatchAllReduce(hipStream_t stream, T* input, T* output, int size, bool use_new = true) {
    RankData* ptrs = get_buffer_RD(stream, input);
    
    auto d = packed_t<T>::P::size;
    if (size % d != 0) {
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple of " +
          std::to_string(d));
    }
    if (kMaxBlocks > 80) {
      throw std::runtime_error("max supported block limit is 80");
    }
    
    auto bytes = size * sizeof(T);
    size /= d;
    
    int blocks = 16;
    int threads = 512;
    bool call_1stage = false;
    bool call_2stage = false;
    
    if (world_size_ == 2) {
      call_1stage = true;
    }
    else if (full_nvlink_) {
      if ((world_size_ <= 4 && bytes < 160 * 1024) || 
          (world_size_ <= 8 && bytes < 80 * 1024)) {
        call_1stage = true;
      }
      else {
        call_2stage = true;
      }
    }
    
    if (call_1stage) {
      blocks = std::min(kMaxBlocks, (size + (threads / world_size_) - 1) / (threads / world_size_));
    }
    else if (call_2stage) {
      blocks = std::min(kMaxBlocks, (size / world_size_ + (threads / world_size_) - 1) / (threads / world_size_));
    }
    
#define KL(ngpus, name) \
    name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);

#define dispatch(ngpus, name) \
    do { \
      if (bytes % 128 == 0 && world_size_ != 6) { \
        KL(ngpus, name) \
      } \
      else { \
        KL(ngpus, name##_naive) \
      } \
    } while(0)

#define REDUCE_CASE(ngpus) \
    case ngpus: { \
      if (call_1stage) { \
        dispatch(ngpus, cross_device_reduce_1stage); \
      } \
      else if (call_2stage) { \
        dispatch(ngpus, cross_device_reduce_2stage); \
      } \
      break; \
    }
    
    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
    default:
      throw std::runtime_error(
          "custom allreduce only supports num gpus in (2,4,6,8). Actual num gpus = " +
          std::to_string(world_size_));
    }
    
#undef REDUCE_CASE
#undef dispatch
#undef KL
  }

  ~CommKernelEntity() {
    for (auto& [handle, ptr] : ipc_handles_) {
      if (ptr) {
        HIP_CALL(hipIpcCloseMemHandle(ptr));
      }
    }
  }

private:

  char* open_ipc_handle(const hipIpcMemHandle_t* ipc_handle) {
    // Convert hipIpcMemHandle_t to IPC_KEY for use as map key
    auto [it, new_handle] =
      ipc_handles_.insert({*((const IPC_KEY*)ipc_handle), nullptr});
    // insert success
    if (new_handle) {
      char* ipc_ptr;
      HIP_CALL(hipIpcOpenMemHandle((void**)&ipc_ptr,
                                   *((const hipIpcMemHandle_t *)ipc_handle),
                                   hipIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_) {
      throw std::runtime_error(
          "Rank data buffer is overflow by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }
  }

  RankData* get_buffer_RD(hipStream_t stream, void* input) {
    RankData* ptrs;
    auto it = buffers_.find(input);
    if (it != buffers_.end()) {
      ptrs = it->second;
    }
    else {
      hipStreamCaptureStatus status;
      HIP_CALL(hipStreamIsCapturing(stream, &status));
      if (status == hipStreamCaptureStatusActive) {
        ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
        graph_unreg_buffers_.push_back(input);
      }
      else {
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      }
    }
    return ptrs;
  }
};

