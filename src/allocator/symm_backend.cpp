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

// symm_backend.cpp -- expose the mori symmetric heap as a torch SymmetricMemory backend.
//
// register_availability() makes "MORI" selectable, after which torch drives everything:
// symm_mem.empty -> alloc, symm_mem.rendezvous -> rendezvous, and torch.ops.symm_mem.*
// run on mori memory. Registering means supplying rendezvous, not reusing torch's --
// cheap here, since shmem already bootstrapped the mapping and rendezvous is just
// ShmemPtrP2p() per PE.
//
// ShmemPtrP2p returns 0 for a PE reached over RDMA, so buffer_ptrs()[pe] is null exactly
// for peers that are not load/store accessible -- the same contract NCCL expresses via
// ncclGetLsaPointer, and what world_within_direct_access() aggregates.

#include <hip/hip_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "mori/shmem/shmem.hpp"

namespace mori {
namespace allocator {

namespace {

using c10d::symmetric_memory::SymmetricMemory;
using c10d::symmetric_memory::SymmetricMemoryAllocator;

#define MORI_HIP_CHECK(expr)                                                  \
  do {                                                                        \
    hipError_t _e = (expr);                                                   \
    TORCH_CHECK(_e == hipSuccess, #expr, " failed: ", hipGetErrorString(_e)); \
  } while (0)

// Match torch's signal pad size so layouts stay comparable across backends.
constexpr size_t kSignalPadBytes = 9216;

size_t RoundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// Raw HIP: torch ships both c10::cuda and c10::hip, and the wrong one drags
// cuda_runtime_api.h into a ROCm build.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_idx) {
    if (hipGetDevice(&prev_) != hipSuccess) prev_ = -1;
    if (prev_ >= 0 && prev_ != device_idx) {
      restore_ = (hipSetDevice(device_idx) == hipSuccess);
    }
  }
  ~DeviceGuard() {
    if (restore_) (void)hipSetDevice(prev_);
  }

 private:
  int prev_ = -1;
  bool restore_ = false;
};

// One symmetric allocation, before any rendezvous.
struct Block {
  void* ptr = nullptr;
  size_t buffer_size = 0;  // what the caller asked for
  size_t total_size = 0;   // buffer + signal pad
  int device_idx = 0;
  std::optional<std::string> default_group_name;
  c10::intrusive_ptr<SymmetricMemory> symm;  // cached; rendezvous is once per allocation
};

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(std::vector<void*> buffers, std::vector<void*> signal_pads,
                      size_t buffer_size, int rank, int world_size, int device_idx)
      : buffers_(std::move(buffers)),
        signal_pads_(std::move(signal_pads)),
        buffer_size_(buffer_size),
        rank_(rank),
        world_size_(world_size),
        device_(c10::DeviceType::CUDA, device_idx) {
    rank_to_global_rank_.resize(world_size_);
    for (int r = 0; r < world_size_; ++r) {
      // mori PE numbering is the group's rank numbering
      rank_to_global_rank_[r] = r;
    }

    const size_t arr_bytes = world_size_ * sizeof(void*);
    MORI_HIP_CHECK(hipMalloc(&buffers_dev_, arr_bytes));
    MORI_HIP_CHECK(hipMalloc(&signal_pads_dev_, arr_bytes));
    MORI_HIP_CHECK(hipMemcpy(buffers_dev_, buffers_.data(), arr_bytes, hipMemcpyHostToDevice));
    MORI_HIP_CHECK(
        hipMemcpy(signal_pads_dev_, signal_pads_.data(), arr_bytes, hipMemcpyHostToDevice));
    MORI_HIP_CHECK(hipMalloc(&rank_to_global_rank_dev_, world_size_ * sizeof(int)));
    MORI_HIP_CHECK(hipMemcpy(rank_to_global_rank_dev_, rank_to_global_rank_.data(),
                             world_size_ * sizeof(int), hipMemcpyHostToDevice));
  }

  ~MoriSymmetricMemory() override {
    // Buffers belong to the shmem heap; only the device-side arrays are ours.
    if (buffers_dev_ != nullptr) (void)hipFree(buffers_dev_);
    if (signal_pads_dev_ != nullptr) (void)hipFree(signal_pads_dev_);
    if (rank_to_global_rank_dev_ != nullptr) (void)hipFree(rank_to_global_rank_dev_);
  }

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override { return signal_pads_; }
  void** get_buffer_ptrs_dev() override { return buffers_dev_; }
  void** get_signal_pad_ptrs_dev() override { return signal_pads_dev_; }
  size_t get_buffer_size() override { return buffer_size_; }
  size_t get_offset() override { return 0; }

  // mori has no NVLS-style multicast object.
  bool has_multicast_support() override { return false; }
  void* get_multicast_ptr() override { return nullptr; }

  int get_rank() override { return rank_; }
  int get_world_size() override { return world_size_; }
  c10::Device get_device() override { return device_; }

  const std::vector<int>& get_rank_to_global_rank() override { return rank_to_global_rank_; }
  int* get_rank_to_global_rank_dev() override { return rank_to_global_rank_dev_; }

  // False when any peer is RDMA-only. Kernels dereferencing buffer_ptrs must check it.
  bool world_within_direct_access() override {
    for (void* p : buffers_) {
      if (p == nullptr) return false;
    }
    return true;
  }

  // Coarser than torch's signal-pad spin: drains the device and syncs the whole world
  // rather than a channel. Correct, but not channel-isolated.
  void barrier(int /*channel*/, size_t /*timeout_ms*/) override {
    MORI_HIP_CHECK(hipDeviceSynchronize());
    shmem::ShmemBarrierAll();
  }

  void put_signal(int /*dst_rank*/, int /*channel*/, size_t /*timeout_ms*/) override {
    TORCH_CHECK(false,
                "mori symm backend: put_signal is not implemented yet; use barrier() for "
                "synchronisation");
  }

  void wait_signal(int /*src_rank*/, int /*channel*/, size_t /*timeout_ms*/) override {
    TORCH_CHECK(false,
                "mori symm backend: wait_signal is not implemented yet; use barrier() for "
                "synchronisation");
  }

 private:
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  size_t buffer_size_;
  int rank_;
  int world_size_;
  c10::Device device_;
  std::vector<int> rank_to_global_rank_;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  int* rank_to_global_rank_dev_ = nullptr;
};

class MoriSymmAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx,
              const std::optional<std::string>& group_name) override {
    TORCH_CHECK(shmem::ShmemIsInitialized(),
                "mori symm backend: shmem is not initialised. Call "
                "mori.shmem.shmem_torch_process_group_init() before allocating.");

    DeviceGuard guard(device_idx);

    // Buffer and signal pad share one allocation, as torch's backends do.
    const size_t total = RoundUp(size + kSignalPadBytes, 256);
    void* ptr = shmem::ShmemMalloc(total);
    TORCH_CHECK(ptr != nullptr, "mori symm backend: ShmemMalloc(", total, ") failed");
    MORI_HIP_CHECK(hipMemset(ptr, 0, total));

    auto block = std::make_shared<Block>();
    block->ptr = ptr;
    block->buffer_size = size;
    block->total_size = total;
    block->device_idx = device_idx;
    block->default_group_name = group_name;

    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[ptr] = std::move(block);
    return ptr;
  }

  void free(void* ptr) override {
    std::shared_ptr<Block> block;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = blocks_.find(ptr);
      if (it == blocks_.end()) return;
      block = it->second;
      blocks_.erase(it);
    }
    // ShmemFree is collective; ranks must release in the same order.
    shmem::ShmemFree(block->ptr);
  }

  size_t get_alloc_size(void* ptr) override {
    auto block = FindBlock(ptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    return block->buffer_size;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr, const std::optional<std::string>& group_name) override {
    auto block = FindBlock(ptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    if (block->symm != nullptr) {
      return block->symm;
    }

    auto name = group_name.has_value() ? group_name : block->default_group_name;
    TORCH_CHECK(name.has_value(),
                "mori symm backend: group_name given neither at allocation nor rendezvous");

    auto& info = c10d::symmetric_memory::get_group_info(*name);
    const int rank = info.rank;
    const int world_size = info.world_size;

    // shmem already bootstrapped the mapping, so the group must be the shmem world.
    TORCH_CHECK(rank == shmem::ShmemMyPe() && world_size == shmem::ShmemNPes(),
                "mori symm backend: group '", *name, "' is (rank ", rank, "/", world_size,
                ") but shmem was initialised as (PE ", shmem::ShmemMyPe(), "/",
                shmem::ShmemNPes(), "). The group must match the shmem world.");

    DeviceGuard guard(block->device_idx);

    // Zero means that PE is RDMA-reached, i.e. not load/store accessible from here.
    std::vector<void*> buffers(world_size, nullptr);
    std::vector<void*> signal_pads(world_size, nullptr);
    for (int pe = 0; pe < world_size; ++pe) {
      const uint64_t peer =
          shmem::ShmemPtrP2p(reinterpret_cast<uint64_t>(block->ptr), rank, pe);
      if (peer == 0) continue;
      buffers[pe] = reinterpret_cast<void*>(peer);
      signal_pads[pe] = reinterpret_cast<void*>(peer + block->buffer_size);
    }
    TORCH_CHECK(buffers[rank] != nullptr,
                "mori symm backend: local peer pointer resolution failed");

    auto symm = c10::make_intrusive<MoriSymmetricMemory>(std::move(buffers),
                                                         std::move(signal_pads),
                                                         block->buffer_size, rank, world_size,
                                                         block->device_idx);
    block->symm = symm;
    return symm;
  }

  bool has_multicast_support(int /*device_idx*/) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "MORI"; }

 private:
  std::shared_ptr<Block> FindBlock(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    return it == blocks_.end() ? nullptr : it->second;
  }

  std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<Block>> blocks_;
};

}  // namespace

// Idempotent.
void RegisterTorchSymmBackend() {
  static c10::intrusive_ptr<MoriSymmAllocator> allocator = c10::make_intrusive<MoriSymmAllocator>();
  static bool registered = [] {
    c10d::symmetric_memory::register_availability("MORI", allocator);
    return true;
  }();
  (void)registered;
}

}  // namespace allocator
}  // namespace mori

// Importing the extension registers the backend.
PYBIND11_MODULE(mori_torch_symm, m) {
  mori::allocator::RegisterTorchSymmBackend();
  m.def("register_backend", &mori::allocator::RegisterTorchSymmBackend,
        "Register the MORI symmetric memory backend with torch (idempotent)");
  m.attr("backend_name") = "MORI";
}
