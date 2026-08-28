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

// symm_backend.cpp -- a torch SymmetricMemory backend backed by CCO LSA windows.
//
// register_availability() makes "MORI" selectable; torch then drives symm_mem.empty ->
// alloc, symm_mem.rendezvous -> rendezvous, and torch.ops.symm_mem.* on the result.
//
// Every torch allocation remains an independent HIP VMM allocation, so Python GC may
// release tensors in a different order on each rank. rendezvous imports that externally
// owned allocation into a group-scoped CCO communicator. CCO owns the flat LSA mapping;
// torch's pointer array is derived from the same ccoWindow and remains API-compatible.
//
// The torch interface still exposes only buffer_ptrs / buffer_ptrs_dev. MORI-specific
// kernels can additionally request the owning CcoLsaWindow and use ccoWindowDevice
// arithmetic without publishing another private flat-base/stride ABI. See ROCm/mori#557.

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/version.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/cco/cco.hpp"
#include "mori/utils/hip_compat.hpp"

// The SymmetricMemory interface is not stable across torch minors -- methods move
// between pure virtual, virtual-with-default and non-virtual -- so the few places that
// differ are gated rather than written for one release.
#if !defined(TORCH_VERSION_MAJOR) || !defined(TORCH_VERSION_MINOR)
#error "mori's torch SymmetricMemory backend needs <torch/version.h> to define TORCH_VERSION_*"
#endif
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
#error "mori's torch SymmetricMemory backend requires torch >= 2.9"
#endif
#define MORI_TORCH_AT_LEAST(major, minor) \
  (TORCH_VERSION_MAJOR > (major) ||       \
   (TORCH_VERSION_MAJOR == (major) && TORCH_VERSION_MINOR >= (minor)))

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

#define MORI_CCO_CHECK(expr)                                         \
  do {                                                               \
    int _e = (expr);                                                 \
    TORCH_CHECK(_e == 0, #expr, " failed with CCO error code ", _e); \
  } while (0)

// Match torch's signal pad size so layouts stay comparable across backends.
// Signal-pad support. barrier()/put_signal()/wait_signal() are unimplemented, so by
// default no pad is reserved: the ops report unsupported and every window costs exactly
// its buffer. Physical backing is 2 MiB-paged, so a 9216 B pad appended to a page-aligned
// request costs a whole extra page -- 2 MiB on a 64 MiB window. Build with
// -DMORI_SYMM_SIGNAL_PAD=1 to reserve torch's pad once the ops exist.
#ifndef MORI_SYMM_SIGNAL_PAD
#define MORI_SYMM_SIGNAL_PAD 0
#endif
#if MORI_SYMM_SIGNAL_PAD
constexpr size_t kSignalPadBytes = 9216;
#else
constexpr size_t kSignalPadBytes = 0;
#endif

size_t RoundUp(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

// Teardown can run while the HIP runtime is already unwinding, in which case any VMM call
// segfaults. is_finalizing() catches torch's own shutdown; this catches the rest by
// probing the runtime first. Leaking there is deliberate -- the process is exiting.
bool RuntimeUsable() {
  if (c10d::symmetric_memory::is_finalizing()) return false;
  int dev = -1;
  return hipGetDevice(&dev) == hipSuccess;
}

// Raw HIP: torch ships both c10::cuda and c10::hip; the wrong one pulls cuda_runtime_api.h.
class DeviceGuard {
 public:
  explicit DeviceGuard(int dev) {
    if (hipGetDevice(&prev_) != hipSuccess) prev_ = -1;
    if (prev_ >= 0 && prev_ != dev) restore_ = (hipSetDevice(dev) == hipSuccess);
  }
  ~DeviceGuard() {
    if (restore_) (void)hipSetDevice(prev_);
  }

 private:
  int prev_ = -1;
  bool restore_ = false;
};

// ROCm 7.1 spells this requestedHandleType; newer releases add a union with the plural.
// The singular name exists in both.
hipMemAllocationProp MakeProp(int dev, hipMemAllocationHandleType handle_type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = handle_type;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  return prop;
}

void SetRwAccess(void* ptr, size_t size, int dev) {
  hipMemAccessDesc ad = {};
  ad.location.type = hipMemLocationTypeDevice;
  ad.location.id = dev;
  ad.flags = hipMemAccessFlagsProtReadWrite;
  MORI_HIP_CHECK(hipMemSetAccess(ptr, size, &ad, 1));
}

// Probed once per device: the capability attribute enum is not stable across releases.
hipMemAllocationHandleType ProbeHandleType(int dev) {
  static std::mutex mu;
  static std::unordered_map<int, hipMemAllocationHandleType> cache;
  std::lock_guard<std::mutex> lock(mu);
  if (auto it = cache.find(dev); it != cache.end()) return it->second;

  hipMemAllocationHandleType chosen = hipMemHandleTypePosixFileDescriptor;
  const char* force_fd = std::getenv("MORI_CCO_FORCE_FD");
  if (force_fd == nullptr || std::atoi(force_fd) == 0) {
    auto prop = MakeProp(dev, hipMemHandleTypeFabricCompat);
    size_t gran = 0;
    if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended) ==
            hipSuccess &&
        gran > 0) {
      hipMemGenericAllocationHandle_t h{};
      if (hipMemCreate(&h, gran, &prop, 0) == hipSuccess) {
        hipMemFabricHandle_compat_t blob;
        if (hipMemExportToShareableHandle(&blob, h, hipMemHandleTypeFabricCompat, 0) ==
            hipSuccess) {
          chosen = hipMemHandleTypeFabricCompat;
        }
        (void)hipMemRelease(h);
      }
    }
  }
  (void)hipGetLastError();  // the probe is expected to fail on gfx9
  cache[dev] = chosen;
  return chosen;
}

size_t CcoPerRankVmmSize() {
  const char* raw = std::getenv("MORI_SYMM_CCO_VMM_SIZE");
  if (raw == nullptr || raw[0] == '\0') return 0;  // CCO defaults to total GPU memory.
  char* end = nullptr;
  unsigned long long value = std::strtoull(raw, &end, 0);
  TORCH_CHECK(end != raw && end != nullptr && *end == '\0',
              "MORI_SYMM_CCO_VMM_SIZE must be an integer byte count, got '", raw, "'");
  return static_cast<size_t>(value);
}

// What each rank publishes through the torch Store before entering CCO's collective
// WindowRegister. This gives a useful error instead of letting different tensor sizes
// reach a lower-level handle exchange.
struct RendezvousReq {
  size_t alloc_size;
  size_t buffer_size;
  int device_idx;
  int handle_type;
  int global_rank;
};

class CcoGroupContext {
 public:
  CcoGroupContext(std::string group_name, c10::intrusive_ptr<c10d::Store> store, int rank,
                  int world_size, int device_idx)
      : group_name_(std::move(group_name)),
        store_(std::move(store)),
        rank_(rank),
        world_size_(world_size),
        device_idx_(device_idx) {
    DeviceGuard guard(device_idx_);
    const int64_t ticket = store_->add("mori_symm_cco_context_ticket/" + group_name_, 1);
    TORCH_CHECK(ticket > 0, "mori symm backend: invalid CCO context ticket ", ticket);
    incarnation_ = static_cast<uint64_t>((ticket - 1) / world_size_);
    const std::string incarnation_suffix = "/i" + std::to_string(incarnation_);
    window_exchange_ = std::make_unique<c10d::symmetric_memory::StoreExchange>(
        "mori_symm_cco_window/" + group_name_ + incarnation_suffix);

    mori::cco::ccoUniqueId local_uid{};
    if (rank_ == 0) MORI_CCO_CHECK(mori::cco::ccoGetUniqueId(&local_uid));

    c10d::symmetric_memory::StoreExchange bootstrap_exchange("mori_symm_cco_bootstrap/" +
                                                             group_name_ + incarnation_suffix);
    auto all_uids = bootstrap_exchange.all_gather(store_, rank_, world_size_, local_uid);
    MORI_CCO_CHECK(mori::cco::ccoCommCreateLsaOnly(all_uids[0], world_size_, rank_,
                                                   CcoPerRankVmmSize(), &comm_));
    try {
      TORCH_CHECK(comm_ != nullptr,
                  "mori symm backend: ccoCommCreate returned a null communicator");
      info_ = CCO_COMM_INFO_INITIALIZER;
      MORI_CCO_CHECK(mori::cco::ccoCommGetInfo(comm_, &info_));
      TORCH_CHECK(info_.rank == rank_ && info_.worldSize == world_size_,
                  "mori symm backend: CCO communicator rank/size mismatch");
    } catch (...) {
      if (comm_ != nullptr) (void)mori::cco::ccoCommDestroy(comm_);
      comm_ = nullptr;
      throw;
    }
  }

  ~CcoGroupContext() { Close(); }

  void Close() {
    if (comm_ == nullptr || !RuntimeUsable()) return;
    DeviceGuard guard(device_idx_);
    (void)mori::cco::ccoCommDestroy(comm_);
    comm_ = nullptr;
  }

  CcoGroupContext(const CcoGroupContext&) = delete;
  CcoGroupContext& operator=(const CcoGroupContext&) = delete;

  std::vector<int> ValidateWindow(size_t alloc_size, size_t buffer_size, int handle_type,
                                  int global_rank) {
    RendezvousReq local{alloc_size, buffer_size, device_idx_, handle_type, global_rank};
    auto reqs = window_exchange_->all_gather(store_, rank_, world_size_, local);
    std::vector<int> rank_to_global_rank(world_size_);
    for (int r = 0; r < world_size_; ++r) {
      TORCH_CHECK(
          reqs[r].alloc_size == local.alloc_size && reqs[r].buffer_size == local.buffer_size,
          "mori symm backend: rank ", r, " allocated ", reqs[r].buffer_size,
          " bytes but this rank allocated ", local.buffer_size,
          "; symm_mem.empty must be symmetric across ranks");
      TORCH_CHECK(reqs[r].handle_type == local.handle_type,
                  "mori symm backend: mixed CCO handle types in one process group "
                  "(rank ",
                  r, " uses ", reqs[r].handle_type, ", local rank uses ", local.handle_type, ")");
      rank_to_global_rank[r] = reqs[r].global_rank;
    }
    return rank_to_global_rank;
  }

  mori::cco::ccoComm* comm() const { return comm_; }
  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int lsa_rank() const { return info_.lsaRank; }
  int lsa_size() const { return info_.lsaSize; }
  int lsa_start() const { return info_.lsaStart; }
  size_t per_rank_size() const { return info_.perRankSize; }
  int device_idx() const { return device_idx_; }
  std::mutex& window_mutex() { return window_mutex_; }

 private:
  std::string group_name_;
  c10::intrusive_ptr<c10d::Store> store_;
  int rank_;
  int world_size_;
  int device_idx_;
  uint64_t incarnation_ = 0;
  mori::cco::ccoComm* comm_ = nullptr;
  mori::cco::ccoCommInfo info_ = CCO_COMM_INFO_INITIALIZER;
  std::unique_ptr<c10d::symmetric_memory::StoreExchange> window_exchange_;
  std::mutex window_mutex_;
};

class CcoWindowState {
 public:
  CcoWindowState(std::shared_ptr<CcoGroupContext> group, mori::cco::ccoWindow_t handle,
                 void* local_ptr)
      : group_(std::move(group)), handle_(handle), local_ptr_(local_ptr) {}

  ~CcoWindowState() { Close(); }

  CcoWindowState(const CcoWindowState&) = delete;
  CcoWindowState& operator=(const CcoWindowState&) = delete;

  void Close() {
    if ((handle_ == nullptr && local_ptr_ == nullptr) || !RuntimeUsable()) return;
    DeviceGuard guard(group_->device_idx());
    std::lock_guard<std::mutex> lock(group_->window_mutex());
    if (handle_ != nullptr) {
      (void)mori::cco::ccoWindowDeregister(group_->comm(), handle_);
      handle_ = nullptr;
    }
    if (local_ptr_ != nullptr) {
      (void)mori::cco::ccoMemFree(group_->comm(), local_ptr_);
      local_ptr_ = nullptr;
    }
  }

  void* peer_ptr(int pe) const { return mori::cco::ccoGetPeerPtr(group_->comm(), local_ptr_, pe); }
  mori::cco::ccoWindow_t handle() const { return handle_; }
  void* local_ptr() const { return local_ptr_; }
  const std::shared_ptr<CcoGroupContext>& group() const { return group_; }

 private:
  std::shared_ptr<CcoGroupContext> group_;
  mori::cco::ccoWindow_t handle_ = nullptr;
  void* local_ptr_ = nullptr;
};

class AllocationState {
 public:
  explicit AllocationState(int device_idx) : device_idx_(device_idx) {}

  ~AllocationState() { Close(); }

  void SetHandle(hipMemGenericAllocationHandle_t handle) {
    handle_ = handle;
    handle_live_ = true;
  }
  void SetAddress(void* ptr, size_t size) {
    ptr_ = ptr;
    size_ = size;
    address_reserved_ = true;
  }
  void MarkMapped() { mapped_ = true; }
  void SetExportFd(int fd) { export_fd_ = fd; }

  void Close() {
    if (!address_reserved_ && !handle_live_) return;
    if (!RuntimeUsable()) return;
    DeviceGuard guard(device_idx_);
    (void)hipDeviceSynchronize();
    if (mapped_) {
      (void)hipMemUnmap(ptr_, size_);
      mapped_ = false;
    }
    if (address_reserved_) {
      (void)hipMemAddressFree(ptr_, size_);
      ptr_ = nullptr;
      address_reserved_ = false;
    }
    if (handle_live_) {
      (void)hipMemRelease(handle_);
      handle_live_ = false;
    }
    // ROCm 7.14 ties a shareable FD's lifetime to the physical allocation:
    // closing it before the final hipMemRelease can leak or crash that release.
    if (export_fd_ >= 0) {
      (void)::close(export_fd_);
      export_fd_ = -1;
    }
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
  hipMemGenericAllocationHandle_t handle_{};
  bool handle_live_ = false;
  bool address_reserved_ = false;
  bool mapped_ = false;
  int export_fd_ = -1;
  int device_idx_ = 0;
};

class DeviceArrayState {
 public:
  static std::shared_ptr<DeviceArrayState> Create(int device_idx, const std::vector<void*>& buffers,
                                                  const std::vector<void*>& signal_pads,
                                                  const std::vector<int>& rank_to_global_rank,
                                                  bool with_signal_pad) {
    auto state = std::shared_ptr<DeviceArrayState>(new DeviceArrayState(device_idx));
    DeviceGuard guard(device_idx);
    const size_t ptr_bytes = buffers.size() * sizeof(void*);
    MORI_HIP_CHECK(hipMalloc(&state->buffers_dev_, ptr_bytes));
    MORI_HIP_CHECK(
        hipMemcpy(state->buffers_dev_, buffers.data(), ptr_bytes, hipMemcpyHostToDevice));
    if (with_signal_pad) {
      MORI_HIP_CHECK(hipMalloc(&state->signal_pads_dev_, ptr_bytes));
      MORI_HIP_CHECK(
          hipMemcpy(state->signal_pads_dev_, signal_pads.data(), ptr_bytes, hipMemcpyHostToDevice));
    }
    MORI_HIP_CHECK(
        hipMalloc(&state->rank_to_global_rank_dev_, rank_to_global_rank.size() * sizeof(int)));
    MORI_HIP_CHECK(hipMemcpy(state->rank_to_global_rank_dev_, rank_to_global_rank.data(),
                             rank_to_global_rank.size() * sizeof(int), hipMemcpyHostToDevice));
    return state;
  }

  ~DeviceArrayState() { Close(); }

  void Close() {
    if (closed_) return;
    closed_ = true;
    if (!RuntimeUsable()) return;
    DeviceGuard guard(device_idx_);
    (void)hipDeviceSynchronize();
    if (buffers_dev_) {
      (void)hipFree(buffers_dev_);
      buffers_dev_ = nullptr;
    }
    if (signal_pads_dev_) {
      (void)hipFree(signal_pads_dev_);
      signal_pads_dev_ = nullptr;
    }
    if (rank_to_global_rank_dev_) {
      (void)hipFree(rank_to_global_rank_dev_);
      rank_to_global_rank_dev_ = nullptr;
    }
  }

  void** buffers() const { return buffers_dev_; }
  void** signal_pads() const { return signal_pads_dev_; }
  int* rank_to_global_rank() const { return rank_to_global_rank_dev_; }

 private:
  explicit DeviceArrayState(int device_idx) : device_idx_(device_idx) {}

  int device_idx_;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  int* rank_to_global_rank_dev_ = nullptr;
  bool closed_ = false;
};

class MoriSymmetricMemory;

struct Block {
  void* ptr = nullptr;
  size_t buffer_size = 0;
  size_t alloc_size = 0;
  int device_idx = 0;
  hipMemGenericAllocationHandle_t handle{};
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
  std::optional<std::string> default_group_name;
  std::mutex lifecycle_mutex;
  std::optional<std::string> rendezvous_group_name;
  std::shared_ptr<AllocationState> allocation;
  c10::intrusive_ptr<MoriSymmetricMemory> symm;
};

constexpr const char* kNoSignalPad =
    "Signal-pad support is compiled out (MORI_SYMM_SIGNAL_PAD=0), so no pad is reserved "
    "in the window; rebuild with -DMORI_SYMM_SIGNAL_PAD=1 to allocate it.";

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(std::shared_ptr<CcoWindowState> window,
                      std::shared_ptr<AllocationState> allocation, size_t buffer_size,
                      std::vector<int> rank_to_global_rank)
      : window_(std::move(window)),
        allocation_(std::move(allocation)),
        buffer_size_(buffer_size),
        rank_(window_->group()->rank()),
        world_size_(window_->group()->world_size()),
        device_(c10::DeviceType::CUDA, window_->group()->device_idx()),
        rank_to_global_rank_(std::move(rank_to_global_rank)),
        world_within_direct_access_(window_->group()->lsa_size() == world_size_) {
    if (rank_to_global_rank_.size() != static_cast<size_t>(world_size_)) {
      rank_to_global_rank_.resize(world_size_);
      std::iota(rank_to_global_rank_.begin(), rank_to_global_rank_.end(), 0);
    }
    buffers_.reserve(world_size_);
    signal_pads_.reserve(world_size_);
    for (int r = 0; r < world_size_; ++r) {
      auto* slot = static_cast<char*>(window_->peer_ptr(r));
      buffers_.push_back(slot);
      signal_pads_.push_back(kSignalPadBytes && slot != nullptr ? slot + buffer_size_ : nullptr);
    }

    device_arrays_ = DeviceArrayState::Create(device_.index(), buffers_, signal_pads_,
                                              rank_to_global_rank_, kSignalPadBytes != 0);
  }

  ~MoriSymmetricMemory() override { Close(); }

  void Close() {
    if (closed_) return;
    closed_ = true;
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    // Usually reached from free(), which has already synchronised; not when a Python
    // reference outlived the tensor. Sync on the window's own device either way.
    DeviceGuard guard(device_.index());
    (void)hipDeviceSynchronize();
    device_arrays_->Close();
    device_arrays_.reset();
    // The external-owner registration borrows this allocation handle, so remove
    // CCO's peer/local aliases before releasing torch's original mapping.
    window_.reset();
    allocation_->Close();
    allocation_.reset();
  }

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override {
    TORCH_CHECK(kSignalPadBytes != 0, kNoSignalPad);
    return signal_pads_;
  }
  void** get_buffer_ptrs_dev() override {
    return device_arrays_ ? device_arrays_->buffers() : nullptr;
  }
  void** get_signal_pad_ptrs_dev() override {
    TORCH_CHECK(kSignalPadBytes != 0, kNoSignalPad);
    return device_arrays_ ? device_arrays_->signal_pads() : nullptr;
  }
  size_t get_buffer_size() override { return buffer_size_; }
  size_t get_offset() override { return 0; }
  // Every alloc() is its own VMM allocation, so torch's storage starts at the window
  // base and the pad, when reserved, sits directly after the buffer.
  //
  // torch 2.9 declares this pure virtual, so it must be defined; 2.10 turned it into a
  // concrete base method, where 'override' is itself a compile error. Hence the guard
  // rather than defining it unconditionally.
#if MORI_TORCH_AT_LEAST(2, 10)
  // provided by the base class
#else
  size_t get_signal_pad_size() override { return kSignalPadBytes; }
#endif

  bool has_multicast_support() override { return false; }
  void* get_multicast_ptr() override { return nullptr; }

  int get_rank() override { return rank_; }
  int get_world_size() override { return world_size_; }
  c10::Device get_device() override { return device_; }

  const std::vector<int>& get_rank_to_global_rank() override { return rank_to_global_rank_; }
  int* get_rank_to_global_rank_dev() override {
    return device_arrays_ ? device_arrays_->rank_to_global_rank() : nullptr;
  }

  const std::shared_ptr<DeviceArrayState>& device_arrays() const { return device_arrays_; }

  bool world_within_direct_access() override { return world_within_direct_access_; }

  uintptr_t cco_window_handle() const { return reinterpret_cast<uintptr_t>(window_->handle()); }
  uintptr_t cco_local_ptr() const { return reinterpret_cast<uintptr_t>(window_->local_ptr()); }
  uintptr_t cco_peer_ptr(int pe) const {
    TORCH_CHECK(pe >= 0 && pe < world_size_, "peer rank ", pe, " is outside [0, ", world_size_,
                ")");
    return reinterpret_cast<uintptr_t>(window_->peer_ptr(pe));
  }
  int cco_lsa_rank() const { return window_->group()->lsa_rank(); }
  int cco_lsa_size() const { return window_->group()->lsa_size(); }
  int cco_lsa_start() const { return window_->group()->lsa_start(); }
  size_t cco_per_rank_size() const { return window_->group()->per_rank_size(); }

  void barrier(int, size_t) override {
    TORCH_CHECK(false,
                "mori symm backend: barrier is not implemented; synchronise on the "
                "host with dist.barrier() for now. ",
                kNoSignalPad);
  }
  void put_signal(int, int, size_t) override {
    TORCH_CHECK(false, "mori symm backend: put_signal is not implemented. ", kNoSignalPad);
  }
  void wait_signal(int, int, size_t) override {
    TORCH_CHECK(false, "mori symm backend: wait_signal is not implemented. ", kNoSignalPad);
  }

 private:
  std::shared_ptr<CcoWindowState> window_;
  std::shared_ptr<AllocationState> allocation_;
  size_t buffer_size_;
  int rank_;
  int world_size_;
  c10::Device device_;
  std::vector<int> rank_to_global_rank_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  std::shared_ptr<DeviceArrayState> device_arrays_;
  bool world_within_direct_access_ = false;
  bool closed_ = false;
};

// MORI-specific companion to torch's pointer-array handle. Holding this object pins the
// underlying MoriSymmetricMemory, so the raw GPU ccoWindow_t cannot dangle while a kernel
// is using it.
class CcoLsaWindow {
 public:
  explicit CcoLsaWindow(c10::intrusive_ptr<MoriSymmetricMemory> owner) : owner_(std::move(owner)) {}

  uintptr_t window_handle() const { return owner_->cco_window_handle(); }
  uintptr_t local_ptr() const { return owner_->cco_local_ptr(); }
  uintptr_t peer_ptr(int pe) const { return owner_->cco_peer_ptr(pe); }
  int rank() const { return owner_->get_rank(); }
  int world_size() const { return owner_->get_world_size(); }
  int lsa_rank() const { return owner_->cco_lsa_rank(); }
  int lsa_size() const { return owner_->cco_lsa_size(); }
  int lsa_start() const { return owner_->cco_lsa_start(); }
  size_t per_rank_size() const { return owner_->cco_per_rank_size(); }

 private:
  c10::intrusive_ptr<MoriSymmetricMemory> owner_;
};

struct CcoGroupKey {
  std::string name;
  const void* store = nullptr;
  int device_idx = 0;

  bool operator==(const CcoGroupKey& other) const {
    return name == other.name && store == other.store && device_idx == other.device_idx;
  }
};

struct CcoGroupKeyHash {
  size_t operator()(const CcoGroupKey& key) const {
    size_t h = std::hash<std::string>{}(key.name);
    h ^= std::hash<const void*>{}(key.store) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.device_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct CcoGroupEntry {
  bool creating = true;
  std::shared_ptr<CcoGroupContext> context;
  std::exception_ptr error;
  std::condition_variable ready;
};

class MoriSymmAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx, const std::optional<std::string>& group_name) override {
    DeviceGuard guard(device_idx);

    const auto handle_type = ProbeHandleType(device_idx);
    auto prop = MakeProp(device_idx, handle_type);
    size_t gran = 0;
    MORI_HIP_CHECK(
        hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
    const size_t alloc_size = RoundUp(size + kSignalPadBytes, gran);

    auto allocation = std::make_shared<AllocationState>(device_idx);
    hipMemGenericAllocationHandle_t handle{};
    MORI_HIP_CHECK(hipMemCreate(&handle, alloc_size, &prop, 0));
    allocation->SetHandle(handle);

    void* ptr = nullptr;
    MORI_HIP_CHECK(hipMemAddressReserve(&ptr, alloc_size, gran, nullptr, 0));
    allocation->SetAddress(ptr, alloc_size);
    MORI_HIP_CHECK(hipMemMap(ptr, alloc_size, 0, handle, 0));
    allocation->MarkMapped();
    SetRwAccess(ptr, alloc_size, device_idx);
    MORI_HIP_CHECK(hipMemset(ptr, 0, alloc_size));
    int export_fd = -1;
    if (handle_type == hipMemHandleTypePosixFileDescriptor) {
      MORI_HIP_CHECK(hipMemExportToShareableHandle(&export_fd, handle,
                                                   hipMemHandleTypePosixFileDescriptor, 0));
      allocation->SetExportFd(export_fd);
    }

    auto block = std::make_shared<Block>();
    block->ptr = ptr;
    block->buffer_size = size;
    block->alloc_size = alloc_size;
    block->device_idx = device_idx;
    block->handle = handle;
    block->handle_type = handle_type;
    block->default_group_name = group_name;
    block->allocation = std::move(allocation);

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
    if (!RuntimeUsable()) return;
    std::lock_guard<std::mutex> lifecycle_lock(block->lifecycle_mutex);
    // Unmapping a range a kernel is still writing is a page fault, not a stale read, and
    // torch's contract lets a symmetric tensor die with work outstanding. Sync here
    // rather than in the window's destructor, which a never-rendezvous'd block has none of.
    DeviceGuard guard(block->device_idx);
    (void)hipDeviceSynchronize();
    // A surviving torch/CCO handle retains both AllocationState and the CCO window.
    // Otherwise MoriSymmetricMemory closes them in the driver-safe order.
    block->symm.reset();
    block->allocation.reset();
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

    auto name = group_name.has_value() ? group_name : block->default_group_name;
    TORCH_CHECK(name.has_value(),
                "mori symm backend: group_name given neither at allocation nor rendezvous");
    std::unique_lock<std::mutex> lifecycle_lock(block->lifecycle_mutex);
    if (block->symm != nullptr) {
      TORCH_CHECK(block->rendezvous_group_name == name,
                  "mori symm backend: allocation already rendezvoused with group '",
                  block->rendezvous_group_name.value_or("<unknown>"),
                  "', cannot reuse it with group '", *name, "'");
      return block->symm;
    }
    auto& info = c10d::symmetric_memory::get_group_info(*name);

    auto group = GetOrCreateGroup(*name, info, block->device_idx);
    DeviceGuard guard(block->device_idx);
    mori::cco::ccoWindow_t window_handle = nullptr;
    void* local_alias = nullptr;
    int global_rank = info.rank;
    try {
      auto global_group = c10d::resolve_process_group("0");
      if (global_group) global_rank = global_group->getRank();
    } catch (const c10::Error&) {
      // Custom groups created through set_group_info may have no native world
      // process group. Their logical rank is the best available mapping.
    }
    std::vector<int> rank_to_global_rank;
    {
      std::lock_guard<std::mutex> lock(group->window_mutex());
      rank_to_global_rank = group->ValidateWindow(
          block->alloc_size, block->buffer_size, static_cast<int>(block->handle_type), global_rank);
      mori::cco::ccoWindowRegisterOptions options = CCO_WINDOW_REGISTER_OPTIONS_INITIALIZER;
      options.flags = mori::cco::CCO_WINDOW_REGISTER_LSA_ONLY;
      MORI_CCO_CHECK(mori::cco::ccoWindowRegisterExternal(
          group->comm(), block->ptr, block->alloc_size, &options, &window_handle, &local_alias));
    }

    std::shared_ptr<CcoWindowState> window;
    try {
      window = std::make_shared<CcoWindowState>(group, window_handle, local_alias);
    } catch (...) {
      std::lock_guard<std::mutex> lock(group->window_mutex());
      (void)mori::cco::ccoWindowDeregister(group->comm(), window_handle);
      (void)mori::cco::ccoMemFree(group->comm(), local_alias);
      throw;
    }
    TrackResources(window, block->allocation);

    auto symm = c10::make_intrusive<MoriSymmetricMemory>(
        std::move(window), block->allocation, block->buffer_size, std::move(rank_to_global_rank));
    TrackDeviceArrays(symm->device_arrays());
    block->symm = symm;
    block->rendezvous_group_name = *name;
    return symm;
  }

  bool has_multicast_support(int) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "MORI"; }

  CcoLsaWindow GetCcoWindow(void* ptr) {
    auto block = FindBlock(ptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a live mori allocation");
    std::lock_guard<std::mutex> lifecycle_lock(block->lifecycle_mutex);
    TORCH_CHECK(block->symm != nullptr,
                "mori symm backend: tensor has not been rendezvoused yet; call "
                "symm_mem.rendezvous(tensor, group) first");
    return CcoLsaWindow(block->symm);
  }

  // Drop every live allocation while python and HIP are still up. Registered as an
  // atexit hook: destructors that run during interpreter shutdown are too late, and
  // segfault even though is_finalizing() is still false.
  void Shutdown() {
    std::unordered_map<void*, std::shared_ptr<Block>> taken;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      taken.swap(blocks_);
    }
    if (!RuntimeUsable()) return;
    std::vector<std::shared_ptr<DeviceArrayState>> live_device_arrays;
    std::vector<std::shared_ptr<CcoWindowState>> live_windows;
    std::vector<std::shared_ptr<AllocationState>> live_allocations;
    {
      std::lock_guard<std::mutex> lock(resource_mutex_);
      for (auto& weak : live_device_arrays_) {
        if (auto resource = weak.lock()) live_device_arrays.push_back(std::move(resource));
      }
      for (auto& weak : live_windows_) {
        if (auto resource = weak.lock()) live_windows.push_back(std::move(resource));
      }
      for (auto& weak : live_allocations_) {
        if (auto resource = weak.lock()) live_allocations.push_back(std::move(resource));
      }
      live_device_arrays_.clear();
      live_windows_.clear();
      live_allocations_.clear();
    }
    // Windows borrow their external allocation handles, so close every CCO
    // alias before releasing any original allocation.
    for (auto& arrays : live_device_arrays) arrays->Close();
    for (auto& window : live_windows) window->Close();
    for (auto& allocation : live_allocations) allocation->Close();

    for (auto& item : taken) {
      auto& block = item.second;
      std::lock_guard<std::mutex> lifecycle_lock(block->lifecycle_mutex);
      DeviceGuard guard(block->device_idx);
      (void)hipDeviceSynchronize();
      // Force-close even when a Python CcoLsaWindow still owns the wrapper; its
      // later destructor is idempotent and must not touch HIP after atexit.
      if (block->symm) block->symm->Close();
      block->symm.reset();
      if (block->allocation) block->allocation->Close();
      block->allocation.reset();
    }
    std::unordered_map<CcoGroupKey, std::shared_ptr<CcoGroupEntry>, CcoGroupKeyHash> groups;
    {
      std::lock_guard<std::mutex> lock(group_mutex_);
      groups.swap(groups_);
      group_devices_.clear();
    }
    for (auto& item : groups) {
      auto& entry = item.second;
      if (entry && entry->context) entry->context->Close();
    }
    groups.clear();
  }

  std::shared_ptr<Block> FindBlock(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    return it == blocks_.end() ? nullptr : it->second;
  }

 private:
  void TrackDeviceArrays(const std::shared_ptr<DeviceArrayState>& arrays) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    live_device_arrays_.erase(std::remove_if(live_device_arrays_.begin(), live_device_arrays_.end(),
                                             [](const auto& weak) { return weak.expired(); }),
                              live_device_arrays_.end());
    live_device_arrays_.push_back(arrays);
  }

  void TrackResources(const std::shared_ptr<CcoWindowState>& window,
                      const std::shared_ptr<AllocationState>& allocation) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    live_windows_.erase(std::remove_if(live_windows_.begin(), live_windows_.end(),
                                       [](const auto& weak) { return weak.expired(); }),
                        live_windows_.end());
    live_allocations_.erase(std::remove_if(live_allocations_.begin(), live_allocations_.end(),
                                           [](const auto& weak) { return weak.expired(); }),
                            live_allocations_.end());
    live_windows_.push_back(window);
    live_allocations_.push_back(allocation);
  }

  std::shared_ptr<CcoGroupContext> GetOrCreateGroup(const std::string& name,
                                                    c10d::symmetric_memory::GroupInfo& info,
                                                    int device_idx) {
    CcoGroupKey key{name, info.store.get(), device_idx};
    std::shared_ptr<CcoGroupEntry> entry;
    {
      std::unique_lock<std::mutex> lock(group_mutex_);
      auto& devices_by_group = group_devices_[info.store.get()];
      auto device_it = devices_by_group.find(name);
      if (device_it == devices_by_group.end()) {
        devices_by_group.emplace(name, device_idx);
      } else {
        TORCH_CHECK(device_it->second == device_idx,
                    "mori CCO LSA Stage 2 supports one local device per process group; group '",
                    name, "' already uses device ", device_it->second, " but device ", device_idx,
                    " was requested");
      }
      auto it = groups_.find(key);
      if (it == groups_.end()) {
        entry = std::make_shared<CcoGroupEntry>();
        groups_.emplace(key, entry);
      } else {
        entry = it->second;
        entry->ready.wait(lock, [&] { return !entry->creating; });
        if (entry->error) std::rethrow_exception(entry->error);
        return entry->context;
      }
    }

    try {
      auto group = std::make_shared<CcoGroupContext>(name, info.store, info.rank, info.world_size,
                                                     device_idx);
      {
        std::lock_guard<std::mutex> lock(group_mutex_);
        entry->context = group;
        entry->creating = false;
      }
      entry->ready.notify_all();
      return group;
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(group_mutex_);
        entry->error = std::current_exception();
        entry->creating = false;
      }
      entry->ready.notify_all();
      throw;
    }
  }

  std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<Block>> blocks_;
  std::mutex resource_mutex_;
  std::vector<std::weak_ptr<DeviceArrayState>> live_device_arrays_;
  std::vector<std::weak_ptr<CcoWindowState>> live_windows_;
  std::vector<std::weak_ptr<AllocationState>> live_allocations_;
  std::mutex group_mutex_;
  std::unordered_map<const void*, std::unordered_map<std::string, int>> group_devices_;
  std::unordered_map<CcoGroupKey, std::shared_ptr<CcoGroupEntry>, CcoGroupKeyHash> groups_;
};

c10::intrusive_ptr<MoriSymmAllocator>& AllocatorSingleton() {
  // Immortal on purpose: register_availability() parks a reference in a libtorch-owned
  // registry that outlives this extension's statics.
  static auto* inst =
      new c10::intrusive_ptr<MoriSymmAllocator>(c10::make_intrusive<MoriSymmAllocator>());
  return *inst;
}

// Arithmetic peer addressing, which torch's interface has no place for.

void Shutdown() { AllocatorSingleton()->Shutdown(); }

CcoLsaWindow GetCcoWindow(uintptr_t storage_ptr) {
  return AllocatorSingleton()->GetCcoWindow(reinterpret_cast<void*>(storage_ptr));
}

std::string HandleTypeName(int dev) {
  return ProbeHandleType(dev) == hipMemHandleTypeFabricCompat ? "fabric" : "posix_fd";
}

}  // namespace

// Idempotent.
void RegisterTorchSymmBackend() {
  static bool registered = [] {
    c10d::symmetric_memory::register_availability("MORI", AllocatorSingleton());
    return true;
  }();
  (void)registered;
}

}  // namespace allocator
}  // namespace mori

// Importing the extension registers the backend.
PYBIND11_MODULE(mori_torch_symm, m) {
  mori::allocator::RegisterTorchSymmBackend();
  pybind11::class_<mori::allocator::CcoLsaWindow>(m, "CcoLsaWindow")
      .def_property_readonly("window_handle", &mori::allocator::CcoLsaWindow::window_handle)
      .def_property_readonly("local_ptr", &mori::allocator::CcoLsaWindow::local_ptr)
      .def("peer_ptr", &mori::allocator::CcoLsaWindow::peer_ptr, pybind11::arg("peer"))
      .def_property_readonly("rank", &mori::allocator::CcoLsaWindow::rank)
      .def_property_readonly("world_size", &mori::allocator::CcoLsaWindow::world_size)
      .def_property_readonly("lsa_rank", &mori::allocator::CcoLsaWindow::lsa_rank)
      .def_property_readonly("lsa_size", &mori::allocator::CcoLsaWindow::lsa_size)
      .def_property_readonly("lsa_start", &mori::allocator::CcoLsaWindow::lsa_start)
      .def_property_readonly("per_rank_size", &mori::allocator::CcoLsaWindow::per_rank_size);
  m.def("register_backend", &mori::allocator::RegisterTorchSymmBackend,
        "Register the MORI symmetric memory backend with torch (idempotent)");
  m.def("shutdown", &mori::allocator::Shutdown,
        "Release every live symmetric allocation (registered as an atexit hook)");
  m.def("get_cco_window", &mori::allocator::GetCcoWindow, pybind11::arg("storage_ptr"),
        "Return an owning CCO LSA view for a rendezvoused MORI allocation");
  m.def("handle_type", &mori::allocator::HandleTypeName,
        "'fabric' or 'posix_fd' -- what this device can export");
  m.attr("backend_name") = "MORI";
  m.attr("lsa_stage2_api_version") = 1;
  // Whether the window carries torch's signal pad. False by default, and then anything
  // that synchronises on the device -- barrier(), put/wait_signal(), and torch's own
  // symm_mem collectives, which barrier internally -- raises instead.
  m.attr("signal_pad_supported") = mori::allocator::kSignalPadBytes != 0;
}
