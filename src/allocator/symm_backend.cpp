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

// symm_backend.cpp -- a torch SymmetricMemory backend on plain HIP VMM.
//
// register_availability() makes "MORI" selectable; torch then drives symm_mem.empty ->
// alloc, symm_mem.rendezvous -> rendezvous, and torch.ops.symm_mem.* on the result.
//
// Deliberately does not use mori's shmem or cco allocators: both keep peer offsets
// aligned only while every rank allocates AND frees in the same order, which torch cannot
// hold (tensors die on Python GC, whose order is not synchronised across ranks). Here each
// allocation is independent and free() is local, so divergent free order is harmless.
//
// Peers are published torch's way, as the buffer_ptrs / buffer_ptrs_dev array. They happen
// to sit in one flat span (peer(r) == flat_base + r*stride) because that is the cheapest
// way to map them, but that layout is not part of the API here: exposing it belongs with
// mori's cco window, which already defines it. See ROCm/mori#557.
//
// Handle type is probed per device -- fabric where supported, POSIX fd otherwise (gfx9 has
// none). Fabric handles are portable bytes and ride the torch Store; fds need SCM_RIGHTS.

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/utils/hip_compat.hpp"

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
  // KNOWN GAP: releasing a rendezvous'd window segfaults at world_size >= 4 (fine at 2),
  // somewhere in the unmap/release path. Until that is understood, teardown is off by
  // default -- symmetric buffers are few and long-lived, so leaking them is far less
  // harmful than crashing. MORI_SYMM_TEARDOWN=1 re-enables it for debugging.
  static const bool teardown = [] {
    const char* e = getenv("MORI_SYMM_TEARDOWN");
    return e && *e == '1';
  }();
  if (!teardown) return false;
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
  auto prop = MakeProp(dev, hipMemHandleTypeFabricCompat);
  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended) ==
          hipSuccess &&
      gran > 0) {
    hipMemGenericAllocationHandle_t h{};
    if (hipMemCreate(&h, gran, &prop, 0) == hipSuccess) {
      hipMemFabricHandle_compat_t blob;
      if (hipMemExportToShareableHandle(&blob, h, hipMemHandleTypeFabricCompat, 0) == hipSuccess) {
        chosen = hipMemHandleTypeFabricCompat;
      }
      (void)hipMemRelease(h);
    }
  }
  (void)hipGetLastError();  // the probe is expected to fail on gfx9
  cache[dev] = chosen;
  return chosen;
}

// An fd is an index into one process's table, so it needs SCM_RIGHTS. torch has an
// IpcChannel for this but does not export it, hence this local equivalent.
class FdChannel {
 public:
  explicit FdChannel(int rank) : path_(Path(getpid(), rank)) {
    sock_ = ::socket(AF_UNIX, SOCK_DGRAM, 0);
    TORCH_CHECK(sock_ >= 0, "mori symm backend: socket() failed");
    ::unlink(path_.c_str());
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
    TORCH_CHECK(::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                "mori symm backend: bind(", path_, ") failed");
  }

  ~FdChannel() {
    if (sock_ >= 0) ::close(sock_);
    ::unlink(path_.c_str());
  }

  static std::string Path(int pid, int rank) {
    return "/tmp/mori_symm_" + std::to_string(pid) + "_" + std::to_string(rank);
  }

  // The payload carries the sender's rank: datagrams from different owners can arrive in
  // any order, so the receiver must not assume one round per owner.
  void SendFd(const std::string& dst, int fd, int sender_rank) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, dst.c_str(), sizeof(addr.sun_path) - 1);

    int tag = sender_rank;
    iovec iov{&tag, sizeof(tag)};
    char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_name = &addr;
    msg.msg_namelen = sizeof(addr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

    // The peer may not have bound yet.
    ssize_t n = -1;
    for (int retry = 0; retry < 400 && n < 0; ++retry) {
      n = ::sendmsg(sock_, &msg, 0);
      if (n < 0) ::usleep(5000);
    }
    TORCH_CHECK(n >= 0, "mori symm backend: sendmsg to ", dst, " failed");
  }

  // Returns (sender_rank, fd).
  std::pair<int, int> RecvFd() {
    int tag = -1;
    iovec iov{&tag, sizeof(tag)};
    char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    TORCH_CHECK(::recvmsg(sock_, &msg, 0) >= 0, "mori symm backend: recvmsg failed");
    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    TORCH_CHECK(cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS,
                "mori symm backend: no SCM_RIGHTS in received message");
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return {tag, fd};
  }

 private:
  std::string path_;
  int sock_ = -1;
};

// What each rank publishes through the torch Store.
struct RendezvousReq {
  size_t alloc_size;
  size_t buffer_size;
  int device_idx;
  int pid;
  hipMemFabricHandle_compat_t fabric;  // meaningful only on the fabric path
};

struct Block {
  void* ptr = nullptr;
  size_t buffer_size = 0;
  size_t alloc_size = 0;
  int device_idx = 0;
  hipMemGenericAllocationHandle_t handle{};
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
  std::optional<std::string> default_group_name;
  c10::intrusive_ptr<SymmetricMemory> symm;
};

constexpr const char* kNoSignalPad =
    "Signal-pad support is compiled out (MORI_SYMM_SIGNAL_PAD=0), so no pad is reserved "
    "in the window; rebuild with -DMORI_SYMM_SIGNAL_PAD=1 to allocate it.";

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(char* flat_base, size_t span, size_t stride, size_t buffer_size,
                      std::vector<hipMemGenericAllocationHandle_t> handles, int rank,
                      int world_size, int device_idx)
      : flat_base_(flat_base),
        span_(span),
        stride_(stride),
        buffer_size_(buffer_size),
        handles_(std::move(handles)),
        rank_(rank),
        world_size_(world_size),
        device_(c10::DeviceType::CUDA, device_idx) {
    rank_to_global_rank_.resize(world_size_);
    buffers_.reserve(world_size_);
    signal_pads_.reserve(world_size_);
    for (int r = 0; r < world_size_; ++r) {
      rank_to_global_rank_[r] = r;
      char* slot = flat_base_ + static_cast<size_t>(r) * stride_;
      buffers_.push_back(slot);
      signal_pads_.push_back(kSignalPadBytes ? slot + buffer_size_ : nullptr);
    }

    const size_t arr = world_size_ * sizeof(void*);
    MORI_HIP_CHECK(hipMalloc(&buffers_dev_, arr));
    MORI_HIP_CHECK(hipMemcpy(buffers_dev_, buffers_.data(), arr, hipMemcpyHostToDevice));
    if (kSignalPadBytes) {
      MORI_HIP_CHECK(hipMalloc(&signal_pads_dev_, arr));
      MORI_HIP_CHECK(hipMemcpy(signal_pads_dev_, signal_pads_.data(), arr, hipMemcpyHostToDevice));
    }
    MORI_HIP_CHECK(hipMalloc(&rank_to_global_rank_dev_, world_size_ * sizeof(int)));
    MORI_HIP_CHECK(hipMemcpy(rank_to_global_rank_dev_, rank_to_global_rank_.data(),
                             world_size_ * sizeof(int), hipMemcpyHostToDevice));
  }

  ~MoriSymmetricMemory() override {
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    (void)hipDeviceSynchronize();
    if (buffers_dev_) (void)hipFree(buffers_dev_);
    if (signal_pads_dev_) (void)hipFree(signal_pads_dev_);
    if (rank_to_global_rank_dev_) (void)hipFree(rank_to_global_rank_dev_);
    for (int r = 0; r < world_size_; ++r) {
      (void)hipMemUnmap(flat_base_ + static_cast<size_t>(r) * stride_, stride_);
      (void)hipMemRelease(handles_[r]);
    }
    (void)hipMemAddressFree(flat_base_, span_);
  }

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override {
    TORCH_CHECK(kSignalPadBytes != 0, kNoSignalPad);
    return signal_pads_;
  }
  void** get_buffer_ptrs_dev() override { return buffers_dev_; }
  void** get_signal_pad_ptrs_dev() override {
    TORCH_CHECK(kSignalPadBytes != 0, kNoSignalPad);
    return signal_pads_dev_;
  }
  size_t get_buffer_size() override { return buffer_size_; }
  size_t get_offset() override { return 0; }

  bool has_multicast_support() override { return false; }
  void* get_multicast_ptr() override { return nullptr; }

  int get_rank() override { return rank_; }
  int get_world_size() override { return world_size_; }
  c10::Device get_device() override { return device_; }

  const std::vector<int>& get_rank_to_global_rank() override { return rank_to_global_rank_; }
  int* get_rank_to_global_rank_dev() override { return rank_to_global_rank_dev_; }

  // Every rank is mapped for load/store; there is no RDMA fallback in this backend.
  bool world_within_direct_access() override { return true; }

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
  char* flat_base_;
  size_t span_;
  size_t stride_;
  size_t buffer_size_;
  std::vector<hipMemGenericAllocationHandle_t> handles_;
  int rank_;
  int world_size_;
  c10::Device device_;
  std::vector<int> rank_to_global_rank_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  int* rank_to_global_rank_dev_ = nullptr;
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

    hipMemGenericAllocationHandle_t handle{};
    MORI_HIP_CHECK(hipMemCreate(&handle, alloc_size, &prop, 0));

    void* ptr = nullptr;
    MORI_HIP_CHECK(hipMemAddressReserve(&ptr, alloc_size, gran, nullptr, 0));
    MORI_HIP_CHECK(hipMemMap(ptr, alloc_size, 0, handle, 0));
    SetRwAccess(ptr, alloc_size, device_idx);
    MORI_HIP_CHECK(hipMemset(ptr, 0, alloc_size));

    auto block = std::make_shared<Block>();
    block->ptr = ptr;
    block->buffer_size = size;
    block->alloc_size = alloc_size;
    block->device_idx = device_idx;
    block->handle = handle;
    block->handle_type = handle_type;
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
    // Local only: the flat span is owned by the SymmetricMemory object.
    if (!RuntimeUsable()) return;
    (void)hipMemUnmap(block->ptr, block->alloc_size);
    (void)hipMemAddressFree(block->ptr, block->alloc_size);
    (void)hipMemRelease(block->handle);
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
    if (block->symm != nullptr) return block->symm;

    auto name = group_name.has_value() ? group_name : block->default_group_name;
    TORCH_CHECK(name.has_value(),
                "mori symm backend: group_name given neither at allocation nor rendezvous");
    auto& info = c10d::symmetric_memory::get_group_info(*name);
    const int rank = info.rank;
    const int world_size = info.world_size;

    DeviceGuard guard(block->device_idx);
    const bool use_fabric = (block->handle_type == hipMemHandleTypeFabricCompat);

    RendezvousReq local{};
    local.alloc_size = block->alloc_size;
    local.buffer_size = block->buffer_size;
    local.device_idx = block->device_idx;
    local.pid = getpid();
    if (use_fabric) {
      MORI_HIP_CHECK(hipMemExportToShareableHandle(&local.fabric, block->handle,
                                                   hipMemHandleTypeFabricCompat, 0));
    }
    auto reqs = store_exchange_.all_gather(info.store, rank, world_size, local);
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(
          reqs[r].alloc_size == local.alloc_size && reqs[r].buffer_size == local.buffer_size,
          "mori symm backend: rank ", r, " allocated ", reqs[r].buffer_size,
          " bytes but this rank "
          "allocated ",
          local.buffer_size, "; symm_mem.empty must be symmetric across ranks");
    }

    const size_t stride = block->alloc_size;
    const size_t span = static_cast<size_t>(world_size) * stride;
    auto prop = MakeProp(block->device_idx, block->handle_type);
    size_t gran = 0;
    MORI_HIP_CHECK(
        hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));

    void* flat = nullptr;
    MORI_HIP_CHECK(hipMemAddressReserve(&flat, span, gran, nullptr, 0));
    auto map_slot = [&](int r, hipMemGenericAllocationHandle_t h) {
      char* slot = static_cast<char*>(flat) + static_cast<size_t>(r) * stride;
      MORI_HIP_CHECK(hipMemMap(slot, stride, 0, h, 0));
      SetRwAccess(slot, stride, block->device_idx);
    };

    std::vector<hipMemGenericAllocationHandle_t> handles(world_size);

    // A second alias of our own allocation, so the stride is uniform across all ranks.
    MORI_HIP_CHECK(hipMemRetainAllocationHandle(&handles[rank], block->ptr));
    map_slot(rank, handles[rank]);

    if (use_fabric) {
      for (int r = 0; r < world_size; ++r) {
        if (r == rank) continue;
        MORI_HIP_CHECK(hipMemImportFromShareableHandle(&handles[r], &reqs[r].fabric,
                                                       hipMemHandleTypeFabricCompat));
        map_slot(r, handles[r]);
      }
    } else {
      // Send ours to everyone, then take world-1 fds in whatever order they land.
      FdChannel chan(rank);
      int fd = -1;
      MORI_HIP_CHECK(hipMemExportToShareableHandle(&fd, block->handle,
                                                   hipMemHandleTypePosixFileDescriptor, 0));
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer != rank) chan.SendFd(FdChannel::Path(reqs[peer].pid, peer), fd, rank);
      }
      ::close(fd);

      for (int i = 0; i < world_size - 1; ++i) {
        auto [owner, peer_fd] = chan.RecvFd();
        TORCH_CHECK(owner >= 0 && owner < world_size && owner != rank,
                    "mori symm backend: bad owner tag ", owner, " in fd exchange");
        MORI_HIP_CHECK(hipMemImportFromShareableHandle(
            &handles[owner], reinterpret_cast<void*>(static_cast<intptr_t>(peer_fd)),
            hipMemHandleTypePosixFileDescriptor));
        ::close(peer_fd);
        map_slot(owner, handles[owner]);
      }
    }

    auto symm = c10::make_intrusive<MoriSymmetricMemory>(static_cast<char*>(flat), span, stride,
                                                         block->buffer_size, std::move(handles),
                                                         rank, world_size, block->device_idx);
    block->symm = symm;
    return symm;
  }

  bool has_multicast_support(int) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "MORI"; }

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
    for (auto& [ptr, block] : taken) {
      block->symm.reset();  // unmaps the flat span
      (void)hipMemUnmap(block->ptr, block->alloc_size);
      (void)hipMemAddressFree(block->ptr, block->alloc_size);
      (void)hipMemRelease(block->handle);
    }
  }

  std::shared_ptr<Block> FindBlock(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    return it == blocks_.end() ? nullptr : it->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<Block>> blocks_;
  c10d::symmetric_memory::StoreExchange store_exchange_{"mori_symm_backend"};
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
  m.def("register_backend", &mori::allocator::RegisterTorchSymmBackend,
        "Register the MORI symmetric memory backend with torch (idempotent)");
  m.def("shutdown", &mori::allocator::Shutdown,
        "Release every live symmetric allocation (registered as an atexit hook)");
  m.def("handle_type", &mori::allocator::HandleTypeName,
        "'fabric' or 'posix_fd' -- what this device can export");
  m.attr("backend_name") = "MORI";
  // Whether the window carries torch's signal pad. False by default, and then anything
  // that synchronises on the device -- barrier(), put/wait_signal(), and torch's own
  // symm_mem collectives, which barrier internally -- raises instead.
  m.attr("signal_pad_supported") = mori::allocator::kSignalPadBytes != 0;
}
