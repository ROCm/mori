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
// alloc() is plain HIP VMM, kept independent per rank; CCO owns the peer mapping from
// rendezvous() onward. Window teardown is local -- ccoWindowDeregister runs no collective
// -- so tensors may die in whatever order Python's GC picks. The POSIX-fd path does embed
// the local slot offset in its rendezvous socket path, so a genuinely rank-divergent free
// order can make a *later* register fail; it fails loudly rather than silently.
//
// Peers are published torch's way, as the buffer_ptrs / buffer_ptrs_dev array. They happen
// to sit in one flat span (peer(r) == flat_base + r*stride) because that is the cheapest
// way to map them, but that layout is not part of the API here: exposing it belongs with
// mori's cco window, which already defines it. See ROCm/mori#557.
//
// The peer exchange is CCO's: rendezvous hands the tensor to ccoWindowRegister, which
// aliases it into the flat LSA space and returns a window. buffer_ptrs are then
// ccoGetPeerPtr results, so host and device (Window.lsa_ptr) agree by construction.
// Handle type is still probed per device for the allocation itself -- fabric where
// supported, POSIX fd otherwise (gfx9 has none).

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/version.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mori/cco/cco.hpp>
#include <mutex>
#include <string>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

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

// torch's signal pad size. It gets its own cco allocation and window rather than being
// appended to the user's buffer, which is what NCCL's backend does too: appending 9216 B
// to a page-aligned request cost a whole extra 2 MiB page, and that cost is why the pad
// used to be compiled out.
constexpr size_t kSignalPadBytes = 9216;

size_t RoundUp(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

using mori::cco::ccoComm;
using mori::cco::ccoWindow_t;

#define MORI_CCO_CHECK(expr)                                       \
  do {                                                             \
    int _rc = (expr);                                              \
    TORCH_CHECK(_rc == 0, #expr, " failed with cco status ", _rc); \
  } while (0)

// cco reserves its flat VA once per communicator and quantises the per-rank stride
// up to 4 GiB, so this is a floor rather than a budget: any request at or below
// 4 GiB produces the same reservation. It bounds live symmetric memory per rank,
// not the number of allocations over time -- slots are recycled on free.
size_t PerRankVmmSize() {
  static const size_t v = [] {
    const char* e = getenv("MORI_SYMM_PER_RANK_VMM");
    return e ? static_cast<size_t>(std::strtoull(e, nullptr, 0)) : (size_t{4} << 30);
  }();
  return v;
}

// One communicator per torch process group, made on the first rendezvous and kept
// for the process. torch calls rendezvous once per allocation, whereas ccoCommCreate
// reserves the flat VA once, so the two lifetimes do not line up without this.
struct CcoGroup {
  ccoComm* comm = nullptr;

  // group -> key -> devcomm, the shape torch's NCCL backend uses. A DevComm is
  // parameterised by ccoDevCommRequirements -- signal counts, QP counts, connection
  // type -- and those are properties of the algorithm that will run, not of the group,
  // so two collectives on one group need two of them. NCCL keys this on
  // __builtin_FUNCTION(); callers here pass the key explicitly.
  struct DevCommEntry {
    mori::cco::ccoDevComm host{};
    mori::cco::ccoDevComm* device = nullptr;  // kernel-argument copy
    int lsa_barrier_count = 0;                // what it was built with
  };
  // Serialises everything this process does with `comm`. cco's own allocMutex is
  // documented as covering allocTable/windows/windowTableEntries against concurrent
  // MemAlloc/MemFree/WindowRegister/WindowDeregister (include/mori/cco/cco.hpp), but
  // only MemAlloc/MemImport/MemFree actually take it -- the window paths do not. That
  // did not matter while cco was driven from explicit single-threaded user code; it
  // does now, because torch drops the last reference to a tensor on whatever thread
  // happens to hold it, so ~MoriSymmetricMemory can deregister a window while another
  // thread is registering one. Also guards dev_comms, so there is no lock ordering.
  std::mutex mu;
  std::unordered_map<std::string, DevCommEntry> dev_comms;

  ~CcoGroup() {
    if (!comm || c10d::symmetric_memory::is_finalizing()) return;
    for (auto& [key, e] : dev_comms) {
      if (e.device) mori::cco::ccoDevCommFreeDeviceCopy(e.device);
      (void)mori::cco::ccoDevCommDestroy(comm, &e.host);
    }
    (void)mori::cco::ccoCommDestroy(comm);
  }
};

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

// All that still crosses the Store here: the size agreement torch requires. The
// handle exchange that used to live in this struct is ccoWindowRegister's job now.
struct SizeCheck {
  size_t alloc_size;
  size_t buffer_size;
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

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(std::shared_ptr<CcoGroup> group, ccoWindow_t win, void* local_ptr,
                      ccoWindow_t pad_win, void* pad_ptr, std::vector<void*> buffers,
                      std::vector<void*> signal_pads, size_t buffer_size, int rank, int world_size,
                      int device_idx)
      : group_(std::move(group)),
        win_(win),
        local_ptr_(local_ptr),
        pad_win_(pad_win),
        pad_ptr_(pad_ptr),
        buffers_(std::move(buffers)),
        signal_pads_(std::move(signal_pads)),
        buffer_size_(buffer_size),
        rank_(rank),
        world_size_(world_size),
        device_(c10::DeviceType::CUDA, device_idx) {
    rank_to_global_rank_.resize(world_size_);
    for (int r = 0; r < world_size_; ++r) rank_to_global_rank_[r] = r;

    const size_t arr = world_size_ * sizeof(void*);
    MORI_HIP_CHECK(hipMalloc(&buffers_dev_, arr));
    MORI_HIP_CHECK(hipMemcpy(buffers_dev_, buffers_.data(), arr, hipMemcpyHostToDevice));
    MORI_HIP_CHECK(hipMalloc(&signal_pads_dev_, arr));
    MORI_HIP_CHECK(hipMemcpy(signal_pads_dev_, signal_pads_.data(), arr, hipMemcpyHostToDevice));
    MORI_HIP_CHECK(hipMalloc(&rank_to_global_rank_dev_, world_size_ * sizeof(int)));
    MORI_HIP_CHECK(hipMemcpy(rank_to_global_rank_dev_, rank_to_global_rank_.data(),
                             world_size_ * sizeof(int), hipMemcpyHostToDevice));
  }

  ~MoriSymmetricMemory() override {
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    DeviceGuard guard(device_.index());
    (void)hipDeviceSynchronize();
    if (buffers_dev_) (void)hipFree(buffers_dev_);
    if (signal_pads_dev_) (void)hipFree(signal_pads_dev_);
    if (rank_to_global_rank_dev_) (void)hipFree(rank_to_global_rank_dev_);
    // Purely local: ccoWindowDeregister unmaps this rank's view of its peers and drops
    // the handles it imported, with no collective inside, so this can run on whichever
    // thread drops the last reference and in whatever order across ranks.
    if (group_ && group_->comm) {
      std::lock_guard<std::mutex> lock(group_->mu);
      (void)mori::cco::ccoWindowDeregister(group_->comm, pad_win_);
      (void)mori::cco::ccoMemFree(group_->comm, pad_ptr_);
      (void)mori::cco::ccoWindowDeregister(group_->comm, win_);
      (void)mori::cco::ccoMemFree(group_->comm, local_ptr_);
    }
  }

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override { return signal_pads_; }
  void** get_buffer_ptrs_dev() override { return buffers_dev_; }
  void** get_signal_pad_ptrs_dev() override { return signal_pads_dev_; }
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
  int* get_rank_to_global_rank_dev() override { return rank_to_global_rank_dev_; }

  // True only if cco mapped every peer. A null entry means that rank needs RDMA, which
  // this backend cannot drive yet -- but reporting it honestly is what lets torch fall
  // back rather than dereference a null pointer.
  bool world_within_direct_access() override {
    for (void* p : buffers_)
      if (p == nullptr) return false;
    return true;
  }

  // cco's barrier is a host collective, while torch's is meant to be stream-ordered,
  // so the current stream is drained first. That makes this stronger than asked for --
  // every rank's prior GPU work is complete before any rank returns -- and heavier.
  // A device-side barrier belongs with a DevComm, which this backend does not own yet.
  void barrier(int channel, size_t /*timeout_ms*/) override {
    TORCH_CHECK(channel == 0,
                "mori symm backend: only channel 0 exists; cco's barrier is comm-wide");
    TORCH_CHECK(group_ && group_->comm, "mori symm backend: window has no communicator");
    DeviceGuard guard(device_.index());
    MORI_HIP_CHECK(hipStreamSynchronize(c10::hip::getCurrentHIPStream()));
    MORI_CCO_CHECK(mori::cco::ccoBarrierAll(group_->comm));
  }

  // Point-to-point signalling needs cco signals, which live on a ccoDevComm that this
  // backend does not create yet. torch's own collectives do not call these -- they
  // synchronise through the signal pad above.
  void put_signal(int, int, size_t) override {
    TORCH_CHECK(false, "mori symm backend: put_signal is not implemented yet");
  }
  void wait_signal(int, int, size_t) override {
    TORCH_CHECK(false, "mori symm backend: wait_signal is not implemented yet");
  }

  // Backend-specific, deliberately not part of torch's interface -- the same shape as
  // NCCLSymmetricMemory::get_window(). A kernel that wants flat-VA addressing takes the
  // window and calls cco's device API on it instead of walking buffer_ptrs.
  ccoWindow_t get_window() const { return win_; }
  ccoWindow_t get_signal_pad_window() const { return pad_win_; }

 private:
  std::shared_ptr<CcoGroup> group_;  // keeps the communicator alive for this window
  ccoWindow_t win_;
  void* local_ptr_;  // cco's flat-VA alias of the torch tensor
  ccoWindow_t pad_win_;
  void* pad_ptr_;  // the signal pad: cco's own allocation, not part of the tensor
  size_t buffer_size_;
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
    const size_t alloc_size = RoundUp(size, gran);

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
    if (!RuntimeUsable()) return;
    // Unmapping a range a kernel is still writing is a page fault, not a stale read, and
    // torch's contract lets a symmetric tensor die with work outstanding. Sync here
    // rather than in the window's destructor, which a never-rendezvous'd block has none of.
    DeviceGuard guard(block->device_idx);
    (void)hipDeviceSynchronize();
    // Our own reference goes first, the window second. cco exported this allocation to
    // a shareable fd when it imported it, and closes that fd inside ccoMemFree; on
    // ROCm 7.14 the fd has to outlive every hipMemRelease of the allocation it names,
    // ours included, or the physical memory is never returned. cco's retained handle
    // keeps the allocation alive across these three calls.
    (void)hipMemUnmap(block->ptr, block->alloc_size);
    (void)hipMemAddressFree(block->ptr, block->alloc_size);
    (void)hipMemRelease(block->handle);
    block->symm.reset();
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
    // Resolve the process group directly rather than going through torch's GroupInfo
    // registry. That registry is only ever populated by enable_symm_mem_for_group(),
    // which torch has deprecated -- and torch's own CUDA and NCCL backends both take
    // this route instead, so callers no longer have to make that deprecated call.
    auto group = c10d::resolve_process_group(*name);
    const int rank = group->getRank();
    const int world_size = group->getSize();
    auto store = group->getStore();

    DeviceGuard guard(block->device_idx);

    // torch's contract is that every rank hands in the same size. cco's register is
    // collective and would deadlock or mismap on a mismatch, so say so first.
    SizeCheck local{block->alloc_size, block->buffer_size};
    auto sizes = store_exchange_.all_gather(store, rank, world_size, local);
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(
          sizes[r].alloc_size == local.alloc_size && sizes[r].buffer_size == local.buffer_size,
          "mori symm backend: rank ", r, " allocated ", sizes[r].buffer_size,
          " bytes but this rank allocated ", local.buffer_size,
          "; symm_mem.empty must be symmetric across ranks");
    }

    auto cco_group = GetOrCreateGroup(*name, store, rank, world_size);

    // Overload C: retain the tensor's VMM handle, alias it into the flat LSA slot,
    // and register the window. No copy, and no second peer exchange of our own --
    // this is the exchange the backend used to hand-roll over SCM_RIGHTS.
    // Held across the registers and the peer-pointer reads: see CcoGroup::mu.
    std::unique_lock<std::mutex> cco_lock(cco_group->mu);

    ccoWindow_t win{};
    void* local_ptr = nullptr;
    MORI_CCO_CHECK(mori::cco::ccoWindowRegister(cco_group->comm, block->ptr, block->alloc_size,
                                                &win, &local_ptr));

    // torch's signal pad, as its own allocation and window rather than a tail on the
    // user's buffer. cco owns this one outright, so it is overload B.
    void* pad_ptr = nullptr;
    MORI_CCO_CHECK(mori::cco::ccoMemAlloc(cco_group->comm, kSignalPadBytes, &pad_ptr));
    // Not hipMemset: ccoMemAlloc memory is fabric-exportable, and hipMemset returns
    // hipErrorOutOfMemory on UALink fabric pools (ROCm 7.15). cco hit this first and
    // routes its own DevComm resource window around it -- see CcoZeroWindowMem in
    // src/cco/cco_init.cpp. A host->device copy is also synchronous, so the pad is
    // observably zero before the collective register below lets a peer write to it.
    {
      const std::vector<char> zeros(kSignalPadBytes, 0);
      MORI_HIP_CHECK(hipMemcpy(pad_ptr, zeros.data(), kSignalPadBytes, hipMemcpyHostToDevice));
    }
    ccoWindow_t pad_win{};
    MORI_CCO_CHECK(
        mori::cco::ccoWindowRegister(cco_group->comm, pad_ptr, kSignalPadBytes, &pad_win));

    // buffer_ptrs is now one source of truth with the device-side lsa_ptr: both are
    // flatBase + peerLsaRank*stride + slotOffset. A peer outside the LSA team comes back
    // null, and that is left as null on purpose -- it is how torch's own backends report
    // a peer that has to be reached by RDMA rather than by load/store.
    std::vector<void*> buffers(world_size), pads(world_size);
    for (int r = 0; r < world_size; ++r) {
      buffers[r] = mori::cco::ccoGetPeerPtr(cco_group->comm, local_ptr, r);
      pads[r] = mori::cco::ccoGetPeerPtr(cco_group->comm, pad_ptr, r);
    }
    TORCH_CHECK(buffers[rank] != nullptr,
                "mori symm backend: cco did not map this rank's own window");
    cco_lock.unlock();

    auto symm = c10::make_intrusive<MoriSymmetricMemory>(
        cco_group, win, local_ptr, pad_win, pad_ptr, std::move(buffers), std::move(pads),
        block->buffer_size, rank, world_size, block->device_idx);
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
      DeviceGuard guard(block->device_idx);
      (void)hipDeviceSynchronize();
      block->symm.reset();  // unmaps the flat span
      (void)hipMemUnmap(block->ptr, block->alloc_size);
      (void)hipMemAddressFree(block->ptr, block->alloc_size);
      (void)hipMemRelease(block->handle);
    }
  }

  // A device communicator for this group, created on first use and cached under `key`.
  // Deliberately not built during rendezvous: rendezvous knows a buffer and a group, and
  // nothing about how many signals or QPs the kernel that follows will want.
  uint64_t DevCommPtr(const std::string& group_name, const std::string& key,
                      int lsa_barrier_count) {
    std::shared_ptr<CcoGroup> group;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cco_groups_.find(group_name);
      TORCH_CHECK(it != cco_groups_.end(), "mori symm backend: no communicator for group ",
                  group_name, ". Have you rendezvoused a tensor with this group?");
      group = it->second;
    }

    int dev = -1;
    MORI_HIP_CHECK(hipGetDevice(&dev));
    DeviceGuard guard(dev);

    std::lock_guard<std::mutex> lock(group->mu);
    auto it = group->dev_comms.find(key);
    if (it != group->dev_comms.end()) {
      // The key is supposed to encode the parameterisation, so a mismatch means two
      // callers disagree about what this key means. Returning the first one's DevComm
      // would hand the second a barrier array smaller than the ids its kernel uses.
      TORCH_CHECK(it->second.lsa_barrier_count == lsa_barrier_count,
                  "mori symm backend: dev_comm key '", key,
                  "' already exists with "
                  "lsa_barrier_count=",
                  it->second.lsa_barrier_count, ", cannot serve ", lsa_barrier_count,
                  "; use a different key");
    }
    if (it == group->dev_comms.end()) {
      CcoGroup::DevCommEntry e{};
      mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
      // Intra-node only for now: no RDMA connections, just LSA barriers. Scale-out is
      // where the caller will need to choose these, which is the reason for the key.
      reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
      reqs.lsaBarrierCount = lsa_barrier_count;
      MORI_CCO_CHECK(mori::cco::ccoDevCommCreate(group->comm, &reqs, &e.host));
      e.device = mori::cco::ccoDevCommCopyToDevice(&e.host);
      if (e.device == nullptr) {
        (void)mori::cco::ccoDevCommDestroy(group->comm, &e.host);
        TORCH_CHECK(false, "mori symm backend: ccoDevCommCopyToDevice failed");
      }
      e.lsa_barrier_count = lsa_barrier_count;
      it = group->dev_comms.emplace(key, e).first;
    }
    return reinterpret_cast<uint64_t>(it->second.device);
  }

  // The backend-specific escape hatch, keyed by the tensor pointer because torch's
  // handle has nowhere to carry it. Returns the cco window a kernel can address through.
  uint64_t WindowHandle(void* ptr, bool signal_pad) {
    auto block = FindBlock(ptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    TORCH_CHECK(block->symm != nullptr,
                "mori symm backend: rendezvous this tensor before asking for its window");
    auto* mem = static_cast<MoriSymmetricMemory*>(block->symm.get());
    return reinterpret_cast<uint64_t>(signal_pad ? mem->get_signal_pad_window()
                                                 : mem->get_window());
  }

  std::shared_ptr<Block> FindBlock(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(ptr);
    return it == blocks_.end() ? nullptr : it->second;
  }

  // One communicator per group, built on first use. The uid rides torch's own
  // Store, so cco's rendezvous needs no second bootstrap channel.
  std::shared_ptr<CcoGroup> GetOrCreateGroup(const std::string& name,
                                             const c10::intrusive_ptr<c10d::Store>& store, int rank,
                                             int world_size) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cco_groups_.find(name);
      if (it != cco_groups_.end()) return it->second;
    }

    mori::cco::ccoUniqueId uid{};
    if (rank == 0) MORI_CCO_CHECK(mori::cco::ccoGetUniqueId(&uid));
    auto ids = store_exchange_.all_gather(store, rank, world_size, uid);

    auto group = std::make_shared<CcoGroup>();
    MORI_CCO_CHECK(
        mori::cco::ccoCommCreate(ids[0], world_size, rank, PerRankVmmSize(), &group->comm));
    // ccoCommCreate leaves a tolerated probe failure latched in HIP's per-thread
    // sticky slot. torch checks that slot around every launch, so without this the
    // caller's next kernel is blamed for it. Consuming it here, at the call site that
    // caused it, is the narrow form -- not a blanket clear at the Python boundary.
    (void)hipGetLastError();

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = cco_groups_.emplace(name, std::move(group));
    return it->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<Block>> blocks_;
  std::unordered_map<std::string, std::shared_ptr<CcoGroup>> cco_groups_;
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

uint64_t WindowHandle(uint64_t data_ptr, bool signal_pad) {
  return AllocatorSingleton()->WindowHandle(reinterpret_cast<void*>(data_ptr), signal_pad);
}

uint64_t DevCommPtr(const std::string& group_name, const std::string& key, int lsa_barrier_count) {
  return AllocatorSingleton()->DevCommPtr(group_name, key, lsa_barrier_count);
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
namespace py = pybind11;

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
  m.def("window_handle", &mori::allocator::WindowHandle, py::arg("data_ptr"),
        py::arg("signal_pad") = false,
        "cco window handle for a rendezvous'd tensor (backend-specific; torch's "
        "SymmetricMemory has no slot for it)");
  // The pad is its own cco window now, so it is always there.
  m.def("dev_comm", &mori::allocator::DevCommPtr, py::arg("group_name"), py::arg("key") = "default",
        py::arg("lsa_barrier_count") = 1,
        "device ccoDevComm pointer for a group, cached per (group, key)");
  m.attr("signal_pad_supported") = true;
}
