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
// -- so tensors may die in whatever order Python's GC picks.
//
// Peers are published torch's way, as the buffer_ptrs / buffer_ptrs_dev array. This
// rank's entry is the tensor's own address; the peers' happen to sit in one flat span
// (peer(r) == flat_base + r*stride) because that is the cheapest way to map them, but that
// layout is not part of the API here: exposing it belongs with mori's cco window, which
// already defines it. See ROCm/mori#557.
//
// The peer exchange is CCO's: rendezvous hands the tensor to ccoWindowRegister, which
// aliases it into the flat LSA space and returns a window. Peer buffer_ptrs are then
// ccoGetPeerPtr results, so host and device (Window.lsa_ptr) agree by construction.
// Handle type is still probed per device for the allocation itself -- fabric where
// supported, POSIX fd otherwise (gfx9 has none).
//
// One CCO comm per (process group, device); one window, pad and peer table per (allocation, group).

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/version.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mori/cco/cco.hpp>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/utils/hip_compat.hpp"
#include "symm_signal.hpp"

// torch >= 2.15 orders every signal-pad op of a group across streams; join it when present.
#if __has_include(<torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>)
#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>
#define MORI_HAVE_GROUP_STREAM_GUARD 1
#else
#define MORI_HAVE_GROUP_STREAM_GUARD 0
#endif

// The SymmetricMemory interface is not stable across torch minors -- methods move
// between pure virtual, virtual-with-default and non-virtual, and new virtuals appear --
// so this file is written to compile against all of them rather than for one release.
#if !defined(TORCH_VERSION_MAJOR) || !defined(TORCH_VERSION_MINOR)
#error "mori's torch SymmetricMemory backend needs <torch/version.h> to define TORCH_VERSION_*"
#endif
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
#error "mori's torch SymmetricMemory backend requires torch >= 2.9"
#endif
#define MORI_TORCH_AT_LEAST(major, minor) \
  (TORCH_VERSION_MAJOR > (major) ||       \
   (TORCH_VERSION_MAJOR == (major) && TORCH_VERSION_MINOR >= (minor)))

// The torch build this module was compiled against; the loader refuses a mismatch.
#ifndef MORI_TORCH_FINGERPRINT
#define MORI_TORCH_FINGERPRINT unknown
#endif
#define MORI_STR_IMPL(x) #x
#define MORI_STR(x) MORI_STR_IMPL(x)

namespace mori {
namespace allocator {
namespace {

using c10d::symmetric_memory::SymmetricMemory;
using c10d::symmetric_memory::SymmetricMemoryAllocator;
using mori::cco::ccoComm;
using mori::cco::ccoWindow_t;

#define MORI_HIP_CHECK(expr)                                                  \
  do {                                                                        \
    hipError_t _e = (expr);                                                   \
    TORCH_CHECK(_e == hipSuccess, #expr, " failed: ", hipGetErrorString(_e)); \
  } while (0)

#define MORI_CCO_CHECK(expr)                                       \
  do {                                                             \
    int _rc = (expr);                                              \
    TORCH_CHECK(_rc == 0, #expr, " failed with cco status ", _rc); \
  } while (0)

// The pad gets its own cco allocation and window rather than being appended to the
// user's buffer, which is what NCCL's backend does too: appending it to a page-aligned
// request cost a whole extra 2 MiB page, and that cost is why the pad used to be
// compiled out.
//
// Its size comes from torch, not from a constant: set_signal_pad_size() is documented
// for kernels that need more than the 9216 B default. Sizing the allocation from
// anything else means torch hands out a pad tensor larger than what is backed.
size_t SignalPadBytes() {
#if MORI_TORCH_AT_LEAST(2, 10)
  return c10d::symmetric_memory::get_signal_pad_size();
#else
  return 9216;  // torch 2.9: fixed, 32 blocks x 72 ranks x 4 B, with no getter
#endif
}

size_t RoundUp(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

constexpr size_t k4GiB = size_t{1} << 32;

bool EnvFlag(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  std::string s(v);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

// Teardown can run while the HIP runtime is already unwinding, in which case any VMM call
// segfaults. is_finalizing() catches torch's own shutdown; this catches the rest by
// probing the runtime first. Leaking there is deliberate -- the process is exiting.
bool RuntimeUsable() {
  if (c10d::symmetric_memory::is_finalizing()) return false;
  int dev = -1;
  return hipGetDevice(&dev) == hipSuccess;
}

// Raw HIP: torch ships both c10::cuda and c10::hip; the wrong one pulls in CUDA's runtime header.
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

hipStream_t CurrentStream(int dev) {
  return c10::hip::getCurrentHIPStream(static_cast<c10::DeviceIndex>(dev)).stream();
}

// Zeroes VMM memory on `stream` and waits for it. hipMemset returns hipErrorOutOfMemory on
// UALink fabric-exportable pools (ROCm 7.15; cco's CcoZeroWindowMem), so those get copies.
void ZeroAndSync(void* ptr, size_t bytes, bool fabric, hipStream_t stream) {
  if (!fabric) {
    MORI_HIP_CHECK(hipMemsetAsync(ptr, 0, bytes, stream));
  } else if (bytes > 0) {
    const size_t chunk = std::min<size_t>(bytes, size_t{4} << 20);
    const std::vector<char> zeros(chunk, 0);
    for (size_t off = 0; off < bytes; off += chunk) {
      MORI_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(ptr) + off, zeros.data(),
                                    std::min(chunk, bytes - off), hipMemcpyHostToDevice, stream));
    }
  }
  MORI_HIP_CHECK(hipStreamSynchronize(stream));
}

// alloc() and a first rendezvous() sync the stream and use the Store; neither can be captured.
void CheckNotCapturing(int dev, const char* op, const char* hint) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  const hipError_t err = hipStreamIsCapturing(CurrentStream(dev), &status);
  if (err != hipSuccess) (void)hipGetLastError();  // e.g. legacy stream under global capture
  TORCH_CHECK(err == hipSuccess && status == hipStreamCaptureStatusNone, "mori symm backend: ", op,
              " is not supported while the current stream is capturing a graph. ", hint);
}

// Store all-gather counted per store; a process-wide count (torch < 2.15) desyncs subgroups.
class StoreExchange {
 public:
  explicit StoreExchange(std::string prefix) : prefix_(std::move(prefix)) {}

  template <typename T>
  std::vector<T> all_gather(const c10::intrusive_ptr<c10d::Store>& store, int rank, int world_size,
                            const T& val) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::string base = prefix_ + "/" + std::to_string(NextSeq(store)) + "/";
    std::vector<std::string> keys;
    keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) keys.push_back(base + std::to_string(r));
    const auto* bytes = reinterpret_cast<const uint8_t*>(&val);
    store->set(keys[rank], std::vector<uint8_t>(bytes, bytes + sizeof(T)));
    auto payloads = store->multiGet(keys);
    std::vector<T> out(world_size);
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(payloads[r].size() == sizeof(T), "mori symm backend: rank ", r,
                  " published a payload of the wrong size; mixed mori versions?");
      std::memcpy(&out[r], payloads[r].data(), sizeof(T));
    }
    return out;
  }

 private:
  // Keyed by address; the weak ref pins it against reuse, dead stores are pruned here.
  size_t NextSeq(const c10::intrusive_ptr<c10d::Store>& store) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = seqs_.begin(); it != seqs_.end();) {
      it = it->second.ref.expired() ? seqs_.erase(it) : std::next(it);
    }
    const auto key = reinterpret_cast<uintptr_t>(store.get());
    auto it = seqs_.find(key);
    if (it == seqs_.end()) {
      it = seqs_.emplace(key, Entry{c10::weak_intrusive_ptr<c10d::Store>(store), 0}).first;
    }
    return it->second.seq++;
  }

  struct Entry {
    c10::weak_intrusive_ptr<c10d::Store> ref;
    size_t seq;
  };
  const std::string prefix_;
  std::mutex mu_;
  std::unordered_map<uintptr_t, Entry> seqs_;
};

// Backend::use_pg_for_symm_mem_rendezvous (absent in older torch): exchange over the group.
template <typename B, typename = void>
struct HasPgRendezvousFlag : std::false_type {};
template <typename B>
struct HasPgRendezvousFlag<
    B, std::void_t<decltype(std::declval<const B&>().getUsePgForSymmMemRendezvous())>>
    : std::true_type {};

template <typename B>
bool PgRendezvousRequested(const B& backend) {
  if constexpr (HasPgRendezvousFlag<B>::value) {
    return backend.getUsePgForSymmMemRendezvous();
  } else {
    return false;
  }
}

// Asked of the group's CUDA backend, which is set even when the default backend is not.
bool UsePgForRendezvous(const c10::intrusive_ptr<c10d::ProcessGroup>& pg) {
  try {
    return PgRendezvousRequested(*pg->getBackend(c10::DeviceType::CUDA));
  } catch (const std::exception&) {
    return false;  // the group has no CUDA backend
  }
}

// all_gather_single replaced the deprecated _allgather_base; same op, same trace entry.
template <typename P, typename = void>
struct HasAllGatherSingle : std::false_type {};
template <typename P>
struct HasAllGatherSingle<P, std::void_t<decltype(std::declval<P&>().all_gather_single(
                                 std::declval<at::Tensor&>(), std::declval<at::Tensor&>()))>>
    : std::true_type {};

template <typename P>
c10::intrusive_ptr<c10d::Work> AllGatherSingle(P& pg, at::Tensor& out, at::Tensor& in) {
  if constexpr (HasAllGatherSingle<P>::value) {
    return pg.all_gather_single(out, in);
  } else {
    return pg._allgather_base(out, in);
  }
}

// One rendezvous' all-gathers: the Store by default, the group's own collective on request.
class RendezvousExchange {
 public:
  RendezvousExchange(StoreExchange& store_exchange, c10::intrusive_ptr<c10d::ProcessGroup> pg,
                     int dev)
      : store_exchange_(store_exchange),
        pg_(std::move(pg)),
        dev_(dev),
        via_pg_(UsePgForRendezvous(pg_)) {}

  template <typename T>
  std::vector<T> all_gather(const T& val) {
    static_assert(std::is_trivially_copyable_v<T>);
    const int world_size = pg_->getSize();
    if (!via_pg_) {
      return store_exchange_.all_gather(pg_->getStore(), pg_->getRank(), world_size, val);
    }
    const auto opts = at::TensorOptions().dtype(at::kByte).device(
        c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(dev_)));
    at::Tensor in = at::empty({static_cast<int64_t>(sizeof(T))}, opts);
    in.copy_(at::from_blob(const_cast<T*>(&val), {static_cast<int64_t>(sizeof(T))}, at::kByte));
    at::Tensor out = at::empty({static_cast<int64_t>(sizeof(T)) * world_size}, opts);
    AllGatherSingle(*pg_, out, in)->wait();
    const at::Tensor host = out.cpu();
    std::vector<T> res(world_size);
    std::memcpy(res.data(), host.data_ptr(), sizeof(T) * world_size);
    return res;
  }

 private:
  StoreExchange& store_exchange_;
  c10::intrusive_ptr<c10d::ProcessGroup> pg_;
  int dev_;
  bool via_pg_;
};

// Per-rank flat VA of a comm (cco reserves lsaSize x this): device memory, capped for big teams.
size_t PerRankVa(size_t device_mem, int lsa_bound) {
  if (const char* e = std::getenv("MORI_SYMM_PER_RANK_VMM")) {
    return static_cast<size_t>(std::strtoull(e, nullptr, 0));  // 0: cco uses device memory
  }
  constexpr int kUncappedTeam = 8;
  constexpr size_t kFlatVaBudget = size_t{4} << 40;  // per communicator, past kUncappedTeam
  size_t v = device_mem;
  if (lsa_bound > kUncappedTeam) {
    v = std::min(v, std::max(k4GiB, kFlatVaBudget / lsa_bound / k4GiB * k4GiB));
  }
  return RoundUp(std::max(v, k4GiB), k4GiB);
}

// What every rank publishes on the first rendezvous of an allocation with a group.
struct RendezvousInfo {
  uint64_t alloc_size;
  uint64_t buffer_size;
  uint64_t comm_epoch;  // this rank's communicator for the group, 0 if it has none
  uint64_t device_mem;
  int32_t device_idx;
  int32_t global_rank;  // rank in the default group, -1 if there is none
  int32_t fabric;
  char host[HOST_NAME_MAX + 1];
};

std::atomic<int> g_live_comms{0};

// One comm per (process group, device); lives while the group is registered or a handle holds it.
struct CcoGroup {
  ccoComm* comm = nullptr;
  std::string name;
  int device = -1;
  int rank = -1;
  int world_size = 0;
  // Drawn by rank 0; a comm is reused only if every rank reports the same epoch.
  uint64_t epoch = 0;
  std::vector<int> rank_to_global_rank;
  int* rank_to_global_rank_dev = nullptr;

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
  // happens to hold it, so a window can be deregistered while another thread is
  // registering one. Also guards dev_comms, so there is no lock ordering.
  std::mutex mu;
  std::unordered_map<std::string, DevCommEntry> dev_comms;

  ~CcoGroup() {
    if (comm == nullptr) return;
    g_live_comms.fetch_sub(1);
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    DeviceGuard guard(device);
    (void)hipDeviceSynchronize();
    if (rank_to_global_rank_dev) (void)hipFree(rank_to_global_rank_dev);
    for (auto& [key, e] : dev_comms) {
      if (e.device) mori::cco::ccoDevCommFreeDeviceCopy(e.device);
      (void)mori::cco::ccoDevCommDestroy(comm, &e.host);
    }
    (void)mori::cco::ccoCommDestroy(comm);
  }
};

void SweepDeadGroups() noexcept;

struct Block;

// One (allocation, group) rendezvous: windows, own signal pad, peer tables; shared by its handles.
struct PeerAlloc {
  std::shared_ptr<Block> block;     // this rank's own mapping, released after the windows
  std::shared_ptr<CcoGroup> group;  // keeps the communicator alive for these windows
  ccoWindow_t win{};
  void* alias = nullptr;  // cco's flat-VA alias of the tensor, what ccoMemFree takes
  ccoWindow_t pad_win{};
  void* pad_ptr = nullptr;  // the signal pad: cco's own allocation, not part of the tensor
  size_t pad_size = 0;      // what torch asked for when this pad was made
  size_t buffer_size = 0;
  int device = -1;
  std::vector<void*> buffers;
  std::vector<void*> pads;
  void** tables_dev = nullptr;  // buffers then pads, world_size entries each

  ~PeerAlloc() {
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    if (group && group->comm) {
      DeviceGuard guard(device);
      (void)hipDeviceSynchronize();
      if (tables_dev) (void)hipFree(tables_dev);
      // Purely local: ccoWindowDeregister unmaps this rank's view of its peers and drops
      // the handles it imported, with no collective inside, so this can run on whichever
      // thread drops the last reference and in whatever order across ranks.
      std::lock_guard<std::mutex> lock(group->mu);
      if (pad_win) (void)mori::cco::ccoWindowDeregister(group->comm, pad_win);
      if (pad_ptr) (void)mori::cco::ccoMemFree(group->comm, pad_ptr);
      if (win) (void)mori::cco::ccoWindowDeregister(group->comm, win);
      if (alias) (void)mori::cco::ccoMemFree(group->comm, alias);
    }
    // A handle outliving its process group is the last thing keeping the comm alive.
    SweepDeadGroups();
  }
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

struct Block {
  void* ptr = nullptr;
  size_t buffer_size = 0;
  size_t alloc_size = 0;
  int device_idx = 0;
  hipMemGenericAllocationHandle_t handle{};
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
  std::optional<std::string> default_group_name;
  // Group name -> its rendezvous of this block. Guarded by the allocator's mutex.
  std::unordered_map<std::string, std::shared_ptr<PeerAlloc>> peers;

  // Unmaps this rank's view and drops the allocation, once. Handles hold the block, so a
  // tensor freed under a live handle keeps its memory until the last handle goes.
  void Release() noexcept {
    if (ptr == nullptr) return;
    void* p = std::exchange(ptr, nullptr);
    if (!RuntimeUsable()) return;  // leak rather than crash on the way out
    DeviceGuard guard(device_idx);
    // Unmapping a range a kernel is still writing is a page fault, not a stale read.
    (void)hipDeviceSynchronize();
    (void)hipMemUnmap(p, alloc_size);
    (void)hipMemAddressFree(p, alloc_size);
    (void)hipMemRelease(handle);
  }
  ~Block() { Release(); }
};

// No 'override' on has_allocation (virtual from 2.12), get_group_name (2.16) and
// get_signal_pad_size (virtual in 2.9 and after pytorch#204686): each overrides where virtual.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(std::shared_ptr<PeerAlloc> peer, size_t offset)
      : peer_(std::move(peer)),
        group_(peer_->group),
        win_(peer_->win),
        pad_win_(peer_->pad_win),
        pad_size_(peer_->pad_size),
        buffers_(peer_->buffers),
        signal_pads_(peer_->pads),
        buffers_dev_(peer_->tables_dev),
        signal_pads_dev_(peer_->tables_dev + peer_->buffers.size()),
        buffer_size_(peer_->buffer_size),
        offset_(offset),
        rank_(group_->rank),
        world_size_(group_->world_size),
        device_(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(peer_->device)) {}

  std::vector<void*> get_buffer_ptrs() override { return buffers_; }
  std::vector<void*> get_signal_pad_ptrs() override { return signal_pads_; }
  void** get_buffer_ptrs_dev() override { return buffers_dev_; }
  void** get_signal_pad_ptrs_dev() override { return signal_pads_dev_; }
  size_t get_buffer_size() override { return buffer_size_; }
  // Where the rendezvous'd tensor starts in its allocation; buffer_ptrs are the bases.
  size_t get_offset() override { return offset_; }
  size_t get_signal_pad_size() { return pad_size_; }
  std::string get_group_name() { return group_->name; }

  bool has_multicast_support() override { return false; }
  void* get_multicast_ptr() override { return nullptr; }

  int get_rank() override { return rank_; }
  int get_world_size() override { return world_size_; }
  c10::Device get_device() override { return device_; }

  const std::vector<int>& get_rank_to_global_rank() override { return group_->rank_to_global_rank; }
  int* get_rank_to_global_rank_dev() override { return group_->rank_to_global_rank_dev; }

  // True only if cco mapped every peer. A null entry means that rank needs RDMA, which
  // this backend cannot drive yet -- but reporting it honestly is what lets torch fall
  // back rather than dereference a null pointer.
  bool world_within_direct_access() override {
    for (void* p : buffers_)
      if (p == nullptr) return false;
    return true;
  }

  // Stream-ordered device kernels on the signal pad, same protocol as torch's backends.
  // Peers outside the LSA team have no pad: barrier() meets them on the host, signals refuse.
  void barrier(int channel, size_t timeout_ms) override {
    CheckChannel(channel);
    DeviceGuard guard(device_.index());
    if (!AllPadsMapped()) return HostBarrier();
    SignalPadStreamGuard order(group_->name);
    MORI_HIP_CHECK(LaunchSignalPadBarrier(PadsDev(), channel, rank_, world_size_, timeout_ms,
                                          c10::hip::getCurrentHIPStream().stream()));
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    CheckChannel(channel);
    CheckRank(dst_rank);
    CheckPadMapped(dst_rank, "put_signal");
    DeviceGuard guard(device_.index());
    SignalPadStreamGuard order(group_->name);
    MORI_HIP_CHECK(LaunchSignalPadPut(PadsDev(), dst_rank, channel, rank_, world_size_, timeout_ms,
                                      c10::hip::getCurrentHIPStream().stream()));
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    CheckChannel(channel);
    CheckRank(src_rank);
    CheckPadMapped(src_rank, "wait_signal");
    DeviceGuard guard(device_.index());
    SignalPadStreamGuard order(group_->name);
    MORI_HIP_CHECK(LaunchSignalPadWait(PadsDev(), src_rank, channel, rank_, world_size_, timeout_ms,
                                       c10::hip::getCurrentHIPStream().stream()));
  }

  // Backend-specific, deliberately not part of torch's interface -- the same shape as
  // NCCLSymmetricMemory::get_window(). A kernel that wants flat-VA addressing takes the
  // window and calls cco's device API on it instead of walking buffer_ptrs.
  ccoWindow_t get_window() const { return win_; }
  ccoWindow_t get_signal_pad_window() const { return pad_win_; }

 private:
#if MORI_HAVE_GROUP_STREAM_GUARD
  using SignalPadStreamGuard = c10d::symmetric_memory::GroupStreamGuard;
#else
  // Older torch orders nothing across streams, its own backends included.
  struct SignalPadStreamGuard {
    explicit SignalPadStreamGuard(const std::string&) {}
  };
#endif

  uint32_t** PadsDev() const { return reinterpret_cast<uint32_t**>(signal_pads_dev_); }

  // Bounded by the pad actually allocated, and worded like torch's check_channel().
  void CheckChannel(int channel) const {
    TORCH_CHECK(channel >= 0,
                "channel for barrier(), put_signal() and wait_signal() must be non-negative (got ",
                channel, ")");
    const size_t num_channels = pad_size_ / (sizeof(uint32_t) * world_size_);
    TORCH_CHECK(static_cast<size_t>(channel) < num_channels,
                "The maximum supported channel for barrier(), put_signal() and wait_signal() is ",
                static_cast<int64_t>(num_channels) - 1, " (got ", channel, ")");
  }

  // Worded like torch's check_rank(), which torch < 2.14 does not have.
  void CheckRank(int rank) const {
    TORCH_CHECK(rank >= 0 && rank < world_size_, "rank must be in [0, ", world_size_, ") (got ",
                rank, ")");
  }

  // A null pad is a peer outside the LSA team: a kernel would fault on it or wait forever.
  bool AllPadsMapped() const {
    return std::all_of(signal_pads_.begin(), signal_pads_.end(),
                       [](void* p) { return p != nullptr; });
  }

  void CheckPadMapped(int peer, const char* op) const {
    TORCH_CHECK(signal_pads_[peer] != nullptr, "mori symm backend: ", op, "() with rank ", peer,
                ", which is not load/store reachable from rank ", rank_,
                " (outside its LSA team; see world_within_direct_access())");
  }

  // cco's host barrier spans the comm, so a group with unreachable peers still meets.
  void HostBarrier() {
    CheckNotCapturing(device_.index(), "barrier() on a group with peers outside the LSA team",
                      "Those peers have no signal pad mapping, so the barrier runs on the host.");
    std::lock_guard<std::mutex> lock(group_->mu);  // the socket bootstrap is not thread-safe
    MORI_HIP_CHECK(hipStreamSynchronize(c10::hip::getCurrentHIPStream().stream()));
    MORI_CCO_CHECK(mori::cco::ccoBarrierAll(group_->comm));
  }

  std::shared_ptr<PeerAlloc> peer_;  // owns everything below
  std::shared_ptr<CcoGroup> group_;
  ccoWindow_t win_;
  ccoWindow_t pad_win_;
  size_t pad_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  size_t buffer_size_;
  size_t offset_;
  int rank_;
  int world_size_;
  c10::Device device_;
};

bool GroupRegistered(const std::string& name) {
  try {
    (void)c10d::resolve_process_group(name);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// torch names the default group "0" (NVSHMEM's backend relies on the same convention).
int GlobalRank() {
  try {
    return c10d::resolve_process_group("0")->getRank();
  } catch (const std::exception&) {
    return -1;
  }
}

class MoriSymmAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx, const std::optional<std::string>& group_name) override {
    // "cuda" without an index means the current device, as in torch's CUDA backend.
    if (device_idx < 0) MORI_HIP_CHECK(hipGetDevice(&device_idx));
    DeviceGuard guard(device_idx);
    CheckNotCapturing(device_idx, "alloc",
                      "Allocate with torch.distributed._symmetric_memory.empty() before "
                      "starting capture.");
    SweepDeadGroups();

    const auto handle_type = ProbeHandleType(device_idx);
    auto prop = MakeProp(device_idx, handle_type);
    size_t gran = 0;
    MORI_HIP_CHECK(
        hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
    const size_t alloc_size = RoundUp(std::max<size_t>(size, 1), gran);

    hipMemGenericAllocationHandle_t handle{};
    MORI_HIP_CHECK(hipMemCreate(&handle, alloc_size, &prop, 0));
    void* ptr = nullptr;
    bool reserved = false, mapped = false;
    try {
      MORI_HIP_CHECK(hipMemAddressReserve(&ptr, alloc_size, gran, nullptr, 0));
      reserved = true;
      MORI_HIP_CHECK(hipMemMap(ptr, alloc_size, 0, handle, 0));
      mapped = true;
      SetRwAccess(ptr, alloc_size, device_idx);
      // On the caller's stream, and complete before a rendezvous can expose it to peers.
      ZeroAndSync(ptr, alloc_size, handle_type == hipMemHandleTypeFabricCompat,
                  CurrentStream(device_idx));
    } catch (...) {
      if (mapped) (void)hipMemUnmap(ptr, alloc_size);
      if (reserved) (void)hipMemAddressFree(ptr, alloc_size);
      (void)hipMemRelease(handle);
      throw;
    }

    auto block = std::make_shared<Block>();
    block->ptr = ptr;
    block->buffer_size = size;
    block->alloc_size = alloc_size;
    block->device_idx = device_idx;
    block->handle = handle;
    block->handle_type = handle_type;
    block->default_group_name = group_name;

    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[reinterpret_cast<uintptr_t>(ptr)] = std::move(block);
    return ptr;
  }

  // Live handles keep the memory, as torch's backends do: the last of tensor and handles frees it.
  void free(void* ptr) override {
    std::shared_ptr<Block> block;
    std::unordered_map<std::string, std::shared_ptr<PeerAlloc>> peers;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = blocks_.find(reinterpret_cast<uintptr_t>(ptr));
      if (it == blocks_.end()) return;
      block = std::move(it->second);
      blocks_.erase(it);
      peers.swap(block->peers);
      held_.erase(std::remove_if(held_.begin(), held_.end(),
                                 [](const std::weak_ptr<Block>& b) { return b.expired(); }),
                  held_.end());
      held_.push_back(block);  // for Shutdown(), should a handle outlive this call
    }
    // Torch's contract lets a symmetric tensor die with work outstanding; each release syncs.
    peers.clear();
    block.reset();
    SweepDeadGroups();
  }

  size_t get_alloc_size(void* ptr) override {
    auto block = FindExact(ptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    return block->buffer_size;
  }

  // Interior pointers count: a MemPool hands out tensors from inside our blocks.
  bool has_allocation(void* ptr) { return FindCovering(ptr, nullptr) != nullptr; }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr, const std::optional<std::string>& group_name) override {
    size_t offset = 0;
    auto block = FindCovering(ptr, &offset);
    if (block == nullptr) {
      // Null like torch's backends: low-contention ops probe arbitrary tensors with this.
      TORCH_WARN(
          "mori symm backend: pointer is not within any symmetric allocation; was "
          "the tensor allocated with torch.distributed._symmetric_memory.empty()?");
      return nullptr;
    }
    std::string name;
    if (group_name.has_value() && !group_name->empty()) {
      name = *group_name;
    } else {
      TORCH_CHECK(block->default_group_name.has_value() && !block->default_group_name->empty(),
                  "mori symm backend: group_name given neither at allocation nor rendezvous");
      name = *block->default_group_name;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = block->peers.find(name);
      if (it != block->peers.end()) {
        return c10::make_intrusive<MoriSymmetricMemory>(it->second, offset);
      }
    }
    auto peer = FirstRendezvous(*block, name);
    peer->block = block;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      block->peers.emplace(name, peer);
    }
    return c10::make_intrusive<MoriSymmetricMemory>(std::move(peer), offset);
  }

  bool has_multicast_support(int) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "MORI"; }

  // Drop every live allocation and communicator while python and HIP are still up.
  // Registered as an atexit hook: destructors that run during interpreter shutdown are
  // too late, and segfault even though is_finalizing() is still false.
  void Shutdown() {
    std::map<uintptr_t, std::shared_ptr<Block>> blocks;
    std::map<GroupKey, std::shared_ptr<CcoGroup>> groups;
    std::vector<std::weak_ptr<Block>> held;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      blocks.swap(blocks_);
      groups.swap(groups_);
      held.swap(held_);
    }
    if (!RuntimeUsable()) return;
    // Our mapping first, then each group's window; freed tensors a handle still holds too.
    for (auto& [addr, block] : blocks) {
      block->Release();
      block->peers.clear();
    }
    for (auto& weak : held) {
      if (auto block = weak.lock()) block->Release();
    }
  }

  // Drop comms of unregistered groups (each dies with its last handle); epochs keep ranks agreed.
  void SweepDeadGroups() {
    std::vector<GroupKey> keys;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& [key, group] : groups_) keys.push_back(key);
    }
    std::vector<std::shared_ptr<CcoGroup>> dead;
    for (const auto& key : keys) {
      if (GroupRegistered(key.first)) continue;
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = groups_.find(key);
      if (it == groups_.end()) continue;
      dead.push_back(std::move(it->second));
      groups_.erase(it);
    }
  }

  // A device communicator for this group, created on first use and cached under `key`.
  // Deliberately not built during rendezvous: rendezvous knows a buffer and a group, and
  // nothing about how many signals or QPs the kernel that follows will want.
  uint64_t DevCommPtr(const std::string& group_name, const std::string& key,
                      int lsa_barrier_count) {
    int dev = -1;
    MORI_HIP_CHECK(hipGetDevice(&dev));
    SweepDeadGroups();
    auto group = LookupGroup(group_name, dev);
    TORCH_CHECK(group != nullptr, "mori symm backend: no communicator for group ", group_name,
                " on device ", dev, ". Have you rendezvoused a tensor with this group?");
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
  uint64_t WindowHandle(void* ptr, bool signal_pad, const std::string& group_name) {
    auto block = FindCovering(ptr, nullptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    std::shared_ptr<PeerAlloc> peer;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!group_name.empty()) {
        auto it = block->peers.find(group_name);
        if (it != block->peers.end()) peer = it->second;
      } else {
        TORCH_CHECK(block->peers.size() <= 1, "mori symm backend: this tensor is rendezvous'd ",
                    "with ", block->peers.size(), " groups; say which one");
        if (!block->peers.empty()) peer = block->peers.begin()->second;
      }
    }
    TORCH_CHECK(peer != nullptr, "mori symm backend: rendezvous this tensor",
                group_name.empty() ? "" : " with group " + group_name,
                " before asking for its window");
    return reinterpret_cast<uint64_t>(signal_pad ? peer->pad_win : peer->win);
  }

  // Diagnostics: torch keeps rank_to_global_rank C++-only.
  std::vector<int> RankToGlobalRank(void* ptr, const std::string& group_name) {
    auto block = FindCovering(ptr, nullptr);
    TORCH_CHECK(block != nullptr, "mori symm backend: pointer is not a mori allocation");
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = block->peers.find(group_name);
    TORCH_CHECK(it != block->peers.end(), "mori symm backend: not rendezvous'd with group ",
                group_name);
    return it->second->group->rank_to_global_rank;
  }

 private:
  using GroupKey = std::pair<std::string, int>;  // (process group name, device)

  std::shared_ptr<Block> FindExact(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(reinterpret_cast<uintptr_t>(ptr));
    return it == blocks_.end() ? nullptr : it->second;
  }

  std::shared_ptr<Block> FindCovering(void* ptr, size_t* offset) {
    const auto p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.upper_bound(p);
    if (it == blocks_.begin()) return nullptr;
    --it;
    const size_t off = p - it->first;
    if (off != 0 && off >= it->second->buffer_size) return nullptr;
    if (offset) *offset = off;
    return it->second;
  }

  std::shared_ptr<CcoGroup> LookupGroup(const std::string& name, int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = groups_.find({name, device});
    return it == groups_.end() ? nullptr : it->second;
  }

  // Collective: map `block` into the group's comm, creating it unless all ranks share one.
  std::shared_ptr<PeerAlloc> FirstRendezvous(const Block& block, const std::string& name) {
    const int dev = block.device_idx;
    DeviceGuard guard(dev);
    CheckNotCapturing(dev, "the first rendezvous of a tensor with a group",
                      "Rendezvous once eagerly before starting capture; later calls on the "
                      "same tensor and group reuse the handle.");
    SweepDeadGroups();
    // Resolve the process group directly rather than going through torch's GroupInfo
    // registry. That registry is only ever populated by enable_symm_mem_for_group(),
    // which torch has deprecated -- and torch's own CUDA and NCCL backends both take
    // this route instead, so callers no longer have to make that deprecated call.
    auto pg = c10d::resolve_process_group(name);
    const int rank = pg->getRank();
    const int world_size = pg->getSize();
    RendezvousExchange exchange(store_exchange_, pg, dev);

    auto cached = LookupGroup(name, dev);
    if (cached && (cached->rank != rank || cached->world_size != world_size)) cached = nullptr;

    RendezvousInfo local{};
    local.alloc_size = block.alloc_size;
    local.buffer_size = block.buffer_size;
    local.comm_epoch = cached ? cached->epoch : 0;
    size_t device_mem = 0;
    MORI_HIP_CHECK(hipDeviceTotalMem(&device_mem, dev));
    local.device_mem = device_mem;
    local.device_idx = dev;
    local.global_rank = GlobalRank();
    local.fabric = block.handle_type == hipMemHandleTypeFabricCompat ? 1 : 0;
    (void)gethostname(local.host, HOST_NAME_MAX);
    auto infos = exchange.all_gather(local);

    // torch's contract is that every rank hands in the same size. cco's register is
    // collective and would deadlock or mismap on a mismatch, so say so first.
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(
          infos[r].alloc_size == local.alloc_size && infos[r].buffer_size == local.buffer_size,
          "mori symm backend: rank ", r, " allocated ", infos[r].buffer_size,
          " bytes but this rank allocated ", local.buffer_size,
          "; symm_mem.empty must be symmetric across ranks");
    }
    if (!EnvFlag("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES")) {
      std::set<std::pair<std::string, int>> devices;
      for (const auto& info : infos) devices.emplace(info.host, info.device_idx);
      TORCH_CHECK(devices.size() == static_cast<size_t>(world_size),
                  "mori symm backend: rendezvous detected allocations from overlapping devices "
                  "on different ranks; set TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1 if that "
                  "is intended");
    }

    bool reuse = cached != nullptr;
    for (const auto& info : infos) reuse = reuse && info.comm_epoch == cached->epoch;
    if (!reuse) cached.reset();  // let CreateGroup release it first
    auto group = reuse ? cached : CreateGroup(name, dev, exchange, rank, world_size, infos);
    return MapBlock(group, block);
  }

  // Collective: a new comm for (name, dev), replacing this rank's cached one.
  std::shared_ptr<CcoGroup> CreateGroup(const std::string& name, int dev,
                                        RendezvousExchange& exchange, int rank, int world_size,
                                        const std::vector<RendezvousInfo>& infos) {
    {
      // Released first so its VA is back before the new reservation, if nothing holds it.
      std::shared_ptr<CcoGroup> old;
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = groups_.find({name, dev});
      if (it != groups_.end()) {
        old = std::move(it->second);
        groups_.erase(it);
      }
    }

    // Bound the LSA team by ranks per host, or by the group where a fabric (vPOD) spans hosts.
    size_t device_mem = 0;
    std::map<std::string, int> per_host;
    bool fabric = false;
    for (const auto& info : infos) {
      device_mem = std::max<size_t>(device_mem, info.device_mem);
      per_host[info.host]++;
      fabric = fabric || info.fabric != 0;
    }
    int lsa_bound = world_size;
    if (!fabric) {
      lsa_bound = 0;
      for (const auto& [host, n] : per_host) lsa_bound = std::max(lsa_bound, n);
    }

    struct CommId {
      mori::cco::ccoUniqueId uid;
      uint64_t epoch;
    };
    CommId mine{};
    if (rank == 0) {
      MORI_CCO_CHECK(mori::cco::ccoGetUniqueId(&mine.uid));
      std::random_device rd;
      mine.epoch = ((static_cast<uint64_t>(rd()) << 32) | rd()) | 1;  // never 0
    }
    // The uid rides torch's own channel, so cco's rendezvous needs no second one.
    auto ids = exchange.all_gather(mine);

    auto group = std::make_shared<CcoGroup>();
    group->name = name;
    group->device = dev;
    group->rank = rank;
    group->world_size = world_size;
    group->epoch = ids[0].epoch;
    // Keep the caller's CPU affinity; load/store peers need no NIC or GPU-NIC topology.
    mori::cco::ccoCommConfig config = CCO_COMM_CONFIG_INITIALIZER;
    config.cpuAffinity = 0;
    config.lsaOnly = 1;
    MORI_CCO_CHECK(mori::cco::ccoCommCreate(
        ids[0].uid, world_size, rank, PerRankVa(device_mem, lsa_bound), &config, &group->comm));
    g_live_comms.fetch_add(1);
    // ccoCommCreate leaves a tolerated probe failure latched in HIP's per-thread
    // sticky slot. torch checks that slot around every launch, so without this the
    // caller's next kernel is blamed for it. Consuming it here, at the call site that
    // caused it, is the narrow form -- not a blanket clear at the Python boundary.
    (void)hipGetLastError();

    bool have_global = true;
    for (const auto& info : infos) have_global = have_global && info.global_rank >= 0;
    group->rank_to_global_rank.resize(world_size);
    for (int r = 0; r < world_size; ++r) {
      group->rank_to_global_rank[r] = have_global ? infos[r].global_rank : r;
    }
    const size_t bytes = world_size * sizeof(int);
    MORI_HIP_CHECK(hipMalloc(&group->rank_to_global_rank_dev, bytes));
    const hipStream_t stream = CurrentStream(dev);
    MORI_HIP_CHECK(hipMemcpyAsync(group->rank_to_global_rank_dev, group->rank_to_global_rank.data(),
                                  bytes, hipMemcpyHostToDevice, stream));
    MORI_HIP_CHECK(hipStreamSynchronize(stream));

    std::lock_guard<std::mutex> lock(mutex_);
    groups_[{name, dev}] = group;
    return group;
  }

  // Collective over the group: alias `block` and a fresh signal pad into its comm.
  std::shared_ptr<PeerAlloc> MapBlock(const std::shared_ptr<CcoGroup>& group, const Block& block) {
    const int rank = group->rank;
    const int world_size = group->world_size;
    auto peer = std::make_shared<PeerAlloc>();
    peer->group = group;
    peer->buffer_size = block.buffer_size;
    peer->device = block.device_idx;
    peer->pad_size = SignalPadBytes();
    const hipStream_t stream = CurrentStream(block.device_idx);

    // Held across the registers and the peer-pointer reads: see CcoGroup::mu.
    std::unique_lock<std::mutex> cco_lock(group->mu);
    // Overload C: retain the tensor's VMM handle, alias it into the flat LSA slot,
    // and register the window. No copy, and no second peer exchange of our own.
    MORI_CCO_CHECK(mori::cco::ccoWindowRegister(group->comm, block.ptr, block.alloc_size,
                                                &peer->win, &peer->alias));

    // torch's signal pad, as its own allocation and window rather than a tail on the
    // user's buffer, and one per (allocation, group) so groups never see each other's
    // signals. cco owns this one outright, so it is overload B.
    MORI_CCO_CHECK(mori::cco::ccoMemAlloc(group->comm, peer->pad_size, &peer->pad_ptr));
    // Zero before the collective register below lets a peer write to it. cco picks the
    // pad's handle type itself, so it is zeroed the way that works for fabric too.
    ZeroAndSync(peer->pad_ptr, peer->pad_size, /*fabric=*/true, stream);
    MORI_CCO_CHECK(
        mori::cco::ccoWindowRegister(group->comm, peer->pad_ptr, peer->pad_size, &peer->pad_win));

    // Peers come from the flat LSA span, so they agree with the device-side lsa_ptr. A
    // peer outside the LSA team comes back null, and that is left as null on purpose --
    // it is how torch's own backends report a peer that has to be reached by RDMA.
    peer->buffers.resize(world_size);
    peer->pads.resize(world_size);
    for (int r = 0; r < world_size; ++r) {
      peer->buffers[r] = mori::cco::ccoGetPeerPtr(group->comm, peer->alias, r);
      peer->pads[r] = mori::cco::ccoGetPeerPtr(group->comm, peer->pad_ptr, r);
    }
    TORCH_CHECK(peer->buffers[rank] != nullptr,
                "mori symm backend: cco did not map this rank's own window");
    cco_lock.unlock();
    // This rank's own entry is the tensor itself, not cco's alias of it.
    peer->buffers[rank] = block.ptr;

    std::vector<void*> tables(peer->buffers);
    tables.insert(tables.end(), peer->pads.begin(), peer->pads.end());
    const size_t bytes = tables.size() * sizeof(void*);
    MORI_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&peer->tables_dev), bytes));
    MORI_HIP_CHECK(
        hipMemcpyAsync(peer->tables_dev, tables.data(), bytes, hipMemcpyHostToDevice, stream));
    MORI_HIP_CHECK(hipStreamSynchronize(stream));
    return peer;
  }

  std::mutex mutex_;
  std::map<uintptr_t, std::shared_ptr<Block>> blocks_;  // ordered for covering lookups
  std::vector<std::weak_ptr<Block>> held_;              // freed by torch, still held by a handle
  std::map<GroupKey, std::shared_ptr<CcoGroup>> groups_;
  StoreExchange store_exchange_{"mori_symm_backend"};
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

c10::intrusive_ptr<MoriSymmAllocator>& AllocatorSingleton() {
  // Immortal on purpose: register_availability() parks a reference in a libtorch-owned
  // registry that outlives this extension's statics.
  static auto* inst =
      new c10::intrusive_ptr<MoriSymmAllocator>(c10::make_intrusive<MoriSymmAllocator>());
  return *inst;
}

void SweepDeadGroups() noexcept {
  try {
    AllocatorSingleton()->SweepDeadGroups();
  } catch (...) {
  }
}

void Shutdown() { AllocatorSingleton()->Shutdown(); }

uint64_t WindowHandle(uint64_t data_ptr, bool signal_pad, const std::string& group_name) {
  return AllocatorSingleton()->WindowHandle(reinterpret_cast<void*>(data_ptr), signal_pad,
                                            group_name);
}

uint64_t DevCommPtr(const std::string& group_name, const std::string& key, int lsa_barrier_count) {
  return AllocatorSingleton()->DevCommPtr(group_name, key, lsa_barrier_count);
}

std::vector<int> RankToGlobalRank(uint64_t data_ptr, const std::string& group_name) {
  return AllocatorSingleton()->RankToGlobalRank(reinterpret_cast<void*>(data_ptr), group_name);
}

std::string HandleTypeName(int dev) {
  return ProbeHandleType(dev) == hipMemHandleTypeFabricCompat ? "fabric" : "posix_fd";
}

// Idempotent; not run at import, so a module built for another torch never registers.
void RegisterTorchSymmBackend() {
  static bool registered = [] {
    c10d::symmetric_memory::register_availability("MORI", AllocatorSingleton());
    return true;
  }();
  (void)registered;
}

}  // namespace
}  // namespace allocator
}  // namespace mori

namespace py = pybind11;

PYBIND11_MODULE(mori_torch_symm, m) {
  m.def("register_backend", &mori::allocator::RegisterTorchSymmBackend,
        "Register the MORI symmetric memory backend with torch (idempotent)");
  m.def("shutdown", &mori::allocator::Shutdown,
        "Release every live symmetric allocation (registered as an atexit hook)");
  m.def("handle_type", &mori::allocator::HandleTypeName,
        "'fabric' or 'posix_fd' -- what this device can export");
  m.attr("backend_name") = "MORI";
  m.attr("torch_fingerprint") = MORI_STR(MORI_TORCH_FINGERPRINT);
  m.def("window_handle", &mori::allocator::WindowHandle, py::arg("data_ptr"),
        py::arg("signal_pad") = false, py::arg("group_name") = "",
        "cco window handle for a rendezvous'd tensor (backend-specific; torch's "
        "SymmetricMemory has no slot for it)");
  m.def("dev_comm", &mori::allocator::DevCommPtr, py::arg("group_name"), py::arg("key") = "default",
        py::arg("lsa_barrier_count") = 1,
        "device ccoDevComm pointer for a group, cached per (group, key)");
  m.def(
      "_live_comms", [] { return mori::allocator::g_live_comms.load(); },
      "communicators currently alive (diagnostics)");
  m.def("_rank_to_global_rank", &mori::allocator::RankToGlobalRank, py::arg("data_ptr"),
        py::arg("group_name"), "a handle's rank_to_global_rank (diagnostics)");
  // The pad is its own cco window, so it is always there.
  m.attr("signal_pad_supported") = true;
}
