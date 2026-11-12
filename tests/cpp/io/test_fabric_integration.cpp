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
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <msgpack.hpp>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"
#include "src/io/fabric/backend_impl.hpp"
#include "src/io/fabric/completion_guard.hpp"

using namespace mori::io;

// Fail allocations on the submitting thread only; completion threads and
// cleanup can still run. Sweep our C++ budgets around actual GPU submissions.
namespace {
thread_local int allocationsBeforeFailure = -1;
thread_local bool allocationFailureTriggered = false;
}  // namespace
void* operator new(std::size_t bytes) {
  if (allocationsBeforeFailure >= 0) {
    Dl_info site{};
    // Inject our C++ bookkeeping allocations. HIP/HSA runtimes also call
    // operator new and can corrupt their internal state when it throws; their
    // allocator/fault behavior is outside this completion regression.
    const bool ours = dladdr(__builtin_return_address(0), &site) != 0 &&
                      site.dli_fname != nullptr &&
                      (std::strstr(site.dli_fname, "libmori_io") != nullptr ||
                       std::strstr(site.dli_fname, "test_fabric_integration") != nullptr);
    if (ours) {
      if (allocationsBeforeFailure == 0) {
        allocationsBeforeFailure = -1;
        allocationFailureTriggered = true;
        throw std::bad_alloc();
      }
      --allocationsBeforeFailure;
    }
  }
  if (void* result = std::malloc(bytes == 0 ? 1 : bytes)) return result;
  throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct AllocationFailureScope {
  explicit AllocationFailureScope(int budget) {
    allocationsBeforeFailure = budget;
    allocationFailureTriggered = false;
  }
  ~AllocationFailureScope() { allocationsBeforeFailure = -1; }
};

void RequireReady(std::future<void>& future, const char* message) {
  if (future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) return;
  // A regression can deadlock both threads. Exit without destroying joinable
  // threads or waiting indefinitely for their destructors.
  std::cerr << "FAIL: " << message << '\n';
  std::_Exit(1);
}

struct ProgressLatch {
  bool armed = false;
  std::mutex progressMutex;
  std::mutex callsMutex;
  std::condition_variable callsChanged;
  size_t calls = 0;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();

  void Enter() {
    std::lock_guard<std::mutex> lock(callsMutex);
    ++calls;
    callsChanged.notify_all();
  }
  void WaitCalls(size_t count) {
    std::unique_lock<std::mutex> lock(callsMutex);
    if (callsChanged.wait_for(lock, std::chrono::seconds(5), [&] { return calls >= count; }))
      return;
    std::cerr << "FAIL: progress waiter did not enter\n";
    std::_Exit(1);
  }
};

// An event-like completion callback: one progress caller owns the completion
// mutex until the asynchronous operation is permitted to finish. This recreates
// FABRIC's event wait / terminal observer lock ordering without GPU timing races.
class LatchingBackendSession final : public BackendSession {
 public:
  LatchingBackendSession(ProgressLatch& latch, const char* source, char* destination)
      : latch_(latch), source_(source), destination_(destination) {}

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId, bool isRead) override {
    Check(isRead, "latching test only implements reads");
    auto progress = [this, status, localOffset, remoteOffset, size] {
      // Session submission polls Failed() before returning to the caller.
      // Arm only after it returns so the latch controls the concurrent waits.
      if (!latch_.armed) return;
      latch_.Enter();
      std::lock_guard<std::mutex> lock(latch_.progressMutex);
      if (completed_) return;
      latch_.released.wait();
      std::memcpy(destination_ + localOffset, source_ + remoteOffset, size);
      completed_ = true;
      status->SetCode(StatusCode::SUCCESS);
    };
    status->SetWaitCallback(progress);
    status->SetProgressCallback(std::move(progress));
    status->SetCode(StatusCode::IN_PROGRESS);
  }
  void BatchReadWrite(const SizeVec&, const SizeVec&, const SizeVec&, TransferStatus*,
                      TransferUniqueId, bool) override {
    throw std::logic_error("latching test does not batch");
  }
  bool Alive() const override { return true; }

 private:
  ProgressLatch& latch_;
  const char* source_;
  char* destination_;
  bool completed_ = false;
};

class LatchingEngineSession final : public IOEngineSession {
 public:
  LatchingEngineSession(IOEngine& owner, const MemoryDesc& local,
                        std::shared_ptr<BackendSession> backend) {
    engine = &owner;
    localMemory = local;
    backendSess = std::move(backend);
  }
};

template <class Function>
std::thread CompletionThread(std::promise<void>& result, Function work) {
  return std::thread([&result, work = std::move(work)] {
    try {
      work();
      result.set_value();
    } catch (...) {
      result.set_exception(std::current_exception());
    }
  });
}

void TestConcurrentRegistration(IOEngine& reader) {
  for (bool destroyStatus : {false, true}) {
    std::array<char, 64> source{}, destination{};
    source.fill(37);
    auto local =
        reader.RegisterMemory(destination.data(), destination.size(), -1, MemoryLocationType::CPU);
    ProgressLatch latch;
    LatchingEngineSession session(
        reader, local,
        std::make_shared<LatchingBackendSession>(latch, source.data(), destination.data()));
    struct SignalledStatus : TransferStatus {
      std::promise<void>* destroying;
      explicit SignalledStatus(std::promise<void>* signal) : destroying(signal) {}
      ~SignalledStatus() {
        if (destroying) destroying->set_value();
      }
    };
    std::promise<void> destructionEntered, firstDone, secondDone;
    auto entered = destructionEntered.get_future();
    auto first = firstDone.get_future(), second = secondDone.get_future();
    auto status = std::make_unique<SignalledStatus>(destroyStatus ? &destructionEntered : nullptr);
    session.Read(0, 0, source.size(), status.get(), reader.AllocateTransferUniqueId());
    latch.armed = true;
    auto firstThread = CompletionThread(firstDone, [&] {
      if (destroyStatus)
        reader.DeregisterMemory(local);
      else
        status->Wait();
    });
    latch.WaitCalls(1);
    auto secondThread = CompletionThread(secondDone, [&] {
      if (destroyStatus)
        status.reset();
      else
        reader.DeregisterMemory(local);
    });
    if (destroyStatus)
      RequireReady(entered, "status destructor did not enter");
    else
      latch.WaitCalls(2);
    const bool blocked =
        second.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    latch.release.set_value();
    RequireReady(first, "completion/deregister deadlocked");
    RequireReady(second, "concurrent deregister/destruction deadlocked");
    firstThread.join();
    secondThread.join();
    first.get();
    second.get();
    Check(blocked, "deregister or status destruction returned before pending completion");
    Check(source == destination, "deregister did not finish the data transfer");
  }
}

void TestCompletionGuard() {
  auto resource = std::make_shared<int>(1);
  std::weak_ptr<int> weak = resource;
  TransferStatus drained;
  fabric::internal::SettleSubmissionError(&drained, "launch failed", true, [hold = resource]() {});
  resource.reset();
  Check(weak.expired(), "confirmed drain must release retained resources");
  Check(drained.Failed() && !drained.ResourcesQuarantined(), "confirmed error must settle");

  resource = std::make_shared<int>(2);
  weak = resource;
  TransferStatus unconfirmed;
  fabric::internal::SettleSubmissionError(&unconfirmed, "fence failed", false,
                                          [hold = resource]() {});
  resource.reset();
  Check(!weak.expired(), "unconfirmed fence must retain resources");
  Check(unconfirmed.Failed() && unconfirmed.ResourcesQuarantined(), "quarantine flag missing");

  resource = std::make_shared<int>(3);
  weak = resource;
  auto prepared = std::make_unique<fabric::internal::QuarantineEntry>();
  prepared->region = resource;
  resource.reset();
  {
    AllocationFailureScope fail(0);
    fabric::internal::QuarantineResources(std::move(prepared));
  }
  Check(!weak.expired(), "preallocated quarantine lost backing under host allocation pressure");
}

void TestWireIdentity() {
  MemoryDesc descriptor;
  descriptor.loc = MemoryLocationType::GPU;
  descriptor.registeredBackendMask = BackendBit(BackendType::FABRIC);
  descriptor.fabricHandle[3] = 17;
  descriptor.fabricOffset = 4096;
  descriptor.fabricAllocSize = 65536;
  msgpack::sbuffer bytes;
  msgpack::pack(bytes, descriptor);
  auto object = msgpack::unpack(bytes.data(), bytes.size());
  MemoryDesc roundTrip;
  object.get().convert(roundTrip);
  Check(roundTrip == descriptor, "fabric descriptor lost capability or offsets");
  roundTrip.fabricHandle[3]++;
  Check(!(roundTrip == descriptor), "fabric handle changes must affect identity");
  // An older append-only array omits the mask and keeps its compatibility default.
  auto legacy = object.get();
  legacy.via.array.size--;
  MemoryDesc old;
  legacy.convert(old);
  Check(old.registeredBackendMask == kAllBackendMask, "legacy registration mask default changed");
}

void TestSimRegistrationAndBatch() {
  setenv("MORI_IO_SIM_ENABLE", "1", 1);
  setenv("MORI_IO_SIM_BACKEND", "rdma", 1);
  setenv("MORI_IO_SIM_RDMA_LATENCY_US", "50000", 1);
  IOEngine owner("fabric-test-owner", {"127.0.0.1", 0});
  IOEngine reader("fabric-test-reader", {"127.0.0.1", 0});
  RdmaBackendConfig config;
  owner.CreateBackend(BackendType::RDMA, config);
  reader.CreateBackend(BackendType::RDMA, config);
  owner.RegisterRemoteEngine(reader.GetEngineDesc());
  reader.RegisterRemoteEngine(owner.GetEngineDesc());
  std::array<char, 4096> src{}, dst{};
  auto remote = owner.RegisterMemory(src.data(), src.size(), -1, MemoryLocationType::CPU);
  auto local = reader.RegisterMemory(dst.data(), dst.size(), -1, MemoryLocationType::CPU);
  auto excluded = reader.RegisterMemoryWithBackends(dst.data(), 1024, -1, MemoryLocationType::CPU,
                                                    BackendBit(BackendType::FABRIC));
  Check(excluded.registeredBackendMask == 0, "excluded backend was registered");
  Check(!reader.GetSelectedBackend(excluded, remote), "excluded RDMA endpoint routed to RDMA");
  Check(reader.GetSelectedBackend(local, remote) == BackendType::RDMA, "expected simulated RDMA");
  reader.SetTransferBackendMask(BackendBit(BackendType::FABRIC));
  Check(!reader.GetSelectedBackend(local, remote), "strict FABRIC must prohibit fallback");
  reader.SetTransferBackendMask(kAllBackendMask);
  TestConcurrentRegistration(reader);

  for (bool read : {true, false}) {
    std::array<TransferStatus, 2> statuses;
    TransferStatusPtrVec pointers{&statuses[0], &statuses[1]};
    TransferUniqueIdVec ids{reader.AllocateTransferUniqueId(), reader.AllocateTransferUniqueId()};
    MemDescVec locals{excluded, local}, remotes{remote, remote};
    BatchSizeVec offsets{{0}, {0}}, sizes{{32}, {32}};
    if (read)
      reader.BatchRead(locals, offsets, remotes, offsets, sizes, pointers, ids);
    else
      reader.BatchWrite(locals, offsets, remotes, offsets, sizes, pointers, ids);
    statuses[1].Wait();
    Check(statuses[0].Failed(), "unavailable group must be terminal failure");
    Check(statuses[1].Succeeded(), "unavailable group truncated later valid group");
  }

  TransferStatus pending;
  reader.Read(local, 0, remote, 0, 32, &pending, reader.AllocateTransferUniqueId());
  // Deregister drives a pending status to completion before freeing its registration.
  reader.DeregisterMemory(local);
  Check(pending.Succeeded(), "deregister returned before submitted work settled");
  TransferStatus after;
  reader.Read(local, 0, remote, 0, 32, &after, reader.AllocateTransferUniqueId());
  Check(after.Failed(), "deregistered local region accepted new work");
  bool rejectedRemote = false;
  try {
    reader.DeregisterMemory(remote);
  } catch (const std::invalid_argument&) {
    rejectedRemote = true;
  }
  Check(rejectedRemote, "remote invalidation entered local memory namespace");
  reader.DeregisterRemoteEngine(owner.GetEngineDesc());
  Check(!reader.GetSelectedBackend(excluded, remote), "invalidated peer routed new work");
  reader.DeregisterMemory(excluded);
  owner.DeregisterMemory(remote);
}

void TestGpuAllocationFailures(IOEngine& reader, const MemoryDesc& remote, void* destination,
                               size_t bytes, const char* pattern) {
  size_t injected = 0, successful = 0;
  for (int budget = 0; budget <= 48; ++budget) {
    auto local = reader.RegisterMemoryWithBackends(destination, bytes, 0, MemoryLocationType::GPU,
                                                   BackendBit(BackendType::FABRIC));
    TransferStatus status;
    try {
      AllocationFailureScope fail(budget);
      if (budget % 2 == 0) {
        reader.Read(local, 0, remote, 0, bytes, &status, reader.AllocateTransferUniqueId());
      } else {
        TransferStatusPtrVec pointers{&status};
        TransferUniqueIdVec ids{reader.AllocateTransferUniqueId()};
        reader.BatchRead({local}, {{0, bytes / 2}}, {remote}, {{0, bytes / 2}},
                         {{bytes / 4, bytes / 4}}, pointers, ids);
      }
    } catch (const std::bad_alloc&) {
    }
    // Some HIP/runtime layers translate an allocation exception to a terminal
    // error instead of propagating it. Count the injection itself in either case.
    const bool allocationFailed = allocationFailureTriggered;
    if (allocationFailed)
      ++injected;
    else
      ++successful;
    // A post-enqueue allocation failure formerly left IN_PROGRESS without any
    // progress/wait callback. Bounded Wait exposes that defect without hanging.
    Check(status.WaitFor(100) != StatusCode::IN_PROGRESS,
          "allocation failure stranded a submitted GPU status without completion");
    status.Wait();
    Check(allocationFailed || status.Succeeded(), "uninjected GPU submission failed");
    Check(!status.ResourcesQuarantined(), "host allocation failure did not drain GPU work");
    reader.DeregisterMemory(local);
    if (status.Succeeded()) {
      std::vector<char> actual(bytes);
      Check(hipMemcpy(actual.data(), destination, bytes, hipMemcpyDeviceToHost) == hipSuccess,
            "allocation regression download failed");
      const size_t checked = budget % 2 == 0 ? bytes : bytes / 4;
      Check(std::memcmp(actual.data(), pattern, checked) == 0,
            "allocation regression copied incorrect bytes");
      if (budget % 2 != 0)
        Check(std::memcmp(actual.data() + bytes / 2, pattern + bytes / 2, bytes / 4) == 0,
              "allocation regression batch copied incorrect second range");
    }
  }
  Check(injected > 0 && successful > 0, "allocation sweep missed failure or successful submission");
}

// Exhaust every pair of source/destination offsets modulo 16, including
// prefixes shorter than a word and sizes on either side of vector/page ends.
void TestCopyBoundaries(const char* modulePath) {
  constexpr size_t stride = 8192;
  constexpr size_t cases = 16 * 16;
  constexpr size_t bytes = stride * cases;
  void *source = nullptr, *destination = nullptr, *deviceSegments = nullptr;
  struct Segment {
    const char* src;
    char* dst;
    size_t size;
  };
  Check(hipMalloc(&source, bytes) == hipSuccess, "boundary source allocation");
  Check(hipMalloc(&destination, bytes) == hipSuccess, "boundary destination allocation");
  Check(hipMalloc(&deviceSegments, cases * sizeof(Segment)) == hipSuccess, "boundary descriptors");
  std::vector<unsigned char> input(bytes), actual(bytes), expected(bytes);
  for (size_t i = 0; i < bytes; ++i) input[i] = (i * 13 + (i >> 8) * 7) % 251;
  Check(hipMemcpy(source, input.data(), bytes, hipMemcpyHostToDevice) == hipSuccess,
        "boundary source initialization");
  hipModule_t module{};
  hipFunction_t contiguous{}, batch{};
  Check(hipModuleLoad(&module, modulePath) == hipSuccess, "boundary module");
  Check(hipModuleGetFunction(&contiguous, module, "fabricCopyKernel") == hipSuccess,
        "boundary contiguous function");
  Check(hipModuleGetFunction(&batch, module, "fabricBatchCopyKernel") == hipSuccess,
        "boundary batch function");
  for (size_t length : {0u, 1u, 2u, 3u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 255u, 256u, 257u,
                        4095u, 4096u, 4097u}) {
    std::fill(expected.begin(), expected.end(), 0xd3);
    std::vector<Segment> segments;
    for (size_t si = 0; si < 16; ++si) {
      for (size_t di = 0; di < 16; ++di) {
        const size_t base = (si * 16 + di) * stride + 32;
        segments.push_back({static_cast<char*>(source) + base + si,
                            static_cast<char*>(destination) + base + di, length});
        std::memcpy(expected.data() + base + di, input.data() + base + si, length);
      }
    }
    Check(hipMemcpy(deviceSegments, segments.data(), cases * sizeof(Segment),
                    hipMemcpyHostToDevice) == hipSuccess,
          "boundary descriptors upload");
    // The one-segment entry point and both small/large batched grids must obey
    // the same byte coverage. The entire allocation verifies all guard gaps.
    for (unsigned blocks : {0u, 1u, 8u, 128u}) {
      Check(hipMemset(destination, 0xd3, bytes) == hipSuccess, "boundary destination reset");
      if (blocks == 0) {
        for (auto segment : segments) {
          void* args[] = {&segment.dst, &segment.src, &segment.size};
          Check(hipModuleLaunchKernel(contiguous, 8, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr) ==
                    hipSuccess,
                "boundary contiguous launch");
        }
      } else {
        size_t count = cases;
        void* args[] = {&deviceSegments, &count, &blocks};
        Check(hipModuleLaunchKernel(batch, cases * blocks, 1, 1, 256, 1, 1, 0, nullptr, args,
                                    nullptr) == hipSuccess,
              "boundary batch launch");
      }
      Check(hipDeviceSynchronize() == hipSuccess, "boundary copy drain");
      Check(hipMemcpy(actual.data(), destination, bytes, hipMemcpyDeviceToHost) == hipSuccess,
            "boundary result download");
      Check(actual == expected, "boundary payload or guard mismatch");
    }
  }
  Check(hipModuleUnload(module) == hipSuccess, "boundary module unload");
  Check(hipFree(deviceSegments) == hipSuccess, "boundary descriptors free");
  Check(hipFree(destination) == hipSuccess, "boundary destination free");
  Check(hipFree(source) == hipSuccess, "boundary source free");
  std::cout << "PASS copy boundaries: 19 sizes x 256 alignments x 4 launch modes\n";
}

void TestGpuFabric(const char* module) {
  unsetenv("MORI_IO_SIM_ENABLE");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP GPU: no HIP devices\n";
    return;
  }
  IOEngine owner("gpu-fabric-owner", {"127.0.0.1", 0});
  IOEngine reader("gpu-fabric-reader", {"127.0.0.1", 0});
  FabricBackendConfig config;
  config.cacheBatchDescriptors = true;
  owner.CreateBackend(BackendType::FABRIC, config);
  reader.CreateBackend(BackendType::FABRIC, config);
  owner.LoadFabricCopyModule(module);
  reader.LoadFabricCopyModule(module);
  if (!reader.IsFabricReady(0)) {
    std::cout << "SKIP GPU: device 0 fabric or module unavailable\n";
    return;
  }
  TestCopyBoundaries(module);
  constexpr size_t bytes = 65536;
  void* source = FabricMalloc(bytes, 0);
  Check(source != nullptr, "fabric VMM allocation failed on ready device");
  void* destination = nullptr;
  Check(hipMalloc(&destination, bytes) == hipSuccess, "GPU destination allocation failed");
  void* defaultStreamConsumer = nullptr;
  Check(hipMalloc(&defaultStreamConsumer, bytes) == hipSuccess,
        "default-stream consumer allocation failed");
  std::array<char, bytes> pattern{}, actual{};
  for (size_t i = 0; i < bytes; i++) pattern[i] = static_cast<char>(i * 13);
  Check(hipMemcpy(source, pattern.data(), bytes, hipMemcpyHostToDevice) == hipSuccess,
        "source initialization failed");
  auto remote = owner.RegisterMemoryWithBackends(source, bytes, 0, MemoryLocationType::GPU,
                                                 BackendBit(BackendType::FABRIC));
  auto local = reader.RegisterMemoryWithBackends(destination, bytes, 0, MemoryLocationType::GPU,
                                                 BackendBit(BackendType::FABRIC));
  reader.RegisterRemoteEngine(owner.GetEngineDesc());
  reader.SetTransferBackendMask(BackendBit(BackendType::FABRIC));
  Check(reader.GetSelectedBackend(local, remote) == BackendType::FABRIC, "GPU test fell back");
  TransferStatus status;
  TransferStatusPtrVec pointers{&status};
  TransferUniqueIdVec ids{reader.AllocateTransferUniqueId()};
  SizeVec srcOffsets{17, 4096, 8192}, dstOffsets{33, 16384, 32768}, lengths{81, 2048, 4096};
  // Repeated descriptors must reuse only completed slots; changes in count or
  // offsets must upload a new layout. Verify guards as well as payload.
  for (int round = 0; round < 6; ++round) {
    const bool changed = round >= 2 && round < 4;
    SizeVec from = changed ? SizeVec{19, 4098} : srcOffsets;
    SizeVec to = changed ? SizeVec{35, 16386} : dstOffsets;
    SizeVec count = changed ? SizeVec{83, 2050} : lengths;
    Check(hipMemset(destination, 0x5a, bytes) == hipSuccess, "layout guard initialization");
    TransferStatus repeat;
    TransferStatusPtrVec repeatPointers{&repeat};
    TransferUniqueIdVec repeatIds{reader.AllocateTransferUniqueId()};
    reader.BatchRead({local}, {to}, {remote}, {from}, {count}, repeatPointers, repeatIds);
    repeat.Wait();
    Check(repeat.Succeeded(), "repeated descriptor copy failed");
    Check(hipMemcpy(actual.data(), destination, bytes, hipMemcpyDeviceToHost) == hipSuccess,
          "layout download");
    for (size_t at = 0; at < bytes; ++at) {
      char expected = 0x5a;
      for (size_t i = 0; i < count.size(); ++i)
        if (at >= to[i] && at - to[i] < count[i]) expected = pattern[from[i] + at - to[i]];
      Check(actual[at] == expected, "stale cached descriptor or guard overwrite");
    }
  }
  reader.BatchRead({local}, {dstOffsets}, {remote}, {srcOffsets}, {lengths}, pointers, ids);
  hipStream_t stream = nullptr;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess, "stream creation");
  Check(status.WaitOnStream(reinterpret_cast<void*>(stream)) == StatusCode::SUCCESS,
        "consumer stream dependency unavailable");
  Check(status.WaitOnStream(nullptr) == StatusCode::SUCCESS,
        "HIP default-stream dependency unavailable");
  Check(hipMemcpyAsync(defaultStreamConsumer, destination, bytes, hipMemcpyDeviceToDevice,
                       nullptr) == hipSuccess,
        "default-stream GPU consumer enqueue failed");
  // Cache invalidation must retain imported VA until this submission finishes.
  reader.DeregisterRemoteEngine(owner.GetEngineDesc());
  status.Wait();
  Check(status.Succeeded(), "in-flight copy failed after peer invalidation");
  Check(hipStreamSynchronize(nullptr) == hipSuccess, "default-stream GPU consumer failed");
  Check(hipMemcpy(actual.data(), defaultStreamConsumer, bytes, hipMemcpyDeviceToHost) == hipSuccess,
        "copy verification download");
  for (size_t segment = 0; segment < lengths.size(); segment++) {
    for (size_t i = 0; i < lengths[segment]; i++) {
      Check(actual[dstOffsets[segment] + i] == pattern[srcOffsets[segment] + i],
            "batch kernel copied incorrect bytes");
    }
  }
  Check(hipStreamSynchronize(stream) == hipSuccess, "consumer stream synchronization failed");
  Check(hipStreamDestroy(stream) == hipSuccess, "consumer stream destruction failed");
  reader.RegisterRemoteEngine(owner.GetEngineDesc());
  TestGpuAllocationFailures(reader, remote, destination, bytes, pattern.data());
  auto abandonedLocal = reader.RegisterMemoryWithBackends(
      destination, bytes, 0, MemoryLocationType::GPU, BackendBit(BackendType::FABRIC));
  reader.RegisterRemoteEngine(owner.GetEngineDesc());
  {
    TransferStatus abandoned;
    reader.Read(abandonedLocal, 0, remote, 0, bytes, &abandoned, reader.AllocateTransferUniqueId());
    // Dropping an unobserved event retains both registration and import holds.
  }
  reader.DeregisterRemoteEngine(owner.GetEngineDesc());
  Check(hipDeviceSynchronize() == hipSuccess, "abandoned transfer device drain failed");
  bool quarantined = false;
  try {
    reader.DeregisterMemory(abandonedLocal);
  } catch (const std::runtime_error&) {
    quarantined = true;
  }
  Check(quarantined, "abandoned pending status released its local registration");
  reader.DeregisterMemory(local);
  owner.DeregisterMemory(remote);
  Check(hipFree(defaultStreamConsumer) == hipSuccess, "default-stream consumer release failed");
  Check(hipFree(destination) == hipSuccess, "GPU destination release failed");
  FabricFree(source);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(argc == 1 || (argc == 3 && std::string(argv[1]) == "--gpu-module"),
          "usage: test_fabric_integration [--gpu-module PATH]");
    TestCompletionGuard();
    TestWireIdentity();
    if (argc == 3 && std::string(argv[1]) == "--gpu-module")
      TestGpuFabric(argv[2]);
    else
      TestSimRegistrationAndBatch();
    std::cout << "PASS fabric integration\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
