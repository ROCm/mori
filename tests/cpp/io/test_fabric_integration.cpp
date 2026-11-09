#include <hip/hip_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <msgpack.hpp>
#include <stdexcept>
#include <thread>

#include "mori/io/io.hpp"
#include "src/io/fabric/backend_impl.hpp"
#include "src/io/fabric/completion_guard.hpp"

using namespace mori::io;

namespace {
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
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
  owner.CreateBackend(BackendType::FABRIC, config);
  reader.CreateBackend(BackendType::FABRIC, config);
  owner.LoadFabricCopyModule(module);
  reader.LoadFabricCopyModule(module);
  if (!reader.IsFabricReady(0)) {
    std::cout << "SKIP GPU: device 0 fabric or module unavailable\n";
    return;
  }
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
