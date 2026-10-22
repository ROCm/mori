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

// test_fence.cpp
//
// Verifies IOEngine::FenceRemoteEngine: after a receiver fences a peer, that peer
// can no longer write into the receiver's registered memory.
//
// The scenario this exists for: a WRITE target gets no completion, so when it gives
// up on a transfer it cannot know whether the sender is dead or only slow. Before
// the target hands the buffer to someone else it has to make sure a late write
// cannot land. These tests drive that with plain CPU memory and engines in a single
// process, so they need an RDMA NIC but no GPU and no inference stack.
//
// Cases:
//   1. baseline            an unfenced peer's write lands
//   2. stale writer        after the fence, the same peer's write must NOT land
//   3. scoped to one peer  another peer still writes normally
//   4. unknown / repeated  fencing nobody, or fencing twice, is a harmless no-op
//   5. initiator reconnect the fencing engine can still initiate to the fenced peer
//   6. peer restart        a restarted peer with the same key reconnects and writes
//
// Every case runs over host memory and, when a GPU is visible, again over device
// memory, so the NIC is also shown to stop writing into VRAM.
//
// Prerequisites: at least one active RDMA device. Without one the test reports SKIP.
//
// Usage:
//   ./test_fence
//
// A stale write is expected to fail only after the NIC exhausts its transport retries
// (about half a second with the defaults); MORI_IO_QP_TIMEOUT / MORI_IO_QP_RETRY_CNT
// lengthen that.

#include <arpa/inet.h>
#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"
#include "src/io/rdma/backend_impl.hpp"

using namespace mori::io;

namespace {

struct TestFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define CHECK(cond, msg)                                                                  \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      throw TestFailure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +   \
                        (msg) + " [" #cond "]");                                          \
    }                                                                                     \
  } while (0)

constexpr size_t kBufSize = 4096;
constexpr int kTransferTimeoutMs = 10000;
// Longer than the NIC's default retry budget, so a stale write has run its course.
constexpr int kStaleWriteTimeoutMs = 30000;

const char* const kProducer = "fence_producer";
const char* const kReceiver = "fence_receiver";
const char* const kOther = "fence_other";

/* -------------------------------------------------------------------------- */
/*                                  Utilities                                  */
/* -------------------------------------------------------------------------- */

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    close(fd);
    return -1;
  }
  int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

bool WaitTerminal(TransferStatus* status, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!status->Init() && !status->InProgress()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

std::unique_ptr<IOEngine> CreateEngine(const std::string& name) {
  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.qpPerTransfer = 1;
  rdmaCfg.enableNotification = true;

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = GetFreePort();
  if (cfg.port <= 0) throw TestFailure("cannot allocate a port for " + name);
  auto engine = std::make_unique<IOEngine>(name, cfg);
  engine->CreateBackend(BackendType::RDMA, rdmaCfg);
  return engine;
}

// Where the buffers under test live. Set once per pass over the cases in main().
MemoryLocationType g_loc = MemoryLocationType::CPU;
constexpr int kGpuDevice = 0;

#define HIP_OK(expr)                                                                   \
  do {                                                                                 \
    hipError_t e_ = (expr);                                                            \
    if (e_ != hipSuccess) {                                                            \
      throw TestFailure(std::string(#expr) + " failed: " + hipGetErrorString(e_));     \
    }                                                                                  \
  } while (0)

bool GpuAvailable() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

// Buffer filled with `fill`, registered with one engine. Page-aligned host memory in
// the CPU pass, device memory in the GPU pass (the RDMA NIC then writes into VRAM).
struct Buffer {
  Buffer(IOEngine* owner, uint8_t fill) : engine(owner), loc(g_loc) {
    if (loc == MemoryLocationType::GPU) {
      HIP_OK(hipSetDevice(kGpuDevice));
      HIP_OK(hipMalloc(&data, kBufSize));
    } else {
      if (posix_memalign(&data, 4096, kBufSize) != 0) throw TestFailure("posix_memalign failed");
    }
    Fill(fill);
    desc = engine->RegisterMemory(data, kBufSize, loc == MemoryLocationType::GPU ? kGpuDevice : -1,
                                  loc);
  }
  ~Buffer() {
    engine->DeregisterMemory(desc);
    if (loc == MemoryLocationType::GPU) {
      (void)hipFree(data);
    } else {
      free(data);
    }
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void Fill(uint8_t v) {
    if (loc == MemoryLocationType::GPU) {
      HIP_OK(hipMemset(data, v, kBufSize));
      HIP_OK(hipDeviceSynchronize());
    } else {
      memset(data, v, kBufSize);
    }
  }
  bool AllEqual(uint8_t v) const {
    std::vector<uint8_t> host(kBufSize);
    if (loc == MemoryLocationType::GPU) {
      HIP_OK(hipMemcpy(host.data(), data, kBufSize, hipMemcpyDeviceToHost));
    } else {
      memcpy(host.data(), data, kBufSize);
    }
    for (size_t i = 0; i < kBufSize; ++i) {
      if (host[i] != v) return false;
    }
    return true;
  }

  IOEngine* engine;
  MemoryLocationType loc;
  void* data{nullptr};
  MemoryDesc desc;
};

// Writes `src` (owned by `from`) into `dst` (owned by another engine) and waits.
// Returns false if the write did not complete successfully within the timeout.
bool WriteAndWait(IOEngine* from, const Buffer& src, const Buffer& dst,
                  int timeoutMs = kTransferTimeoutMs) {
  TransferStatus status;
  from->Write(src.desc, 0, dst.desc, 0, kBufSize, &status, from->AllocateTransferUniqueId());
  if (!WaitTerminal(&status, timeoutMs)) return false;
  return status.Succeeded();
}

void Link(IOEngine* a, IOEngine* b) {
  a->RegisterRemoteEngine(b->GetEngineDesc());
  b->RegisterRemoteEngine(a->GetEngineDesc());
}

/* -------------------------------------------------------------------------- */
/*                                    Cases                                    */
/* -------------------------------------------------------------------------- */

// 1-3 share one set of engines, because the interesting state is what the earlier
// steps left behind: cached sessions and live QPs.
struct Rig {
  Rig()
      : receiver(CreateEngine(kReceiver)),
        producer(CreateEngine(kProducer)),
        other(CreateEngine(kOther)) {
    Link(producer.get(), receiver.get());
    Link(other.get(), receiver.get());
  }
  std::unique_ptr<IOEngine> receiver;
  std::unique_ptr<IOEngine> producer;
  std::unique_ptr<IOEngine> other;
};

void TestBaselineWriteLands(Rig& rig) {
  Buffer src(rig.producer.get(), 0xA1);
  Buffer dst(rig.receiver.get(), 0x00);

  CHECK(WriteAndWait(rig.producer.get(), src, dst), "unfenced write must succeed");
  CHECK(dst.AllEqual(0xA1), "unfenced write must land in the receiver's buffer");
}

void TestStaleWriterCannotLand(Rig& rig) {
  Buffer src(rig.producer.get(), 0xA2);
  Buffer dst(rig.receiver.get(), 0x00);

  // Establish the connection and a cached session, as a real producer would have
  // by the time the receiver gives up on it.
  CHECK(WriteAndWait(rig.producer.get(), src, dst), "warm-up write must succeed");
  CHECK(dst.AllEqual(0xA2), "warm-up write must land");

  // The receiver gives up on this transfer and wants its buffer back.
  dst.Fill(0x00);
  FenceResult fence = rig.receiver->FenceRemoteEngine(kProducer);
  CHECK(fence.supported, "RDMA backend must support fencing");
  CHECK(fence.Ok(), "fence must succeed: " + fence.message);
  CHECK(fence.endpointsFenced >= 1, "fence must have cut at least one connection");
  CHECK(fence.endpointsFailed == 0, "no connection may be left uncut");

  // The producer still holds its half of the connection and does not know it was
  // cut. This is the late write the fence exists to stop.
  src.Fill(0xBB);
  auto stale = std::make_unique<TransferStatus>();
  rig.producer->Write(src.desc, 0, dst.desc, 0, kBufSize, stale.get(),
                      rig.producer->AllocateTransferUniqueId());
  const bool terminal = WaitTerminal(stale.get(), kStaleWriteTimeoutMs);

  // The property that matters: nothing arrived, whatever the sender saw.
  CHECK(dst.AllEqual(0x00), "a fenced peer's write must not land in the buffer");
  if (terminal) {
    CHECK(!stale->Succeeded(), "a write through a fenced connection must not report success");
    printf("           stale write ended as: %s\n", stale->Message().c_str());
  } else {
    printf("           stale write still retrying after %d ms; nothing landed\n",
           kStaleWriteTimeoutMs);
    // The NIC still owns this status; leaking it is safer than freeing it under a
    // completion that may yet arrive.
    (void)stale.release();
  }
}

void TestOtherPeersUnaffected(Rig& rig) {
  Buffer src(rig.other.get(), 0xC3);
  Buffer dst(rig.receiver.get(), 0x00);

  // Peer `other` was never fenced; it must keep working after `producer` was.
  CHECK(WriteAndWait(rig.other.get(), src, dst), "an unfenced peer must still write");
  CHECK(dst.AllEqual(0xC3), "an unfenced peer's write must land");
}

void TestUnknownAndRepeatedFence(Rig& rig) {
  FenceResult unknown = rig.receiver->FenceRemoteEngine("fence_nobody");
  CHECK(unknown.Ok(), "fencing an unknown peer must be ok");
  CHECK(unknown.endpointsFenced == 0, "nothing to cut for an unknown peer");

  // `producer` was fenced in an earlier case; its connections are already gone.
  FenceResult again = rig.receiver->FenceRemoteEngine(kProducer);
  CHECK(again.Ok(), "fencing twice must be ok");
  CHECK(again.endpointsFenced == 0, "a second fence has no connections left to cut");
}

void TestInitiatorReconnectsAfterFence(Rig& rig) {
  // The receiver as initiator: it writes into `other`, building its own connection.
  Buffer src(rig.receiver.get(), 0xD4);
  Buffer dst(rig.other.get(), 0x00);
  CHECK(WriteAndWait(rig.receiver.get(), src, dst), "initiator write must succeed");
  CHECK(dst.AllEqual(0xD4), "initiator write must land");

  // Fencing `other` retires the receiver's cached session toward it as well. If it
  // were kept, the next write would go out on a dead QP and fail.
  FenceResult fence = rig.receiver->FenceRemoteEngine(kOther);
  CHECK(fence.Ok(), "fence must succeed: " + fence.message);
  CHECK(fence.endpointsFenced >= 1, "the receiver's own connection to `other` must be cut");

  dst.Fill(0x00);
  src.Fill(0xE5);
  CHECK(WriteAndWait(rig.receiver.get(), src, dst),
        "after a fence the initiator must rebuild its connection and succeed");
  CHECK(dst.AllEqual(0xE5), "the write over the rebuilt connection must land");
}

void TestRestartedPeerReconnects(Rig& rig) {
  Buffer dst(rig.receiver.get(), 0x00);

  // `producer` was fenced and its old process is gone; a restarted one comes back
  // under the same key with a new address and fresh QPs.
  rig.producer.reset();
  rig.producer = CreateEngine(kProducer);
  Link(rig.producer.get(), rig.receiver.get());

  Buffer src(rig.producer.get(), 0xF6);
  CHECK(WriteAndWait(rig.producer.get(), src, dst),
        "a restarted peer must be able to write after having been fenced");
  CHECK(dst.AllEqual(0xF6), "the restarted peer's write must land");
}

/* -------------------------------------------------------------------------- */
/*                                    Main                                     */
/* -------------------------------------------------------------------------- */

struct TestCase {
  const char* name;
  std::function<void(Rig&)> fn;
};

}  // namespace

int main() {
  printf("Running fence tests...\n\n");

  if (!RdmaBackend::HasActiveDevices()) {
    printf("SKIP: no active RDMA devices\n");
    return 0;
  }

  const std::vector<TestCase> cases = {
      {"1. baseline: an unfenced peer's write lands", TestBaselineWriteLands},
      {"2. stale writer: a write after the fence must not land", TestStaleWriterCannotLand},
      {"3. scoped: other peers are unaffected", TestOtherPeersUnaffected},
      {"4. unknown / repeated fence is a no-op", TestUnknownAndRepeatedFence},
      {"5. initiator reconnects after fencing a peer", TestInitiatorReconnectsAfterFence},
      {"6. restarted peer with the same key reconnects", TestRestartedPeerReconnects},
  };

  int passed = 0;
  int failed = 0;

  // The same cases over host memory and, when a GPU is present, over device memory.
  // A fresh set of engines per pass keeps one pass's connections out of the other.
  std::vector<MemoryLocationType> passes = {MemoryLocationType::CPU};
  if (GpuAvailable()) {
    passes.push_back(MemoryLocationType::GPU);
  } else {
    printf("NOTE: no GPU visible; running the CPU pass only\n\n");
  }

  for (MemoryLocationType loc : passes) {
    g_loc = loc;
    printf("=== buffers in %s memory ===\n\n", loc == MemoryLocationType::GPU ? "GPU" : "CPU");
    try {
      Rig rig;
      for (const auto& tc : cases) {
        printf("  [%s]\n", tc.name);
        try {
          tc.fn(rig);
          printf("           PASS\n\n");
          ++passed;
        } catch (const std::exception& e) {
          printf("           FAIL: %s\n\n", e.what());
          ++failed;
        }
      }
    } catch (const std::exception& e) {
      printf("FAIL: could not set up engines: %s\n", e.what());
      return 1;
    }
  }

  printf("%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
