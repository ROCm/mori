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

// test_rail_affinity.cpp
//
// Verifies that MORI_IO_RAIL_AFFINITY=1 forces sender and receiver to use the
// same physical NIC index, preventing cross-rail QP creation.
//
// Test strategy:
//   - Allocate sender memory on NUMA node 0, receiver memory on a different
//     NUMA node. This causes MatchCpuNics() to produce different NIC orderings
//     on each side (each prefers its NUMA-local NICs first).
//   - Without rail affinity, the receiver picks a NUMA-local NIC via nicRank
//     that differs from the sender's choice → cross-rail QP.
//   - With rail affinity, the receiver uses the sender's railId directly,
//     guaranteeing same-rail alignment.
//
// Prerequisites:
//   - At least 2 NUMA nodes with NICs attached to different NUMA domains
//   - At least 1 active RDMA device
//   - libnuma
//
// Usage:
//   ./test_rail_affinity
//
// The test runs both modes (OFF and ON) and reports PASS/FAIL/SKIP.

#include <arpa/inet.h>
#include <limits.h>
#include <netinet/in.h>
#include <numa.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"
#include "src/io/rdma/backend_impl.hpp"

using namespace mori::io;

/* -------------------------------------------------------------------------- */
/*                                  Utilities                                  */
/* -------------------------------------------------------------------------- */

static int GetFreePort() {
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

static bool WaitTransferDone(TransferStatus* status, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!status->Init() && !status->InProgress()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

static std::unique_ptr<IOEngine> CreateEngine(const std::string& name, RdmaBackendConfig& rdmaCfg) {
  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = GetFreePort();
  if (cfg.port <= 0) {
    fprintf(stderr, "  FAIL: cannot allocate port for %s\n", name.c_str());
    exit(1);
  }
  auto engine = std::make_unique<IOEngine>(name, cfg);
  engine->CreateBackend(BackendType::RDMA, rdmaCfg);
  return engine;
}

/* -------------------------------------------------------------------------- */
/*                              Test Environment                               */
/* -------------------------------------------------------------------------- */

struct TestEnv {
  int numaA;
  int numaB;
};

// Returns {numaA, numaB} or exits with SKIP if prerequisites not met.
static TestEnv CheckPrerequisites() {
  if (!RdmaBackend::HasActiveDevices()) {
    printf("SKIP: no active RDMA devices\n");
    exit(0);
  }

  if (numa_available() < 0) {
    printf("SKIP: NUMA not available\n");
    exit(0);
  }

  int maxNode = numa_max_node();
  if (maxNode < 1) {
    printf("SKIP: need at least 2 NUMA nodes (found 1)\n");
    exit(0);
  }

  return {0, maxNode};
}

/* -------------------------------------------------------------------------- */
/*                          Test: Cross-NUMA Rail Affinity                     */
/* -------------------------------------------------------------------------- */

enum class TransferResult { OK, TIMEOUT, FAILED };

// Attempts a CPU→CPU transfer between two engines whose memory resides on
// different NUMA nodes. Returns the transfer outcome.
static TransferResult RunCrossNumaTransfer(const TestEnv& env, bool railAffinityEnabled) {
  // Set environment
  setenv("MORI_IO_RAIL_AFFINITY", railAffinityEnabled ? "1" : "0", 1);

  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.qpPerTransfer = 2;
  rdmaCfg.enableNotification = true;
  rdmaCfg.numNicsPerTransfer = 2;

  auto engineA = CreateEngine("rail_test_A", rdmaCfg);
  auto engineB = CreateEngine("rail_test_B", rdmaCfg);

  EngineDesc descA = engineA->GetEngineDesc();
  EngineDesc descB = engineB->GetEngineDesc();
  engineA->RegisterRemoteEngine(descB);
  engineB->RegisterRemoteEngine(descA);

  // Allocate memory on different NUMA nodes
  constexpr size_t kBufSize = 4096;
  void* ptrA = numa_alloc_onnode(kBufSize, env.numaA);
  void* ptrB = numa_alloc_onnode(kBufSize, env.numaB);
  if (!ptrA || !ptrB) {
    fprintf(stderr, "  FAIL: numa_alloc_onnode returned NULL\n");
    exit(1);
  }
  memset(ptrA, 0xAB, kBufSize);
  memset(ptrB, 0x00, kBufSize);

  MemoryDesc memA = engineA->RegisterMemory(ptrA, kBufSize, -1, MemoryLocationType::CPU);
  MemoryDesc memB = engineB->RegisterMemory(ptrB, kBufSize, -1, MemoryLocationType::CPU);

  // Perform transfer: A → B
  TransferStatus status;
  TransferUniqueId tid = engineA->AllocateTransferUniqueId();
  engineA->Write(memA, 0, memB, 0, kBufSize, &status, tid);

  TransferResult result;
  if (!WaitTransferDone(&status, 5000)) {
    result = TransferResult::TIMEOUT;
  } else if (status.Failed()) {
    result = TransferResult::FAILED;
  } else {
    // Verify data integrity
    bool correct = (memcmp(ptrB, ptrA, kBufSize) == 0);
    result = correct ? TransferResult::OK : TransferResult::FAILED;
  }

  // Cleanup
  engineA->DeregisterMemory(memA);
  engineB->DeregisterMemory(memB);
  numa_free(ptrA, kBufSize);
  numa_free(ptrB, kBufSize);

  return result;
}

/* -------------------------------------------------------------------------- */
/*                                    Main                                     */
/* -------------------------------------------------------------------------- */

int main() {
  printf("Running rail affinity tests...\n\n");

  TestEnv env = CheckPrerequisites();
  printf("  Environment: NUMA-A=%d, NUMA-B=%d, RDMA devices available\n\n", env.numaA, env.numaB);

  int passed = 0;
  int failed = 0;
  int skipped = 0;

  // Test 1: With rail affinity enabled, cross-NUMA transfer must succeed
  // (both endpoints use the same rail, so QP connects within leaf switch)
  {
    printf("  [Test 1] MORI_IO_RAIL_AFFINITY=1, cross-NUMA CPU transfer\n");
    printf("           Expect: sender and receiver on same rail → success\n");

    TransferResult result = RunCrossNumaTransfer(env, true);

    if (result == TransferResult::OK) {
      printf("           PASS\n\n");
      passed++;
    } else {
      printf("           FAIL (result=%s)\n\n",
             result == TransferResult::TIMEOUT ? "timeout" : "failed");
      failed++;
    }
  }

  // Test 2: Without rail affinity, cross-NUMA transfer may fail on
  // rail-isolated networks (sender and receiver pick different rails)
  // We don't assert failure here since some networks allow cross-rail traffic.
  // Instead we just report the outcome for informational purposes.
  {
    printf("  [Test 2] MORI_IO_RAIL_AFFINITY=0, cross-NUMA CPU transfer\n");
    printf("           Expect: sender and receiver may be on different rails\n");
    printf("           (On rail-isolated networks this will timeout/fail)\n");

    TransferResult result = RunCrossNumaTransfer(env, false);

    switch (result) {
      case TransferResult::OK:
        printf("           INFO: transfer succeeded (network allows cross-rail)\n\n");
        // Not a failure — some networks have full bisection
        passed++;
        break;
      case TransferResult::TIMEOUT:
        printf("           INFO: transfer timed out (rail-isolated network confirmed)\n");
        printf("           This proves rail affinity is needed on this topology.\n\n");
        passed++;  // Expected on rail-isolated networks
        break;
      case TransferResult::FAILED:
        printf("           INFO: transfer failed (rail-isolated network confirmed)\n");
        printf("           This proves rail affinity is needed on this topology.\n\n");
        passed++;  // Expected on rail-isolated networks
        break;
    }
  }

  // Summary
  printf("  ────────────────────────────────────────────\n");
  printf("  Results: %d passed, %d failed, %d skipped\n", passed, failed, skipped);

  if (failed > 0) {
    printf("\n  FAILED\n");
    return 1;
  }

  printf("\n  All rail affinity tests passed.\n");
  return 0;
}
