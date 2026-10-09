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

// Drives one RDMA loopback write between two in-process engines under each
// injected fault and checks how the transfer ends. Requires a build with
// -DENABLE_IO_FAULT_INJECTION=ON, one active RDMA NIC and one GPU.

#include <arpa/inet.h>
#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/fault_injector.hpp"

using namespace mori::io;

namespace {

constexpr size_t kXferBytes = 1 << 20;
constexpr int kFailFastMs = 5000;
constexpr int kHangProbeMs = 3000;

struct TestSkip : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  socklen_t len = sizeof(addr);
  int port = -1;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
      getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
    port = ntohs(addr.sin_port);
  }
  close(fd);
  return port;
}

struct GpuBuffer {
  IOEngine* owner{nullptr};
  MemoryDesc desc{};
  void* ptr{nullptr};

  GpuBuffer(IOEngine* engine, int fill) : owner(engine) {
    HIP_RUNTIME_CHECK(hipSetDevice(0));
    HIP_RUNTIME_CHECK(hipMalloc(&ptr, kXferBytes));
    HIP_RUNTIME_CHECK(hipMemset(ptr, fill, kXferBytes));
    desc = engine->RegisterMemory(ptr, kXferBytes, 0, MemoryLocationType::GPU);
  }
  ~GpuBuffer() {
    owner->DeregisterMemory(desc);
    HIP_RUNTIME_CHECK(hipFree(ptr));
  }
};

// Two connected engines plus a source/destination buffer pair. Field order
// matters: buffers deregister before their engines are destroyed.
struct LoopbackRig {
  std::unique_ptr<IOEngine> initiator;
  std::unique_ptr<IOEngine> target;
  std::unique_ptr<GpuBuffer> src;
  std::unique_ptr<GpuBuffer> dst;

  LoopbackRig() {
    if (!RdmaBackend::HasActiveDevices()) throw TestSkip("requires an active RDMA device");
    int gpus = 0;
    if (hipGetDeviceCount(&gpus) != hipSuccess || gpus == 0) throw TestSkip("requires a GPU");

    initiator = MakeEngine("fault_initiator");
    target = MakeEngine("fault_target");
    initiator->RegisterRemoteEngine(target->GetEngineDesc());
    target->RegisterRemoteEngine(initiator->GetEngineDesc());
    src = std::make_unique<GpuBuffer>(initiator.get(), 0xAB);
    dst = std::make_unique<GpuBuffer>(target.get(), 0x00);
  }

  ~LoopbackRig() {
    src.reset();
    dst.reset();
  }

  // Posts one write and waits up to timeoutMs; returns the final code.
  StatusCode Write(TransferStatus* status, int timeoutMs) {
    initiator->Write(src->desc, 0, dst->desc, 0, kXferBytes, status,
                     initiator->AllocateTransferUniqueId());
    return status->WaitFor(timeoutMs);
  }

  bool DestinationMatchesSource() {
    std::vector<unsigned char> host(kXferBytes);
    HIP_RUNTIME_CHECK(hipMemcpy(host.data(), dst->ptr, kXferBytes, hipMemcpyDeviceToHost));
    for (unsigned char b : host) {
      if (b != 0xAB) return false;
    }
    return true;
  }

 private:
  static std::unique_ptr<IOEngine> MakeEngine(const std::string& key) {
    IOEngineConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = GetFreePort();
    Require(cfg.port > 0, "failed to allocate a tcp port");
    auto engine = std::make_unique<IOEngine>(key, cfg);
    RdmaBackendConfig rdmaCfg{};
    rdmaCfg.enableNotification = false;  // keep each write to a single data CQE
    engine->CreateBackend(BackendType::RDMA, rdmaCfg);
    return engine;
  }
};

struct ScopedFault {
  explicit ScopedFault(const std::string& spec) {
    FaultInjector::Instance().Arm(ParseFaultRule(spec));
  }
  ~ScopedFault() { FaultInjector::Instance().Disarm(); }
};

void RequireFailsFast(const std::string& spec) {
  TransferStatus status;
  LoopbackRig rig;
  ScopedFault fault(spec);
  auto start = std::chrono::steady_clock::now();
  StatusCode code = rig.Write(&status, kFailFastMs);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                  start)
                .count();
  Require(FaultInjector::Instance().Fired() == 1, spec + ": fault did not fire exactly once");
  Require(code == StatusCode::ERR_RDMA_OP,
          spec + ": expected ERR_RDMA_OP within " + std::to_string(kFailFastMs) +
              " ms, got code=" + std::to_string(static_cast<int>(code)));
  std::printf("    %s -> failed in %lld ms: %s\n", spec.c_str(), static_cast<long long>(ms),
              status.Message().c_str());
}

void CaseParseRule() {
  FaultRule r = ParseFaultRule("cqe_drop:op=write:skip=3:count=2:qpn=17");
  Require(r.kind == FaultKind::CqeDrop && r.wcOpcode == IBV_WC_RDMA_WRITE && r.skip == 3 &&
              r.count == 2 && r.qpn == 17,
          "cqe_drop spec parsed wrong");
  r = ParseFaultRule("cqe_error");
  Require(r.value == IBV_WC_RETRY_EXC_ERR, "cqe_error should default to retry_exc");
  r = ParseFaultRule("cqe_error:value=rem_access");
  Require(r.value == IBV_WC_REM_ACCESS_ERR, "named wc status not parsed");
  for (const char* bad : {"no_such_kind", "cqe_drop:skip", "cqe_drop:bogus=1", "qp_error:skip=x"}) {
    bool threw = false;
    try {
      ParseFaultRule(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Require(threw, std::string("expected parse error for '") + bad + "'");
  }
}

void CaseDisarmedWriteSucceeds() {
  TransferStatus status;
  LoopbackRig rig;
  StatusCode code = rig.Write(&status, kFailFastMs);
  Require(code == StatusCode::SUCCESS, "baseline write failed: " + status.Message());
  Require(rig.DestinationMatchesSource(), "baseline write corrupted data");
}

void CasePostSendFailFailsFast() { RequireFailsFast("post_send_fail"); }

void CaseCqeErrorFailsFast() { RequireFailsFast("cqe_error:op=write:value=retry_exc"); }

void CaseQpErrorFailsFast() { RequireFailsFast("qp_error"); }

// A lost completion is the silent-hang class: MORI has no per-WR deadline, so
// the transfer stays IN_PROGRESS forever. This pins today's behavior; flip it
// to RequireFailsFast once a completion deadline lands.
void CaseCqeDropHangsWithoutDeadline() {
  TransferStatus status;
  LoopbackRig rig;
  ScopedFault fault("cqe_drop:op=write");
  StatusCode code = rig.Write(&status, kHangProbeMs);
  Require(FaultInjector::Instance().Fired() == 1, "cqe_drop did not fire");
  Require(code == StatusCode::IN_PROGRESS,
          "expected the transfer to hang, got code=" + std::to_string(static_cast<int>(code)));
  std::printf("    cqe_drop -> still IN_PROGRESS after %d ms (known hang)\n", kHangProbeMs);
}

}  // namespace

int main() {
  // Engines re-read the IO level from the environment; keep the injector's
  // "armed"/"injected" warnings visible unless the caller chose a level.
  setenv("MORI_IO_LOG_LEVEL", "warn", /*overwrite=*/0);
  SetLogLevel("warn");
  struct TestCase {
    const char* name;
    std::function<void()> run;
  };
  std::vector<TestCase> cases = {
      {"parse_rule", CaseParseRule},
      {"disarmed_write_succeeds", CaseDisarmedWriteSucceeds},
      {"post_send_fail_fails_fast", CasePostSendFailFailsFast},
      {"cqe_error_fails_fast", CaseCqeErrorFailsFast},
      {"qp_error_fails_fast", CaseQpErrorFailsFast},
      {"cqe_drop_hangs_without_deadline", CaseCqeDropHangsWithoutDeadline},
  };

  int failed = 0;
  for (const TestCase& tc : cases) {
    try {
      tc.run();
      std::printf("[PASS] %s\n", tc.name);
    } catch (const TestSkip& e) {
      std::printf("[SKIP] %s: %s\n", tc.name, e.what());
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s: %s\n", tc.name, e.what());
      failed++;
    }
  }
  std::printf("==== test_fault_injection: %zu cases, %d failed ====\n", cases.size(), failed);
  return failed == 0 ? 0 : 1;
}
