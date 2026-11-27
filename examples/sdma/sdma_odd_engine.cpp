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
//
// EXPERIMENTAL: can a GPU kernel start a copy on an ODD SDMA engine?
//
// One GPU, one queue per engine, a device-posted COPY + FENCE on each. On a
// driver without ee1379e6d7da an odd engine's first queues get a doorbell that
// is not 32B aligned, the kernel's write lands on the previous engine, and the
// copy never starts -- the same doorbell value written by the host then starts
// it. MORI_SDMA_ALIGN_DEVICE_DOORBELL=1 makes SdmaQueue burn those queues until
// the doorbell is aligned, which is what this example checks:
//
//   sdma_odd_engine                             # as KFD hands the queues out
//   MORI_SDMA_ALIGN_DEVICE_DOORBELL=1 sdma_odd_engine
//
// usage: sdma_odd_engine [engines, default 0,1,2,3,4,5] [queues per engine,
//                         default 1] [hip device, default 0]
// Exits nonzero if a queue that should run does not, or if a copy lands wrong.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/env_utils.hpp"

static constexpr size_t kCopyBytes = 4096;
static constexpr double kWaitMs = 200.0;
static constexpr int kConnectChannels = 2;

// One lane posts COPY + FENCE and rings the doorbell: the whole submit path runs
// on the device, which is what the odd engines fail at.
__global__ void PostCopy(anvil::SdmaQueueDeviceHandle* dh, void* src, void* dst, size_t bytes,
                         HSAuint64* flag, uint32_t value) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  anvil::SdmaQueueDeviceHandle handle = *dh;

  uint64_t offset = 0;
  const uint64_t base =
      handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_FENCE), offset);
  uint64_t pendingWptr = base;

  auto copy = anvil::CreateCopyPacket(src, dst, bytes);
  handle.placePacket<SDMA_PKT_COPY_LINEAR>(copy, pendingWptr, offset);
  auto fence = anvil::CreateFencePacket(flag, value);
  handle.placePacket<SDMA_PKT_FENCE>(fence, pendingWptr, 0);

  handle.submitPacket(base, pendingWptr);
}

static double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  std::vector<uint32_t> engines;
  if (argc > 1) {
    std::stringstream ss(argv[1]);
    std::string tok;
    while (std::getline(ss, tok, ',')) engines.push_back(std::atoi(tok.c_str()));
  } else {
    engines = {0, 1, 2, 3, 4, 5};
  }
  const int perEngine = argc > 2 ? std::atoi(argv[2]) : 1;
  const int device = argc > 3 ? std::atoi(argv[3]) : 0;
  const bool fixOn = mori::env::IsEnvVarEnabled("MORI_SDMA_ALIGN_DEVICE_DOORBELL");

  CHECK_HIP_ERROR(hipSetDevice(device));
  anvil::anvil.init();
  const uint32_t node = anvil::AnvilLib::hsakmtNodeForHipDevice(device);

  void* src = nullptr;
  void* dst = nullptr;
  CHECK_HIP_ERROR(hipMalloc(&src, kCopyBytes));
  CHECK_HIP_ERROR(hipMalloc(&dst, kCopyBytes));
  CHECK_HIP_ERROR(hipMemset(src, 0x5a, kCopyBytes));
  CHECK_HIP_ERROR(hipMemset(dst, 0, kCopyBytes));
  // The FENCE is written by the engine and polled by the host, so it must live
  // in coherent host memory.
  uint64_t* flag = nullptr;
  CHECK_HIP_ERROR(hipHostMalloc((void**)&flag, sizeof(uint64_t), hipHostMallocCoherent));
  *flag = 0;

  printf("MORI_SDMA_ALIGN_DEVICE_DOORBELL=%d, device %d, %zu B self-copy\n", fixOn ? 1 : 0, device,
         kCopyBytes);
  printf("%-6s %-18s %-7s %-10s %s\n", "eng/ch", "doorbell", "db%32", "device", "host rings it");

  int failures = 0;
  uint32_t value = 0;
  // One device-posted copy on `queue`, into `dstBuf`. Reports and scores it.
  auto testQueue = [&](const char* label, anvil::SdmaQueue& queue, void* dstBuf) {
    anvil::SdmaQueueDeviceHandle handle{};
    CHECK_HIP_ERROR(
        hipMemcpy(&handle, queue.deviceHandle(), sizeof(handle), hipMemcpyDeviceToHost));
    const uintptr_t db = reinterpret_cast<uintptr_t>(handle.doorbell);
    const bool aligned = (db % 32) == 0;

    CHECK_HIP_ERROR(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(dstBuf), 0, kCopyBytes));
    ++value;
    PostCopy<<<1, 64>>>(queue.deviceHandle(), src, dstBuf, kCopyBytes, flag, value);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    const double t0 = NowMs();
    while (*flag != value && NowMs() - t0 < kWaitMs) {
    }
    const bool ran = *flag == value;

    // Tell a lost doorbell from a bad packet: the host writes the same value the
    // kernel already wrote, and also drains the queue before it is destroyed.
    const char* afterHost = "-";
    if (!ran) {
      *handle.doorbell = *handle.wptr;
      const double t1 = NowMs();
      while (*flag != value && NowMs() - t1 < kWaitMs) {
      }
      afterHost = (*flag == value) ? "COMPLETES" : "still stuck";
    }

    bool copied = false;
    if (*flag == value) {
      std::vector<uint8_t> got(kCopyBytes);
      CHECK_HIP_ERROR(hipMemcpy(got.data(), dstBuf, kCopyBytes, hipMemcpyDeviceToHost));
      copied = got.front() == 0x5a && got.back() == 0x5a;
    }

    printf("%-6s 0x%-16lx %-7lu %-10s %s%s\n", label, (unsigned long)db, (unsigned long)(db % 32),
           ran ? "OK" : "STUCK", afterHost, (*flag == value && !copied) ? "  (BAD DATA)" : "");

    // With the fix on every queue must run, aligned or not: past an engine's
    // first 32B boundary the alignment of the doorbell no longer matters.
    if ((fixOn || aligned) && !ran) ++failures;
    if (*flag == value && !copied) ++failures;
  };

  // All kept alive to the end: the second queue of an engine is the interesting
  // one (the burn must not repeat per queue), a destroyed queue hands its
  // doorbell slot straight back, and holding them lets `<odd engine> 4` leave
  // the engine full, which is what makes connect() below take its fallback.
  std::vector<std::unique_ptr<anvil::SdmaQueue>> queues;
  for (uint32_t e : engines) {
    const size_t first = queues.size();
    for (int q = 0; q < perEngine; ++q)
      queues.push_back(std::make_unique<anvil::SdmaQueue>(node, e));
    for (size_t i = first; i < queues.size(); ++i)
      testQueue(std::to_string(e).c_str(), *queues[i], dst);
  }

  // The real path: connect() picks the engines, creates the channels and hands
  // the burned doorbell slots back. The channels must still run after that.
  int peerCount = 0;
  CHECK_HIP_ERROR(hipGetDeviceCount(&peerCount));
  if (peerCount > 1) {
    const int peer = (device + 1) % peerCount;
    CHECK_HIP_ERROR(hipDeviceEnablePeerAccess(peer, 0));
    void* peerDst = nullptr;
    CHECK_HIP_ERROR(hipSetDevice(peer));
    CHECK_HIP_ERROR(hipMalloc(&peerDst, kCopyBytes));
    CHECK_HIP_ERROR(hipSetDevice(device));

    const int srcNode = anvil::AnvilLib::nodeForHipDevice(device);
    const int dstNode = anvil::AnvilLib::nodeForHipDevice(peer);
    anvil::anvil.connect(srcNode, dstNode, kConnectChannels);
    printf("connect(): device %d -> %d, %d channels\n", device, peer, kConnectChannels);
    for (int c = 0; c < kConnectChannels; ++c) {
      anvil::SdmaQueue* q = anvil::anvil.getSdmaQueue(srcNode, dstNode, c);
      if (q == nullptr) {
        printf("ch%-4d (no queue)\n", c);
        ++failures;
        continue;
      }
      testQueue(("ch" + std::to_string(c)).c_str(), *q, peerDst);
    }
  }

  printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}
