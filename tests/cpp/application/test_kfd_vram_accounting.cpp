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

// How the amdgpu driver counts VRAM that a NIC pins (ROCm/mori#721, known-issues
// Issue 4), and whether the HIP GEM allocator keeps registration from costing VRAM.
//
// KfdVramAccounting registers an S-byte buffer and measures what changes:
//   free  - hipMemGetInfo (KFD's free VRAM)
//   limit - the largest allocation the driver still accepts (the KFD allocation limit)
// On a correct driver registration changes neither. Expected on hipMalloc buffers:
//
//                     dma-buf free/limit   peer memory free/limit
//   amdgpu-dkms 30.x      -S / -S               0 / 0
//   31.10 - 31.60         -S / -S               0 / -S
//   fixed                  0 / 0                0 / 0
//
// Those two cases only report unless MORI_KFD_EXPECT=30x|31x|fixed asserts a verdict.
// HipGemDmabuf asserts on every driver: the GEM buffer is charged once, when it is
// allocated, and registering it changes nothing.
//
// Needs a GPU and an RDMA device that accepts dma-buf MRs (MORI_RDMA_DEVICES picks
// it). MORI_KFD_TEST_SIZE_GIB sets S (default 8). Measuring the limit briefly fills
// the GPU, so run it on an idle one (HIP_VISIBLE_DEVICES).
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "infiniband/verbs.h"
#include "mori/application/memory/hip_gem_allocator.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

using namespace mori::application;

namespace {

constexpr size_t kGiB = size_t{1} << 30;
constexpr size_t kMiB = size_t{1} << 20;
// The limit is searched to this granularity.
constexpr size_t kGranule = 256 * kMiB;

size_t TestSize() {
  const char* env = getenv("MORI_KFD_TEST_SIZE_GIB");
  long gib = env ? strtol(env, nullptr, 10) : 8;
  return static_cast<size_t>(gib > 0 ? gib : 8) * kGiB;
}

size_t FreeVram() {
  size_t free = 0, total = 0;
  EXPECT_EQ(hipMemGetInfo(&free, &total), hipSuccess);
  return free;
}

size_t TotalVram() {
  size_t free = 0, total = 0;
  EXPECT_EQ(hipMemGetInfo(&free, &total), hipSuccess);
  return total;
}

// Physical free VRAM of device 0, from the amdgpu sysfs counters.
int64_t PhysicalFree() {
  char bus[64] = {};
  if (hipDeviceGetPCIBusId(bus, sizeof(bus), 0) != hipSuccess) return -1;
  for (char* c = bus; *c; ++c) {
    if (*c >= 'A' && *c <= 'F') *c = static_cast<char>(*c - 'A' + 'a');
  }
  auto read = [&](const char* name) -> int64_t {
    std::string path = std::string("/sys/bus/pci/devices/") + bus + "/" + name;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return -1;
    long long v = -1;
    if (fscanf(f, "%lld", &v) != 1) v = -1;
    fclose(f);
    return v;
  };
  int64_t total = read("mem_info_vram_total"), used = read("mem_info_vram_used");
  return total < 0 || used < 0 ? -1 : total - used;
}

bool Fits(size_t bytes) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  EXPECT_EQ(hipFree(p), hipSuccess);
  return true;
}

// Physical free VRAM once freed BOs have gone: a freed BO that is still busy (e.g. being
// cleared) is destroyed a little later, so the counter lags a large hipFree.
int64_t PhysicalFreeSettled() {
  int64_t last = PhysicalFree();
  for (int i = 0; i < 50; ++i) {
    usleep(100 * 1000);
    int64_t now = PhysicalFree();
    if (now == last) return now;
    last = now;
  }
  return last;
}

// The largest single hipMalloc the driver accepts, to kGranule: KFD checks every
// allocation against its limit, so this is what any later allocation gets. It never
// exceeds what hipMemGetInfo reports, and is at most two double charges below it.
size_t MaxAllocatable(size_t s) {
  const size_t free = FreeVram();
  size_t hi = free + kGiB;
  size_t lo = free > 2 * s + 3 * kGiB ? (free - 2 * s - 3 * kGiB) & ~(kGranule - 1) : 0;
  if (lo > 0 && !Fits(lo)) lo = 0;
  if (Fits(hi)) hi = TotalVram() + kGranule;
  while (hi - lo > kGranule) {
    size_t mid = (lo + (hi - lo) / 2) & ~(kGranule - 1);
    if (mid <= lo) mid = lo + kGranule;
    if (Fits(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

struct Snapshot {
  int64_t free;   // hipMemGetInfo
  int64_t limit;  // largest accepted allocation
};

Snapshot Measure(size_t s) {
  return {static_cast<int64_t>(FreeVram()), static_cast<int64_t>(MaxAllocatable(s))};
}

double Gib(int64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kGiB); }

// "0", "-S" or "?" for a change, to S/4.
std::string Steps(int64_t delta, size_t s) {
  const int64_t q = static_cast<int64_t>(s / 4);
  if (delta > -q && delta < q) return "0";
  if (delta > -static_cast<int64_t>(s) - q && delta < -static_cast<int64_t>(s) + q) return "-S";
  return "?";
}

struct Observed {
  bool measured{false};
  int64_t dFree{0};
  int64_t dLimit{0};
};
Observed gDmabuf, gPeer;

void Report(const char* what, const Snapshot& before, const Snapshot& after, size_t s,
            Observed* out) {
  out->measured = true;
  out->dFree = after.free - before.free;
  out->dLimit = after.limit - before.limit;
  printf("[ KFD      ] %-28s S=%.1f GiB: free %+.2f GiB (%s), limit %+.2f GiB (%s)\n", what,
         Gib(static_cast<int64_t>(s)), Gib(out->dFree), Steps(out->dFree, s).c_str(),
         Gib(out->dLimit), Steps(out->dLimit, s).c_str());
}

std::string AmdgpuVersion() {
  FILE* f = fopen("/sys/module/amdgpu/version", "r");
  if (!f) return "in-tree";
  char buf[128] = {};
  if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
  fclose(f);
  std::string v(buf);
  while (!v.empty() && (v.back() == '\n' || v.back() == ' ')) v.pop_back();
  return v;
}

class RdmaFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    context = std::make_unique<RdmaContext>(RdmaBackendType::IBVerbs);
    for (const ActiveDevicePort& port : GetActiveDevicePortList(context->GetRdmaDeviceList())) {
      if (devices.empty() || devices.back() != port.first) devices.push_back(port.first);
    }
    if (devices.empty()) GTEST_SKIP() << "needs an active RDMA device";
    ctx.reset(devices[0]->CreateRdmaDeviceContext());
    ASSERT_NE(ctx, nullptr);
  }

  std::unique_ptr<RdmaContext> context;
  std::vector<RdmaDevice*> devices;
  std::unique_ptr<RdmaDeviceContext> ctx;
};

// ---------------------------------------------------------------------------------------
// The allocator itself.

class HipGemAllocatorTest : public RdmaFixture {};

TEST_F(HipGemAllocatorTest, AllocLookupFree) {
  const size_t want = 64 * kMiB + 1;
  void* p = mori_hip_gem_malloc(static_cast<ssize_t>(want), 0, nullptr);
  ASSERT_NE(p, nullptr);
  HipGemBuffer gem;
  ASSERT_TRUE(HipGemLookup(static_cast<char*>(p) + 3 * kMiB, &gem, /*wantFd=*/false));
  EXPECT_EQ(gem.base, p);
  EXPECT_GE(gem.size, want);
  EXPECT_EQ(gem.size % (2 * kMiB), 0u);
  EXPECT_EQ(gem.device, 0);
  EXPECT_EQ(gem.dmabufFd, -1);
  EXPECT_FALSE(HipGemLookup(static_cast<char*>(p) + gem.size, nullptr, false));
  EXPECT_FALSE(HipGemLookup(static_cast<char*>(p) - 1, nullptr, false));

  int fd = -1;
  void* base = nullptr;
  size_t size = 0;
  int device = -1;
  ASSERT_EQ(mori_hip_gem_lookup(static_cast<char*>(p) + 1, &base, &size, &fd, &device), 0);
  EXPECT_EQ(base, p);
  EXPECT_GE(fd, 0);
  close(fd);

  mori_hip_gem_free(p, static_cast<ssize_t>(want), 0, nullptr);
  EXPECT_FALSE(HipGemLookup(p, nullptr, false));
}

TEST_F(HipGemAllocatorTest, MemoryIsUsable) {
  const size_t n = 32 * kMiB;
  auto* gemBuf = static_cast<uint8_t*>(mori_hip_gem_malloc(static_cast<ssize_t>(n), 0, nullptr));
  ASSERT_NE(gemBuf, nullptr);
  void* plain = nullptr;
  ASSERT_EQ(hipMalloc(&plain, n), hipSuccess);

  std::vector<uint8_t> host(n), back(n);
  for (size_t i = 0; i < n; ++i) host[i] = static_cast<uint8_t>(i * 131 + 7);
  ASSERT_EQ(hipMemcpy(gemBuf, host.data(), n, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(plain, gemBuf, n, hipMemcpyDeviceToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(back.data(), plain, n, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(memcmp(host.data(), back.data(), n), 0);

  ASSERT_EQ(hipMemset(gemBuf + 4096, 0x5a, n - 8192), hipSuccess);
  ASSERT_EQ(hipMemcpy(back.data(), gemBuf, n, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(back[4095], host[4095]);
  EXPECT_EQ(back[4096], 0x5a);
  EXPECT_EQ(back[n - 4097], 0x5a);
  EXPECT_EQ(back[n - 4096], host[n - 4096]);

  EXPECT_EQ(hipFree(plain), hipSuccess);
  mori_hip_gem_free(gemBuf, static_cast<ssize_t>(n), 0, nullptr);
}

// HIP reports imported memory as device 0's whichever device imported it; the
// allocator's registry has the device the buffer lives on.
TEST_F(HipGemAllocatorTest, DeviceComesFromTheAllocator) {
  int count = 0;
  ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
  if (count < 2) GTEST_SKIP() << "needs two visible GPUs";
  void* p = mori_hip_gem_malloc(static_cast<ssize_t>(16 * kMiB), 1, nullptr);
  ASSERT_NE(p, nullptr);
  HipGemBuffer gem;
  ASSERT_TRUE(HipGemLookup(p, &gem, false));
  EXPECT_EQ(gem.device, 1);
  hipPointerAttribute_t attr{};
  if (hipPointerGetAttributes(&attr, p) == hipSuccess) {
    printf("[ INFO     ] hipPointerGetAttributes reports device %d for a GEM buffer on device 1\n",
           attr.device);
  }
  mori_hip_gem_free(p, static_cast<ssize_t>(16 * kMiB), 1, nullptr);
}

// mori's guard for foreign GEM buffers (IsImportedDeviceMemory in rdma.cpp) relies on
// HIP knowing no allocation range for imported memory.
TEST_F(HipGemAllocatorTest, HipKnowsNoRangeForImportedMemory) {
  void* p = mori_hip_gem_malloc(static_cast<ssize_t>(16 * kMiB), 0, nullptr);
  ASSERT_NE(p, nullptr);
  hipPointerAttribute_t attr{};
  ASSERT_EQ(hipPointerGetAttributes(&attr, p), hipSuccess);
  EXPECT_EQ(attr.type, hipMemoryTypeDevice);
  hipDeviceptr_t base = 0;
  size_t range = 0;
  hipError_t err = hipMemGetAddressRange(&base, &range, reinterpret_cast<hipDeviceptr_t>(p));
  (void)hipGetLastError();
  EXPECT_TRUE(err != hipSuccess || base == 0)
      << "HIP now reports a range for imported memory; IsImportedDeviceMemory no longer "
         "catches foreign GEM buffers";
  mori_hip_gem_free(p, static_cast<ssize_t>(16 * kMiB), 0, nullptr);
}

TEST_F(HipGemAllocatorTest, ExportsItsOwnDmabuf) {
  const size_t n = 64 * kMiB;
  auto* p = static_cast<char*>(mori_hip_gem_malloc(static_cast<ssize_t>(n), 0, nullptr));
  ASSERT_NE(p, nullptr);
  uint64_t offset = 0;
  int fd = TryExportDmabufFd(p + 8 * kMiB, 4 * kMiB, &offset);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(offset, 8 * kMiB);
  close(fd);
  EXPECT_LT(TryExportDmabufFd(p + n - kMiB, 2 * kMiB, &offset), 0) << "range past the buffer";
  mori_hip_gem_free(p, static_cast<ssize_t>(n), 0, nullptr);
}

TEST_F(HipGemAllocatorTest, RegistersThroughItsDmabuf) {
  const size_t n = 256 * kMiB;
  auto* p = static_cast<char*>(mori_hip_gem_malloc(static_cast<ssize_t>(n), 0, nullptr));
  ASSERT_NE(p, nullptr);
  // A miss here would send the buffer to ibv_reg_mr, which the amdgpu peer-memory
  // client cannot take for a GEM BO.
  ASSERT_TRUE(HipGemLookup(p, nullptr, false));
  RdmaMemoryRegion whole = ctx->RegisterRdmaMemoryRegionAuto(p, n);
  EXPECT_EQ(whole.addr, reinterpret_cast<uintptr_t>(p));
  EXPECT_EQ(whole.length, n);
  RdmaMemoryRegion part = ctx->RegisterRdmaMemoryRegion(p + 64 * kMiB, 32 * kMiB);
  EXPECT_EQ(part.addr, reinterpret_cast<uintptr_t>(p + 64 * kMiB));
  EXPECT_EQ(part.length, 32 * kMiB);
  EXPECT_NE(whole.rkey, part.rkey);
  ctx->DeregisterRdmaMemoryRegion(p + 64 * kMiB);
  ctx->DeregisterRdmaMemoryRegion(p);
  mori_hip_gem_free(p, static_cast<ssize_t>(n), 0, nullptr);
}

// ---------------------------------------------------------------------------------------
// Driver accounting.

class KfdVramAccounting : public RdmaFixture {
 protected:
  void SetUp() override {
    RdmaFixture::SetUp();
    if (IsSkipped()) return;
    s = TestSize();
    printf("[ KFD      ] amdgpu %s, %.1f GiB VRAM\n", AmdgpuVersion().c_str(),
           Gib(static_cast<int64_t>(TotalVram())));
  }
  size_t s{0};
};

TEST_F(KfdVramAccounting, HipMallocDmabuf) {
  void* p = nullptr;
  ASSERT_EQ(hipMalloc(&p, s), hipSuccess);
  Snapshot before = Measure(s);
  uint64_t offset = 0;
  int fd = TryExportDmabufFd(p, s, &offset);
  ASSERT_GE(fd, 0) << "no dma-buf export";
  ibv_mr* mr = ibv_reg_dmabuf_mr(ctx->GetIbvPd(), offset, s, reinterpret_cast<uint64_t>(p), fd,
                                 IBV_ACCESS_LOCAL_WRITE);
  close(fd);
  ASSERT_NE(mr, nullptr) << "dma-buf MR rejected: " << strerror(errno);
  Snapshot after = Measure(s);
  Report("hipMalloc + dma-buf MR", before, after, s, &gDmabuf);
  ibv_dereg_mr(mr);
  EXPECT_EQ(hipFree(p), hipSuccess);
}

TEST_F(KfdVramAccounting, HipMallocPeerMemory) {
  void* p = nullptr;
  ASSERT_EQ(hipMalloc(&p, s), hipSuccess);
  Snapshot before = Measure(s);
  ibv_mr* mr = ibv_reg_mr(ctx->GetIbvPd(), p, s, IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr) {
    EXPECT_EQ(hipFree(p), hipSuccess);
    GTEST_SKIP() << "no peer-memory support for VRAM (ibv_reg_mr: " << strerror(errno) << ")";
  }
  Snapshot after = Measure(s);
  Report("hipMalloc + peer-memory MR", before, after, s, &gPeer);
  ibv_dereg_mr(mr);
  EXPECT_EQ(hipFree(p), hipSuccess);
}

TEST_F(KfdVramAccounting, HipGemDmabuf) {
  const int64_t physEmpty = PhysicalFreeSettled();
  Snapshot empty = Measure(s);
  void* p = mori_hip_gem_malloc(static_cast<ssize_t>(s), 0, nullptr);
  ASSERT_NE(p, nullptr);
  ASSERT_TRUE(HipGemLookup(p, nullptr, false));
  Snapshot allocated = Measure(s);
  Observed atAlloc, atReg;
  Report("HIP GEM allocation", empty, allocated, s, &atAlloc);

  std::vector<std::unique_ptr<RdmaDeviceContext>> more;
  ctx->RegisterRdmaMemoryRegionAuto(p, s);
  if (devices.size() > 1) {
    more.emplace_back(devices[1]->CreateRdmaDeviceContext());
    more.back()->RegisterRdmaMemoryRegionAuto(p, s);
  }
  Snapshot registered = Measure(s);
  Report(devices.size() > 1 ? "HIP GEM + dma-buf MR x2" : "HIP GEM + dma-buf MR", allocated,
         registered, s, &atReg);

  // Charged once, when allocated: hipMemGetInfo and the limit both drop by S...
  EXPECT_EQ(Steps(atAlloc.dFree, s), "-S") << "hipMemGetInfo should cover the pinned BO";
  EXPECT_EQ(Steps(atAlloc.dLimit, s), "-S");
  // ...and registering it changes neither.
  EXPECT_EQ(Steps(atReg.dFree, s), "0");
  EXPECT_EQ(Steps(atReg.dLimit, s), "0") << "registration still costs VRAM";

  for (auto& c : more) c->DeregisterRdmaMemoryRegion(p);
  ctx->DeregisterRdmaMemoryRegion(p);
  mori_hip_gem_free(p, static_cast<ssize_t>(s), 0, nullptr);
  const int64_t physFreed = PhysicalFreeSettled();
  Snapshot freed = Measure(s);
  EXPECT_EQ(Steps(freed.free - empty.free, s), "0") << "the pin outlived the free";
  int hipVersion = 0;
  (void)hipRuntimeGetVersion(&hipVersion);
  const std::string returned = Steps(physFreed - physEmpty, s);
  if (hipVersion / 10000000 < 10) {
    // HIP before ROCm 10 keeps memory imported with hipImportExternalMemory mapped until
    // the process exits, whatever is released; ROCm 10.0 gives it back on free.
    printf(
        "[ KFD      ] HIP %d: freed GEM buffer physically returned: %s (expected only from "
        "ROCm 10)\n",
        hipVersion, returned == "0" ? "yes" : "no");
    return;
  }
  EXPECT_EQ(returned, "0") << "the BO outlived its free";
  EXPECT_EQ(Steps(freed.limit - empty.limit, s), "0") << "freeing should return the VRAM";
}

TEST_F(KfdVramAccounting, DriverVerdict) {
  if (!gDmabuf.measured) GTEST_SKIP() << "HipMallocDmabuf did not run";
  const std::string dma = Steps(gDmabuf.dLimit, s);
  const std::string peer = gPeer.measured ? Steps(gPeer.dLimit, s) : "n/a";
  std::string verdict = "unknown";
  if (dma == "0" && (peer == "0" || peer == "n/a")) verdict = "fixed";
  if (dma == "-S" && peer == "0") verdict = "30x";
  if (dma == "-S" && peer == "-S") verdict = "31x";
  if (dma == "-S" && peer == "n/a") verdict = "30x-or-31x (no peer memory to tell)";
  printf("[ KFD      ] amdgpu %s: dma-buf limit %s, peer-memory limit %s -> %s\n",
         AmdgpuVersion().c_str(), dma.c_str(), peer.c_str(), verdict.c_str());
  if (const char* want = getenv("MORI_KFD_EXPECT")) {
    EXPECT_EQ(verdict, std::string(want));
  }
}

}  // namespace
