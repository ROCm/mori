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

// ccoMemImport of a pointer inside its HIP VMM allocation, which is what torch
// 2.14's symm_mem.empty() hands out (its data starts 0x2400 bytes in). Checks:
//   - the alias keeps the offset, the window's winBase includes it, and every
//     LSA peer's data reads back through ccoGetPeerPtr, not the bytes before it;
//   - writes through the alias land in the external buffer;
//   - a FULL DevComm, which needs an iova-0 RDMA MR on every window, fails up
//     front for a window off a page boundary;
//   - ccoMemFree hands the slot back, and ccoCommDestroy unmaps an import
//     nobody freed;
//   - ranks importing at different offsets all fail ccoWindowRegister.
//
// Run: ./test_mem_import [nranks]    (fork mode, one GPU per rank)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t kTorchOffset = 0x2400;  // where torch 2.14 puts symm_mem data
static const size_t kElems = 4096;
static const size_t kBytes = kElems * sizeof(uint32_t);
static const int kHeadByte = 0xA5;  // fills the bytes before the data

static int g_fails = 0;

static void Check(bool ok, const char* what) {
  if (ok) return;
  std::fprintf(stderr, "[rank %d] FAILED: %s\n", g_rank, what);
  g_fails++;
}

static uint32_t Pattern(int rank, size_t i) {
  return static_cast<uint32_t>(rank) * 1000003u + static_cast<uint32_t>(i);
}

// One HIP VMM allocation with the data `offset` bytes in, made the way torch makes one.
struct ExternalBuffer {
  hipMemGenericAllocationHandle_t handle{};
  void* base = nullptr;
  size_t allocSize = 0;
  char* data = nullptr;
};

static ExternalBuffer CreateExternal(const ccoComm* comm, int dev, size_t offset, size_t bytes) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = static_cast<hipMemAllocationHandleType>(comm->handleType);
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
  // ccoMemImport maps whole granules of the comm's granularity.
  gran = std::max(gran, comm->vmmGranularity);

  ExternalBuffer b;
  b.allocSize = (offset + bytes + gran - 1) / gran * gran;
  HIP_CHECK(hipMemCreate(&b.handle, b.allocSize, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&b.base, b.allocSize, gran, nullptr, 0));
  HIP_CHECK(hipMemMap(b.base, b.allocSize, 0, b.handle, 0));
  hipMemAccessDesc access = {};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(b.base, b.allocSize, &access, 1));
  b.data = static_cast<char*>(b.base) + offset;
  return b;
}

static void DestroyExternal(ExternalBuffer& b) {
  HIP_CHECK(hipMemUnmap(b.base, b.allocSize));
  HIP_CHECK(hipMemAddressFree(b.base, b.allocSize));
  HIP_CHECK(hipMemRelease(b.handle));
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  const int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    std::fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }
  if (comm->lsaSize != comm->worldSize) {
    // Across nodes every window carries an MR, which this offset rules out.
    if (rank == 0) std::printf("needs a single-node comm: SKIP\n");
    ccoCommDestroy(comm);
    return 0;
  }

  ExternalBuffer ext = CreateExternal(comm, dev, kTorchOffset, kBytes);
  HIP_CHECK(hipMemset(ext.base, kHeadByte, kTorchOffset));
  std::vector<uint32_t> mine(kElems);
  for (size_t i = 0; i < kElems; i++) mine[i] = Pattern(rank, i);
  HIP_CHECK(hipMemcpy(ext.data, mine.data(), kBytes, hipMemcpyHostToDevice));

  // ── Import + register at the torch offset ──
  ccoWindow_t win = nullptr;
  void* local = nullptr;
  if (ccoWindowRegister(comm, ext.data, kBytes, &win, &local) != 0) {
    std::fprintf(stderr, "[rank %d] WindowRegister of an import %#zx bytes in failed\n", rank,
                 kTorchOffset);
    ccoCommDestroy(comm);
    DestroyExternal(ext);
    return 1;
  }
  // Slots are granularity-aligned, so the alias sits as far into its slot as the data does.
  Check(reinterpret_cast<uintptr_t>(local) % comm->vmmGranularity == kTorchOffset,
        "the alias lost the offset into the allocation");

  ccoWindowDevice wd;
  HIP_CHECK(hipMemcpy(&wd, win, sizeof(wd), hipMemcpyDeviceToHost));
  auto viaWindow = [&](int lsa) {
    return static_cast<void*>(wd.winBase + (static_cast<uint64_t>(lsa) * wd.stride4G << 32));
  };
  Check(viaWindow(wd.lsaRank) == local, "winBase + own slice is not the alias");

  std::vector<uint32_t> got(kElems);
  HIP_CHECK(hipMemcpy(got.data(), local, kBytes, hipMemcpyDeviceToHost));
  Check(got == mine, "the alias does not read the external data");
  const uint32_t probe = 0xC0FFEEu + static_cast<uint32_t>(rank);
  uint32_t seen = 0;
  HIP_CHECK(hipMemcpy(local, &probe, sizeof(probe), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(&seen, ext.data, sizeof(seen), hipMemcpyDeviceToHost));
  Check(seen == probe, "a write through the alias is not in the external buffer");
  HIP_CHECK(hipMemcpy(ext.data, mine.data(), sizeof(uint32_t), hipMemcpyHostToDevice));

  ccoBarrierAll(comm);
  for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
    const int pe = comm->myNodeStart + lsa;
    void* peer = ccoGetPeerPtr(comm, local, pe);
    Check(peer == viaWindow(lsa), "ccoGetPeerPtr and the window disagree");
    HIP_CHECK(hipMemcpy(got.data(), peer, kBytes, hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < kElems; i++) bad += got[i] != Pattern(pe, i);
    if (bad != 0) {
      std::fprintf(stderr,
                   "[rank %d] FAILED: peer %d: %zu of %zu words wrong (got %#x, want %#x)\n", rank,
                   pe, bad, kElems, got[0], Pattern(pe, 0));
      g_fails++;
    }
  }
  ccoBarrierAll(comm);  // every rank is done reading before any unmaps

  // ── A FULL DevComm needs an iova-0 MR on every window ──
  if (comm->handleType == static_cast<int>(hipMemHandleTypePosixFileDescriptor)) {
    ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
    ccoDevComm dc{};
    const int rc = ccoDevCommCreate(comm, &reqs, &dc);
    Check(rc != 0, "a FULL DevComm took a window off a page boundary");
    if (rc == 0) ccoDevCommDestroy(comm, &dc);
  }

  ccoWindowDeregister(comm, win);
  Check(ccoMemFree(comm, local) == 0, "ccoMemFree of the import failed");

  // ── The slot came back: importing again gets the same alias ──
  void* again = nullptr;
  Check(ccoMemImport(comm, ext.data, kBytes, &again) == 0 && again == local,
        "ccoMemFree did not hand the import's slot back");
  if (again != nullptr) ccoMemFree(comm, again);

  // ── Different offsets on different ranks fail everywhere, without hanging ──
  if (nranks > 1) {
    void* skewed = ext.data + (rank == 0 ? 0 : 4096);
    ccoWindow_t skewedWin = nullptr;
    void* skewedLocal = nullptr;
    const int rc = ccoWindowRegister(comm, skewed, kBytes, &skewedWin, &skewedLocal);
    Check(rc != 0, "WindowRegister took imports at different offsets");
    if (rc == 0) {
      ccoWindowDeregister(comm, skewedWin);
      ccoMemFree(comm, skewedLocal);
    }
  }

  // ── ccoCommDestroy unmaps an import nobody freed; freeing the flat VA fails otherwise ──
  void* leaked = nullptr;
  Check(ccoMemImport(comm, ext.data, kBytes, &leaked) == 0, "import before teardown failed");
  ccoCommDestroy(comm);

  HIP_CHECK(hipMemcpy(got.data(), ext.data, kBytes, hipMemcpyDeviceToHost));
  Check(got == mine, "the external data changed");
  DestroyExternal(ext);

  std::printf("[rank %d] %s\n", rank, g_fails == 0 ? "PASSED" : "FAILED");
  return g_fails != 0 ? 1 : 0;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO MemImport at an offset", "/tmp/cco_mem_import_uid", 19883);
}
