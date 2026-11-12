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

// test: ccoMemImport of a pointer inside an external VMM allocation (torch >= 2.14 symm_mem puts
// a 0x2400-byte signal pad before the data).
//
// For each data offset (0 and 0x2400), every rank builds an external allocation, fills the data
// with a rank pattern and the bytes before it with a canary, then imports and registers it:
//   1. the local alias reads the data;
//   2. LSA: every peer's alias (ccoGetPeerPtr) reads that peer's data;
//   3. GDA: an all-to-all put between two imported windows lands at the data, canary intact. The
//      send window's MR is back-filled by ccoDevCommCreate, the recv window's is registered after.
// Then rank 0 imports at a different offset than the others: register must fail on every rank.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "mori/utils/hip_compat.hpp"

static const size_t PER_RANK_VMM_SIZE = 1ULL << 32;
static const size_t COUNT = 1024;  // floats per rank-pair
static const uint8_t CANARY = 0x5a;

template <mori::core::ProviderType PrvdType>
__global__ void GdaAlltoAllKernel(mori::cco::ccoWindow_t sendWin, mori::cco::ccoWindow_t recvWin,
                                  size_t perPairBytes, mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  for (int r = threadIdx.x; r < nRanks; r += blockDim.x) {
    if (r == myRank) continue;
    gda.put(r, recvWin, myRank * perPairBytes, sendWin, r * perPairBytes, perPairBytes,
            ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
  }
  gda.flush(ccoCoopBlock{});
  if (threadIdx.x < nRanks && threadIdx.x != myRank) {
    gda.waitSignal(static_cast<ccoGdaSignal_t>(threadIdx.x), 1);
  }
}

// An external VMM allocation mapped at its own VA, rounded to the recommended granularity like
// torch's symm_mem allocator.
struct ExtBuf {
  hipMemGenericAllocationHandle_t handle{};
  void* base{nullptr};
  size_t size{0};
};

static hipMemAllocationProp ExtProp(int dev, hipMemAllocationHandleType type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = type;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  prop.allocFlags.gpuDirectRDMACapable = 1;
  return prop;
}

// The comm uses fabric handles when they export (same probe as ccoCommCreate), so must we.
static hipMemAllocationHandleType ExtHandleType(int dev) {
  hipMemAllocationProp prop = ExtProp(dev, hipMemHandleTypeFabricCompat);
  size_t gran = 0;
  hipMemGenericAllocationHandle_t h = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended) ==
          hipSuccess &&
      gran > 0 && hipMemCreate(&h, gran, &prop, 0) == hipSuccess) {
    hipMemFabricHandle_compat_t fh;
    hipError_t err = hipMemExportToShareableHandle(&fh, h, hipMemHandleTypeFabricCompat, 0);
    (void)hipMemRelease(h);
    if (err == hipSuccess) return hipMemHandleTypeFabricCompat;
  }
  (void)hipGetLastError();
  return hipMemHandleTypePosixFileDescriptor;
}

static int ExtAlloc(int dev, size_t bytes, ExtBuf* b) {
  static const hipMemAllocationHandleType type = ExtHandleType(dev);
  hipMemAllocationProp prop = ExtProp(dev, type);
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
  b->size = (bytes + gran - 1) / gran * gran;
  HIP_CHECK(hipMemCreate(&b->handle, b->size, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&b->base, b->size, gran, nullptr, 0));
  HIP_CHECK(hipMemMap(b->base, b->size, 0, b->handle, 0));
  hipMemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(b->base, b->size, &acc, 1));
  return 0;
}

static void ExtFree(ExtBuf* b) {
  (void)hipMemUnmap(b->base, b->size);
  (void)hipMemAddressFree(b->base, b->size);
  (void)hipMemRelease(b->handle);
  *b = ExtBuf{};
}

static bool CheckFloats(const char* what, const std::vector<float>& got,
                        const std::vector<float>& want, int skipSlot = -1) {
  for (size_t i = 0; i < want.size(); i++) {
    if (skipSlot >= 0 && static_cast<int>(i / COUNT) == skipSlot) continue;
    if (got[i] != want[i]) {
      fprintf(stderr, "[rank %d] %s mismatch at %zu: got %.0f expected %.0f\n", g_rank, what, i,
              got[i], want[i]);
      return false;
    }
  }
  return true;
}

static bool CheckCanary(const char* what, const ExtBuf& b, size_t offset) {
  if (offset == 0) return true;
  std::vector<uint8_t> pad(offset);
  HIP_CHECK(hipMemcpy(pad.data(), b.base, offset, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < offset; i++) {
    if (pad[i] != CANARY) {
      fprintf(stderr, "[rank %d] %s canary clobbered at byte %zu: 0x%02x\n", g_rank, what, i,
              pad[i]);
      return false;
    }
  }
  return true;
}

static float SendVal(int src, int dst, size_t i) { return src * 100000.0f + dst * 1000.0f + i; }

static bool RunOffset(mori::cco::ccoComm* comm, int dev, int rank, int nranks, size_t offset) {
  using namespace mori::cco;
  const size_t bufBytes = COUNT * nranks * sizeof(float);
  bool ok = true;

  ExtBuf sendExt, recvExt;
  ExtAlloc(dev, offset + bufBytes, &sendExt);
  ExtAlloc(dev, offset + bufBytes, &recvExt);
  char* sendData = static_cast<char*>(sendExt.base) + offset;
  char* recvData = static_cast<char*>(recvExt.base) + offset;

  std::vector<float> hostSend(COUNT * nranks);
  for (int r = 0; r < nranks; r++)
    for (size_t i = 0; i < COUNT; i++) hostSend[r * COUNT + i] = SendVal(rank, r, i);
  HIP_CHECK(hipMemset(sendExt.base, CANARY, sendExt.size));
  HIP_CHECK(hipMemset(recvExt.base, CANARY, recvExt.size));
  HIP_CHECK(hipMemcpy(sendData, hostSend.data(), bufBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(recvData, 0xff, bufBytes));

  ccoWindow_t sendWin = nullptr;
  void* sendAlias = nullptr;
  if (ccoWindowRegister(comm, sendData, bufBytes, &sendWin, &sendAlias) != 0) {
    fprintf(stderr, "[rank %d] offset 0x%zx: import+register send failed\n", rank, offset);
    return false;
  }

  // 1. local alias
  std::vector<float> got(COUNT * nranks);
  HIP_CHECK(hipMemcpy(got.data(), sendAlias, bufBytes, hipMemcpyDeviceToHost));
  ok = CheckFloats("local alias", got, hostSend) && ok;

  // 2. LSA peer aliases
  ccoBarrierAll(comm);
  for (int pe = 0; pe < nranks && ok; pe++) {
    void* peerPtr = ccoGetPeerPtr(comm, sendAlias, pe);
    if (peerPtr == nullptr) continue;
    HIP_CHECK(hipMemcpy(got.data(), peerPtr, bufBytes, hipMemcpyDeviceToHost));
    std::vector<float> want(COUNT * nranks);
    for (int r = 0; r < nranks; r++)
      for (size_t i = 0; i < COUNT; i++) want[r * COUNT + i] = SendVal(pe, r, i);
    ok = CheckFloats("LSA peer read", got, want) && ok;
  }

  // 3. GDA all-to-all between imported windows (FULL needs intranode cross-rail RDMA)
  const bool skipGda = getenv("MORI_CCO_SKIP_GDA_FULL") != nullptr;
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.gdaCounterCount = 0;
  ccoDevComm devComm{};
  if (!skipGda && ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return false;
  }
  ccoWindow_t recvWin = nullptr;
  void* recvAlias = nullptr;
  if (ccoWindowRegister(comm, recvData, bufBytes, &recvWin, &recvAlias) != 0) {
    fprintf(stderr, "[rank %d] offset 0x%zx: import+register recv failed\n", rank, offset);
    return false;
  }

  if (skipGda) {
    printf("[rank %d] offset 0x%zx: GDA put SKIPPED (MORI_CCO_SKIP_GDA_FULL)\n", rank, offset);
  } else if (devComm.gdaConnType == CCO_GDA_CONNECTION_NONE) {
    printf("[rank %d] offset 0x%zx: no GDA connectivity, skipping the put\n", rank, offset);
  } else {
    ccoBarrierAll(comm);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    CCO_GDA_DISPATCH(GdaAlltoAllKernel<P>
                     <<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT * sizeof(float), devComm));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    ccoBarrierAll(comm);

    HIP_CHECK(hipMemcpy(got.data(), recvData, bufBytes, hipMemcpyDeviceToHost));
    std::vector<float> want(COUNT * nranks);
    for (int src = 0; src < nranks; src++)
      for (size_t i = 0; i < COUNT; i++) want[src * COUNT + i] = SendVal(src, rank, i);
    ok = CheckFloats("GDA recv", got, want, /*skipSlot=*/rank) && ok;
    ok = CheckCanary("GDA recv", recvExt, offset) && ok;
  }
  ok = CheckCanary("send", sendExt, offset) && ok;

  if (!skipGda) ccoDevCommDestroy(comm, &devComm);
  ccoWindowDeregister(comm, recvWin);
  ccoWindowDeregister(comm, sendWin);
  ccoMemFree(comm, recvAlias);
  ccoMemFree(comm, sendAlias);
  ExtFree(&recvExt);
  ExtFree(&sendExt);
  printf("[rank %d] offset 0x%zx: %s\n", rank, offset, ok ? "ok" : "FAILED");
  return ok;
}

// Rank 0 imports at another offset: every rank's register must fail, without hanging.
static bool RunMismatch(mori::cco::ccoComm* comm, int dev, int rank) {
  using namespace mori::cco;
  const size_t bytes = 1 << 20;
  const size_t offset = rank == 0 ? 0x2000 : 0x2400;
  ExtBuf ext;
  ExtAlloc(dev, offset + bytes, &ext);
  ccoWindow_t win = nullptr;
  void* alias = nullptr;
  int ret = ccoWindowRegister(comm, static_cast<char*>(ext.base) + offset, bytes, &win, &alias);
  bool ok = (ret != 0);
  if (!ok) {
    fprintf(stderr, "[rank %d] mismatched offsets registered\n", rank);
    ccoWindowDeregister(comm, win);
    ccoMemFree(comm, alias);
  }
  ccoBarrierAll(comm);
  ExtFree(&ext);
  printf("[rank %d] mismatched offsets: %s\n", rank, ok ? "rejected" : "FAILED");
  return ok;
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  bool ok = true;
  // 0x2400 first: only the first FULL DevComm back-fills MRs of already registered windows.
  ok = RunOffset(comm, dev, rank, nranks, 0x2400) && ok;
  ok = RunOffset(comm, dev, rank, nranks, 0) && ok;
  ok = RunMismatch(comm, dev, rank) && ok;

  mori::cco::ccoCommDestroy(comm);
  printf("[rank %d] %s\n", rank, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO mem import offset", "/tmp/cco_mem_import_offset_uid", 0);
}
