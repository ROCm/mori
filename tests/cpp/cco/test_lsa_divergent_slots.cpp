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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Regression for collective window identity being confused with the local VMM
// slot offset. Every rank deliberately occupies a different amount of its local
// flat slice, then collectively imports/registers one logical external window.
// The FD rendezvous must use the collective sequence while each process maps all
// peer allocations at its own (different) local offset.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"

#define CCO_MUST(expr)                                                                       \
  do {                                                                                       \
    if (!(expr)) {                                                                           \
      std::fprintf(stderr, "[rank %d] CHECK FAILED: %s at %s:%d\n", g_rank, #expr, __FILE__, \
                   __LINE__);                                                                \
      std::fflush(stderr);                                                                   \
      _exit(1);                                                                              \
    }                                                                                        \
  } while (0)

namespace {

constexpr size_t kPerRankVmmSize = 256ULL << 20;
constexpr size_t kPayloadBytes = 1ULL << 20;
constexpr size_t kBlockerQuantum = 16ULL << 20;

size_t alignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

struct ExternalAllocation {
  void* ptr{nullptr};
  size_t size{0};
  hipMemGenericAllocationHandle_t handle{nullptr};
};

ExternalAllocation createExternalFdAllocation(int device) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t externalGranularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&externalGranularity, &prop,
                                           hipMemAllocationGranularityRecommended));

  ExternalAllocation allocation;
  allocation.size = alignUp(kPayloadBytes, externalGranularity);
  HIP_CHECK(hipMemCreate(&allocation.handle, allocation.size, &prop, 0));
  HIP_CHECK(
      hipMemAddressReserve(&allocation.ptr, allocation.size, externalGranularity, nullptr, 0));
  HIP_CHECK(hipMemMap(allocation.ptr, allocation.size, 0, allocation.handle, 0));
  // The mapping owns the allocation lifetime now. Match normal VMM ownership:
  // release the creation handle and let ccoMemImport retain its own handle.
  HIP_CHECK(hipMemRelease(allocation.handle));
  allocation.handle = nullptr;

  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(allocation.ptr, allocation.size, &access, 1));
  return allocation;
}

void destroyExternalAllocation(ExternalAllocation* allocation) {
  if (allocation->ptr != nullptr) {
    HIP_CHECK(hipMemUnmap(allocation->ptr, allocation->size));
  }
  if (allocation->handle != nullptr) {
    HIP_CHECK(hipMemRelease(allocation->handle));
    allocation->handle = nullptr;
  }
  if (allocation->ptr != nullptr) {
    HIP_CHECK(hipMemAddressFree(allocation->ptr, allocation->size));
    allocation->ptr = nullptr;
  }
}

uint8_t payloadByte(int worldRank, size_t index) {
  return static_cast<uint8_t>(17 + worldRank * 13 + index * 3);
}

}  // namespace

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  using namespace mori::cco;
  g_rank = rank;

  // This regression specifically targets LocalBootstrap + SCM_RIGHTS. Without
  // the sequence fix, different slot offsets produce different socket paths and
  // the collective registration hangs.
  CCO_MUST(setenv("MORI_CCO_FORCE_FD", "1", 1) == 0);

  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  CCO_MUST(deviceCount > 0);
  const int device = rank % deviceCount;
  HIP_CHECK(hipSetDevice(device));

  ccoComm* comm = nullptr;
  CCO_MUST(ccoCommCreateLsaOnly(uid, nranks, rank, kPerRankVmmSize, &comm) == 0);
  CCO_MUST(comm != nullptr);

  ccoCommInfo info = CCO_COMM_INFO_INITIALIZER;
  CCO_MUST(ccoCommGetInfo(comm, &info) == 0);
  CCO_MUST(info.rank == rank);
  CCO_MUST(info.worldSize == nranks);
  CCO_MUST(info.lsaSize > 0);
  CCO_MUST(info.lsaRank >= 0 && info.lsaRank < info.lsaSize);
  CCO_MUST(info.lsaStart + info.lsaRank == rank);
  CCO_MUST(info.perRankSize >= kPerRankVmmSize);

  if (info.lsaSize < 2) {
    if (rank == 0) {
      std::printf("SKIP: divergent-slot LSA regression needs at least two ranks per LSA team\n");
    }
    CCO_MUST(ccoCommDestroy(comm) == 0);
    return 0;
  }

  ExternalAllocation external = createExternalFdAllocation(device);

  // Occupy a different local prefix on every LSA rank. This is intentionally
  // non-collective and does not consume a window-registration sequence.
  const size_t blockerSize = static_cast<size_t>(info.lsaRank + 1) * kBlockerQuantum;
  void* blocker = nullptr;
  CCO_MUST(ccoMemAlloc(comm, blockerSize, &blocker) == 0);

  std::vector<uint8_t> payload(256);
  for (size_t i = 0; i < payload.size(); i++) payload[i] = payloadByte(rank, i);
  HIP_CHECK(hipMemcpy(external.ptr, payload.data(), payload.size(), hipMemcpyHostToDevice));

  ccoWindowRegisterOptions options = CCO_WINDOW_REGISTER_OPTIONS_INITIALIZER;
  options.flags = CCO_WINDOW_REGISTER_LSA_ONLY;
  ccoWindow_t window = nullptr;
  void* localAlias = nullptr;
  CCO_MUST(ccoWindowRegisterExternal(comm, external.ptr, external.size, &options, &window,
                                     &localAlias) == 0);

  const size_t localGap = static_cast<size_t>(reinterpret_cast<uintptr_t>(localAlias) -
                                              reinterpret_cast<uintptr_t>(blocker));
  CCO_MUST(localGap == blockerSize);
  std::printf("[rank %d] lsaRank=%d localSlotOffset=%zu\n", rank, info.lsaRank, localGap);

  ccoWindowDevice windowHost = {};
  HIP_CHECK(hipMemcpy(&windowHost, window, sizeof(windowHost), hipMemcpyDeviceToHost));
  CCO_MUST(windowHost.lsaRank == info.lsaRank);
  CCO_MUST(localAlias == windowHost.winBase + static_cast<size_t>(info.lsaRank) * info.perRankSize);
  CCO_MUST(windowHost.ibgdaWin.lkey == 0);

  std::vector<uint32_t> peerRkeys(info.worldSize, 1);
  HIP_CHECK(hipMemcpy(peerRkeys.data(), windowHost.ibgdaWin.peerRkeys,
                      peerRkeys.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  CCO_MUST(
      std::all_of(peerRkeys.begin(), peerRkeys.end(), [](uint32_t rkey) { return rkey == 0; }));

  CCO_MUST(ccoBarrierAll(comm) == 0);

  for (int lsa = 0; lsa < info.lsaSize; lsa++) {
    const int pe = info.lsaStart + lsa;
    void* peerPtr = ccoGetPeerPtr(comm, localAlias, pe);
    void* windowPeerPtr = windowHost.winBase + static_cast<size_t>(lsa) * info.perRankSize;
    CCO_MUST(peerPtr == windowPeerPtr);

    std::vector<uint8_t> got(payload.size());
    HIP_CHECK(hipMemcpy(got.data(), peerPtr, got.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); i++) {
      CCO_MUST(got[i] == payloadByte(pe, i));
    }
  }

  for (int pe = 0; pe < info.worldSize; pe++) {
    const bool inLsa = pe >= info.lsaStart && pe < info.lsaStart + info.lsaSize;
    if (!inLsa) CCO_MUST(ccoGetPeerPtr(comm, localAlias, pe) == nullptr);
  }

  HIP_CHECK(hipDeviceSynchronize());
  CCO_MUST(ccoBarrierAll(comm) == 0);

  auto releaseLocalState = [&]() {
    CCO_MUST(ccoWindowDeregister(comm, window) == 0);
    window = nullptr;
    CCO_MUST(ccoMemFree(comm, localAlias) == 0);
    localAlias = nullptr;
    CCO_MUST(ccoMemFree(comm, blocker) == 0);
    blocker = nullptr;
  };

  // LSA leaders release first while other ranks retain their windows. A world
  // barrier between the two waves proves deregistration/free are local and
  // keeps divergent lifetimes deterministic.
  if (info.lsaRank == 0) releaseLocalState();
  CCO_MUST(ccoBarrierAll(comm) == 0);
  if (info.lsaRank != 0) releaseLocalState();
  CCO_MUST(ccoBarrierAll(comm) == 0);

  // Release the original external owners only after every process has dropped
  // its imported peer handles. Serialize VMM teardown across local GPUs because
  // older ROCr versions can race concurrent hsa_amd_vmem_handle_release calls.
  HIP_CHECK(hipDeviceSynchronize());
  for (int owner = 0; owner < info.lsaSize; owner++) {
    if (info.lsaRank == owner) destroyExternalAllocation(&external);
    CCO_MUST(ccoBarrierAll(comm) == 0);
  }
  CCO_MUST(ccoCommDestroy(comm) == 0);
  std::printf("[rank %d] PASSED\n", rank);
  return 0;
}

int main(int argc, char** argv) {
  // Two ranks are the minimal topology that proves distinct local offsets and
  // keeps the default regression focused and inexpensive. An explicit rank
  // count still overrides this for manual stress runs.
  char twoRanks[] = "2";
  char* defaultArgv[] = {argv[0], twoRanks};
  if (argc == 1) {
    argc = 2;
    argv = defaultArgv;
  }
  return ccoTestMain(argc, argv, "CCO LSA divergent slots", "/tmp/cco_lsa_divergent_slots_uid",
                     19885);
}
