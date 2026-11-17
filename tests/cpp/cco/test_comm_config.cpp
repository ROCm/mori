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

// test: ccoCommConfig (cpuAffinity, lsaOnly, forward-compatible size).
//
//   1. A config without the INITIALIZER's magic is rejected before any collective.
//   2. cpuAffinity=0 + lsaOnly=1 on one LSA team: the thread's CPU affinity is
//      untouched, windows get no MR, peers read each other through the flat VA, a
//      default (CROSSNODE) DevComm resolves to NONE and FULL is refused.
//   3. lsaOnly=1 across two LSA teams (MORI_VPOD_ID splits the host): the comm
//      is created, the other team has no peer pointer, CROSSNODE is refused and
//      NONE works.
//   4. A config from an older caller (size ends before lsaOnly) keeps the lsaOnly
//      default even when the bytes past `size` say otherwise.

#include <sched.h>
#include <stdlib.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t WINDOW_SIZE = 4096;
// Rank 0 hands out the ids for comms 3 and 4 through its window in comm 2.
static const size_t kUidSlots = 2;
static const size_t kPatternOffset = kUidSlots * sizeof(ccoUniqueId);

#define FAIL(...)                                \
  do {                                           \
    fprintf(stderr, "[rank %d] FAIL: ", g_rank); \
    fprintf(stderr, __VA_ARGS__);                \
    fprintf(stderr, "\n");                       \
    return 1;                                    \
  } while (0)

static bool AffinityUnchanged(const cpu_set_t& before) {
  cpu_set_t now;
  CPU_ZERO(&now);
  return sched_getaffinity(0, sizeof(now), &now) == 0 && CPU_EQUAL(&before, &now);
}

static int DevCommRc(ccoComm* comm, ccoGdaConnectionType type, ccoGdaConnectionType* resolved) {
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = type;
  reqs.lsaBarrierCount = 1;
  ccoDevComm dc{};
  int rc = ccoDevCommCreate(comm, &reqs, &dc);
  if (rc == 0) {
    if (resolved) *resolved = dc.gdaConnType;
    ccoDevCommDestroy(comm, &dc);
  }
  return rc;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  cpu_set_t before;
  CPU_ZERO(&before);
  if (sched_getaffinity(0, sizeof(before), &before) != 0) FAIL("sched_getaffinity");

  // 1. Uninitialized config: local error, no bootstrap started.
  {
    ccoCommConfig bad;
    std::memset(&bad, 0, sizeof(bad));
    ccoComm* c = nullptr;
    if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &bad, &c) == 0 || c != nullptr)
      FAIL("config without magic was accepted");
  }

  // 2. LSA-only comm on one team that leaves CPU affinity alone.
  ccoCommConfig config = CCO_COMM_CONFIG_INITIALIZER;
  config.cpuAffinity = 0;
  config.lsaOnly = 1;
  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &config, &comm) != 0)
    FAIL("lsaOnly ccoCommCreate");
  if (!AffinityUnchanged(before)) FAIL("cpuAffinity=0 changed the thread's affinity");
  if (!comm->lsaOnly || comm->lsaSize != nranks || comm->windowMrsEnabled)
    FAIL("lsaOnly comm state: lsaOnly=%d lsaSize=%d windowMrs=%d", comm->lsaOnly, comm->lsaSize,
         comm->windowMrsEnabled);

  void* buf = nullptr;
  ccoWindow_t win = nullptr;
  if (ccoWindowRegister(comm, WINDOW_SIZE, &win, &buf) != 0) FAIL("ccoWindowRegister");
  std::vector<uint8_t> pattern(WINDOW_SIZE, static_cast<uint8_t>(rank + 1));
  HIP_CHECK(hipMemcpy(buf, pattern.data(), WINDOW_SIZE, hipMemcpyHostToDevice));
  ccoUniqueId ids[kUidSlots];
  if (rank == 0) {
    for (auto& id : ids)
      if (ccoGetUniqueId(&id) != 0) FAIL("ccoGetUniqueId");
    HIP_CHECK(hipMemcpy(buf, ids, sizeof(ids), hipMemcpyHostToDevice));
  }
  ccoBarrierAll(comm);

  const int peer = (rank + 1) % nranks;
  void* peerBuf = ccoGetPeerPtr(comm, buf, peer);
  if (peerBuf == nullptr) FAIL("no LSA pointer to peer %d", peer);
  uint8_t got = 0;
  HIP_CHECK(
      hipMemcpy(&got, static_cast<uint8_t*>(peerBuf) + kPatternOffset, 1, hipMemcpyDeviceToHost));
  if (got != static_cast<uint8_t>(peer + 1)) FAIL("peer %d read %u, want %u", peer, got, peer + 1);
  const void* rootBuf = rank == 0 ? buf : ccoGetPeerPtr(comm, buf, 0);
  HIP_CHECK(hipMemcpy(ids, rootBuf, sizeof(ids), hipMemcpyDeviceToHost));

  ccoGdaConnectionType resolved = CCO_GDA_CONNECTION_FULL;
  if (DevCommRc(comm, CCO_GDA_CONNECTION_CROSSNODE, &resolved) != 0 ||
      resolved != CCO_GDA_CONNECTION_NONE)
    FAIL("CROSSNODE DevComm on one team: want NONE, got %d", static_cast<int>(resolved));
  if (DevCommRc(comm, CCO_GDA_CONNECTION_FULL, nullptr) == 0)
    FAIL("FULL DevComm accepted on an lsaOnly comm");

  ccoBarrierAll(comm);
  ccoWindowDeregister(comm, win);
  ccoMemFree(comm, buf);
  ccoCommDestroy(comm);

  // 3. LSA-only comm over two LSA teams on this host.
  if (nranks % 2 == 0) {
    const int half = nranks / 2;
    setenv("MORI_VPOD_ID", rank < half ? "0" : "1", 1);
    ccoComm* split = nullptr;
    int rc = ccoCommCreate(ids[0], nranks, rank, PER_RANK_VMM_SIZE, &config, &split);
    unsetenv("MORI_VPOD_ID");
    if (rc != 0) FAIL("lsaOnly ccoCommCreate over two teams");
    if (split->lsaSize != half || split->windowMrsEnabled)
      FAIL("split comm: lsaSize=%d (want %d) windowMrs=%d", split->lsaSize, half,
           split->windowMrsEnabled);
    void* b = nullptr;
    ccoWindow_t w = nullptr;
    if (ccoWindowRegister(split, WINDOW_SIZE, &w, &b) != 0) FAIL("split ccoWindowRegister");
    const int other = (rank + half) % nranks;
    if (ccoGetPeerPtr(split, b, other) != nullptr)
      FAIL("peer %d in the other team has an LSA pointer", other);
    if (DevCommRc(split, CCO_GDA_CONNECTION_CROSSNODE, nullptr) == 0)
      FAIL("CROSSNODE DevComm accepted on an lsaOnly comm spanning teams");
    if (DevCommRc(split, CCO_GDA_CONNECTION_NONE, nullptr) != 0)
      FAIL("NONE DevComm on an lsaOnly comm spanning teams");
    ccoBarrierAll(split);
    ccoWindowDeregister(split, w);
    ccoMemFree(split, b);
    ccoCommDestroy(split);
  }

  // 4. Older caller: size covers cpuAffinity only, so lsaOnly keeps its default (0).
  ccoCommConfig older = CCO_COMM_CONFIG_INITIALIZER;
  older.size = offsetof(ccoCommConfig, lsaOnly);
  older.cpuAffinity = 0;
  older.lsaOnly = 1;  // past `size`: must be ignored
  ccoComm* comm2 = nullptr;
  if (ccoCommCreate(ids[1], nranks, rank, PER_RANK_VMM_SIZE, &older, &comm2) != 0)
    FAIL("ccoCommCreate with an older config");
  if (!AffinityUnchanged(before)) FAIL("older config: cpuAffinity=0 changed the affinity");
  if (comm2->lsaOnly) FAIL("lsaOnly past the caller's size was read");
  ccoCommDestroy(comm2);

  printf("[rank %d] PASS\n", rank);
  return 0;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO comm config", "/tmp/cco_comm_config_uid", 0);
}
