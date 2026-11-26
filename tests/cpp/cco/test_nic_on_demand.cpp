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

// test: cco opens the NICs only when a peer needs RDMA.
//
//   A. One node: CommCreate, a window and NONE / CROSSNODE / RAIL DevComms leave
//      the NICs closed; a FULL DevComm (NIC loopback) opens them. FULL is skipped
//      under MORI_CCO_SKIP_GDA_FULL, like the other FULL tests.
//   B. The same ranks split into two nodes with MORI_NODE_ID: peers outside the
//      LSA team need RDMA, so CommCreate opens the NICs. Runs only when A was one
//      node and nranks is even. On a scale-up fabric the LSA team still spans
//      both halves, and the NICs stay closed.
//
// "Opened" means the Context built its RdmaContext, which happens whether or not
// the host has a NIC, so the checks hold on NIC-less hosts too.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/application/bootstrap/base_bootstrap.hpp"
#include "mori/application/context/context.hpp"
#include "mori/cco/cco.hpp"

using namespace mori::cco;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

static bool NicOpened(ccoComm* comm) { return comm->ctx->GetRdmaContext() != nullptr; }

static bool Expect(int rank, const char* phase, bool opened, bool expected) {
  if (opened == expected) return true;
  fprintf(stderr, "[rank %d] FAIL %s: NICs %s, expected %s\n", rank, phase,
          opened ? "opened" : "closed", expected ? "opened" : "closed");
  return false;
}

static bool DevCommRoundTrip(ccoComm* comm, ccoGdaConnectionType type) {
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = type;
  ccoDevComm dc{};
  if (ccoDevCommCreate(comm, &reqs, &dc) != 0) return false;
  ccoDevCommDestroy(comm, &dc);
  return true;
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  // ── A: the comm as launched ──
  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] ccoCommCreate failed\n", rank);
    return 1;
  }
  const bool oneNode = comm->lsaSize == comm->worldSize;
  bool ok = Expect(rank, "A CommCreate", NicOpened(comm), !oneNode);

  ccoWindow_t win = nullptr;
  void* buf = nullptr;
  if (ccoWindowRegister(comm, 4096, &win, &buf) != 0) {
    fprintf(stderr, "[rank %d] ccoWindowRegister failed\n", rank);
    ok = false;
  }
  ok = Expect(rank, "A window", NicOpened(comm), !oneNode) && ok;

  const ccoGdaConnectionType noNic[] = {CCO_GDA_CONNECTION_NONE, CCO_GDA_CONNECTION_CROSSNODE,
                                        CCO_GDA_CONNECTION_RAIL};
  for (ccoGdaConnectionType t : noNic) {
    if (!DevCommRoundTrip(comm, t)) {
      fprintf(stderr, "[rank %d] ccoDevCommCreate(type %d) failed\n", rank, t);
      ok = false;
    }
    char phase[64];
    snprintf(phase, sizeof(phase), "A DevComm type %d", t);
    ok = Expect(rank, phase, NicOpened(comm), !oneNode) && ok;
  }

  if (getenv("MORI_CCO_SKIP_GDA_FULL") == nullptr) {
    if (!DevCommRoundTrip(comm, CCO_GDA_CONNECTION_FULL)) {
      fprintf(stderr, "[rank %d] ccoDevCommCreate(FULL) failed\n", rank);
      ok = false;
    }
    ok = Expect(rank, "A DevComm FULL", NicOpened(comm), true) && ok;
  } else if (rank == 0) {
    printf("A DevComm FULL: SKIPPED (MORI_CCO_SKIP_GDA_FULL)\n");
  }
  printf("[rank %d] A: lsaSize=%d worldSize=%d\n", rank, comm->lsaSize, comm->worldSize);

  // Rank 0's fresh id for B travels over A's bootstrap, so B works in every launch mode.
  const bool runB = oneNode && nranks % 2 == 0;
  ccoUniqueId uidB{};
  if (runB) {
    ccoUniqueId mine{};
    if (rank == 0 && ccoGetUniqueId(&mine) != 0) {
      fprintf(stderr, "[rank 0] ccoGetUniqueId for B failed\n");
      std::abort();
    }
    std::vector<ccoUniqueId> all(nranks);
    comm->bootNet->Allgather(&mine, all.data(), sizeof(mine));
    uidB = all[0];
  }

  ccoWindowDeregister(comm, win);
  ccoMemFree(comm, buf);
  ccoCommDestroy(comm);

  // ── B: two nodes on one host ──
  if (runB) {
    setenv("MORI_NODE_ID", rank < nranks / 2 ? "nic-on-demand-a" : "nic-on-demand-b", 1);
    ccoComm* split = nullptr;
    if (ccoCommCreate(uidB, nranks, rank, PER_RANK_VMM_SIZE, &split) != 0) {
      fprintf(stderr, "[rank %d] ccoCommCreate (split) failed\n", rank);
      return 1;
    }
    ok = Expect(rank, "B CommCreate", NicOpened(split), split->lsaSize < split->worldSize) && ok;
    printf("[rank %d] B: lsaSize=%d worldSize=%d\n", rank, split->lsaSize, split->worldSize);
    ccoCommDestroy(split);
    unsetenv("MORI_NODE_ID");
  } else if (rank == 0) {
    printf("B: SKIPPED (%s)\n", oneNode ? "odd nranks" : "already multi-node");
  }

  if (ok) printf("[rank %d] PASS\n", rank);
  fflush(stdout);  // fork mode leaves through _exit, which does not flush
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO NIC on demand", "/tmp/cco_nic_on_demand_uid", 0);
}
