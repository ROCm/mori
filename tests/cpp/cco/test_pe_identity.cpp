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

// The PE identity table (application::Context::PeIdentities). Each rank is given one
// card through HIP_VISIBLE_DEVICES, so every rank sees "device 0": the identities must
// still name the card each one really has. Needs at least two GPUs; no NIC.

#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// Host-only on purpose: ccoComm is a complete type only in host translation units.
#include "cco_test_harness.hpp"
#include "mori/application/context/context.hpp"
#include "mori/metrics/metrics.hpp"

using mori::application::PeIdentity;

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

static int g_fails = 0;
#define EXPECT(cond, ...)                                 \
  do {                                                    \
    if (!(cond)) {                                        \
      fprintf(stderr, "  [rank %d] FAIL: ", g_rank);      \
      fprintf(stderr, __VA_ARGS__);                       \
      fprintf(stderr, "  (%s:%d)\n", __FILE__, __LINE__); \
      g_fails++;                                          \
    }                                                     \
  } while (0)

// The PCI address of each GPU by physical index, read from sysfs independently of HIP.
static std::vector<std::string> PhysicalGpus() {
  std::vector<std::pair<int, std::string>> cards;
  for (const std::string& name : mori::metrics::detail::ListDir("/sys/class/drm")) {
    if (!mori::metrics::detail::IsDrmCardName(name)) continue;
    const std::string dev = "/sys/class/drm/" + name + "/device";
    if (access((dev + "/xgmi_device_id").c_str(), F_OK) != 0) continue;
    char real[PATH_MAX];
    if (realpath(dev.c_str(), real) == nullptr) continue;
    const std::string path = real;
    cards.emplace_back(atoi(name.c_str() + 4), path.substr(path.find_last_of('/') + 1));
  }
  std::sort(cards.begin(), cards.end());
  std::vector<std::string> out;
  for (const auto& c : cards) out.push_back(mori::metrics::NormalizePciBdf(c.second));
  return out;
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  g_rank = rank;
  // Before the first HIP call; in reverse, so that "rank i has GPU i" does not hold.
  const int visible = nranks - 1 - rank;
  setenv("HIP_VISIBLE_DEVICES", std::to_string(visible).c_str(), 1);

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  EXPECT(numDevices == 1, "one card is visible, HIP reports %d", numDevices);
  HIP_CHECK(hipSetDevice(0));
  char bus[32] = {0};
  HIP_CHECK(hipDeviceGetPCIBusId(bus, sizeof(bus), 0));
  const std::string myPci = mori::metrics::NormalizePciBdf(bus);

  mori::cco::ccoComm* comm = nullptr;
  int ret = mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm);
  if (ret != 0) {
    fprintf(stderr, "[rank %d] ccoCommCreate failed: %d\n", rank, ret);
    return 1;
  }

  const std::vector<PeIdentity>& ids = comm->ctx->PeIdentities();
  EXPECT(static_cast<int>(ids.size()) == nranks, "table has %zu entries", ids.size());

  const std::vector<std::string> physical = PhysicalGpus();
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  std::set<std::string> labels;
  for (int r = 0; r < static_cast<int>(ids.size()); r++) {
    const PeIdentity& id = ids[r];
    EXPECT(id.hostname == host, "rank %d host '%s', expected '%s'", r, id.hostname.c_str(), host);
    EXPECT(id.gpuIndex >= 0 && id.gpuIndex < static_cast<int>(physical.size()),
           "rank %d gpuIndex %d out of range (%zu physical GPUs)", r, id.gpuIndex, physical.size());
    if (id.gpuIndex >= 0 && id.gpuIndex < static_cast<int>(physical.size())) {
      EXPECT(physical[id.gpuIndex] == id.gpuPci, "rank %d: index %d is %s, it advertised %s", r,
             id.gpuIndex, physical[id.gpuIndex].c_str(), id.gpuPci.c_str());
    }
    EXPECT(id.GpuLabel() == std::to_string(id.gpuIndex), "rank %d label '%s'", r,
           id.GpuLabel().c_str());
    labels.insert(id.GpuLabel());
  }
  EXPECT(static_cast<int>(labels.size()) == nranks, "%zu distinct gpu labels for %d ranks",
         labels.size(), nranks);
  EXPECT(ids[rank].gpuPci == myPci, "own entry %s, HIP says %s", ids[rank].gpuPci.c_str(),
         myPci.c_str());

  if (rank == 0) {
    for (int r = 0; r < static_cast<int>(ids.size()); r++) {
      printf("  rank %d: host %s gpu %s (%s) nic %s/%s\n", r, ids[r].hostname.c_str(),
             ids[r].GpuLabel().c_str(), ids[r].gpuPci.c_str(),
             ids[r].ibdev.empty() ? "-" : ids[r].ibdev.c_str(),
             ids[r].netdev.empty() ? "-" : ids[r].netdev.c_str());
    }
    fflush(stdout);  // the harness leaves a rank through _exit
  }
  mori::cco::ccoCommDestroy(comm);
  return g_fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  // The parent must not initialise HIP: the children choose their card first.
  int nranks = 4;
  if (const char* env = getenv("CCO_TEST_NRANKS")) nranks = atoi(env);
  const int gpus = static_cast<int>(PhysicalGpus().size());
  if (gpus < 2) {
    printf("SKIP: needs at least two GPUs (found %d)\n", gpus);
    return 0;
  }
  if (nranks > gpus) nranks = gpus;
  // sysfs can list GPUs this user may not open (a login node): ask HIP, in a child.
  pid_t probe = fork();
  if (probe == 0) {
    int n = 0;
    _exit(hipGetDeviceCount(&n) == hipSuccess && n >= 2 ? 0 : 1);
  }
  int probeStatus = 0;
  waitpid(probe, &probeStatus, 0);
  if (!WIFEXITED(probeStatus) || WEXITSTATUS(probeStatus) != 0) {
    printf("SKIP: HIP cannot use two GPUs here\n");
    return 0;
  }
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank") || !strcmp(argv[i], "--gen-uid")) {
      return ccoTestMain(argc, argv, "CCO PE identity", "/tmp/cco_pe_identity_uid", 0);
    }
  }
  return ccoTestForkMode(nranks, "CCO PE identity", "/tmp/cco_pe_identity_uid", 0);
}
