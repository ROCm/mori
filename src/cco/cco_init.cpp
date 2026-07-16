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
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "mori/application/application.hpp"  // Context, BootstrapNetwork
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/memory/va_manager.hpp"  // HeapVAManager
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/cco/cco.hpp"  // public, self-contained (opaque ccoComm fwd-decl)
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace cco {

namespace {
// Serialize HIP VMM calls across threads in one process. ROCm ROCR can race on
// concurrent hsa_amd_vmem_map / hipMemSetAccess from SPMT callers.
std::recursive_mutex& vmmProcessMutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

struct vmmProcessLock {
  std::lock_guard<std::recursive_mutex> guard{vmmProcessMutex()};
};
}  // namespace

// ccoProviderType is cco's self-contained copy of core::ProviderType; the cast
// below relies on a 1:1 mapping, so guard it (this TU sees both enums).
static_assert(static_cast<int>(CCO_PROVIDER_UNKNOWN) ==
                  static_cast<int>(core::ProviderType::Unknown),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_MLX5) == static_cast<int>(core::ProviderType::MLX5),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_BNXT) == static_cast<int>(core::ProviderType::BNXT),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_PSD) == static_cast<int>(core::ProviderType::PSD),
              "ccoProviderType drifted from core::ProviderType");
static_assert(static_cast<int>(CCO_PROVIDER_IBVERBS) ==
                  static_cast<int>(core::ProviderType::IBVERBS),
              "ccoProviderType drifted from core::ProviderType");

// ccoFabricHandle_t (cco.hpp, self-contained) and hipMemFabricHandle_compat_t
// (hip_compat.hpp) are both 64-byte PODs; guard layout compatibility.
static_assert(sizeof(ccoFabricHandle_t) == sizeof(hipMemFabricHandle_compat_t),
              "ccoFabricHandle_t / hipMemFabricHandle_compat_t size mismatch");

// Out-of-line dtor for the unique_ptr<HeapVAManager> member: ccoComm is defined
// in cco.hpp with HeapVAManager only forward-declared, so its destruction must
// be emitted here where HeapVAManager (va_manager.hpp) is complete.
ccoComm::~ccoComm() = default;

static size_t AlignUp(size_t x, size_t align) { return (x + align - 1) & ~(align - 1); }

// Symmetric-window VMM allocation type. Default = uncached (fine-grained),
// matching mori-shmem's hipDeviceMallocUncached heap: P2P remote reads/writes
// stay coherent over the fabric without coarse-grained L2 coherence handling.
// Opt out with CCO_UNCACHED_WINDOW=0 (reverts to coarse-grained pinned memory).
// MUST be used for BOTH the granularity query and the actual hipMemCreate so
// the granularity matches the allocation.
static hipMemAllocationType CcoWindowAllocType() {
  static const bool cached = [] {
    const char* e = getenv("CCO_UNCACHED_WINDOW");
    return e && atoi(e) == 0;
  }();
  return cached ? hipMemAllocationTypePinned : hipMemAllocationTypeUncached;
}

// Local slot base = the VA where this rank's slice of the flat VA starts.
// Used as HeapVAManager's baseAddr so Allocate() returns dereferenceable
// localVa directly. Guaranteed non-zero because flatBase comes from
// hipMemAddressReserve.
static uintptr_t LocalSlotBase(const ccoComm* comm) {
  return reinterpret_cast<uintptr_t>(comm->flatBase) +
         static_cast<uintptr_t>(comm->lsaRank) * comm->perRankSize;
}

/* ========================================================================== */
/*                              ccoCommCreate                              */
/* ========================================================================== */

int ccoGetUniqueId(ccoUniqueId* uniqueId) {
  if (!uniqueId) return -1;
  static_assert(sizeof(application::UniqueId) <= sizeof(ccoUniqueId),
                "ccoUniqueId must be large enough to hold application::UniqueId");
  // Encode rank 0's socket rendezvous endpoint into the id; non-root ranks
  // connect here during ccoCommCreate, so the address+port must be concrete.
  // Pick a free port by random probe-bind (zero-config, no fixed port to
  // collide) — same scheme as shmem's ShmemGetUniqueId. Interface from
  // MORI_SOCKET_IFNAME, else the first non-loopback NIC. Caller broadcasts the
  // POD id to every rank out-of-band.
  const char* ifname = std::getenv("MORI_SOCKET_IFNAME");
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> portDis(10000, 60000);
  constexpr int kMaxPortRetries = 20;

  try {
    for (int attempt = 0; attempt < kMaxPortRetries; attempt++) {
      int port = portDis(gen);
      int probeFd = socket(AF_INET, SOCK_STREAM, 0);
      if (probeFd < 0) continue;
      int opt = 1;
      setsockopt(probeFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      struct sockaddr_in probeAddr{};
      probeAddr.sin_family = AF_INET;
      probeAddr.sin_port = htons(static_cast<uint16_t>(port));
      probeAddr.sin_addr.s_addr = htonl(INADDR_ANY);
      if (bind(probeFd, reinterpret_cast<struct sockaddr*>(&probeAddr), sizeof(probeAddr)) == 0) {
        close(probeFd);
        application::UniqueId appUid =
            ifname
                ? application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(ifname, port)
                : application::SocketBootstrapNetwork::GenerateUniqueIdWithLocalAddr(port);
        std::memset(uniqueId, 0, sizeof(*uniqueId));
        std::memcpy(uniqueId, &appUid, sizeof(appUid));
        return 0;
      }
      close(probeFd);
    }
  } catch (const std::exception& e) {
    MORI_SHMEM_ERROR("ccoGetUniqueId failed: {} (set MORI_SOCKET_IFNAME=<iface>)", e.what());
    return -1;
  }
  MORI_SHMEM_ERROR("ccoGetUniqueId: no free port after {} attempts", kMaxPortRetries);
  return -1;
}

// Internal bootstrap helper: caller-provided transport (ownership transferred to
// the comm; Finalize()d + deleted in ccoCommDestroy). Not part of the public API
// — the ccoUniqueId overload below builds the built-in socket bootstrap and
// delegates here.
static int ccoCommCreateImpl(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                             ccoComm** outComm);

// Self-contained overload: build cco's built-in socket bootstrap from the id and
// delegate to the internal helper, which takes ownership (the socket bootstrap is
// Finalize()d + deleted in ccoCommDestroy).
int ccoCommCreate(const ccoUniqueId& uniqueId, int nRanks, int rank, size_t perRankVmmSize,
                  ccoComm** outComm) {
  if (!outComm || nRanks <= 0 || rank < 0 || rank >= nRanks) return -1;
  application::UniqueId appUid;
  std::memcpy(&appUid, &uniqueId, sizeof(appUid));
  auto* boot = new application::SocketBootstrapNetwork(appUid, rank, nRanks);
  return ccoCommCreateImpl(boot, perRankVmmSize, outComm);
}

// The LSA team is always the contiguous rank range [myNodeStart, myNodeStart+
// lsaSize) — its width is set by the topology step (same host by default, same
// vPOD when grouping is enabled). Membership + flat-VA index derive purely from
// that range, so the same formula covers intra-node, cross-node-vPOD, and
// whole-world layouts.
static bool CcoCanLsaMapPeer(const ccoComm* comm, int pe) {
  if (pe == comm->rank) return false;
  return pe >= comm->myNodeStart && pe < comm->myNodeStart + comm->lsaSize;
}

static int CcoPeToLsaRank(const ccoComm* comm, int pe) { return pe - comm->myNodeStart; }

// ── LSA-team (vPOD) topology detection ───────────────────────────────────────
// A vPOD is a scale-up fabric domain (AMD UALink) that may span multiple hosts;
// its GPUs are directly flat-VA (P2P) interconnectable. We mirror RCCL's MNNVL
// clique model: group ranks by (ppod_id UUID, vpod_id) read from the per-GPU
// UALink sysfs, gated on the accelerator being ACTIVE/READY. ppod_id (a UUID) is
// what makes the key globally unique — hive_id collides across hosts.
//   Precedence: MORI_CCO_FABRIC_DISABLE=1 -> force host-only LSA
//               MORI_VPOD_ID=<n>          -> explicit override (all-or-none)
//               MORI_CCO_FABRIC_CROSSNODE_LSA=1 -> whole world is one team
//               auto                      -> UALink (ppod_id, vpod_id)
//               else                      -> group by host (default)
enum CcoLsaMode { CCO_LSA_HOST = 0, CCO_LSA_MANUAL = 1, CCO_LSA_FABRIC = 2 };

struct CcoLsaKey {
  int mode;                  // CcoLsaMode
  int vpodId;                // manual value or UALink vpod_id
  int vpodSize;              // UALink vpod_size (0 if n/a)
  unsigned char ppodId[16];  // UALink ppod_id UUID (zeros if n/a)
};

static bool CcoReadSysfsLine(const std::string& path, std::string& out) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char buf[128] = {0};
  bool ok = fgets(buf, sizeof(buf), f) != nullptr;
  fclose(f);
  if (!ok) return false;
  out = buf;
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
    out.pop_back();
  return true;
}

// Parse "9217fed9-c6cf-4c9e-9d9c-7110b90917cc" into 16 bytes; false (zeros) if
// it isn't a well-formed, non-zero UUID.
static bool CcoParseUuid(const std::string& s, unsigned char out[16]) {
  memset(out, 0, 16);
  int b = 0;
  unsigned hi = 0;
  bool haveHi = false, any = false;
  for (char c : s) {
    if (c == '-') continue;
    int v;
    if (c >= '0' && c <= '9')
      v = c - '0';
    else if (c >= 'a' && c <= 'f')
      v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      v = c - 'A' + 10;
    else
      return false;
    if (!haveHi) {
      hi = v;
      haveHi = true;
    } else {
      if (b >= 16) return false;
      out[b] = static_cast<unsigned char>((hi << 4) | v);
      if (out[b]) any = true;
      b++;
      haveHi = false;
    }
  }
  return b == 16 && !haveHi && any;
}

// Read the local GPU's UALink fabric identity from sysfs, mirroring RCCL's
// alt_rsmi ARSMI_get_fabric_info. True only if the fabric is present and the
// accelerator is ACTIVE/READY (i.e. usable for cross-node LSA).
static bool CcoReadFabricKey(int hipDev, CcoLsaKey* key) {
  char bdf[32] = {0};
  if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), hipDev) != hipSuccess) return false;
  for (char* p = bdf; *p; ++p) *p = static_cast<char>(tolower(*p));
  std::string dir = std::string("/sys/bus/pci/devices/") + bdf + "/ualink";

  std::string link, state, ppod;
  if (!CcoReadSysfsLine(dir + "/link_type", link)) return false;  // not a UALink GPU
  if (link != "UALoE" && link != "UALLink") return false;
  if (!CcoReadSysfsLine(dir + "/accel_state", state)) return false;
  if (state != "active" && state != "ready") return false;  // not usable yet
  if (!CcoReadSysfsLine(dir + "/ppod_id", ppod)) return false;
  if (!CcoParseUuid(ppod, key->ppodId)) return false;  // zero/garbage UUID

  std::string v;
  key->vpodId = CcoReadSysfsLine(dir + "/vpod_id", v) ? atoi(v.c_str()) : 0;
  key->vpodSize = CcoReadSysfsLine(dir + "/vpod_size", v) ? atoi(v.c_str()) : 0;
  key->mode = CCO_LSA_FABRIC;
  return true;
}

// Compute this rank's LSA-team key by the precedence above.
static void CcoComputeLsaKey(int hipDev, CcoLsaKey* key) {
  memset(key, 0, sizeof(*key));
  key->mode = CCO_LSA_HOST;
  const char* dis = getenv("MORI_CCO_FABRIC_DISABLE");
  if (dis && atoi(dis) != 0) return;  // forced host-only
  const char* mv = getenv("MORI_VPOD_ID");
  if (mv && *mv) {
    key->mode = CCO_LSA_MANUAL;
    key->vpodId = atoi(mv);
    return;
  }
  const char* w = getenv("MORI_CCO_FABRIC_CROSSNODE_LSA");
  if (w && atoi(w) != 0) {  // whole world = one team
    key->mode = CCO_LSA_MANUAL;
    key->vpodId = 0;
    return;
  }
  CcoReadFabricKey(hipDev, key);  // sets FABRIC on success, leaves HOST otherwise
}

// Two ranks share an LSA team iff their keys match. HOST mode is resolved by the
// sameHost predicate instead (this returns false for it).
static bool CcoLsaKeySame(const CcoLsaKey& a, const CcoLsaKey& b) {
  if (a.mode != b.mode) return false;
  if (a.mode == CCO_LSA_MANUAL) return a.vpodId == b.vpodId;
  if (a.mode == CCO_LSA_FABRIC)
    return a.vpodId == b.vpodId && memcmp(a.ppodId, b.ppodId, sizeof(a.ppodId)) == 0;
  return false;
}

// By default a cross-node hipMemSetAccess failure is fatal at register time
// (fail loudly instead of leaving a silently-unusable peer mapping that faults
// mid-kernel). MORI_CCO_FABRIC_LENIENT=1 restores continue-on-failure, for
// fabrics where the import itself already grants access.
static bool CcoFabricLenient() {
  const char* e = getenv("MORI_CCO_FABRIC_LENIENT");
  return e && atoi(e) != 0;
}

// Zero a symmetric-window buffer WITHOUT hipMemset. On UALink fabric-exportable
// pools hipMemset returns hipErrorOutOfMemory (observed on ROCm 7.15), so we
// initialize via chunked host->device copies from a small zeroed staging buffer
// (host-only HIP runtime API — mori_cco links hip::host, not device code).
static hipError_t CcoZeroWindowMem(void* ptr, size_t bytes) {
  if (bytes == 0) return hipSuccess;
  const size_t chunk = std::min<size_t>(bytes, 4u << 20);  // 4 MB staging
  std::vector<char> zeros(chunk, 0);
  char* dst = static_cast<char*>(ptr);
  for (size_t off = 0; off < bytes; off += chunk) {
    size_t n = std::min(chunk, bytes - off);
    hipError_t e = hipMemcpy(dst + off, zeros.data(), n, hipMemcpyHostToDevice);
    if (e != hipSuccess) return e;
  }
  return hipSuccess;
}

static int ccoCommCreateImpl(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                             ccoComm** outComm) {
  auto* comm = new ccoComm();
  *outComm = comm;

  // Step 1: bootstrap
  comm->bootNet = bootNet;
  comm->bootNet->Initialize();
  comm->rank = comm->bootNet->GetLocalRank();
  comm->worldSize = comm->bootNet->GetWorldSize();

  // Derive a shared group ID (rank 0's pid) for unique LocalBootstrap socket paths
  int64_t myPid = static_cast<int64_t>(getpid());
  std::vector<int64_t> allPids(comm->worldSize);
  comm->bootNet->Allgather(&myPid, allPids.data(), sizeof(int64_t));
  comm->groupId = allPids[0];

  MORI_SHMEM_TRACE("ccoCommCreate: rank={} worldSize={} groupId={}", comm->rank, comm->worldSize,
                   comm->groupId);

  // Step 2: context (RDMA endpoints + transport-type negotiation).
  comm->ctx = new application::Context(*comm->bootNet);
  comm->defaultNumQpPerPe = comm->ctx->GetNumQpPerPe();

  // Cache the bound device once (used by topology detection below and later by
  // ccoMemAlloc / ccoWindowRegister). Callers MUST keep the calling thread bound
  // to this device for any later CCO API on this comm.
  HIP_RUNTIME_CHECK(hipGetDevice(&comm->hipDev));
  comm->peerHipDevs.assign(comm->worldSize, comm->hipDev);
  comm->bootNet->Allgather(&comm->hipDev, comm->peerHipDevs.data(), sizeof(int));

  // Step 2.5: detect LSA (Local Symmetric Access) team topology.
  // Membership is a HARDWARE fact — peers whose memory this rank can load/store
  // through a flat VA — captured before transport policy: same host by default,
  // or the same vPOD (a scale-up UALink fabric domain that may span hosts) when
  // fabric/manual grouping applies (see CcoComputeLsaKey).
  //
  // HARD CONTRACT — violations are fatal:
  //   (a) LSA-major contiguous ranks (team peers form a single block)
  //   (b) every rank observes the same lsaSize
  // Both are required by the flat-VA formula `lsaFlatBase + lsaRank * stride`.

  // Each rank computes its LSA-team key (host / manual vPOD / UALink fabric),
  // then we allgather + decide the job-wide grouping. Manual (MORI_VPOD_ID) is
  // all-or-none; UALink fabric grouping only kicks in when EVERY rank reports a
  // ready fabric (else fall back to host LSA + RDMA, like RCCL disabling MNNVL).
  CcoLsaKey myKey;
  CcoComputeLsaKey(comm->hipDev, &myKey);
  std::vector<CcoLsaKey> allKeys(comm->worldSize);
  comm->bootNet->Allgather(&myKey, allKeys.data(), sizeof(CcoLsaKey));

  int manualCnt = 0, fabricCnt = 0;
  for (const auto& k : allKeys) {
    if (k.mode == CCO_LSA_MANUAL)
      manualCnt++;
    else if (k.mode == CCO_LSA_FABRIC)
      fabricCnt++;
  }
  bool vpodMode;
  if (manualCnt > 0) {
    if (manualCnt != comm->worldSize) {
      MORI_SHMEM_ERROR(
          "ccoCommCreate: MORI_VPOD_ID set on {}/{} ranks — manual vPOD grouping must "
          "be all-or-none across the job.",
          manualCnt, comm->worldSize);
      delete comm->ctx;
      comm->bootNet->Finalize();
      delete comm;
      *outComm = nullptr;
      return -1;
    }
    vpodMode = true;
  } else if (fabricCnt == comm->worldSize) {
    vpodMode = true;  // every rank on a ready UALink fabric
  } else {
    vpodMode = false;  // host LSA + RDMA (default / no fabric / mixed readiness)
    if (fabricCnt > 0)
      MORI_SHMEM_INFO(
          "ccoCommCreate: UALink fabric ready on only {}/{} ranks — using host LSA + RDMA",
          fabricCnt, comm->worldSize);
  }
  bool lsaSpansHosts = false;
  {
    int lsaCount = 0;
    int firstInTeam = comm->rank;
    int lastInTeam = comm->rank;
    for (int pe = 0; pe < comm->worldSize; pe++) {
      const auto& cap = comm->ctx->GetPeerCapabilities(pe);
      const bool sameHost = (pe == comm->rank) || cap.sameHost;
      const bool inTeam = vpodMode ? CcoLsaKeySame(allKeys[pe], myKey) : sameHost;
      if (inTeam) {
        if (pe < firstInTeam) firstInTeam = pe;
        if (pe > lastInTeam) lastInTeam = pe;
        lsaCount++;
        if (!sameHost) lsaSpansHosts = true;  // team member on another host
      }
    }

    if (lastInTeam - firstInTeam + 1 != lsaCount) {
      MORI_SHMEM_ERROR(
          "ccoCommCreate: non-contiguous lsa membership "
          "(rank {}: first={} last={} count={}). CCO requires "
          "LSA-major contiguous rank layout. Reorder ranks in your "
          "launch (mpirun -host A:N,B:N or equivalent), or align MORI_VPOD_ID "
          "with contiguous rank blocks.",
          comm->rank, firstInTeam, lastInTeam, lsaCount);
      delete comm->ctx;
      comm->bootNet->Finalize();
      delete comm;
      *outComm = nullptr;
      return -1;
    }

    std::vector<int> allLsaSizes(comm->worldSize);
    comm->bootNet->Allgather(&lsaCount, allLsaSizes.data(), sizeof(int));
    for (int r = 0; r < comm->worldSize; r++) {
      if (allLsaSizes[r] != lsaCount) {
        MORI_SHMEM_ERROR(
            "ccoCommCreate: heterogeneous lsa sizes detected "
            "(my rank {} sees lsaSize={}, rank {} sees lsaSize={}). "
            "CCO requires uniform LSA-team size across all ranks.",
            comm->rank, lsaCount, r, allLsaSizes[r]);
        delete comm->ctx;
        comm->bootNet->Finalize();
        delete comm;
        *outComm = nullptr;
        return -1;
      }
    }

    comm->lsaSize = lsaCount;
    comm->myNodeStart = firstInTeam;
    comm->lsaRank = comm->rank - firstInTeam;

    // Sanity: on a UALink fabric the team should match the reported vpod_size
    // (a mismatch just means a subset of the vPOD was launched — informational).
    if (myKey.mode == CCO_LSA_FABRIC && myKey.vpodSize > 0 && lsaCount != myKey.vpodSize) {
      MORI_SHMEM_INFO(
          "ccoCommCreate: LSA team size {} != UALink vpod_size {} (subset of the vPOD "
          "launched?)",
          lsaCount, myKey.vpodSize);
    }

    MORI_SHMEM_INFO(
        "ccoCommCreate: lsa topology rank={} lsaSize={} lsaRank={} lsaStart={} "
        "mode={} spansHosts={}",
        comm->rank, comm->lsaSize, comm->lsaRank, comm->myNodeStart,
        (myKey.mode == CCO_LSA_FABRIC ? "fabric" : (vpodMode ? "manual" : "host")), lsaSpansHosts);
  }

  // Step 3: reserve flat VA. Always 4GB-aligned so stride4G = perRankSize >> 32
  // is lossless. perRankVmmSize == 0 defaults to GPU total memory.
  if (perRankVmmSize == 0) {
    size_t freeMem = 0, totalMem = 0;
    HIP_RUNTIME_CHECK(hipMemGetInfo(&freeMem, &totalMem));
    perRankVmmSize = totalMem;
  }
  perRankVmmSize = AlignUp(perRankVmmSize, 1ULL << 32);
  comm->perRankSize = perRankVmmSize;

  // Probe fabric handle support: try to allocate + export with the fabric
  // handle type. If it works, all subsequent allocations use fabric handles
  // (64-byte tokens exchangeable via Allgather) instead of dma-buf FDs
  // (which require Unix socket + SCM_RIGHTS).
  {
    hipMemAllocationProp probeProp = {};
    probeProp.type = hipMemAllocationTypePinned;
    probeProp.requestedHandleType = hipMemHandleTypeFabricCompat;
    probeProp.location.type = hipMemLocationTypeDevice;
    probeProp.location.id = comm->hipDev;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    probeProp.allocFlags.gpuDirectRDMACapable = 1;
#endif

    size_t probeGranularity = 0;
    hipError_t probeErr = hipMemGetAllocationGranularity(&probeGranularity, &probeProp,
                                                         hipMemAllocationGranularityRecommended);
    if (probeErr == hipSuccess && probeGranularity > 0) {
      hipMemGenericAllocationHandle_t probeHandle = 0;
      probeErr = hipMemCreate(&probeHandle, probeGranularity, &probeProp, 0);
      if (probeErr == hipSuccess) {
        hipMemFabricHandle_compat_t probeFabric;
        probeErr = hipMemExportToShareableHandle(&probeFabric, probeHandle,
                                                 hipMemHandleTypeFabricCompat, 0);
        (void)hipMemRelease(probeHandle);
        if (probeErr == hipSuccess) {
          comm->handleType = static_cast<int>(hipMemHandleTypeFabricCompat);
          MORI_SHMEM_INFO("ccoCommCreate: fabric handle probe succeeded");
        }
      }
    }
    if (comm->handleType != static_cast<int>(hipMemHandleTypeFabricCompat)) {
      MORI_SHMEM_INFO("ccoCommCreate: fabric handle probe failed, using FD path");
    }
  }

  // Cross-node LSA: when the LSA team (vPOD) spans hosts, its peers are only
  // reachable through fabric handles mapped into the flat VA. lsaSize/myNodeStart
  // already describe the (contiguous) team; here we just require fabric and flag
  // that peer mapping must go cross-node (fabric import, no hipDeviceEnablePeer-
  // Access). A scale-up fabric is mandatory — fail loudly if it's missing.
  if (lsaSpansHosts) {
    if (comm->handleType != static_cast<int>(hipMemHandleTypeFabricCompat)) {
      MORI_SHMEM_ERROR(
          "ccoCommCreate: LSA team spans hosts (cross-node vPOD, lsaSize={}) but fabric "
          "handle support is unavailable — cannot map peer VAs across nodes. Fix the "
          "fabric/driver, or scope MORI_VPOD_ID to a single host.",
          comm->lsaSize);
      delete comm->ctx;
      comm->bootNet->Finalize();
      delete comm;
      *outComm = nullptr;
      return -1;
    }
    comm->fabricCrossNodeLsa = true;
    MORI_SHMEM_INFO("ccoCommCreate: cross-node vPOD LSA enabled (lsaSize={} spans hosts)",
                    comm->lsaSize);
  }

  // Query granularity with the SAME allocProp MemAlloc will use — granularity
  // can shift when requestedHandleType (FD export vs fabric) is enabled.
  hipMemAllocationProp allocProp = {};
  allocProp.type = CcoWindowAllocType();
  allocProp.requestedHandleType = static_cast<hipMemAllocationHandleType>(comm->handleType);
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = comm->hipDev;

  size_t granularity = 0;
  // Flat VA covers the LSA team (world-wide when fabricCrossNodeLsa).
  size_t totalVaSize = static_cast<size_t>(comm->lsaSize) * perRankVmmSize;
  {
    vmmProcessLock vmmLock;
    HIP_RUNTIME_CHECK(hipMemGetAllocationGranularity(&granularity, &allocProp,
                                                     hipMemAllocationGranularityRecommended));
    comm->vmmGranularity = granularity;
    HIP_RUNTIME_CHECK(hipMemAddressReserve(&comm->flatBase, totalVaSize, granularity, nullptr, 0));
  }
  MORI_SHMEM_TRACE(
      "ccoCommCreate: flatBase={} totalVA={} (lsaSize={} x perRankSize={}) granularity={}",
      comm->flatBase, totalVaSize, comm->lsaSize, perRankVmmSize, granularity);

  // Per-rank slot allocator. baseAddr is THIS rank's slot in the flat VA,
  // so vaManager->Allocate() returns a dereferenceable localVa directly.
  // flatBase + lsaRank*perRankSize is granularity-aligned (perRankSize is
  // 4 GiB-aligned) and non-zero (kernel-allocated VA), satisfying
  // HeapVAManager's invariants.
  comm->vaManager.reset(new application::HeapVAManager(LocalSlotBase(comm), perRankVmmSize, 0));

  // Step 4: SDMA queue setup. Materialize only if the user opted in
  // (MORI_ENABLE_SDMA) AND at least one peer has SDMA-capable hardware.
  bool anySdmaCapable = false;
  for (int pe = 0; pe < comm->worldSize; pe++) {
    if (comm->ctx->GetPeerCapabilities(pe).canSDMA) {
      anySdmaCapable = true;
      break;
    }
  }
  if (comm->ctx->IsSdmaEnabled() && anySdmaCapable) {
    comm->sdmaNumQueue = anvil::GetSdmaNumChannels();
    comm->ctx->EnsureSdmaTransport();

    // sdmaDevHandles is lsaSize × sdmaNumQueue, indexed by lsaRank. Assumes
    // ranks bind 1:1 to GPUs within a node (rank lsa ⇒ GPU lsa).
    int srcDeviceId = comm->hipDev;
    size_t numSlots = static_cast<size_t>(comm->lsaSize) * comm->sdmaNumQueue;
    HIP_RUNTIME_CHECK(
        hipMalloc(&comm->sdmaDevHandles, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));
    HIP_RUNTIME_CHECK(
        hipMemset(comm->sdmaDevHandles, 0, numSlots * sizeof(anvil::SdmaQueueDeviceHandle*)));

    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      int pe = comm->myNodeStart + lsa;
      if (!comm->ctx->GetPeerCapabilities(pe).canSDMA) continue;
      int dstDeviceId = lsa;
      for (int q = 0; q < comm->sdmaNumQueue; q++) {
        auto* handle = anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
        HIP_RUNTIME_CHECK(hipMemcpy(&comm->sdmaDevHandles[lsa * comm->sdmaNumQueue + q], &handle,
                                    sizeof(handle), hipMemcpyHostToDevice));
      }
    }
  } else {
    comm->sdmaNumQueue = 0;
  }

  // RDMA QP endpoints are NOT pre-allocated here. ccoDevCommCreate builds
  // a fresh QP set per DevComm via ctx->CreateAdditionalEndpoints, sized by
  // reqs.gdaContextCount, so multiple DevComms can coexist with independent
  // QP state.

  MORI_SHMEM_INFO(
      "ccoCommCreate: rank={}/{} groupId={} flatBase={} perRankSize={} "
      "granularity={} defaultNumQpPerPe={} sdmaNumQueue={} rdma={} fabric={}",
      comm->rank, comm->worldSize, comm->groupId, comm->flatBase, comm->perRankSize,
      comm->vmmGranularity, comm->defaultNumQpPerPe, comm->sdmaNumQueue,
      comm->ctx->RdmaTransportEnabled(),
      comm->handleType == static_cast<int>(hipMemHandleTypeFabricCompat));
  return 0;
}

/* ========================================================================== */
/*                             ccoCommDestroy                              */
/* ========================================================================== */

int ccoCommDestroy(ccoComm* comm) {
  if (!comm) return 0;

  MORI_SHMEM_TRACE("ccoCommDestroy: rank={}", comm->rank);

  // Safety net for callers that didn't pair every WindowRegister with a
  // matching Deregister: walk and properly deregister each straggler so
  // peer-imported handles, peer VA mappings, RDMA MRs, and GPU shadow
  // structs all get released. Each Deregister removes from comm->windows,
  // so iterate via .back() until empty.
  while (!comm->windows.empty()) {
    ccoWindowHost* wh = comm->windows.back();
    if (!wh || !wh->devPtr) {
      delete wh;
      comm->windows.pop_back();
      continue;
    }
    MORI_SHMEM_WARN(
        "ccoCommDestroy: window {} not deregistered by caller; "
        "auto-deregistering",
        wh->localPtr);
    (void)ccoWindowDeregister(comm, wh->devPtr);
  }

  // Safety net for callers that allocated symmetric memory but never freed it:
  // unmap + release each straggler (same as ccoMemFree) so no mappings remain
  // in the flat VA. hipMemAddressFree fails if any sub-range is still mapped.
  for (auto& [ptr, meta] : comm->allocTable) {
    vmmProcessLock vmmLock;
    (void)hipMemUnmap(ptr, meta.size);
    (void)hipMemRelease(meta.physHandle);
    if (!meta.isFabric && meta.shareFd >= 0) close(meta.shareFd);
  }
  comm->allocTable.clear();

  if (comm->sdmaDevHandles) HIP_RUNTIME_CHECK(hipFree(comm->sdmaDevHandles));

  // Release flat VA — sized to match the reservation in ccoCommCreate.
  if (comm->flatBase) {
    size_t totalVaSize = static_cast<size_t>(comm->lsaSize) * comm->perRankSize;
    vmmProcessLock vmmLock;
    HIP_RUNTIME_CHECK(hipMemAddressFree(comm->flatBase, totalVaSize));
  }

  delete comm->ctx;
  comm->bootNet->Finalize();
  delete comm->bootNet;

  delete comm;
  return 0;
}

/* ========================================================================== */
/*                              ccoMemAlloc                                */
/* ========================================================================== */

int ccoMemAlloc(ccoComm* comm, size_t size, void** outPtr) {
  if (outPtr == nullptr) {
    MORI_SHMEM_ERROR("ccoMemAlloc: outPtr is NULL");
    return -1;
  }
  if (size == 0) {
    *outPtr = nullptr;
    return 0;
  }

  size_t alignedSize = AlignUp(size, comm->vmmGranularity);

  // Reserve a slot via first-fit in the per-rank HeapVAManager. The returned
  // address IS the local VA for this rank's slot — directly dereferenceable.
  // 0 is the failure sentinel; baseAddr was set to flatBase + lsaRank*perRankSize
  // which is non-zero, so 0 unambiguously means failure.
  uintptr_t slotAddr = comm->vaManager->Allocate(alignedSize, comm->vmmGranularity);
  if (slotAddr == 0) {
    MORI_SHMEM_ERROR(
        "ccoMemAlloc: slot exhausted (no contiguous {} bytes free in perRankSize={}). "
        "Increase perRankVmmSize at ccoCommCreate or free unused allocations.",
        alignedSize, comm->perRankSize);
    return -1;
  }
  // slotOffset is the offset within the rank's perRankSize slot; needed for
  // peer-VA computation (peer's localVa = flatBase + peerLsaRank*stride + slotOffset).
  size_t slotOffset = static_cast<size_t>(slotAddr - LocalSlotBase(comm));

  MORI_SHMEM_TRACE("ccoMemAlloc: rank={} size={} alignedSize={} slotOffset={}", comm->rank, size,
                   alignedSize, slotOffset);

  // Return the reserved slot to the vaManager on any failure after this point.
  auto rollbackSlot = [&]() { (void)comm->vaManager->Free(slotAddr); };

  const bool useFabric = (comm->handleType == static_cast<int>(hipMemHandleTypeFabricCompat));
  hipMemAllocationProp allocProp = {};
  allocProp.type = CcoWindowAllocType();
  allocProp.requestedHandleType = static_cast<hipMemAllocationHandleType>(comm->handleType);
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = comm->hipDev;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  allocProp.allocFlags.gpuDirectRDMACapable = 1;
#endif

  hipMemGenericAllocationHandle_t physHandle = 0;
  hipError_t err = hipSuccess;
  void* localVa = reinterpret_cast<void*>(slotAddr);
  {
    vmmProcessLock vmmLock;
    err = hipMemCreate(&physHandle, alignedSize, &allocProp, 0);
    if (err != hipSuccess) {
      MORI_SHMEM_ERROR("ccoMemAlloc: hipMemCreate failed: {} ({})", static_cast<int>(err),
                       hipGetErrorString(err));
      rollbackSlot();
      return -1;
    }

    err = hipMemMap(localVa, alignedSize, 0, physHandle, 0);
    if (err != hipSuccess) {
      MORI_SHMEM_ERROR("ccoMemAlloc: hipMemMap failed: {} ({})", static_cast<int>(err),
                       hipGetErrorString(err));
      (void)hipMemRelease(physHandle);
      rollbackSlot();
      return -1;
    }
  }

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = comm->hipDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int retry = 0; retry < 5; retry++) {
    {
      vmmProcessLock vmmLock;
      err = hipMemSetAccess(localVa, alignedSize, &accessDesc, 1);
    }
    if (err == hipSuccess) break;
    usleep(1000 * (1 << retry));
  }
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemSetAccess failed after retries: {} ({})",
                     static_cast<int>(err), hipGetErrorString(err));
    {
      vmmProcessLock vmmLock;
      (void)hipMemUnmap(localVa, alignedSize);
      (void)hipMemRelease(physHandle);
    }
    rollbackSlot();
    return -1;
  }

  // Export a shareable handle for WindowRegister (P2P mapping + RDMA MR).
  // Fabric path: 64-byte token (exchangeable via Allgather — no Unix sockets).
  // FD path: dma-buf FD (exchanged via LocalBootstrapNetwork + SCM_RIGHTS).
  ccoComm::AllocMeta meta;
  meta.physHandle = physHandle;
  meta.isFabric = useFabric;
  meta.slotOffset = slotOffset;
  meta.size = alignedSize;

  if (useFabric) {
    err = hipMemExportToShareableHandle(&meta.fabricHandle, physHandle,
                                        hipMemHandleTypeFabricCompat, 0);
  } else {
    meta.shareFd = -1;
    err = hipMemExportToShareableHandle(reinterpret_cast<void*>(&meta.shareFd), physHandle,
                                        hipMemHandleTypePosixFileDescriptor, 0);
  }
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("ccoMemAlloc: hipMemExportToShareableHandle failed: {} ({})",
                     static_cast<int>(err), hipGetErrorString(err));
    {
      vmmProcessLock vmmLock;
      (void)hipMemUnmap(localVa, alignedSize);
      (void)hipMemRelease(physHandle);
    }
    rollbackSlot();
    return -1;
  }
  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    comm->allocTable[localVa] = meta;
  }

  *outPtr = localVa;
  MORI_SHMEM_TRACE("ccoMemAlloc: done, localPtr={}", localVa);
  return 0;
}

/* ========================================================================== */
/*                              ccoMemFree                                 */
/* ========================================================================== */

int ccoMemFree(ccoComm* comm, void* ptr) {
  if (ptr == nullptr) return 0;

  // Snapshot meta + return the slot to vaManager, then drop the cco mutex
  // before the (potentially slow) hipMem* calls so concurrent MemAlloc
  // isn't blocked. vaManager->Free takes its own mutex internally.
  ccoComm::AllocMeta meta;
  {
    std::lock_guard<std::mutex> lock(comm->allocMutex);
    auto it = comm->allocTable.find(ptr);
    if (it == comm->allocTable.end()) {
      MORI_SHMEM_WARN("ccoMemFree: ptr {} not found", ptr);
      return -1;
    }
    meta = it->second;
    comm->allocTable.erase(it);
  }
  // ptr == LocalSlotBase(comm) + meta.slotOffset == the address vaManager handed out.
  (void)comm->vaManager->Free(reinterpret_cast<uintptr_t>(ptr));

  size_t alignedSize = meta.size;

  MORI_SHMEM_TRACE("ccoMemFree: rank={} ptr={} size={}", comm->rank, ptr, alignedSize);

  {
    vmmProcessLock vmmLock;
    hipError_t err = hipMemUnmap(ptr, alignedSize);
    if (err != hipSuccess) {
      MORI_SHMEM_WARN("ccoMemFree: local hipMemUnmap failed: {} ({})", static_cast<int>(err),
                      hipGetErrorString(err));
    }
    err = hipMemRelease(meta.physHandle);
    if (err != hipSuccess) {
      MORI_SHMEM_WARN("ccoMemFree: hipMemRelease failed: {} ({})", static_cast<int>(err),
                      hipGetErrorString(err));
    }
  }

  if (!meta.isFabric && meta.shareFd >= 0) close(meta.shareFd);

  return 0;
}

/* ========================================================================== */
/*                         ccoWindowRegister (ptr)                         */
/* ========================================================================== */

int ccoWindowRegister(ccoComm* comm, void* ptr, size_t size, ccoWindow_t* outWin) {
  auto it = comm->allocTable.find(ptr);
  if (it == comm->allocTable.end()) {
    MORI_SHMEM_ERROR("ccoWindowRegister: ptr {} not in allocTable", ptr);
    return -1;
  }

  auto& meta = it->second;
  size_t slotOffset = meta.slotOffset;
  void* localPtr = ptr;
  int worldSize = comm->worldSize;
  int rank = comm->rank;
  const bool useFabric = meta.isFabric;

  size_t alignedSize = meta.size;

  MORI_SHMEM_TRACE("ccoWindowRegister: rank={} ptr={} size={} slotOffset={} fabric={}", rank, ptr,
                   size, slotOffset, useFabric);

  // P2P imported handles — collected during the exchange loop below,
  // ownership later transferred to ccoWindowHost so Deregister can release.
  std::vector<hipMemGenericAllocationHandle_t> p2pImportedHandles;

  // P2P: exchange handles with same-node peers and map their slots into
  // the LSA flat VA.
  std::vector<int> p2pPeers;
  for (int pe = 0; pe < worldSize; pe++) {
    if (CcoCanLsaMapPeer(comm, pe)) {
      p2pPeers.push_back(pe);
    }
  }

  if (!p2pPeers.empty()) {
    // Common peer-mapping helpers shared by both fabric and FD paths.
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = comm->hipDev;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    struct MappedPeer {
      hipMemGenericAllocationHandle_t handle;
      void* peerVa;
    };
    std::vector<MappedPeer> mappedPeers;
    mappedPeers.reserve(p2pPeers.size());

    auto rollbackMappedPeers = [&]() {
      vmmProcessLock vmmLock;
      for (auto& mp : mappedPeers) {
        (void)hipMemUnmap(mp.peerVa, alignedSize);
        (void)hipMemRelease(mp.handle);
      }
      mappedPeers.clear();
    };

    // Import + map a single peer's handle into our flat VA. Returns 0 on
    // success, -1 on failure (rolls back all prior mappings).
    auto mapPeer = [&](int pe, hipMemGenericAllocationHandle_t importedHandle) -> int {
      int peerLsaRank = CcoPeToLsaRank(comm, pe);
      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(peerLsaRank) * comm->perRankSize + slotOffset;
      const bool crossNodePeer =
          comm->fabricCrossNodeLsa && !comm->ctx->GetPeerCapabilities(pe).sameHost;
      // hipDeviceEnablePeerAccess is intra-node only; cross-node fabric P2P
      // does not use this API (peerDev indices are local and collide across nodes).
      if (!crossNodePeer && pe < static_cast<int>(comm->peerHipDevs.size())) {
        int peerDev = comm->peerHipDevs[pe];
        if (peerDev != comm->hipDev) {
          hipError_t peerErr = hipDeviceEnablePeerAccess(peerDev, 0);
          // Consume the sticky last-error: hipDeviceEnablePeerAccess leaves the
          // runtime's last-error set (e.g. AlreadyEnabled / NotSupported), which
          // a later torch op would pick up via hipGetLastError() and raise as
          // "operation not supported" (observed on gfx950). Mirrors the sibling
          // ccoDevCommCreate call site.
          (void)hipGetLastError();
          if (peerErr != hipSuccess && peerErr != hipErrorPeerAccessAlreadyEnabled) {
            MORI_SHMEM_WARN("ccoWindowRegister: hipDeviceEnablePeerAccess PE {} dev {} failed: {}",
                            pe, peerDev, static_cast<int>(peerErr));
          }
        }
      }
      {
        vmmProcessLock vmmLock;
        hipError_t mapErr = hipMemMap(peerVa, alignedSize, 0, importedHandle, 0);
        if (mapErr != hipSuccess) {
          MORI_SHMEM_ERROR("ccoWindowRegister: hipMemMap PE {} failed: {}", pe,
                           static_cast<int>(mapErr));
          (void)hipMemRelease(importedHandle);
          rollbackMappedPeers();
          return -1;
        }
      }
      hipError_t setErr = hipSuccess;
      for (int retry = 0; retry < 5; retry++) {
        {
          vmmProcessLock vmmLock;
          setErr = hipMemSetAccess(peerVa, alignedSize, &accessDesc, 1);
        }
        if (setErr == hipSuccess) break;
        usleep(1000 * (1 << retry));
      }
      if (setErr != hipSuccess) {
        if (crossNodePeer && CcoFabricLenient()) {
          // Opt-in leniency: some fabrics grant load/store access at import time
          // and return non-success from hipMemSetAccess. Trust the mapping.
          MORI_SHMEM_WARN(
              "ccoWindowRegister: hipMemSetAccess PE {} cross-node failed: {} "
              "(MORI_CCO_FABRIC_LENIENT=1 — continuing, import may already grant access)",
              pe, static_cast<int>(setErr));
        } else {
          // Fail loudly at register time rather than leaving a peer VA that
          // faults on first device access. For cross-node this most likely means
          // the fabric is not a scale-up (load/store) domain.
          MORI_SHMEM_ERROR("ccoWindowRegister: hipMemSetAccess PE {} failed after retries: {}{}",
                           pe, static_cast<int>(setErr),
                           crossNodePeer ? " (cross-node fabric not load/store-reachable?)" : "");
          {
            vmmProcessLock vmmLock;
            (void)hipMemUnmap(peerVa, alignedSize);
            (void)hipMemRelease(importedHandle);
          }
          rollbackMappedPeers();
          return -1;
        }
      }
      mappedPeers.push_back({importedHandle, peerVa});
      return 0;
    };

    if (useFabric) {
      // Fabric path: Allgather 64-byte fabric handles via the bootstrap
      // network — no Unix sockets, no SCM_RIGHTS.
      std::vector<hipMemFabricHandle_compat_t> allFabricHandles(worldSize);
      comm->bootNet->Allgather(&meta.fabricHandle, allFabricHandles.data(),
                               sizeof(hipMemFabricHandle_compat_t));

      for (int pe : p2pPeers) {
        hipMemGenericAllocationHandle_t importedHandle;
        hipError_t err;
        {
          // Import must share the VMM serialization: ROCr races on concurrent
          // import vs map/setaccess across SPMT threads -> hipMemSetAccess
          // InvalidValue. #455 serialized map/setaccess but not import.
          vmmProcessLock vmmLock;
          err = hipMemImportFromShareableHandle(&importedHandle, &allFabricHandles[pe],
                                                hipMemHandleTypeFabricCompat);
        }
        if (err != hipSuccess) {
          MORI_SHMEM_ERROR("ccoWindowRegister: fabric import from PE {} failed: {}", pe,
                           static_cast<int>(err));
          rollbackMappedPeers();
          return -1;
        }
        if (mapPeer(pe, importedHandle) != 0) return -1;
      }
    } else {
      // FD path (fallback): exchange dma-buf FDs via LocalBootstrapNetwork +
      // SCM_RIGHTS (one Unix socket pair per peer).
      int shareFd = meta.shareFd;

      std::vector<int> sortedGroup = p2pPeers;
      sortedGroup.push_back(rank);
      std::sort(sortedGroup.begin(), sortedGroup.end());

      int myPeerRank = 0;
      for (int i = 0; i < static_cast<int>(sortedGroup.size()); i++) {
        if (sortedGroup[i] == rank) {
          myPeerRank = i;
          break;
        }
      }
      int p2pWorldSize = static_cast<int>(sortedGroup.size());

      std::string socketPath = "/tmp/mori_cco_" + std::to_string(comm->groupId) + "_" +
                               std::to_string(slotOffset) + "_g" + std::to_string(sortedGroup[0]) +
                               "_";

      application::LocalBootstrapNetwork localBoot(myPeerRank, p2pWorldSize, socketPath);
      localBoot.Initialize();

      std::vector<int> myFds = {shareFd};
      std::vector<std::vector<int>> allFds;
      if (!localBoot.ExchangeFileDescriptors(myFds, allFds)) {
        MORI_SHMEM_ERROR("ccoWindowRegister: P2P FD exchange failed");
        localBoot.Finalize();
        return -1;
      }

      auto closePeerFds = [&]() {
        for (int i = 0; i < static_cast<int>(allFds.size()); i++) {
          if (i == myPeerRank) continue;
          for (int fd : allFds[i]) {
            if (fd >= 0) close(fd);
          }
        }
        allFds.clear();
      };

      std::vector<int> globalToPeer(worldSize, -1);
      for (int i = 0; i < p2pWorldSize; i++) {
        globalToPeer[sortedGroup[i]] = i;
      }

      auto bailFd = [&]() {
        rollbackMappedPeers();
        closePeerFds();
        localBoot.Finalize();
      };

      for (int pe : p2pPeers) {
        int pr = globalToPeer[pe];
        if (pr < 0 || pr >= static_cast<int>(allFds.size())) {
          MORI_SHMEM_ERROR("ccoWindowRegister: PE {} missing in FD exchange result", pe);
          bailFd();
          return -1;
        }
        int peerFd = allFds[pr][0];
        if (peerFd < 0) {
          MORI_SHMEM_ERROR("ccoWindowRegister: PE {} delivered invalid FD ({})", pe, peerFd);
          bailFd();
          return -1;
        }

        hipMemGenericAllocationHandle_t importedHandle;
        hipError_t err;
        {
          vmmProcessLock vmmLock;  // serialize import w/ map/setaccess (SPMT race)
          err = hipMemImportFromShareableHandleCompat(&importedHandle, peerFd,
                                                      hipMemHandleTypePosixFileDescriptor);
        }
        if (err != hipSuccess) {
          MORI_SHMEM_ERROR("ccoWindowRegister: import from PE {} failed: {}", pe,
                           static_cast<int>(err));
          bailFd();
          return -1;
        }
        if (mapPeer(pe, importedHandle) != 0) {
          closePeerFds();
          localBoot.Finalize();
          return -1;
        }
      }

      closePeerFds();
      localBoot.Finalize();
    }

    p2pImportedHandles.reserve(mappedPeers.size());
    for (auto& mp : mappedPeers) p2pImportedHandles.push_back(mp.handle);
  }

  // RDMA MR registration + rkey Allgather.
  uint32_t lkey = 0;
  uint32_t localRkey = 0;

  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx) {
    application::RdmaMemoryRegion mr;
    if (useFabric) {
      mr = rdmaDevCtx->RegisterRdmaMemoryRegionAuto(localPtr, size);
    } else if (comm->iovaZeroMode) {
      mr = rdmaDevCtx->RegisterRdmaMemoryRegionDmabufIova0(localPtr, size, meta.shareFd);
    } else {
      mr = rdmaDevCtx->RegisterRdmaMemoryRegionDmabuf(localPtr, size, meta.shareFd);
    }
    lkey = mr.lkey;
    localRkey = mr.rkey;
  }

  // Allgather rkeys into a std::vector so an exception in Allgather doesn't
  // leak the host buffer (HIP_RUNTIME_CHECKs below abort the process anyway,
  // but bootNet->Allgather is throwing).
  std::vector<uint32_t> peerRkeys_host(worldSize, 0);
  peerRkeys_host[rank] = localRkey;
  comm->bootNet->Allgather(&localRkey, peerRkeys_host.data(), sizeof(uint32_t));

  // SDMA signal pool is per-DevComm (materialized by ccoDevCommCreate); kernels
  // look up signals via devComm->sdma.

  uint32_t* peerRkeys_gpu = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&peerRkeys_gpu, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(peerRkeys_gpu, peerRkeys_host.data(), sizeof(uint32_t) * worldSize,
                              hipMemcpyHostToDevice));

  ccoWindowDevice hostShadow = {};
  hostShadow.winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  hostShadow.stride4G = static_cast<uint32_t>(comm->perRankSize >> 32);
  hostShadow.lsaRank = comm->lsaRank;
  hostShadow.ibgdaWin.peerRkeys = peerRkeys_gpu;
  hostShadow.ibgdaWin.lkey = lkey;

  ccoWindowDevice* devPtr = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&devPtr, sizeof(ccoWindowDevice)));
  HIP_RUNTIME_CHECK(hipMemcpy(devPtr, &hostShadow, sizeof(ccoWindowDevice), hipMemcpyHostToDevice));

  // Publish into the per-comm window table (drives findWindow lookups).
  ccoComm::WindowTableEntry tableEntry;
  tableEntry.base = reinterpret_cast<uintptr_t>(localPtr);
  tableEntry.size = static_cast<uintptr_t>(size);
  tableEntry.devPtr = devPtr;
  comm->windowTableEntries.push_back(tableEntry);

  auto* wh = new ccoWindowHost();
  wh->localPtr = localPtr;
  wh->size = size;
  wh->devPtr = devPtr;
  wh->peerRkeys_gpu = peerRkeys_gpu;
  wh->peerImportedHandles = std::move(p2pImportedHandles);
  comm->windows.push_back(wh);

  *outWin = devPtr;

  char* winBase = static_cast<char*>(comm->flatBase) + slotOffset;
  MORI_SHMEM_INFO(
      "ccoWindowRegister: rank={} win={} winBase={} size={} slotOffset={} lkey={} fabric={}", rank,
      (void*)devPtr, (void*)winBase, size, slotOffset, lkey, useFabric);
  for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
    int pe = comm->myNodeStart + lsa;
    void* peerVa = winBase + static_cast<size_t>(lsa) * comm->perRankSize;
    MORI_SHMEM_INFO("  LSA[{}] (PE {}): flatVA={} rkey={}", lsa, pe, peerVa, peerRkeys_host[pe]);
  }
  // Peers outside the LSA team (different vPOD / host) are reached via RDMA.
  for (int pe = 0; pe < worldSize; pe++) {
    if (pe >= comm->myNodeStart && pe < comm->myNodeStart + comm->lsaSize) continue;
    MORI_SHMEM_INFO("  XNODE PE {}: rkey={} (RDMA via iova=0)", pe, peerRkeys_host[pe]);
  }
  // peerRkeys_host is std::vector — destructs cleanly at scope exit.

  return 0;
}

/* ========================================================================== */
/*                      ccoWindowRegister (convenience)                    */
/* ========================================================================== */

int ccoWindowRegister(ccoComm* comm, size_t size, ccoWindow_t* outWin, void** localPtr) {
  void* ptr = nullptr;
  int ret = ccoMemAlloc(comm, size, &ptr);
  if (ret != 0) return ret;

  ret = ccoWindowRegister(comm, ptr, size, outWin);
  if (ret != 0) {
    ccoMemFree(comm, ptr);
    return ret;
  }

  *localPtr = ptr;
  return 0;
}

/* ========================================================================== */
/*                          ccoWindowDeregister                            */
/* ========================================================================== */

int ccoWindowDeregister(ccoComm* comm, ccoWindow_t win) {
  ccoWindowHost* wh = nullptr;
  size_t idx = 0;
  for (size_t i = 0; i < comm->windows.size(); i++) {
    if (comm->windows[i]->devPtr == win) {
      wh = comm->windows[i];
      idx = i;
      break;
    }
  }
  if (!wh) {
    MORI_SHMEM_WARN("ccoWindowDeregister: win {} not found", (void*)win);
    return -1;
  }

  MORI_SHMEM_TRACE("ccoWindowDeregister: rank={} ptr={}", comm->rank, wh->localPtr);

  // Unmap the P2P peer slots that WindowRegister mapped (ENOMAP is fine).
  auto allocIt = comm->allocTable.find(wh->localPtr);
  if (allocIt != comm->allocTable.end()) {
    size_t slotOff = allocIt->second.slotOffset;
    size_t allocSize = allocIt->second.size;
    vmmProcessLock vmmLock;
    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      if (lsa == comm->lsaRank) continue;
      int pe = comm->myNodeStart + lsa;  // global pe (matches register's p2pPeers)
      if (!CcoCanLsaMapPeer(comm, pe)) continue;
      void* peerVa = static_cast<char*>(comm->flatBase) +
                     static_cast<size_t>(lsa) * comm->perRankSize + slotOff;
      (void)hipMemUnmap(peerVa, allocSize);
    }
  }

  // Drop refcount on each peer's imported handle. hipMemUnmap above
  // detaches VA mappings but doesn't release the handle itself.
  {
    vmmProcessLock vmmLock;
    for (auto handle : wh->peerImportedHandles) {
      (void)hipMemRelease(handle);
    }
  }
  wh->peerImportedHandles.clear();

  auto& entries = comm->windowTableEntries;
  entries.erase(
      std::remove_if(entries.begin(), entries.end(),
                     [win](const ccoComm::WindowTableEntry& e) { return e.devPtr == win; }),
      entries.end());

  application::RdmaDeviceContext* rdmaDevCtx = comm->ctx->GetRdmaDeviceContext();
  if (rdmaDevCtx) rdmaDevCtx->DeregisterRdmaMemoryRegion(wh->localPtr);

  if (wh->peerRkeys_gpu) HIP_RUNTIME_CHECK(hipFree(wh->peerRkeys_gpu));
  if (wh->devPtr) HIP_RUNTIME_CHECK(hipFree(wh->devPtr));

  comm->windows.erase(comm->windows.begin() + idx);
  delete wh;
  return 0;
}

/* ========================================================================== */
/*                            ccoDevCommCreate                             */
/* ========================================================================== */

int ccoDevCommCreate(ccoComm* comm, const ccoDevCommRequirements* reqs, ccoDevComm* outDevComm) {
  MORI_SHMEM_TRACE("ccoDevCommCreate: rank={}", comm->rank);

  // Forward-compat: validate {magic, version}.
  if (reqs == nullptr) {
    MORI_SHMEM_ERROR(
        "ccoDevCommCreate: reqs is NULL — must initialize via "
        "CCO_DEV_COMM_REQUIREMENTS_INITIALIZER");
    return -1;
  }
  if (reqs->magic != CCO_API_MAGIC) {
    MORI_SHMEM_ERROR(
        "ccoDevCommCreate: reqs->magic mismatch (got {:#x}, expect {:#x}) — "
        "must initialize via CCO_DEV_COMM_REQUIREMENTS_INITIALIZER",
        reqs->magic, CCO_API_MAGIC);
    return -1;
  }
  if (reqs->version > CCO_API_VERSION) {
    MORI_SHMEM_WARN("ccoDevCommCreate: reqs->version={} > runtime CCO_API_VERSION={}",
                    reqs->version, CCO_API_VERSION);
  }

  // Resolve connection type. CROSSNODE collapses to NONE on single-node
  // deployments (no cross-node peers exist). RAIL collapses to NONE if it
  // ends up with zero peers (single-node, or self-rail only).
  ccoGdaConnectionType connType = reqs->gdaConnectionType;
  if (connType == CCO_GDA_CONNECTION_CROSSNODE && comm->lsaSize == comm->worldSize) {
    MORI_SHMEM_TRACE("ccoDevCommCreate: single-node, CROSSNODE -> NONE");
    connType = CCO_GDA_CONNECTION_NONE;
  }

  ccoDevComm hostShadow = {};
  hostShadow.rank = comm->rank;
  hostShadow.worldSize = comm->worldSize;
  hostShadow.lsaSize = comm->lsaSize;
  hostShadow.lsaRank = comm->lsaRank;
  hostShadow.myNodeStart = comm->myNodeStart;
  hostShadow.gdaConnType = connType;
  hostShadow.flatBase = comm->flatBase;
  hostShadow.perRankSize = comm->perRankSize;

  // Fresh QP set per DevComm.
  ccoIbgdaContext& ibgda = hostShadow.ibgda;
  int numQpPerPe = reqs->gdaContextCount > 0 ? reqs->gdaContextCount : comm->defaultNumQpPerPe;
  ibgda.numQpPerPe = numQpPerPe;

  size_t numEps = static_cast<size_t>(comm->worldSize) * numQpPerPe;
  core::RdmaEndpointDevice* epsGpu = nullptr;

  // Build the peer mask once based on connType. Context::CreateAdditional /
  // ConnectAdditional take the same mask. Empty mask if NONE.
  //
  // Layout assumption (HARD CONTRACT enforced at CommCreate): ranks are
  // node-major contiguous and lsaSize is uniform across nodes, so each
  // peer's lsaRank is `peer % comm->lsaSize`.
  std::vector<bool> peerMask;
  if (connType != CCO_GDA_CONNECTION_NONE) {
    peerMask.assign(comm->worldSize, false);
    for (int peer = 0; peer < comm->worldSize; peer++) {
      if (peer == comm->rank) continue;
      const auto& cap = comm->ctx->GetPeerCapabilities(peer);
      switch (connType) {
        case CCO_GDA_CONNECTION_FULL:
          peerMask[peer] = cap.canRDMA;
          break;
        case CCO_GDA_CONNECTION_CROSSNODE:
          peerMask[peer] = cap.canRDMA && !cap.sameHost;
          break;
        case CCO_GDA_CONNECTION_RAIL: {
          const int myLsaRank = comm->lsaRank;
          const int peerLsaRank = peer % comm->lsaSize;
          peerMask[peer] = cap.canRDMA && !cap.sameHost && (peerLsaRank == myLsaRank);
          break;
        }
        default:
          break;
      }
    }
    // Collapse to NONE if the resolved mask is empty (e.g. RAIL on single
    // node, or CROSSNODE that lost all peers).
    if (std::none_of(peerMask.begin(), peerMask.end(), [](bool b) { return b; })) {
      MORI_SHMEM_TRACE(
          "ccoDevCommCreate: resolved peer mask is empty, "
          "downgrading connType {} -> NONE",
          static_cast<int>(connType));
      connType = CCO_GDA_CONNECTION_NONE;
      peerMask.clear();
    }
    hostShadow.gdaConnType = connType;  // may have been collapsed above
  }

  if (connType != CCO_GDA_CONNECTION_NONE && comm->ctx->RdmaTransportEnabled()) {
    // Collective: every rank must call CreateAdditionalEndpoints together.
    auto newEps = comm->ctx->CreateAdditionalEndpoints(numQpPerPe, peerMask);
    comm->ctx->ConnectAdditionalEndpoints(newEps, numQpPerPe, peerMask);

    // Note: post-Connect RTS verification via ibv_query_qp doesn't work for
    // direct-verbs providers (bnxt, mlx5), which keep QPs in their own
    // containers and leave ibvHandle.qp null. Provider-side QueryQpState is
    // a TODO; for now, rely on modify_qp's internal check + the transport
    // map dump (MORI_CCO_LOG_TRANSPORT) for visibility.

    std::vector<core::RdmaEndpointDevice> epsHost(numEps);
    for (size_t i = 0; i < numEps; i++) {
      epsHost[i].vendorId = newEps[i].vendorId;
      epsHost[i].qpn = newEps[i].handle.qpn;
      epsHost[i].wqHandle = newEps[i].wqHandle;
      epsHost[i].cqHandle = newEps[i].cqHandle;
      epsHost[i].atomicIbuf = newEps[i].atomicIbuf;
      // Cache the GDA provider from the first connected endpoint (empty peer
      // slots keep vendorId==Unknown) as an informational parameter on the comm.
      if (comm->providerType == CCO_PROVIDER_UNKNOWN) {
        core::ProviderType p = epsHost[i].GetProviderType();
        if (p != core::ProviderType::Unknown) comm->providerType = static_cast<ccoProviderType>(p);
      }
    }

    HIP_RUNTIME_CHECK(hipMalloc(&epsGpu, numEps * sizeof(core::RdmaEndpointDevice)));
    HIP_RUNTIME_CHECK(hipMemcpy(epsGpu, epsHost.data(), numEps * sizeof(core::RdmaEndpointDevice),
                                hipMemcpyHostToDevice));
  }
  ibgda.endpoints = epsGpu;

  // Resource window backing this DevComm's session state (IBGDA
  // signal/shadows/counter pool, LSA barrier inbox+state). signalBufOffset is
  // pinned to 0 so a peer's RDMA atomic-add uses raddr = signal_slot_id * 8.
  // Allocated before the windowTable build below so findWindow() sees it.
  // GDA-Rail barriers are only usable with cross-node peers AND RDMA QPs.
  int nNodes = comm->worldSize / comm->lsaSize;
  bool gdaRailUsable = (connType != CCO_GDA_CONNECTION_NONE) && (nNodes > 1);

  int signalCountUser = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaSignalCount;
  int counterCount = (connType == CCO_GDA_CONNECTION_NONE) ? 0 : reqs->gdaCounterCount;
  int lsaBarrierCount = reqs->lsaBarrierCount;
  int railGdaBarrierCount = gdaRailUsable ? reqs->railGdaBarrierCount : 0;
  int hybridBarrierCount = reqs->barrierCount;
  // hybrid Rail half is only active when we have cross-rail peers + RDMA.
  int hybridRailBarrierCount = gdaRailUsable ? hybridBarrierCount : 0;

  // Signal slot assignment:
  //   [0 .. signalCountUser)                 — user-visible signal slots
  //   [signalCountUser .. +A)                — railGdaBarrier (A = N*nNodes)
  //   [.. +B)                                — hybridRailGdaBarrier (B = N*nNodes)
  uint32_t railGdaBarrierSignal0 = static_cast<uint32_t>(signalCountUser);
  int railGdaBarrierSignals = railGdaBarrierCount * nNodes;
  uint32_t hybridRailBarrierSignal0 =
      railGdaBarrierSignal0 + static_cast<uint32_t>(railGdaBarrierSignals);
  int hybridRailBarrierSignals = hybridRailBarrierCount * nNodes;
  int signalCount = signalCountUser + railGdaBarrierSignals + hybridRailBarrierSignals;
  ibgda.signalCount = signalCount;
  ibgda.counterCount = counterCount;

  auto alignTo = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  auto lsaBarBytes = [&](int n) -> size_t {
    // Multimem epoch/inbox omitted; add when hardware support lands.
    return static_cast<size_t>(n + n * comm->lsaSize) * sizeof(uint32_t);
  };

  struct ResourceWindowLayout {
    size_t signalBufOffset = 0;
    size_t signalShadowsOffset = 0;
    size_t counterBufOffset = 0;
    size_t lsaBarrierOffset = 0;
    size_t lsaBarrierBytes = 0;
    size_t hybridLsaBarrierOffset = 0;
    size_t hybridLsaBarrierBytes = 0;
    size_t totalSize = 0;
  } layout;
  if (lsaBarrierCount > 0) layout.lsaBarrierBytes = lsaBarBytes(lsaBarrierCount);
  if (hybridBarrierCount > 0) layout.hybridLsaBarrierBytes = lsaBarBytes(hybridBarrierCount);

  bool needWindow =
      signalCount > 0 || counterCount > 0 || lsaBarrierCount > 0 || hybridBarrierCount > 0;
  if (needWindow) {
    size_t off = 0;
    layout.signalBufOffset = off;  // pinned at 0
    off += static_cast<size_t>(signalCount) * sizeof(uint64_t);
    off = alignTo(off, 8);
    layout.signalShadowsOffset = off;
    off += static_cast<size_t>(signalCount) * sizeof(uint64_t);
    off = alignTo(off, 8);
    layout.counterBufOffset = off;
    off += static_cast<size_t>(counterCount) * sizeof(uint64_t);
    // LSA barrier slabs: 128B align so peers' P2P stores hit a cache-line-
    // isolated region.
    if (lsaBarrierCount > 0) {
      off = alignTo(off, 128);
      layout.lsaBarrierOffset = off;
      off += layout.lsaBarrierBytes;
    }
    if (hybridBarrierCount > 0) {
      off = alignTo(off, 128);
      layout.hybridLsaBarrierOffset = off;
      off += layout.hybridLsaBarrierBytes;
    }
    layout.totalSize = off;
  }

  void* resourceWindowPtr = nullptr;
  ccoWindow_t resourceWindow = nullptr;
  if (layout.totalSize > 0) {
    if (ccoMemAlloc(comm, layout.totalSize, &resourceWindowPtr) != 0) {
      MORI_SHMEM_ERROR("ccoDevCommCreate: resource window MemAlloc failed");
      if (epsGpu) HIP_RUNTIME_CHECK(hipFree(epsGpu));
      return -1;
    }
    // resourceWindowPtr is fabric-exportable (ccoMemAlloc) — avoid hipMemset,
    // which returns OOM on UALink fabric pools (ROCm 7.15).
    HIP_RUNTIME_CHECK(CcoZeroWindowMem(resourceWindowPtr, layout.totalSize));
    if (ccoWindowRegister(comm, resourceWindowPtr, layout.totalSize, &resourceWindow) != 0) {
      MORI_SHMEM_ERROR("ccoDevCommCreate: resource window Register failed");
      (void)ccoMemFree(comm, resourceWindowPtr);
      if (epsGpu) HIP_RUNTIME_CHECK(hipFree(epsGpu));
      return -1;
    }
    auto* base = static_cast<uint8_t*>(resourceWindowPtr);
    if (signalCount > 0) {
      ibgda.signalBuf = reinterpret_cast<uint64_t*>(base + layout.signalBufOffset);
      ibgda.signalShadows = reinterpret_cast<uint64_t*>(base + layout.signalShadowsOffset);
    }
    if (counterCount > 0) {
      ibgda.counterBuf = reinterpret_cast<uint64_t*>(base + layout.counterBufOffset);
    }
    if (lsaBarrierCount > 0) {
      hostShadow.lsaBarrier.bufOffset = static_cast<uint32_t>(layout.lsaBarrierOffset);
      hostShadow.lsaBarrier.nBarriers = lsaBarrierCount;
    }
    if (hybridBarrierCount > 0) {
      hostShadow.hybridLsaBarrier.bufOffset = static_cast<uint32_t>(layout.hybridLsaBarrierOffset);
      hostShadow.hybridLsaBarrier.nBarriers = hybridBarrierCount;
    }

    // Snapshot the GPU resource-window struct into the DevComm so kernels
    // can read winBase/stride4G/ibgdaWin.{lkey,peerRkeys} straight out of
    // kernel cmem (no extra GPU memory load through the pointer).
    HIP_RUNTIME_CHECK(hipMemcpy(&hostShadow.resourceWindow_inlined, resourceWindow,
                                sizeof(ccoWindowDevice), hipMemcpyDeviceToHost));
  }
  hostShadow.resourceWindow = resourceWindow;

  // GDA barrier handles point into ibgda.signalBuf; no resource-window bytes
  // consumed. Disabled handles stay {0,0}.
  if (railGdaBarrierCount > 0) {
    hostShadow.railGdaBarrier.signal0 = railGdaBarrierSignal0;
    hostShadow.railGdaBarrier.nBarriers = railGdaBarrierCount;
  }
  if (hybridRailBarrierCount > 0) {
    hostShadow.hybridRailGdaBarrier.signal0 = hybridRailBarrierSignal0;
    hostShadow.hybridRailGdaBarrier.nBarriers = hybridRailBarrierCount;
  }

  MORI_SHMEM_TRACE(
      "ccoDevCommCreate: resourceWindow={} ptr={} totalSize={} signals={} "
      "counters={} lsaBar={} lsaBarOff={:#x} hybLsaBar={} hybLsaBarOff={:#x} "
      "railGdaBar={} railGdaSig0={} hybRailGdaBar={} hybRailGdaSig0={}",
      (void*)resourceWindow, resourceWindowPtr, layout.totalSize, signalCount, counterCount,
      lsaBarrierCount, layout.lsaBarrierOffset, hybridBarrierCount, layout.hybridLsaBarrierOffset,
      railGdaBarrierCount, railGdaBarrierSignal0, hybridRailBarrierCount, hybridRailBarrierSignal0);

  // Build window-table linked list on GPU.
  const auto& tableEntries = comm->windowTableEntries;
  size_t numWindows = tableEntries.size();
  size_t numNodes = (numWindows + CCO_WINDOW_TABLE_SIZE - 1) / CCO_WINDOW_TABLE_SIZE;
  if (numNodes == 0) numNodes = 1;

  std::vector<ccoWindowTableNode*> gpuNodes(numNodes, nullptr);
  for (size_t n = 0; n < numNodes; n++) {
    HIP_RUNTIME_CHECK(hipMalloc(&gpuNodes[n], sizeof(ccoWindowTableNode)));
    HIP_RUNTIME_CHECK(hipMemset(gpuNodes[n], 0, sizeof(ccoWindowTableNode)));
  }

  for (size_t n = 0; n < numNodes; n++) {
    ccoWindowTableNode nodeHost = {};
    size_t base = n * CCO_WINDOW_TABLE_SIZE;
    for (int i = 0; i < CCO_WINDOW_TABLE_SIZE; i++) {
      size_t idx = base + i;
      if (idx < numWindows) {
        nodeHost.entries[i].base = tableEntries[idx].base;
        nodeHost.entries[i].size = tableEntries[idx].size;
        nodeHost.entries[i].window = tableEntries[idx].devPtr;
      }
    }
    nodeHost.next = (n + 1 < numNodes) ? gpuNodes[n + 1] : nullptr;
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuNodes[n], &nodeHost, sizeof(ccoWindowTableNode), hipMemcpyHostToDevice));
  }
  hostShadow.windowTable = gpuNodes[0];

  MORI_SHMEM_TRACE("ccoDevCommCreate: windowTable with {} windows in {} nodes", numWindows,
                   numNodes);

  // SDMA signal pool (per-DevComm). Materialized only if comm-level SDMA
  // queues are up. Pool: [lsaSize × sdmaNumQueue × uint64], shared by all
  // windows. Kernels index via devComm->sdma.signalBuf[lsaPeer * sdmaNumQueue + qId].
  //
  // SPMT-safe peer-pointer exchange: hipIpcOpenMemHandle fails when the
  // handle was exported by the same process, so for SPMT we Allgather raw
  // VAs alongside IPC handles and pick per-peer based on SameProcessP2P.
  // (See shmem's SymmMemManager::Register for the same pattern.)
  ccoSdmaContext& sdma = hostShadow.sdma;
  sdma.sdmaNumQueue = static_cast<uint32_t>(comm->sdmaNumQueue);
  if (comm->sdmaNumQueue > 0) {
    size_t poolBytes = static_cast<size_t>(comm->lsaSize) * comm->sdmaNumQueue * sizeof(HSAuint64);
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.signalBuf, poolBytes));
    HIP_RUNTIME_CHECK(hipMemset(sdma.signalBuf, 0, poolBytes));
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.expectSignals, poolBytes));
    HIP_RUNTIME_CHECK(hipMemset(sdma.expectSignals, 0, poolBytes));

    // Use std::vector for host scratch so any exception thrown by
    // bootNet->Allgather (cross-rank comm) doesn't leak heap.
    hipIpcMemHandle_t myHandle;
    HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&myHandle, sdma.signalBuf));
    std::vector<hipIpcMemHandle_t> handles(comm->worldSize);
    comm->bootNet->Allgather(&myHandle, handles.data(), sizeof(hipIpcMemHandle_t));

    // Also Allgather raw VAs — used for same-process peers where IPC fails.
    HSAuint64* myRawVa = sdma.signalBuf;
    std::vector<HSAuint64*> rawVas(comm->worldSize, nullptr);
    comm->bootNet->Allgather(&myRawVa, rawVas.data(), sizeof(HSAuint64*));

    std::vector<HSAuint64*> peerPtrs_host(comm->lsaSize, nullptr);
    peerPtrs_host[comm->lsaRank] = sdma.signalBuf;
    for (int lsa = 0; lsa < comm->lsaSize; lsa++) {
      if (lsa == comm->lsaRank) continue;
      int pe = comm->myNodeStart + lsa;
      if (!comm->ctx->GetPeerCapabilities(pe).canSDMA) continue;

      if (comm->ctx->SameProcessP2P(pe)) {
        // Same process (SPMT): use peer's raw VA, defensively enable peer
        // access for its device. hipIpcMemLazyEnablePeerAccess doesn't run
        // here because we're not opening an IPC handle.
        peerPtrs_host[lsa] = rawVas[pe];
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, rawVas[pe]) == hipSuccess &&
            attr.device != hipInvalidDeviceId) {
          hipError_t peerErr = hipDeviceEnablePeerAccess(attr.device, 0);
          (void)hipGetLastError();
          if (peerErr != hipSuccess && peerErr != hipErrorPeerAccessAlreadyEnabled) {
            MORI_SHMEM_WARN(
                "ccoDevCommCreate: hipDeviceEnablePeerAccess(peer={}, "
                "device={}) failed: {}",
                pe, attr.device, hipGetErrorString(peerErr));
          }
        } else {
          (void)hipGetLastError();
        }
      } else {
        // Cross-process: standard IPC open.
        void* mapped = nullptr;
        HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(&mapped, handles[pe], hipIpcMemLazyEnablePeerAccess));
        peerPtrs_host[lsa] = reinterpret_cast<HSAuint64*>(mapped);
      }
    }
    HIP_RUNTIME_CHECK(hipMalloc(&sdma.peerSignalPtrs, sizeof(HSAuint64*) * comm->lsaSize));
    HIP_RUNTIME_CHECK(hipMemcpy(sdma.peerSignalPtrs, peerPtrs_host.data(),
                                sizeof(HSAuint64*) * comm->lsaSize, hipMemcpyHostToDevice));

    sdma.deviceHandles = comm->sdmaDevHandles;
    MORI_SHMEM_TRACE(
        "ccoDevCommCreate: SDMA pool signalBuf={} expectSignals={} "
        "peerSignalPtrs={} numQueue={}",
        (void*)sdma.signalBuf, (void*)sdma.expectSignals, (void*)sdma.peerSignalPtrs,
        sdma.sdmaNumQueue);
  }

  // Fill the caller-provided host struct in place — no device allocation. It
  // holds device pointers (windowTable, endpoints, resource pools) but lives on
  // the host; kernels take it by value.
  *outDevComm = hostShadow;
  MORI_SHMEM_INFO("ccoDevCommCreate: rank={} windows={} signals={} counters={} resourceWindow={}",
                  comm->rank, numWindows, signalCount, counterCount, (void*)resourceWindow);

  // Optional transport map dump, gated on MORI_CCO_LOG_TRANSPORT. Shows each
  // peer's hardware capability (canP2P/canSDMA/canRDMA) alongside whether
  // this DevComm has materialized resources for that transport — useful for
  // verifying gdaConnectionType behavior end-to-end.
  if (const char* env = std::getenv("MORI_CCO_LOG_TRANSPORT")) {
    if (env[0] != '0') {
      const char* connTypeStr = "?";
      switch (connType) {
        case CCO_GDA_CONNECTION_NONE:
          connTypeStr = "NONE";
          break;
        case CCO_GDA_CONNECTION_FULL:
          connTypeStr = "FULL";
          break;
        case CCO_GDA_CONNECTION_CROSSNODE:
          connTypeStr = "CROSSNODE";
          break;
        case CCO_GDA_CONNECTION_RAIL:
          connTypeStr = "RAIL";
          break;
      }
      const bool sdmaPoolActive =
          (hostShadow.sdma.sdmaNumQueue > 0 && hostShadow.sdma.signalBuf != nullptr);

      // Build the entire table into one string and emit atomically — avoids
      // interleaving when ranks fork-write to the same stderr concurrently.
      std::string buf;
      buf.reserve(256 + 64 * comm->worldSize);
      char line[160];
      snprintf(line, sizeof(line),
               "[cco] DevComm rank=%d/%d connType=%s — transport map "
               "(CAP=hardware capability, ACT=materialized by this DevComm):\n",
               comm->rank, comm->worldSize, connTypeStr);
      buf += line;
      buf += "  peer  | cap                | active\n";
      buf += "  ------+--------------------+--------------------\n";
      const bool rdmaEnabled = comm->ctx->RdmaTransportEnabled();
      for (int peer = 0; peer < comm->worldSize; peer++) {
        if (peer == comm->rank) {
          snprintf(line, sizeof(line), "  %4d* | SELF               | SELF\n", peer);
          buf += line;
          continue;
        }
        const auto& cap = comm->ctx->GetPeerCapabilities(peer);

        // Active = "has resources / connectivity right now":
        //   P2P  — sameHost peer with capability (LSA flat-VA covers
        //          intra-node; window-level FD exchange happens in WindowRegister).
        //   SDMA — this DevComm allocated an SDMA signal pool AND peer canSDMA.
        //   RDMA — this DevComm allocated a QP for peer (depends on connType).
        const bool actP2P = cap.canP2P && cap.sameHost;
        const bool actSDMA = sdmaPoolActive && cap.canSDMA;
        bool actRDMA = false;
        if (connType != CCO_GDA_CONNECTION_NONE && rdmaEnabled &&
            peer < static_cast<int>(peerMask.size())) {
          actRDMA = peerMask[peer];
        }

        auto fmt = [](bool p2p, bool sdma, bool rdma, char* out, size_t n) {
          out[0] = '\0';
          if (p2p) snprintf(out + strlen(out), n - strlen(out), "P2P ");
          if (sdma) snprintf(out + strlen(out), n - strlen(out), "SDMA ");
          if (rdma) snprintf(out + strlen(out), n - strlen(out), "RDMA ");
          if (out[0] == '\0') snprintf(out, n, "(none)");
        };
        char capStr[32], actStr[32];
        fmt(cap.canP2P, cap.canSDMA, cap.canRDMA, capStr, sizeof(capStr));
        fmt(actP2P, actSDMA, actRDMA, actStr, sizeof(actStr));
        snprintf(line, sizeof(line), "  %4d  | %-18s | %-18s%s\n", peer, capStr, actStr,
                 cap.sameHost ? " (intra-node)" : "");
        buf += line;
      }
      fwrite(buf.data(), 1, buf.size(), stderr);
      fflush(stderr);
    }
  }

  return 0;
}

/* ========================================================================== */
/*                           ccoDevCommDestroy                             */
/* ========================================================================== */

int ccoDevCommDestroy(ccoComm* comm, ccoDevComm* devComm) {
  if (!devComm) return 0;

  // devComm is the caller's host struct (filled by ccoDevCommCreate); read its
  // device-pointer fields directly to release the resources they reference.
  ccoDevComm& hostShadow = *devComm;

  auto& ibgda = hostShadow.ibgda;

  // Resource window: undoes ccoMemAlloc + ccoWindowRegister done in
  // DevCommCreate. WindowDeregister handles MR deregister, peer-VA unmap,
  // imported handle release, and frees the GPU ccoWindowDevice. MemFree
  // then releases the physical pages and returns the slot to vaManager.
  // Look up the wh->localPtr before Deregister erases the entry.
  if (hostShadow.resourceWindow && comm) {
    void* resourceWindowLocalPtr = nullptr;
    for (auto* wh : comm->windows) {
      if (wh && wh->devPtr == hostShadow.resourceWindow) {
        resourceWindowLocalPtr = wh->localPtr;
        break;
      }
    }
    (void)ccoWindowDeregister(comm, hostShadow.resourceWindow);
    if (resourceWindowLocalPtr) (void)ccoMemFree(comm, resourceWindowLocalPtr);
  }

  // QP endpoints array. signalBuf/Shadows/counterBuf are sub-pointers into
  // the resource window — they were freed above by ccoWindowDeregister +
  // ccoMemFree, so no separate hipFree needed.
  if (ibgda.endpoints) HIP_RUNTIME_CHECK(hipFree(ibgda.endpoints));

  // SDMA pool cleanup. peerSignalPtrs is a GPU array of host-mapped peer
  // pointers — only the cross-process entries came from hipIpcOpenMemHandle
  // and need a matching close; same-process entries are raw VAs into a peer
  // thread's signalBuf and must NOT be passed to hipIpcCloseMemHandle.
  auto& sdma = hostShadow.sdma;
  if (sdma.peerSignalPtrs) {
    std::vector<HSAuint64*> peerPtrs_host(hostShadow.lsaSize, nullptr);
    HIP_RUNTIME_CHECK(hipMemcpy(peerPtrs_host.data(), sdma.peerSignalPtrs,
                                sizeof(HSAuint64*) * hostShadow.lsaSize, hipMemcpyDeviceToHost));
    for (int lsa = 0; lsa < hostShadow.lsaSize; lsa++) {
      if (lsa == hostShadow.lsaRank) continue;
      if (!peerPtrs_host[lsa]) continue;
      int pe = hostShadow.myNodeStart + lsa;
      if (comm && comm->ctx && comm->ctx->SameProcessP2P(pe)) continue;
      (void)hipIpcCloseMemHandle(peerPtrs_host[lsa]);
    }
    HIP_RUNTIME_CHECK(hipFree(sdma.peerSignalPtrs));
  }
  if (sdma.signalBuf) HIP_RUNTIME_CHECK(hipFree(sdma.signalBuf));
  if (sdma.expectSignals) HIP_RUNTIME_CHECK(hipFree(sdma.expectSignals));

  ccoWindowTableNode* node = hostShadow.windowTable;
  while (node) {
    ccoWindowTableNode nodeHost;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&nodeHost, node, sizeof(ccoWindowTableNode), hipMemcpyDeviceToHost));
    HIP_RUNTIME_CHECK(hipFree(node));
    node = nodeHost.next;
  }

  return 0;
}

/* ========================================================================== */
/*                             ccoBarrierAll                               */
/* ========================================================================== */

int ccoBarrierAll(ccoComm* comm) {
  comm->bootNet->Barrier();
  return 0;
}

ccoDevComm* ccoDevCommCopyToDevice(const ccoDevComm* host) {
  ccoDevComm* device = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&device, sizeof(ccoDevComm)));
  HIP_RUNTIME_CHECK(hipMemcpy(device, host, sizeof(ccoDevComm), hipMemcpyHostToDevice));
  return device;
}

void ccoDevCommFreeDeviceCopy(ccoDevComm* devicePtr) {
  if (devicePtr) HIP_RUNTIME_CHECK(hipFree(devicePtr));
}

}  // namespace cco
}  // namespace mori
