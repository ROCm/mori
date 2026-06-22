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

#include "util.hpp"

#include <cstring>
#include <string>

#include "hip/hip_runtime.h"
#include <mpi.h>

#include "mori/application/utils/check.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori::cco::benchmark {

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  transport is selected by env MORI_DISABLE_P2P:\n"
               "    unset / on / 1  -> ibgda (RDMA)   [default]\n"
               "    off / 0 / false -> lsa   (intra-node P2P)\n"
               "  -b min_bytes   minimum message size\n"
               "  -e max_bytes   maximum message size\n"
               "  -f step        multiply size by this factor each step\n"
               "  -n iters       timed iterations\n"
               "  -w warmup      warmup iterations\n"
               "  -c grid_x      HIP grid x (blocks)\n"
               "  -t threads     threads per block\n"
               "  -s scope       thread | warp | block (default block)\n"
               "  -h             this help\n",
               program != nullptr ? program : "program");
}

int ParseArgs(int argc, char** argv, PerfArgs* out_args) {
  if (out_args == nullptr) {
    return 1;
  }

  *out_args = PerfArgs{};

  auto parse_size = [](const char* s) -> std::size_t {
    char* end = nullptr;
    std::size_t val = std::strtoul(s, &end, 0);
    if (end && *end != '\0') {
      switch (*end | 0x20) {  // tolower
        case 'k':
          val <<= 10;
          break;
        case 'm':
          val <<= 20;
          break;
        case 'g':
          val <<= 30;
          break;
      }
    }
    return val;
  };

  int opt = 0;
  while ((opt = getopt(argc, argv, "hb:e:f:n:w:c:t:s:")) != -1) {
    switch (opt) {
      case 'h':
        return 2;
      case 'b':
        out_args->min_size = parse_size(optarg);
        break;
      case 'e':
        out_args->max_size = parse_size(optarg);
        break;
      case 'f':
        out_args->step_factor = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'n':
        out_args->iters = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'w':
        out_args->warmup = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'c':
        out_args->nblocks = std::atoi(optarg);
        break;
      case 't':
        out_args->threads_per_block = std::atoi(optarg);
        break;
      case 's':
        if (std::strcmp(optarg, "thread") == 0) {
          out_args->put_scope = PutScope::kThread;
        } else if (std::strcmp(optarg, "warp") == 0) {
          out_args->put_scope = PutScope::kWarp;
        } else if (std::strcmp(optarg, "block") == 0) {
          out_args->put_scope = PutScope::kBlock;
        } else {
          return 1;
        }
        break;
      default:
        return 1;
    }
  }

  return 0;
}

static std::string fmt_size(std::size_t bytes) {
  char buf[16];
  std::size_t val;
  const char* unit;
  if (bytes >= (1ULL << 30)) {
    val = bytes >> 30;
    unit = "GB";
  } else if (bytes >= (1ULL << 20)) {
    val = bytes >> 20;
    unit = "MB";
  } else if (bytes >= (1ULL << 10)) {
    val = bytes >> 10;
    unit = "KB";
  } else {
    val = bytes;
    unit = "B ";
  }
  std::snprintf(buf, sizeof(buf), "%3zu %s", val, unit);
  return buf;
}

void PrintPerfTable(const char* test_name, const char* transport_name, const char* scope_name,
                    int grid_x, int block_threads, int warp_size, std::size_t iters,
                    std::size_t warmup, PerfTableMetric metric,
                    const std::vector<PerfTableRow>& rows) {
  const char* scope_col = (scope_name != nullptr && scope_name[0] != '\0') ? scope_name : "none";
  const char* tag = (test_name != nullptr && test_name[0] != '\0') ? test_name : "p2p";

  std::printf("# %s transport=%s scope=%s grid=%d block=%d warpSize=%d iters=%zu warmup=%zu\n", tag,
              transport_name, scope_col, grid_x, block_threads, warp_size, iters, warmup);

  constexpr int kWSize = 10;
  constexpr int kWScope = 8;
  constexpr int kWNum = 12;

  const bool is_bw = (metric == PerfTableMetric::kBandwidthGbps);
  const char* num_header = is_bw ? "Bandwidth" : "Latency";
  const char* unit_str = is_bw ? "GB/s" : "us";

  std::printf("%-*s %-*s %*s %s\n", kWSize, "size", kWScope, "scope", kWNum, num_header, unit_str);

  for (const PerfTableRow& r : rows) {
    std::string sz = fmt_size(r.size_bytes);
    if (r.skipped) {
      std::printf("%-*s %-*s %*s\n", kWSize, sz.c_str(), kWScope, scope_col, kWNum, "skip");
    } else {
      std::printf("%-*s %-*s %*.3f %s\n", kWSize, sz.c_str(), kWScope, scope_col, kWNum, r.value,
                  unit_str);
    }
  }
  std::fflush(stdout);
}

int PerfInit(int argc, char** argv, PerfContext* ctx) {
  std::memset(&ctx->devComm, 0, sizeof(ctx->devComm));
  ctx->comm = nullptr;
  ctx->send_win = nullptr;
  ctx->recv_win = nullptr;
  ctx->send_buf = nullptr;
  ctx->recv_buf = nullptr;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx->world_rank);

  PerfArgs& args = ctx->args;
  int rc = ParseArgs(argc, argv, &args);
  if (rc) {
    if (ctx->world_rank == 0) {
      PrintUsage(argv[0]);
    }
    MPI_Finalize();
    return rc;  // 2 = help, 1 = bad args; caller must NOT call PerfFinalize
  }

  if (args.min_size > args.max_size || args.step_factor < 2 || args.iters < 1 || args.nblocks < 1 ||
      args.threads_per_block < 1) {
    if (ctx->world_rank == 0) {
      std::fprintf(stderr,
                   "Invalid arguments (need iters >= 1, nblocks/threads >= 1, step >= 2).\n");
    }
    MPI_Finalize();
    return 1;
  }
  if (args.min_size % sizeof(double) != 0) {
    args.min_size = (args.min_size + sizeof(double) - 1) / sizeof(double) * sizeof(double);
  }

  // Transport from MORI_DISABLE_P2P. Default (unset) is IBGDA (P2P disabled);
  // an explicit false value ("0"/"false"/"off"/"no") selects LSA. This mirrors
  // mori::env::IsEnvVarEnabled's parsing but defaults to enabled when unset.
  {
    const char* v = std::getenv("MORI_DISABLE_P2P");
    bool p2p_disabled = true;  // default: IBGDA
    if (v != nullptr && v[0] != '\0') {
      p2p_disabled = env::detail::ParseBool(v).value_or(true);
    }
    args.transport = p2p_disabled ? Transport::kIbgda : Transport::kLsa;
  }

  // Local communicator → local rank → device binding. CCO pins the device at
  // ccoCommCreate, so hipSetDevice must happen first.
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &ctx->local_comm);
  MPI_Comm_rank(ctx->local_comm, &ctx->local_rank);

  HIP_RUNTIME_CHECK(hipGetDeviceCount(&ctx->device_count));
  assert(ctx->device_count);
  const int device_id = ctx->local_rank % ctx->device_count;
  HIP_RUNTIME_CHECK(hipSetDevice(device_id));
  HIP_RUNTIME_CHECK(
      hipDeviceGetAttribute(&ctx->device_warp_size, hipDeviceAttributeWarpSize, device_id));

  // Enable peer access for LSA flat-VA loopback / import where available.
  for (int i = 0; i < ctx->device_count; i++) {
    if (i == device_id) continue;
    int can_access = 0;
    HIP_RUNTIME_CHECK(hipDeviceCanAccessPeer(&can_access, device_id, i));
    if (can_access) (void)hipDeviceEnablePeerAccess(i, 0);
  }

  // CCO comm via the cco-native uniqueId API: rank 0 mints the id (its socket
  // rendezvous), MPI broadcasts the POD, all ranks create the comm.
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ccoUniqueId uid;
  if (ctx->world_rank == 0) {
    if (ccoGetUniqueId(&uid) != 0) {
      std::fprintf(stderr, "ccoGetUniqueId failed (set MORI_SOCKET_IFNAME=<iface>)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
  const std::size_t per_rank_vmm = 2 * args.max_size + kVmmSlack;
  if (ccoCommCreate(uid, world_size, ctx->world_rank, per_rank_vmm, &ctx->comm) != 0) {
    if (ctx->world_rank == 0) std::fprintf(stderr, "ccoCommCreate failed\n");
    PerfFinalize(ctx);
    return 1;
  }

  ctx->my_pe = ctx->world_rank;
  ctx->npes = world_size;
  if (ctx->npes != 2) {
    if (ctx->my_pe == 0) {
      std::fprintf(stderr, "CCO p2p benchmark requires exactly 2 PEs (npes=%d)\n", ctx->npes);
    }
    PerfFinalize(ctx);
    return 1;
  }

  // Register send/recv windows (overload A: internal VMM alloc + P2P + RDMA MR).
  if (ccoWindowRegister(ctx->comm, args.max_size, &ctx->send_win, &ctx->send_buf) != 0 ||
      ccoWindowRegister(ctx->comm, args.max_size, &ctx->recv_win, &ctx->recv_buf) != 0) {
    if (ctx->my_pe == 0) std::fprintf(stderr, "ccoWindowRegister failed\n");
    PerfFinalize(ctx);
    return 1;
  }

  // DevComm tuned to the chosen transport.
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (args.transport == Transport::kLsa) {
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
    reqs.gdaSignalCount = 0;
    reqs.gdaCounterCount = 0;
    // Unidirectional bw/lat: only PE 0 writes, host ccoBarrierAll provides
    // cross-rank sync — no device barrier session needed.
    reqs.lsaBarrierCount = 0;
  } else {
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
    // One QP context per block (ginContext=blockIdx): each block drives its own
    // QP, so blocks are independent — no cross-block barrier, and each block
    // flushes its own QP. gdaContextCount must cover the largest block count.
    reqs.gdaContextCount = args.nblocks;
    reqs.gdaSignalCount = 0;
    reqs.gdaCounterCount = 0;
    reqs.lsaBarrierCount = 0;
  }

  if (ccoDevCommCreate(ctx->comm, &reqs, &ctx->devComm) != 0) {
    if (ctx->my_pe == 0) std::fprintf(stderr, "ccoDevCommCreate failed\n");
    PerfFinalize(ctx);
    return 1;
  }

  // Transport feasibility checks.
  if (args.transport == Transport::kLsa) {
    if (ctx->devComm.lsaSize < 2) {
      if (ctx->my_pe == 0) {
        std::fprintf(stderr,
                     "LSA transport requires both PEs on the same node (lsaSize=%d). "
                     "Run -T ibgda for cross-node.\n",
                     ctx->devComm.lsaSize);
      }
      PerfFinalize(ctx);
      return 1;
    }
    // The other PE's index within the LSA team.
    ctx->peer_lsa_rank = ctx->devComm.lsaRank ^ 1;
  } else {
    if (ctx->devComm.gdaConnType == CCO_GDA_CONNECTION_NONE) {
      if (ctx->my_pe == 0) {
        std::fprintf(stderr,
                     "IBGDA transport collapsed to NONE — RDMA loopback unsupported on a single "
                     "node? Run across 2 nodes.\n");
      }
      PerfFinalize(ctx);
      return 1;
    }
  }

  // Announce the resolved transport up front (PE 0 only) so the run is
  // self-identifying without parsing the table header.
  if (ctx->my_pe == 0) {
    if (args.transport == Transport::kLsa) {
      std::printf("[cco-bench] transport = LSA  (intra-node P2P, flat-VA load/store; lsaSize=%d)\n",
                  ctx->devComm.lsaSize);
    } else {
      std::printf(
          "[cco-bench] transport = IBGDA (cross-node RDMA via ccoGda; gdaConnType=%d)\n"
          "            set MORI_DISABLE_P2P=0 to switch to LSA\n",
          static_cast<int>(ctx->devComm.gdaConnType));
    }
    std::fflush(stdout);
  }

  return 0;
}

void PerfFinalize(PerfContext* ctx) {
  if (ctx->local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&ctx->local_comm);
    ctx->local_comm = MPI_COMM_NULL;
  }
  if (ctx->comm != nullptr) {
    ccoDevCommDestroy(ctx->comm, &ctx->devComm);
    // Windows came from the combined ccoWindowRegister(size, &win, &buf)
    // overload, which internally does ccoMemAlloc. Deregister only releases the
    // RDMA MR / GPU structs / peer P2P slots — the local VMM mapping must be
    // freed with ccoMemFree, else the flat VA still has live maps and
    // ccoCommDestroy's hipMemAddressFree fails.
    if (ctx->recv_win) ccoWindowDeregister(ctx->comm, ctx->recv_win);
    if (ctx->send_win) ccoWindowDeregister(ctx->comm, ctx->send_win);
    if (ctx->recv_buf) ccoMemFree(ctx->comm, ctx->recv_buf);
    if (ctx->send_buf) ccoMemFree(ctx->comm, ctx->send_buf);
    ccoCommDestroy(ctx->comm);  // cco's socket bootstrap — does NOT finalize MPI
    ctx->comm = nullptr;
  }
  // We own MPI now (uniqueId bootstrap); finalize it ourselves.
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) MPI_Finalize();
}

void PerfResAlloc(PerfRes* res) {
  HIP_RUNTIME_CHECK(hipEventCreate(&res->start));
  HIP_RUNTIME_CHECK(hipEventCreate(&res->stop));
  HIP_RUNTIME_CHECK(hipMalloc(&res->counter_d, 2 * sizeof(unsigned int)));
}

void PerfResFree(PerfRes* res) {
  HIP_RUNTIME_CHECK(hipEventDestroy(res->start));
  HIP_RUNTIME_CHECK(hipEventDestroy(res->stop));
  if (res->counter_d) HIP_RUNTIME_CHECK(hipFree(res->counter_d));
}

float RunWarmupAndTimed(PerfRes& res, std::size_t warmup, std::size_t iters, LaunchFn launch) {
  if (res.counter_d) HIP_RUNTIME_CHECK(hipMemset(res.counter_d, 0, 2 * sizeof(unsigned int)));
  launch(static_cast<int>(warmup));
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (res.counter_d) HIP_RUNTIME_CHECK(hipMemset(res.counter_d, 0, 2 * sizeof(unsigned int)));
  HIP_RUNTIME_CHECK(hipEventRecord(res.start, nullptr));
  launch(static_cast<int>(iters));
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipEventRecord(res.stop, nullptr));
  HIP_RUNTIME_CHECK(hipEventSynchronize(res.stop));

  float ms = 0.f;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&ms, res.start, res.stop));
  return ms;
}

}  // namespace mori::cco::benchmark
