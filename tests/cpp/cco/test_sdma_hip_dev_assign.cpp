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
// CLI usage:
//   test_sdma_hip_dev_assign [--n_processes N] [--n_local_devices N]
//                            [--visible_devices D0,D1,D2,...] [--torch_style]
// --n_processes      number of OS processes to fork (process_count). Default:
//                     the auto-detected total GPU count (or the length of
//                     --visible_devices, if given) divided by
//                     --n_local_devices (i.e. "use every visible GPU") —
//                     under --torch_style, NOT divided by --n_local_devices
//                     (which is ignored), since each process owns 1 GPU.
// --n_local_devices  number of GPUs owned by each process (devices_per_process).
//                     Default: 1. Ignored when --torch_style is set.
// --visible_devices  comma-separated physical device ordinals to carve up
//                     instead of the default contiguous 0,1,2,...
//                     assignment. Process i gets the n_local_devices-sized
//                     slice at offset i*n_local_devices, in list order — e.g.
//                     --visible_devices 1,2,4,5,7,3,0,6 --n_processes 4
//                     --n_local_devices 2 gives processes HIP_VISIBLE_DEVICES
//                     "1,2", "4,5", "7,3", "0,6" respectively. The list length
//                     must equal n_processes * n_local_devices exactly.
//                     Under --torch_style, the FULL list (unsliced) is given
//                     to every process as HIP_VISIBLE_DEVICES, and only needs
//                     length >= n_processes (extra entries are simply
//                     visible-but-unused, matching real torch behavior).
// --torch_style      launch torch-style instead of jax-style: n_processes
//                     processes, each owning exactly 1 GPU (n_local_devices
//                     is ignored). Every process sees the SAME (unsliced)
//                     visible-device set — the full --visible_devices list if
//                     given, else every native GPU (HIP_VISIBLE_DEVICES left
//                     unset) — and process i binds device ordinal i within
//                     that shared set via hipSetDevice(i), mirroring a real
//                     torch multi-process launch. globalRank = processIdx,
//                     worldSize = n_processes (no SPMT threading, since
//                     there's exactly one GPU per process).
//
// Examples:
//   test_sdma_hip_dev_assign
//     jax-style, all defaults: 1 GPU/process, one process per detected GPU.
//   test_sdma_hip_dev_assign --n_processes 2 --n_local_devices 4
//     jax-style: 2 processes x 4 GPUs each (8-way world), contiguous
//     HIP_VISIBLE_DEVICES "0,1,2,3" / "4,5,6,7".
//   test_sdma_hip_dev_assign --visible_devices 1,2,4,5,7,3,0,6 \
//       --n_processes 4 --n_local_devices 2
//     jax-style with a custom device order: HIP_VISIBLE_DEVICES
//     "1,2" / "4,5" / "7,3" / "0,6" for processes 0..3.
//   test_sdma_hip_dev_assign --torch_style
//     torch-style, all defaults: one process per detected GPU, every
//     process sees every GPU (HIP_VISIBLE_DEVICES unset), rank i binds
//     device ordinal i.
//   test_sdma_hip_dev_assign --torch_style --n_processes 4
//     torch-style: 4 processes, each seeing all native GPUs, binding
//     ordinals 0..3.
//   test_sdma_hip_dev_assign --torch_style \
//       --visible_devices 1,2,4,5,7,3,0,6 --n_processes 4
//     torch-style with a restricted, shared device set: every process gets
//     HIP_VISIBLE_DEVICES "1,2,4,5,7,3,0,6"; ranks 0..3 bind ordinals 0..3
//     within that list (physical devices 1, 2, 4, 5).
//
// NOTE: run in fork mode (default), so no HIP call happens before the slice
// is applied. Requires MORI_ENABLE_SDMA=1; otherwise the comm has no SDMA
// queues and the test SKIPs.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "cco_test_harness.hpp"

// ── inlined from sdma_allgather_common.hpp ───────────────────────────────
//
// All-gather layout (COUNT floats per rank):
//   input[i]            = rank * 1000 + i                      (this rank's chunk)
//   gather[s*COUNT + i] = s * 1000 + i    for every source rank s
// Each rank SDMA-puts its own input chunk into slot `lsaRank` of every peer's
// gather window (self included, via the loopback queue), then quiet()s. When
// all ranks have done so, every rank's gather window holds all N chunks.

static const size_t SDMA_ALLGATHER_COUNT = 16384;
static const size_t SDMA_ALLGATHER_VMM_SIZE = 16ULL * 1024 * 1024;

__global__ void SdmaAllGatherJaxKernel(mori::cco::ccoWindow_t gather, mori::cco::ccoWindow_t input,
                                       size_t chunkBytes, mori::cco::ccoDevComm devComm) {
  int myRank = devComm.lsaRank;
  int nRanks = devComm.lsaSize;

  int p = threadIdx.x;
  if (p >= nRanks) return;
  mori::cco::ccoSdma sdma{devComm};
  sdma.put(p, gather, myRank * chunkBytes, input, 0, chunkBytes);
  sdma.quiet(p);
}

// Per-rank state threaded between setup, launch, and verify.
struct SdmaAllGatherCtx {
  mori::cco::ccoComm* comm{nullptr};
  mori::cco::ccoDevComm devComm{};
  mori::cco::ccoWindow_t inputWin{nullptr};
  mori::cco::ccoWindow_t gatherWin{nullptr};
  void* inputBuf{nullptr};
  void* gatherBuf{nullptr};
  hipStream_t stream{nullptr};
  size_t chunkBytes{0};
  int rank{0};
  int nranks{0};
  bool hasSdma{false};  // false => no SDMA queues; test SKIPs the data phase
};

// Create comm + windows + devComm and (when SDMA is available) a stream.
// Assumes the caller has already bound this rank's GPU. Returns 0 on
// success; on failure returns nonzero and the caller should bail out
// (nothing to tear down that a leaked process/thread exit won't reclaim).
static int SdmaAllGatherSetup(int rank, int nranks, const mori::cco::ccoUniqueId& uid,
                              SdmaAllGatherCtx* ctx) {
  using namespace mori::cco;
  ctx->rank = rank;
  ctx->nranks = nranks;

  int hipDev = -1, visibleDevices = 0;
  HIP_CHECK(hipGetDevice(&hipDev));
  HIP_CHECK(hipGetDeviceCount(&visibleDevices));
  printf("[rank %d/%d] pid=%d hipDev=%d visibleDevices=%d\n", rank, nranks, getpid(), hipDev,
         visibleDevices);
  fflush(stdout);

  if (ccoCommCreate(uid, nranks, rank, SDMA_ALLGATHER_VMM_SIZE, &ctx->comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  ctx->chunkBytes = SDMA_ALLGATHER_COUNT * sizeof(float);
  const size_t gatherBytes = ctx->chunkBytes * nranks;
  if (ccoMemAlloc(ctx->comm, ctx->chunkBytes, &ctx->inputBuf) != 0 ||
      ccoMemAlloc(ctx->comm, gatherBytes, &ctx->gatherBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }

  std::vector<float> hostInput(SDMA_ALLGATHER_COUNT);
  for (size_t i = 0; i < SDMA_ALLGATHER_COUNT; i++)
    hostInput[i] = static_cast<float>(rank * 1000 + i);
  HIP_CHECK(hipMemcpy(ctx->inputBuf, hostInput.data(), ctx->chunkBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(ctx->gatherBuf, 0xff, gatherBytes));

  if (ccoWindowRegister(ctx->comm, ctx->inputBuf, ctx->chunkBytes, &ctx->inputWin) != 0 ||
      ccoWindowRegister(ctx->comm, ctx->gatherBuf, gatherBytes, &ctx->gatherWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // SDMA needs no GDA connectivity; the signal pool is materialized whenever the
  // comm has SDMA queues (set up in ccoCommCreate for canSDMA peers).
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.gdaContextCount = 0;
  reqs.gdaSignalCount = 0;
  reqs.gdaCounterCount = 0;
  if (ccoDevCommCreate(ctx->comm, &reqs, &ctx->devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  ctx->hasSdma = ctx->devComm.sdma.sdmaNumQueue != 0;
  if (ctx->hasSdma) {
    HIP_CHECK(hipStreamCreate(&ctx->stream));
  } else {
    printf("[rank %d] SKIP — no SDMA queues (set MORI_ENABLE_SDMA=1)\n", rank);
  }
  return 0;
}

// Verify the gathered result (when SDMA ran) and tear the comm down. Returns 0
// on pass/skip, 1 on mismatch.
static int VerifyAndTeardown(SdmaAllGatherCtx* ctx) {
  using namespace mori::cco;
  bool ok = true;

  if (ctx->hasSdma) {
    const size_t gatherBytes = ctx->chunkBytes * ctx->nranks;
    std::vector<float> host(SDMA_ALLGATHER_COUNT * ctx->nranks);
    HIP_CHECK(hipMemcpy(host.data(), ctx->gatherBuf, gatherBytes, hipMemcpyDeviceToHost));
    for (int s = 0; s < ctx->nranks && ok; s++) {
      for (size_t i = 0; i < SDMA_ALLGATHER_COUNT; i++) {
        float expected = static_cast<float>(s * 1000 + i);
        if (host[s * SDMA_ALLGATHER_COUNT + i] != expected) {
          fprintf(stderr, "[rank %d] ALLGATHER mismatch [src=%d][%zu]: got %.0f expected %.0f\n",
                  ctx->rank, s, i, host[s * SDMA_ALLGATHER_COUNT + i], expected);
          ok = false;
          break;
        }
      }
    }
    printf("[rank %d] allgather %s\n", ctx->rank, ok ? "PASSED" : "FAILED");
    HIP_CHECK(hipStreamDestroy(ctx->stream));
  }

  ccoDevCommDestroy(ctx->comm, &ctx->devComm);
  ccoWindowDeregister(ctx->comm, ctx->gatherWin);
  ccoWindowDeregister(ctx->comm, ctx->inputWin);
  ccoMemFree(ctx->comm, ctx->gatherBuf);
  ccoMemFree(ctx->comm, ctx->inputBuf);
  ccoCommDestroy(ctx->comm);
  return ok ? 0 : 1;
}

// ── per-thread (per local GPU) rank driver ───────────────────────────────

struct ThreadResult {
  int rank{-1};
  bool passed{false};
  char detail[256]{};
};

// Runs entirely on its own thread. `rank`/`nranks` here ARE the CCO GPU
// rank/world size (globalRank / process_count*devices_per_process) — unlike
// run_test below, no reinterpretation here. `localDevice` is this thread's
// ordinal within the process's (already-sliced) HIP_VISIBLE_DEVICES list
// (0..devices_per_process-1); hipSetDevice is thread-local, so each thread
// must bind its own device before making any other HIP call.
static void RunLocalRank(int localDevice, int rank, int nranks, const mori::cco::ccoUniqueId& uid,
                         ThreadResult* result) {
  // cco_test_harness.hpp's HIP_CHECK macro reports failures via the
  // process-wide g_rank global (not thread-local). Setting it here is
  // best-effort for diagnostics only: under concurrent threads the printed
  // rank in a HIP_CHECK failure message can race, but HIP_CHECK always
  // _exit(1)s the whole process immediately regardless, so this never
  // affects correctness — only which rank number an error message blames.
  g_rank = rank;
  result->rank = rank;
  result->passed = false;

  HIP_CHECK(hipSetDevice(localDevice));

  SdmaAllGatherCtx ctx;
  if (SdmaAllGatherSetup(rank, nranks, uid, &ctx) != 0) {
    snprintf(result->detail, sizeof(result->detail), "setup failed");
    return;
  }

  if (ctx.hasSdma) {
    mori::cco::ccoBarrierAll(ctx.comm);
    SdmaAllGatherJaxKernel<<<1, 64, 0, ctx.stream>>>(ctx.gatherWin, ctx.inputWin, ctx.chunkBytes,
                                                     ctx.devComm);
    HIP_CHECK(hipStreamSynchronize(ctx.stream));
    mori::cco::ccoBarrierAll(ctx.comm);
  }

  bool ok = VerifyAndTeardown(&ctx) == 0;
  result->passed = ok;
  snprintf(result->detail, sizeof(result->detail), ok ? "OK" : "allgather mismatch");
}

// devices_per_process: not known to cco_test_harness.hpp's run_test(rank,
// nranks, uid) signature, so main() parses it from --n_local_devices into
// this global before handing off to ccoTestForkMode.
static int g_devicesPerProcess = 1;

// Optional custom physical-device ordinals from --visible_devices, in the
// order they should be carved up across processes. Empty means "use the
// default contiguous 0,1,2,... assignment" (see run_test below).
static std::vector<int> g_visibleDevices;

// --torch_style: every process sees the same (unsliced) device set and picks
// device ordinal == its rank, instead of jax-style per-process slicing. See
// run_test below.
static bool g_torchStyle = false;

// Invoked ONCE PER FORKED PROCESS by cco_test_harness.hpp's fork mode. Here
// `processIdx`/`numProcesses` are this PROCESS's index / the total process
// count (process_count) — NOT a GPU rank/world size like other tests that
// use this harness. See the file header comment for why.
int run_test(int processIdx, int numProcesses, const mori::cco::ccoUniqueId& uid) {
  g_rank = processIdx;

  if (g_torchStyle) {
    // torch-style: every process shares the SAME (unsliced) visible-device
    // set and picks device ordinal == its own rank, mirroring a real torch
    // multi-process launch where all GPUs are visible to every process.
    // --n_local_devices is ignored (each process owns exactly 1 GPU).
    if (!g_visibleDevices.empty()) {
      std::string vis;
      for (size_t i = 0; i < g_visibleDevices.size(); i++) {
        if (i) vis += ",";
        vis += std::to_string(g_visibleDevices[i]);
      }
      setenv("HIP_VISIBLE_DEVICES", vis.c_str(), /*overwrite=*/1);
    }
    // Else: leave HIP_VISIBLE_DEVICES unset, so native full visibility applies.

    ThreadResult result;
    RunLocalRank(processIdx, processIdx, numProcesses, uid, &result);
    printf("[proc %d][rank %d] %s: %s\n", processIdx, result.rank,
           result.passed ? "PASSED" : "FAILED", result.detail);
    return result.passed ? 0 : 1;
  }

  int devicesPerProcess = g_devicesPerProcess;
  // jax-style: slice HIP visibility to this process's GPU range BEFORE any
  // further HIP call, so local ordinals 0..devicesPerProcess-1 map to the
  // DISTINCT physical GPUs [gpuOffset, gpuOffset+devicesPerProcess) — or, if
  // --visible_devices was given, to the devicesPerProcess-sized slice of that
  // list at the same offset (main() already validated the list length).
  // Leave ROCR_VISIBLE_DEVICES unset so HSA still sees all GPUs (required by
  // the KFD node-id SDMA path).
  const int gpuOffset = processIdx * devicesPerProcess;
  std::string vis;
  for (int i = 0; i < devicesPerProcess; i++) {
    if (i) vis += ",";
    int physicalDev = g_visibleDevices.empty() ? gpuOffset + i : g_visibleDevices[gpuOffset + i];
    vis += std::to_string(physicalDev);
  }
  setenv("HIP_VISIBLE_DEVICES", vis.c_str(), /*overwrite=*/1);

  const int worldSize = numProcesses * devicesPerProcess;
  std::vector<ThreadResult> results(devicesPerProcess);
  std::vector<std::thread> threads;
  threads.reserve(devicesPerProcess);
  for (int local = 0; local < devicesPerProcess; local++) {
    int globalRank = gpuOffset + local;
    threads.emplace_back(RunLocalRank, local, globalRank, worldSize, std::cref(uid),
                         &results[local]);
  }
  for (auto& t : threads) t.join();

  int fail = 0;
  for (auto& r : results) {
    printf("[proc %d][rank %d] %s: %s\n", processIdx, r.rank, r.passed ? "PASSED" : "FAILED",
           r.detail);
    if (!r.passed) fail++;
  }
  return fail > 0 ? 1 : 0;
}

// Total KFD GPU nodes visible on the host, via sysfs (no HIP calls, so this
// is safe to call before HIP_VISIBLE_DEVICES is set for any process/thread).
static int DetectTotalGpuCount() {
  int count = 0;
  for (int i = 0; i < 64; i++) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", i);
    FILE* f = fopen(path, "r");
    if (!f) break;
    unsigned long gpuId = 0;
    if (fscanf(f, "%lu", &gpuId) == 1 && gpuId != 0) count++;
    fclose(f);
  }
  return count;
}

static void PrintUsageAndExit(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--n_processes N] [--n_local_devices N] [--visible_devices D0,D1,...] "
          "[--torch_style]\n"
          "  --n_processes N        number of OS processes to fork (default: "
          "auto-detected GPU count, or --visible_devices length, / --n_local_devices; "
          "not divided under --torch_style)\n"
          "  --n_local_devices N    GPUs owned by each process (default: 1; ignored under "
          "--torch_style)\n"
          "  --visible_devices LIST comma-separated physical device ordinals to carve up "
          "instead of the default contiguous 0,1,2,... assignment; length must equal "
          "n_processes * n_local_devices (jax-style) or be >= n_processes (--torch_style, "
          "unsliced)\n"
          "  --torch_style          torch-style launch: n_processes processes, each owning "
          "1 GPU from a SHARED (unsliced) visible-device set, binding ordinal == rank\n",
          argv0);
  exit(1);
}

// Parses a comma-separated list of non-negative integers, e.g. "1,2,4,5,7,3,0,6".
// Returns false (leaving *out unspecified) on any malformed token or empty list.
static bool ParseVisibleDevicesList(const char* s, std::vector<int>* out) {
  out->clear();
  std::string str(s);
  size_t pos = 0;
  while (pos <= str.size()) {
    size_t comma = str.find(',', pos);
    std::string tok = str.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (tok.empty() ||
        !std::all_of(tok.begin(), tok.end(), [](unsigned char c) { return isdigit(c); }))
      return false;
    out->push_back(atoi(tok.c_str()));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return !out->empty();
}

int main(int argc, char** argv) {
  int nProcesses = -1;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--n_processes") && i + 1 < argc) {
      nProcesses = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--n_local_devices") && i + 1 < argc) {
      g_devicesPerProcess = std::max(1, atoi(argv[++i]));
    } else if (!strcmp(argv[i], "--visible_devices") && i + 1 < argc) {
      if (!ParseVisibleDevicesList(argv[++i], &g_visibleDevices)) {
        fprintf(stderr,
                "--visible_devices: invalid device list '%s' (expected comma-separated "
                "non-negative integers)\n",
                argv[i]);
        return 1;
      }
    } else if (!strcmp(argv[i], "--torch_style")) {
      g_torchStyle = true;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      PrintUsageAndExit(argv[0]);
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      PrintUsageAndExit(argv[0]);
    }
  }

  const int totalDevices =
      g_visibleDevices.empty() ? DetectTotalGpuCount() : static_cast<int>(g_visibleDevices.size());

  if (nProcesses < 0) {
    // --torch_style ignores --n_local_devices (always 1 GPU/process), so the
    // "use every visible GPU" default is just totalDevices, not divided.
    nProcesses =
        g_torchStyle ? std::max(1, totalDevices) : std::max(1, totalDevices / g_devicesPerProcess);
  } else if (nProcesses < 1) {
    fprintf(stderr, "--n_processes must be >= 1\n");
    return 1;
  }

  if (g_torchStyle) {
    // Every process shares the same visible set and picks ordinal == rank, so
    // it only needs >= n_processes entries (unlike jax-style's exact-match
    // slicing) -- extra entries are simply visible-but-unused.
    if (nProcesses > totalDevices) {
      fprintf(stderr, "--torch_style: --n_processes(%d) exceeds available devices (%d)\n",
              nProcesses, totalDevices);
      return 1;
    }
  } else if (!g_visibleDevices.empty() &&
             static_cast<int>(g_visibleDevices.size()) != nProcesses * g_devicesPerProcess) {
    fprintf(stderr,
            "--visible_devices has %zu entries but --n_processes(%d) * --n_local_devices(%d) "
            "= %d\n",
            g_visibleDevices.size(), nProcesses, g_devicesPerProcess,
            nProcesses * g_devicesPerProcess);
    return 1;
  }

  return ccoTestForkMode(nProcesses, "CCO SDMA allgather (various dev assignments)",
                         "/tmp/cco_sdma_allgather_uid", 0);
}
