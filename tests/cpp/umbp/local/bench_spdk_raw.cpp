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
// SPDK Benchmark for umbp SpdkSsdTier and raw SpdkEnv I/O.
//
// Measures sequential write/read bandwidth at multiple chunk sizes, both:
//   1. Raw SPDK I/O (SeqDirectWorker — multi-threaded deep-queue pipeline)
//   2. SpdkSsdTier batch write/read (through the TierBackend interface)
//
// Usage:
//   sudo UMBP_SPDK_BDEV=NVMe0n1 ./spdk_bench [threads] [iodepth] [iterations]
//   sudo ./spdk_bench                  # uses malloc bdev
//
// Env vars: UMBP_SPDK_BDEV, UMBP_SPDK_REACTOR_MASK, UMBP_SPDK_MEM_MB,
//           UMBP_SPDK_NVME_PCI, UMBP_SPDK_NVME_CTRL

#ifdef USE_SPDK

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/common/log.h"
#include "umbp/local/tiers/spdk_ssd_tier.h"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/storage/spdk/spdk_env.h"

using namespace mori::umbp;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Config
// ============================================================================
struct BenchConfig {
  int threads = 4;
  int iodepth = 128;
  int iterations = 3;
  size_t total_bytes = 512ULL * 1024 * 1024;
  size_t backend_keys = 2000;
};

// ============================================================================
// Helpers
// ============================================================================
struct BandwidthResult {
  double total_bytes = 0;
  double total_secs = 0;
  size_t total_ops = 0;
  double BW_MBps() const {
    return total_secs > 0 ? total_bytes / total_secs / (1024.0 * 1024.0) : 0;
  }
};

static void FillPattern(char* buf, size_t len, uint32_t seed) {
  auto* p = reinterpret_cast<uint32_t*>(buf);
  size_t n = len / sizeof(uint32_t);
  for (size_t i = 0; i < n; ++i) p[i] = seed ^ static_cast<uint32_t>(i);
}

static std::string BuildCoreMask(int num_cores) {
  if (num_cores <= 0) num_cores = 1;
  uint64_t mask = 0;
  for (int i = 0; i < num_cores; ++i) mask |= (1ULL << i);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%" PRIx64, mask);
  return buf;
}

static umbp::SpdkEnvConfig MakeEnvConfig(int cores) {
  umbp::SpdkEnvConfig cfg;
  const char* bdev = std::getenv("UMBP_SPDK_BDEV");
  const char* pci = std::getenv("UMBP_SPDK_NVME_PCI");
  const char* ctrl = std::getenv("UMBP_SPDK_NVME_CTRL");

  if (pci && pci[0]) {
    cfg.nvme_pci_addr = pci;
    if (ctrl) cfg.nvme_ctrl_name = ctrl;
    cfg.bdev_name = bdev ? bdev : (cfg.nvme_ctrl_name + "n1");
  } else if (bdev) {
    cfg.bdev_name = bdev;
  } else {
    cfg.use_malloc_bdev = true;
    cfg.bdev_name = "Malloc0";
    cfg.malloc_num_blocks = 262144;
    cfg.malloc_block_size = 4096;
  }
  const char* mask = std::getenv("UMBP_SPDK_REACTOR_MASK");
  cfg.reactor_mask = mask ? mask : BuildCoreMask(cores);
  const char* mem = std::getenv("UMBP_SPDK_MEM_MB");
  cfg.mem_size_mb = mem ? std::atoi(mem) : 256;
  return cfg;
}

// ============================================================================
// Raw SPDK sequential bandwidth worker (deep-queue pipeline)
// ============================================================================
static void SeqDirectWorker(umbp::SpdkEnv& env, size_t io_size, size_t aligned_io, bool is_write,
                            size_t start_offset, size_t addr_range, size_t my_io_ops, int qd,
                            BandwidthResult* out) {
  if (my_io_ops == 0) return;

  auto dma_bufs = std::make_unique<void*[]>(qd);
  int got = env.DmaPoolAllocBatch(dma_bufs.get(), aligned_io, qd, env.GetBlockSize());
  if (got == 0) {
    UMBP_LOG_ERROR("DMA alloc failed");
    return;
  }
  qd = got;

  if (is_write) {
    for (int i = 0; i < qd; ++i) {
      FillPattern(static_cast<char*>(dma_bufs[i]), io_size, static_cast<uint32_t>(i));
      if (aligned_io > io_size)
        std::memset(static_cast<char*>(dma_bufs[i]) + io_size, 0, aligned_io - io_size);
    }
  }

  auto reqs = std::make_unique<umbp::SpdkIoRequest[]>(qd);
  auto batch_ptrs = std::make_unique<umbp::SpdkIoRequest*[]>(qd);

  int submitted = 0, completed_cnt = 0, head = 0, tail = 0;
  size_t next_offset = start_offset;

  auto t0 = Clock::now();

  while (completed_cnt < static_cast<int>(my_io_ops)) {
    int batch_count = 0;
    while (submitted - completed_cnt < qd && submitted < static_cast<int>(my_io_ops)) {
      int slot = head;
      auto& req = reqs[slot];
      req.op = is_write ? umbp::SpdkIoRequest::WRITE : umbp::SpdkIoRequest::READ;
      req.buf = dma_bufs[slot];
      req.offset = next_offset;
      req.nbytes = aligned_io;
      req.completed.store(false, std::memory_order_release);
      req.success = false;
      req._next_batch = nullptr;
      req.dst_iov = nullptr;
      req.dst_iovcnt = 0;
      req.dst_skip = 0;
      req.src_data = nullptr;
      req.src_len = 0;
      req.src_iov = nullptr;
      req.src_iovcnt = 0;

      if (is_write) {
        std::memcpy(dma_bufs[slot], dma_bufs[slot % got], io_size);
      }

      batch_ptrs[batch_count++] = &req;
      submitted++;
      next_offset += io_size;
      if (next_offset - start_offset >= addr_range) next_offset = start_offset;
      head = (head + 1) % qd;

      if (is_write && batch_count >= 8) {
        env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);
        batch_count = 0;
      }
    }

    if (batch_count > 0) env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

    while (completed_cnt < submitted) {
      if (!reqs[tail].completed.load(std::memory_order_acquire)) break;
      out->total_bytes += io_size;
      out->total_ops++;
      completed_cnt++;
      tail = (tail + 1) % qd;
    }
  }

  out->total_secs = std::chrono::duration<double>(Clock::now() - t0).count();
  env.DmaPoolFreeBatch(dma_bufs.get(), aligned_io, qd);
}

static BandwidthResult BenchRawSpdk(size_t chunk_size, size_t total_bytes, bool is_write,
                                    int iodepth, int threads) {
  auto& env = umbp::SpdkEnv::Instance();
  uint32_t bs = env.GetBlockSize();
  size_t aligned_io = (chunk_size + bs - 1) & ~(static_cast<size_t>(bs) - 1);
  size_t total_ops = total_bytes / chunk_size;
  size_t ops_per_thread = total_ops / threads;
  size_t addr_range_per_thread = ops_per_thread * chunk_size;

  std::vector<BandwidthResult> results(threads);
  std::vector<std::thread> pool;

  size_t offset = 0;
  for (int t = 0; t < threads; ++t) {
    size_t my_ops = (t == threads - 1) ? (total_ops - ops_per_thread * t) : ops_per_thread;
    size_t my_range = my_ops * chunk_size;
    int per_qd = iodepth;
    if (per_qd < 1) per_qd = 1;
    constexpr size_t kMaxDmaPerThread = 512ULL * 1024 * 1024;
    int max_by_mem = static_cast<int>(kMaxDmaPerThread / aligned_io);
    if (max_by_mem < 1) max_by_mem = 1;
    if (per_qd > max_by_mem) per_qd = max_by_mem;

    pool.emplace_back(SeqDirectWorker, std::ref(env), chunk_size, aligned_io, is_write, offset,
                      my_range, my_ops, per_qd, &results[t]);
    offset += my_range;
  }

  for (auto& th : pool) th.join();

  BandwidthResult agg;
  double max_secs = 0;
  for (auto& r : results) {
    agg.total_bytes += r.total_bytes;
    agg.total_ops += r.total_ops;
    max_secs = std::max(max_secs, r.total_secs);
  }
  agg.total_secs = max_secs;
  return agg;
}

// ============================================================================
// SpdkSsdTier batch throughput benchmark
// ============================================================================
static BandwidthResult BenchTierBatch(SpdkSsdTier& tier, size_t value_size, int count,
                                      bool is_write) {
  std::vector<std::string> keys(count);
  std::vector<std::vector<char>> bufs(count);
  std::vector<const void*> wptrs(count);
  std::vector<uintptr_t> rptrs(count);
  std::vector<size_t> sizes(count, value_size);

  for (int i = 0; i < count; ++i) {
    keys[i] = "bench_" + std::to_string(i);
    bufs[i].resize(value_size);
    FillPattern(bufs[i].data(), value_size, static_cast<uint32_t>(i));
    wptrs[i] = bufs[i].data();
    rptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
  }

  if (!is_write) {
    auto wr = tier.BatchWrite(keys, wptrs, sizes);
    int ok = 0;
    for (auto b : wr)
      if (b) ++ok;
    if (ok < count) {
      printf("    [warn] only wrote %d/%d keys for read bench\n", ok, count);
    }
  }

  if (is_write) {
    for (auto& k : keys) tier.Evict(k);
  }

  if (!is_write) {
    for (auto& b : bufs) std::memset(b.data(), 0, b.size());
  }

  auto t0 = Clock::now();
  if (is_write) {
    tier.BatchWrite(keys, wptrs, sizes);
  } else {
    tier.BatchReadIntoPtr(keys, rptrs, sizes);
  }
  double secs = std::chrono::duration<double>(Clock::now() - t0).count();

  BandwidthResult res;
  res.total_bytes = static_cast<double>(count) * value_size;
  res.total_secs = secs;
  res.total_ops = count;
  return res;
}

// ============================================================================
// Reporting
// ============================================================================
static void PrintHeader(const char* title) {
  printf("\n");
  for (int i = 0; i < 80; ++i) printf("-");
  printf("\n %s\n", title);
  for (int i = 0; i < 80; ++i) printf("-");
  printf("\n");
}

static double TrimmedMean(std::vector<double>& v) {
  if (v.size() <= 2) {
    return v.empty() ? 0 : v[v.size() / 2];
  }
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
  return sum / (v.size() - 2);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  BenchConfig bench;
  if (argc > 1) bench.threads = std::atoi(argv[1]);
  if (argc > 2) bench.iodepth = std::atoi(argv[2]);
  if (argc > 3) bench.iterations = std::atoi(argv[3]);
  if (argc > 4) bench.backend_keys = std::atoi(argv[4]);

  printf("SPDK Benchmark — threads=%d iodepth=%d iterations=%d keys=%zu\n", bench.threads,
         bench.iodepth, bench.iterations, bench.backend_keys);

  // ---- Initialize SPDK ----
  auto ecfg = MakeEnvConfig(2);
  auto& env = umbp::SpdkEnv::Instance();
  int rc = env.Init(ecfg);
  if (rc != 0) {
    fprintf(stderr, "FATAL: SpdkEnv::Init failed rc=%d\n", rc);
    return 1;
  }
  printf("SpdkEnv: bdev=%s block_size=%u device_size=%zuMB reactors=%d\n", ecfg.bdev_name.c_str(),
         env.GetBlockSize(), static_cast<size_t>(env.GetBdevSize()) / (1024 * 1024),
         env.GetNumReactors());

  // Pre-warm DMA pool
  env.DmaPoolPrewarm(2 * 1024 * 1024, bench.iodepth);

  // ---- Raw SPDK sequential bandwidth ----
  {
    PrintHeader("RAW SPDK SEQUENTIAL BANDWIDTH");
    printf("%12s %12s %12s\n", "ChunkSize", "Write MB/s", "Read MB/s");

    size_t chunks[] = {4096,
                       32 * 1024,
                       128 * 1024,
                       512 * 1024,
                       1024 * 1024,
                       2ULL * 1024 * 1024,
                       8ULL * 1024 * 1024,
                       16ULL * 1024 * 1024,
                       32ULL * 1024 * 1024,
                       64ULL * 1024 * 1024,
                       128ULL * 1024 * 1024,
                       256ULL * 1024 * 1024,
                       512ULL * 1024 * 1024};
    uint64_t bdev_size = env.GetBdevSize();
    for (size_t chunk : chunks) {
      size_t half_bdev = static_cast<size_t>(bdev_size / 2);
      size_t min_for_threads = std::min(chunk * static_cast<size_t>(bench.threads), half_bdev);
      size_t effective_total = std::max({bench.total_bytes, chunk * 2, min_for_threads});
      effective_total = std::min(effective_total, half_bdev);

      std::vector<double> wbw, rbw;
      for (int iter = 0; iter < bench.iterations; ++iter) {
        auto wr = BenchRawSpdk(chunk, effective_total, true, bench.iodepth, bench.threads);
        auto rd = BenchRawSpdk(chunk, effective_total, false, bench.iodepth, bench.threads);
        wbw.push_back(wr.BW_MBps());
        rbw.push_back(rd.BW_MBps());
      }
      if (chunk >= 1024 * 1024) {
        printf("%10zuMB %10.0f %10.0f\n", chunk / (1024 * 1024), TrimmedMean(wbw),
               TrimmedMean(rbw));
      } else {
        printf("%10zuKB %10.0f %10.0f\n", chunk / 1024, TrimmedMean(wbw), TrimmedMean(rbw));
      }
      env.DmaPoolDrain();
    }
  }

  // ---- SpdkSsdTier batch throughput ----
  {
    PrintHeader("SpdkSsdTier BATCH THROUGHPUT");

    UMBPConfig tier_cfg;
    tier_cfg.ssd_backend = "spdk";
    tier_cfg.spdk_bdev_name = ecfg.bdev_name;
    tier_cfg.spdk_reactor_mask = ecfg.reactor_mask;
    tier_cfg.spdk_mem_size_mb = ecfg.mem_size_mb;
    tier_cfg.spdk_nvme_pci_addr = ecfg.nvme_pci_addr;
    tier_cfg.spdk_nvme_ctrl_name = ecfg.nvme_ctrl_name;
    tier_cfg.spdk_io_workers = bench.threads;
    tier_cfg.ssd.capacity_bytes = env.GetBdevSize();

    SpdkSsdTier tier(tier_cfg);
    if (!tier.IsValid()) {
      fprintf(stderr, "ERROR: SpdkSsdTier init failed\n");
    } else {
      printf("%12s %8s %12s %12s\n", "ValueSize", "Count", "Write MB/s", "Read MB/s");

      size_t value_sizes[] = {4096,
                              32 * 1024,
                              128 * 1024,
                              512 * 1024,
                              1024 * 1024,
                              2ULL * 1024 * 1024,
                              8ULL * 1024 * 1024,
                              16ULL * 1024 * 1024,
                              32ULL * 1024 * 1024,
                              64ULL * 1024 * 1024,
                              128ULL * 1024 * 1024,
                              256ULL * 1024 * 1024,
                              512ULL * 1024 * 1024};

      uint64_t bdev_cap = env.GetBdevSize();
      constexpr size_t kMaxTotalData = 512ULL * 1024 * 1024;

      for (size_t vsz : value_sizes) {
        size_t effective_keys = bench.backend_keys;
        if (vsz > 0) {
          size_t max_by_data = std::max<size_t>(2, kMaxTotalData / vsz);
          effective_keys = std::min(effective_keys, max_by_data);
        }
        size_t per_key_overhead = vsz + 4096 + 64;
        size_t max_by_bdev =
            std::max<size_t>(4, static_cast<size_t>(bdev_cap / per_key_overhead / 2));
        effective_keys = std::min(effective_keys, max_by_bdev);
        int count = static_cast<int>(effective_keys);

        std::vector<double> wbw, rbw;
        for (int iter = 0; iter < bench.iterations; ++iter) {
          tier.Clear();
          auto wr = BenchTierBatch(tier, vsz, count, true);
          auto rd = BenchTierBatch(tier, vsz, count, false);
          wbw.push_back(wr.BW_MBps());
          rbw.push_back(rd.BW_MBps());
        }
        if (vsz >= 1024 * 1024) {
          printf("%10zuMB %8d %10.0f %10.0f\n", vsz / (1024 * 1024), count, TrimmedMean(wbw),
                 TrimmedMean(rbw));
        } else {
          printf("%10zuKB %8d %10.0f %10.0f\n", vsz / 1024, count, TrimmedMean(wbw),
                 TrimmedMean(rbw));
        }
      }
    }
  }

  // ---- POSIX SSDTier batch throughput (comparison baseline) ----
  {
    PrintHeader("POSIX SSDTier BATCH THROUGHPUT (fsync per key)");

    std::string posix_dir = "/tmp/umbp_posix_bench";
    std::filesystem::create_directories(posix_dir);

    printf("%12s %8s %12s %12s\n", "ValueSize", "Count", "Write MB/s", "Read MB/s");

    size_t posix_sizes[] = {4096,
                            32 * 1024,
                            128 * 1024,
                            512 * 1024,
                            1024 * 1024,
                            2ULL * 1024 * 1024,
                            8ULL * 1024 * 1024,
                            16ULL * 1024 * 1024,
                            32ULL * 1024 * 1024,
                            64ULL * 1024 * 1024,
                            128ULL * 1024 * 1024,
                            256ULL * 1024 * 1024,
                            512ULL * 1024 * 1024};
    constexpr size_t kPosixMaxTotal = 512ULL * 1024 * 1024;

    for (size_t vsz : posix_sizes) {
      size_t ek = std::min(bench.backend_keys, std::max<size_t>(2, kPosixMaxTotal / vsz));
      int count = static_cast<int>(ek);

      std::vector<double> wbw, rbw;
      for (int iter = 0; iter < bench.iterations; ++iter) {
        UMBPConfig posix_cfg;
        SSDTier posix_tier(posix_dir, 8ULL * 1024 * 1024 * 1024, posix_cfg);

        std::vector<std::string> keys(count);
        std::vector<std::vector<char>> bufs(count);
        std::vector<const void*> wptrs(count);
        std::vector<uintptr_t> rptrs(count);
        std::vector<size_t> sizes(count, vsz);

        for (int i = 0; i < count; ++i) {
          keys[i] = "posix_" + std::to_string(i);
          bufs[i].resize(vsz);
          FillPattern(bufs[i].data(), vsz, static_cast<uint32_t>(i));
          wptrs[i] = bufs[i].data();
          rptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
        }

        auto t0 = Clock::now();
        for (int i = 0; i < count; ++i) posix_tier.Write(keys[i], wptrs[i], sizes[i]);
        double wsecs = std::chrono::duration<double>(Clock::now() - t0).count();
        double wb = static_cast<double>(count) * vsz / wsecs / (1024.0 * 1024.0);
        wbw.push_back(wb);

        for (auto& b : bufs) std::memset(b.data(), 0, b.size());
        auto t1 = Clock::now();
        for (int i = 0; i < count; ++i) posix_tier.ReadIntoPtr(keys[i], rptrs[i], sizes[i]);
        double rsecs = std::chrono::duration<double>(Clock::now() - t1).count();
        double rb = static_cast<double>(count) * vsz / rsecs / (1024.0 * 1024.0);
        rbw.push_back(rb);

        posix_tier.Clear();
      }
      if (vsz >= 1024 * 1024) {
        printf("%10zuMB %8d %10.0f %10.0f\n", vsz / (1024 * 1024), count, TrimmedMean(wbw),
               TrimmedMean(rbw));
      } else {
        printf("%10zuKB %8d %10.0f %10.0f\n", vsz / 1024, count, TrimmedMean(wbw),
               TrimmedMean(rbw));
      }
    }

    std::filesystem::remove_all(posix_dir);
  }

  printf("\nDone.\n");
  env.DmaPoolDrain();
  env.Shutdown();
  return 0;
}

#else  // !USE_SPDK

#include <cstdio>
int main() {
  fprintf(stderr, "spdk_bench requires USE_SPDK build.\n");
  return 1;
}

#endif
