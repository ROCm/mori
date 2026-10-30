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

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "calibrate_common.h"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/device_gather.h"
#include "umbp/common/parallel_for.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/posix_engine.h"
#ifdef UMBP_ENABLE_GDS
#include "umbp/distributed/transfer/gds_engine.h"
#endif
#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/sim/sim.h"
#include "umbp/umbp_client.h"

namespace {
using namespace mori::umbp;
using namespace mori::umbp::calibration;
using MemLoc = mori::io::MemoryLocationType;
using Clock = std::chrono::steady_clock;

struct SsdOptions {
  std::string dir;
  UMBPSsdConfig config;
  bool gpu_caller = false;
  int settle_ms = 10, api_settle_ms = 0;
  size_t read_prefill_mib = 0;
  SsdOptions() { config.io.backend = UMBPIoBackend::Posix; }
};

// Own only the unique fixture directory beneath the explicitly selected drive.
struct DiskFixture {
  std::string dir;
  explicit DiskFixture(const std::string& parent) {
    if (!std::filesystem::is_directory(parent))
      throw std::invalid_argument("--ssd-dir must name an existing storage directory");
    std::string pattern = parent + "/umbp-calibrate-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    if (!mkdtemp(name.data())) throw std::runtime_error("SSD fixture directory creation failed");
    dir = name.data();
  }
  ~DiskFixture() {
    std::error_code error;
    std::filesystem::remove_all(dir, error);
  }
};

void Hip(hipError_t error) {
  if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}

struct Release {
  size_t virtual_bytes = 0;
  bool gpu = false;
  void operator()(char* ptr) const {
    if (virtual_bytes)
      munmap(ptr, virtual_bytes);
    else if (gpu)
      (void)hipFree(ptr);
    else
      std::free(ptr);
  }
};
using Buffer = std::unique_ptr<char, Release>;

size_t HostBytes(size_t bytes) { return (bytes + 4095) & ~size_t{4095}; }

Buffer Allocate(size_t bytes, bool gpu, bool sim) {
  if (sim) {
    const auto memory = AllocateSimMemory(bytes);
    if (!memory.valid()) throw std::bad_alloc();
    return Buffer(static_cast<char*>(memory.ptr), Release{memory.mapped_size, false});
  }
  void* ptr = nullptr;
  if (gpu)
    Hip(hipMalloc(&ptr, bytes));
  else if (posix_memalign(&ptr, 4096, HostBytes(bytes)) != 0)
    throw std::bad_alloc();
  Buffer buffer(static_cast<char*>(ptr), Release{0, gpu});
  if (gpu)
    Hip(hipMemset(ptr, 0, bytes));
  else
    std::memset(ptr, 0, HostBytes(bytes));
  return buffer;
}

// Initialization/readback only; measured GPU transfers go through HbmCopyEngine.
void Payload(char* buffer, bool gpu, std::vector<char>& host, bool read) {
  void* dst = read ? host.data() : buffer;
  const void* src = read ? buffer : host.data();
  if (gpu)
    Hip(hipMemcpy(dst, src, host.size(), read ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
  else
    std::memcpy(dst, src, host.size());
}

void FlushSource(void* ptr, size_t bytes) {
#if defined(__x86_64__)
  for (size_t offset = 0; offset < bytes; offset += 64)
    _mm_clflush(static_cast<char*>(ptr) + offset);
  _mm_mfence();
#else
  throw std::runtime_error("cold calibration requires x86 cache flush; use reuse on this host");
#endif
}

void Check(const std::vector<bool>& result, size_t count) {
  if (result.size() != count || std::count(result.begin(), result.end(), true) != count)
    throw std::runtime_error("UMBP calibration operation failed");
}

void Measure(const std::string& phase, bool cold, int samples, const std::string& profile,
             size_t concurrency, size_t max_bytes, std::vector<size_t> blocks,
             std::vector<size_t> batches, const std::string& path, int device,
             const SsdOptions& ssd_options) {
  const bool training = phase == "train", sim = !profile.empty();
  const bool ssd = path.rfind("file_", 0) == 0 || path.rfind("cpu_file", 0) == 0;
  const bool staged =
      path == "file_cpu_staged" || path == "file_cpu_staged_no_crc" || path == "cpu_file_staged";
  const bool read_file = path.rfind("file_", 0) == 0;
  const bool staged_engine = staged && phase != "api";
  const size_t prefill_bytes = ssd && read_file ? ssd_options.read_prefill_mib << 20 : 0;
  const size_t prefill_chunk = std::min(prefill_bytes, size_t{8} << 20);
  if (ssd && ssd_options.dir.empty()) throw std::invalid_argument("SSD paths require --ssd-dir");
  if (training && staged) setenv("UMBP_SSD_TIMING", "1", 1);
  if (ssd) {
    setenv("UMBP_ENABLE_GDS", path == "file_gpu" ? "1" : "0", 1);
    mori::ModuleLogger::GetInstance().ForceSetModuleLevel("umbp", mori::ModuleLogger::Level::ERROR);
  }
  if (ssd && !sim && ssd_options.config.io.backend == UMBPIoBackend::IoUring) {
    DiskFixture fixture(ssd_options.dir);
    SSDTier check(fixture.dir, size_t{64} << 20, ssd_options.config);
    if (!check.native_async_io())
      throw std::runtime_error("io_uring unavailable; select --ssd-io posix explicitly");
  }
  if (!sim && path == "file_gpu") {
#ifdef UMBP_ENABLE_GDS
    if (!GdsEngine::Available()) throw std::runtime_error("GdsEngine SDK/runtime unavailable");
#else
    throw std::runtime_error("probe was built without GdsEngine SDK headers");
#endif
  }
  const bool host_device = path == "cpu_gpu" || path == "gpu_cpu";
  if (training && concurrency != 1)
    throw std::invalid_argument("concurrent callers are validation workloads only");
  // Framework callers live on GPU: DRAM loads/offloads have opposite directions,
  // while HBM put/get are both GPU→GPU. Engine training follows the selected path.
  const bool gpu_caller =
      phase == "api" && (ssd ? ssd_options.gpu_caller || path == "file_gpu" : path != "cpu_cpu");
  const bool src_gpu = gpu_caller || path == "gpu_cpu" || path == "gpu_gpu";
  const bool dst_gpu = gpu_caller || path == "cpu_gpu" || path == "gpu_gpu" || path == "file_gpu";
  if (!sim && (src_gpu || dst_gpu)) Hip(hipSetDevice(device));
  const bool custom = !blocks.empty() || !batches.empty();
  if (blocks.empty())
    blocks =
        ssd ? (training ? std::vector<size_t>{4096, 65536, 1048576, 4194304}
                        : std::vector<size_t>{8192, 131072, 2097152, 8388608})
            : (training ? std::vector<size_t>{512, 2048, 4096, 65536, 262144, 1048576, 4194304}
                        : std::vector<size_t>{1024, 8192, 32768, 131072, 524288, 2097152, 8388608});
  if (batches.empty())
    batches = ssd ? (training ? std::vector<size_t>{1, 8, 32} : std::vector<size_t>{1, 4, 16})
                  : (training ? std::vector<size_t>{1, 8, 32, 64, 128}
                              : std::vector<size_t>{1, 4, 16, 128});
  size_t cases = 0;
  const bool report_segments = ssd || (path != "cpu_cpu" && phase != "api");
  std::cout << "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns"
            << (report_segments ? ",segments" : "") << ",path"
            << (ssd ? ",model_bytes,caller_gpu,wall_elapsed_ns,ssd_tier_threads" : "") << '\n';
  for (size_t block : blocks) {
    for (size_t count : batches) {
      // Check the aggregate cap before multiplying user-supplied dimensions.
      if (block > max_bytes / concurrency / count) continue;
      const size_t bytes = block * count, total = bytes * concurrency;
      const size_t record_bytes = ssd ? static_cast<size_t>(segment::RecordBytes(20, block)) : 0;
      if (ssd && (record_bytes > max_bytes / concurrency / count ||
                  record_bytes > std::numeric_limits<uint32_t>::max()))
        continue;
      const size_t footprint = ssd ? record_bytes * count * concurrency : 0;
      if (ssd && footprint > std::numeric_limits<size_t>::max() / 16)
        throw std::invalid_argument("SSD fixture capacity overflow");
      size_t prepared_footprint = footprint;
      if (prefill_bytes) {
        const size_t records = 1 + (prefill_bytes - 1) / prefill_chunk;
        const size_t record = segment::RecordBytes(20, prefill_chunk);
        if (records > (std::numeric_limits<size_t>::max() / 16 - footprint) / record)
          throw std::invalid_argument("SSD prefill capacity overflow");
        prepared_footprint += records * record;
      }
      // A sparse concurrent sweep keeps calibration affordable; never used for fitting.
      if (!custom && concurrency > 1 &&
          ((block != 8192 && block != 524288 && block != 2097152) || count == 4))
        continue;
      ++cases;
      auto src = Allocate(total, src_gpu, sim), dst = Allocate(total, dst_gpu, sim);
      Buffer filler(nullptr, Release{});  // Registered memory outlives the client.
      std::vector<char> expected(sim ? 0 : total, 0x5a);
      std::unique_ptr<DiskFixture> disk;
      std::unique_ptr<SSDTier> tier;
      std::unique_ptr<TransferEngine> engine;
      std::unique_ptr<IUMBPClient> client;
      uint64_t payload_ns = 0;
      auto disk_config = ssd_options.config;
      if (path == "file_cpu_staged_no_crc") disk_config.verify_crc = false;
      if (ssd) {
        disk_config.tier_io_threads = EffectiveParallelThreads(disk_config.tier_io_threads);
        disk_config.capacity_bytes = std::max(prepared_footprint * 16, size_t{64} << 20);
        disk_config.segment_size_bytes = std::max(footprint * 2, size_t{8} << 20);
        if (prefill_bytes)
          disk_config.segment_size_bytes =
              std::max(disk_config.segment_size_bytes,
                       static_cast<size_t>(segment::RecordBytes(20, prefill_chunk)));
        disk_config.storage_dir = ssd_options.dir + "/umbp-sim-calibration";
        if (!sim) {
          disk = std::make_unique<DiskFixture>(ssd_options.dir);
          disk_config.storage_dir = disk->dir;
          if (phase != "api") {
            tier = std::make_unique<SSDTier>(disk->dir, disk_config.capacity_bytes, disk_config);
            if (!tier->direct_io_active())
              throw std::runtime_error("SSD calibration requires working O_DIRECT");
          }
        }
      }
      if (phase == "api") {
        UMBPConfig config;
        config.page_size = block;
        config.dram.capacity_bytes = std::max(total * 2, size_t{16} << 20);
        config.dram.prefault_threads = 1;
        config.ssd.enabled = false;
        config.sim.enabled = sim;
        config.sim.profile_path = profile;
        config.sim.trace_path = "/dev/null";
        if (ssd) {
          config.ssd = disk_config;
          config.distributed.emplace();
          config.distributed->medium = UMBPMedium::SSD;
          config.distributed->master_config.node_id = "ssd-calibration";
          config.distributed->master_config.node_address = "127.0.0.1:0";
          if (concurrency * count > static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
            throw std::invalid_argument("too many SSD staging slots");
          config.distributed->ssd_staging_buffer_slots = static_cast<int>(2 * concurrency * count);
          if (prefill_bytes)
            config.distributed->ssd_staging_buffer_slots = static_cast<int>(
                std::max(2 * concurrency * count, 1 + (prefill_chunk - 1) / block));
        } else if (path == "gpu_gpu") {
          config.distributed.emplace();
          config.distributed->medium = UMBPMedium::HBM;
          config.distributed->hbm.device = device;
          config.distributed->hbm.capacity_bytes = config.dram.capacity_bytes;
        }
        client = CreateUMBPClient(config);
        if (!client->RegisterMemory(reinterpret_cast<uintptr_t>(src.get()), total,
                                    src_gpu ? MemLoc::GPU : MemLoc::CPU, src_gpu ? device : -1,
                                    MemoryRegistration::kLocalCopyOnly) ||
            !client->RegisterMemory(reinterpret_cast<uintptr_t>(dst.get()), total,
                                    dst_gpu ? MemLoc::GPU : MemLoc::CPU, dst_gpu ? device : -1,
                                    MemoryRegistration::kLocalCopyOnly))
          throw std::runtime_error("UMBP calibration registration failed");
      } else if (sim) {
        auto simulated = std::make_unique<SimTransferEngine>(profile);
        if (host_device && !src_gpu) simulated->AddHostGatherRegion(src.get(), HostBytes(total));
        if (host_device && !dst_gpu) simulated->AddHostGatherRegion(dst.get(), HostBytes(total));
        engine = std::move(simulated);
      } else if (staged) {
        if (training) tier->SetPayloadTimingObserver([&](uint64_t ns) { payload_ns += ns; });
      } else if (ssd) {
        if (path == "file_gpu") {
#ifdef UMBP_ENABLE_GDS
          if (!GdsEngine::Available())
            throw std::runtime_error("GdsEngine SDK/runtime unavailable");
          engine = std::make_unique<GdsEngine>();
#else
          throw std::runtime_error("probe was built without GdsEngine SDK headers");
#endif
        } else {
          engine = std::make_unique<PosixEngine>();
        }
      } else if (path != "cpu_cpu") {
        auto hbm = std::make_unique<HbmCopyEngine>();
        if (!src_gpu) hbm->AddHostGatherRegion(src.get(), HostBytes(total));
        if (!dst_gpu) hbm->AddHostGatherRegion(dst.get(), HostBytes(total));
        engine = std::move(hbm);
      } else {
        engine = std::make_unique<LocalCopyEngine>();
      }
      struct Request {
        std::vector<std::string> keys;
        std::vector<uintptr_t> srcs, dsts;
        std::vector<const void*> src_data;
        std::vector<TransferItem> items;
        std::vector<RecordReservation> records;
      };
      std::vector<Request> requests(concurrency);
      std::vector<size_t> sizes(count, block);
      for (size_t c = 0; c < concurrency; ++c) {
        auto& request = requests[c];
        for (size_t k = 0; k < count; ++k) {
          const size_t index = c * count + k;
          char* source = src.get() + index * block;
          char* target = dst.get() + index * block;
          // Detect accidental key/caller aliasing in the real-payload check.
          if (!sim)
            std::memcpy(expected.data() + index * block, &index, std::min(block, sizeof(index)));
          request.keys.push_back(std::to_string(index));
          request.srcs.push_back(reinterpret_cast<uintptr_t>(source));
          request.src_data.push_back(source);
          request.dsts.push_back(reinterpret_cast<uintptr_t>(target));
          request.items.push_back(
              {TransferRef::HostBytes(source, block, src_gpu ? MemLoc::GPU : MemLoc::CPU,
                                      src_gpu ? device : -1),
               0,
               TransferRef::HostBytes(target, block, dst_gpu ? MemLoc::GPU : MemLoc::CPU,
                                      dst_gpu ? device : -1),
               0, block, k});
        }
      }
      if (!sim) Payload(src.get(), src_gpu, expected, false);
      const bool retained_read = client && read_file && prefill_bytes;
      if (read_file && (tier || retained_read)) {
        for (const auto& request : requests)
          Check(tier ? tier->BatchWrite(request.keys, request.src_data, sizes)
                     : client->BatchPut(request.keys, request.srcs, sizes),
                count);
        if (prefill_bytes) {
          filler = Allocate(prefill_chunk, false, sim);
          const auto address = reinterpret_cast<uintptr_t>(filler.get());
          if (client && !client->RegisterMemory(address, prefill_chunk, MemLoc::CPU, -1,
                                                MemoryRegistration::kLocalCopyOnly))
            throw std::runtime_error("SSD prefill registration failed");
          for (size_t offset = 0; offset < prefill_bytes;) {
            const size_t size = std::min(prefill_chunk, prefill_bytes - offset);
            const std::vector<std::string> keys{"fill-" + std::to_string(offset / prefill_chunk)};
            Check(tier ? tier->BatchWrite(keys, {filler.get()}, {size})
                       : client->BatchPut(keys, {address}, {size}),
                  1);
            offset += size;
          }
        }
        // Drain seed writes before timing reads, including relaxed durability.
        if (disk)
          for (const auto& file : std::filesystem::recursive_directory_iterator(disk->dir)) {
            if (!file.is_regular_file()) continue;
            const int fd = open(file.path().c_str(), O_RDWR);
            if (fd < 0) throw std::runtime_error("SSD seed file open failed");
            const int result = fsync(fd);
            close(fd);
            if (result != 0) throw std::runtime_error("SSD seed fsync failed");
          }
        // An idle wait does not flush the device cache or fix physical extent placement.
        std::this_thread::sleep_for(std::chrono::milliseconds(ssd_options.settle_ms));
      }
      std::set<int> registered_fds;
      std::vector<int64_t> registration_times;
      auto prepare_files = [&] {
        if (staged_engine || client) return;
        for (size_t c = 0; c < concurrency; ++c) {
          auto& request = requests[c];
          request.items.clear();
          request.records.clear();
          for (size_t k = 0; k < count; ++k) {
            const size_t index = c * count + k;
            RecordLocation location;
            if (sim) {
              location = {1, index * record_bytes + segment::PrefixBytes(20),
                          segment::PaddedValueBytes(block), block, true};
            } else if (read_file) {
              auto found = tier->LocateRecord(request.keys[k]);
              if (!found) throw std::runtime_error("SSD record location missing");
              location = *found;
            } else {
              auto reserved = tier->ReserveRecord(request.keys[k], block);
              if (!reserved) throw std::runtime_error("SSD file reservation failed");
              request.records.push_back(*reserved);
              location = reserved->location;
            }
            TransferRef file;
            if (path == "file_gpu") {
              const bool first = registered_fds.insert(location.fd).second;
              const auto started = Clock::now();
              file =
                  engine->RegisterFile(location.fd, location.value_offset, location.readable_size);
              const auto ns =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
                      .count();
              if (first) registration_times.push_back(ns);
            } else {
              file = TransferRef::File(location.fd, location.value_offset, location.readable_size);
            }
            if (!file.IsFile() || (path == "file_gpu" && !file.gds_handle))
              throw std::runtime_error("SSD file registration failed");
            const auto memory =
                TransferRef::HostBytes(read_file ? dst.get() : src.get(), total,
                                       read_file && dst_gpu ? MemLoc::GPU : MemLoc::CPU,
                                       read_file && dst_gpu ? device : -1);
            request.items.push_back(read_file
                                        ? TransferItem{file, 0, memory, index * block, block, k}
                                        : TransferItem{memory, index * block, file, 0, block, k});
          }
        }
      };
      if (ssd && (read_file || sim)) prepare_files();
      std::vector<size_t> segments(concurrency, ssd ? count : 0);
      if (report_segments && !staged_engine && !client && (!ssd || read_file || sim)) {
        for (size_t c = 0; c < concurrency; ++c) {
          segments[c] = 0;
          const auto geometry = engine->Plan(requests[c].items);
          if (!geometry.rejected_tags.empty()) throw std::runtime_error("rejected transfer");
          for (const auto& plan : geometry.plans) segments[c] += plan.sizes.size();
        }
      }
      auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
      };
      std::vector<std::pair<bool, std::vector<int64_t>>> measurements;
      std::vector<std::vector<int64_t>> wall_measurements;
      std::vector<bool> gathered;
      for (int i = -8; i < samples; ++i) {
        if (client && !retained_read && !client->Clear())
          throw std::runtime_error("UMBP calibration clear failed");
        if (tier && !read_file) {
          tier->Clear();
          if (!staged) prepare_files();
        }
        // GPU caches are not flushed; cold mode applies only to a host source.
        if (cold && !sim && !src_gpu && !read_file) FlushSource(src.get(), total);
        for (bool get : {false, true}) {
          if (retained_read && !get) continue;
          if (staged_engine ? get != read_file : engine && get) continue;
          if (client && read_file && get)
            std::this_thread::sleep_for(std::chrono::milliseconds(ssd_options.api_settle_ms));
          // One training caller makes the real launch counter unambiguous. Sample
          // outside timing, including warmups, but retain labels only for measured calls.
          const auto launches = training && !sim && host_device ? DeviceGatherLaunchCount() : 0;
          std::vector<int64_t> wall(concurrency, 0);
          auto times = Concurrent(concurrency, [&](size_t c) {
            // Newly-created caller threads establish HIP state before timing.
            if (!sim && (src_gpu || dst_gpu)) Hip(hipSetDevice(device));
            const auto& request = requests[c];
            if (staged_engine) {
              if (training) payload_ns = 0;  // Only training installs the observer.
              const auto start = Clock::now();
              if (sim) {
                auto* simulated = dynamic_cast<SimTransferEngine*>(engine.get());
                const auto model_bytes =
                    get ? segment::PaddedValueBytes(block) * count : record_bytes * count;
                if (!simulated->SimulateStorage(get, model_bytes, count,
                                                path != "file_cpu_staged_no_crc"))
                  throw std::runtime_error("simulated SSD operation failed");
              } else if (get) {
                Check(tier->ReadBatchIntoPtr(request.keys, request.dsts, sizes), count);
              } else {
                Check(tier->BatchWrite(request.keys, request.src_data, sizes), count);
              }
              wall[c] = elapsed(start);
              if (training && payload_ns == 0)
                throw std::runtime_error("SSD payload observer missing (oversized batch?)");
              return training ? static_cast<int64_t>(payload_ns) : wall[c];
            }
            if (engine) {
              auto plan = engine->Plan(request.items);
              if (!plan.rejected_tags.empty()) throw std::runtime_error("rejected transfer");
              const auto start = Clock::now();
              auto handle = engine->Submit(std::move(plan.plans));
              if (!handle) throw std::runtime_error("UMBP calibration transfer failed");
              std::vector<TransferFailure> failures;
              handle->Wait(&failures);
              const auto ns = elapsed(start);
              wall[c] = ns;
              if (!failures.empty()) throw std::runtime_error("UMBP calibration copy failed");
              if (ssd && !read_file && !sim)
                for (size_t k = 0; k < count; ++k)
                  if (!tier->CommitRecord(request.keys[k], request.records[k]))
                    throw std::runtime_error("SSD file commit failed");
              return ns;
            }
            const auto start = Clock::now();
            const auto result = get ? client->BatchGet(request.keys, request.dsts, sizes)
                                    : client->BatchPut(request.keys, request.srcs, sizes);
            const auto ns = elapsed(start);
            wall[c] = ns;
            Check(result, count);
            return ns;
          });
          if (i >= 0 && (engine || staged_engine || path == "cpu_cpu" || path == "gpu_gpu" ||
                         get == (ssd ? read_file : path == "cpu_gpu"))) {
            measurements.emplace_back(get, std::move(times));
            wall_measurements.push_back(std::move(wall));
            gathered.push_back(training && !sim && host_device &&
                               DeviceGatherLaunchCount() > launches);
          }
        }
      }
      if (!sim) {
        if (tier && !read_file)
          for (const auto& request : requests)
            Check(tier->ReadBatchIntoPtr(request.keys, request.dsts, sizes), count);
        std::vector<char> actual(total);
        Payload(dst.get(), dst_gpu, actual, true);
        if (actual != expected) throw std::runtime_error("UMBP calibration payload mismatch");
      }
      for (size_t i = 0; i < registration_times.size(); ++i) {
        std::cout << block << ',' << count << ",1," << concurrency << ",register," << i << ','
                  << registration_times[i] << ",0," << path << ",0,0," << registration_times[i]
                  << ',' << disk_config.tier_io_threads << '\n';
      }
      for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& [get, times] = measurements[i];
        for (size_t c = 0; c < times.size(); ++c) {
          std::cout << block << ',' << count << ','
                    << (path == "cpu_cpu" ? LocalCopyThreads(bytes, count) : 1) << ','
                    << concurrency << ','
                    << (engine || staged_engine ? "transfer"
                        : get                   ? "get"
                                                : "put")
                    << ','
                    << (engine || staged_engine || (path != "cpu_cpu" && path != "gpu_gpu") ? i
                                                                                            : i / 2)
                    << ',' << times[c];
          if (report_segments) std::cout << ',' << segments[c];
          std::cout << ',' << path << (gathered[i] ? "_gather" : "");
          if (ssd) {
            const size_t model_bytes =
                staged_engine || client
                    ? (get ? segment::PaddedValueBytes(block) * count : record_bytes * count)
                    : bytes;
            std::cout << ',' << model_bytes << ',' << gpu_caller << ',' << wall_measurements[i][c]
                      << ',' << disk_config.tier_io_threads;
          }
          std::cout << '\n';
        }
      }
    }
  }
  if (cases == 0) throw std::invalid_argument("no workload fits --max-mib for this concurrency");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 ||
        (std::string(argv[1]) != "train" && std::string(argv[1]) != "engine" &&
         std::string(argv[1]) != "api") ||
        (std::string(argv[2]) != "cold" && std::string(argv[2]) != "reuse"))
      throw std::invalid_argument(
          "usage: umbp_sim_probe train|engine|api cold|reuse samples [profile] "
          "[--path cpu_cpu|cpu_gpu|gpu_cpu|gpu_gpu|file_cpu|cpu_file|file_gpu|"
          "file_cpu_staged|file_cpu_staged_no_crc|cpu_file_staged] [--device 0] "
          "[--concurrency 1|2|4] [--max-mib 256] [--block-bytes B,...] [--batches N,...] "
          "[--ssd-dir DIR] [--ssd-io posix|io_uring] [--ssd-durability relaxed|strict] "
          "[--ssd-settle-ms 10] [--ssd-api-settle-ms 0] [--ssd-read-prefill-mib 0] "
          "[--ssd-no-crc] [--gpu-caller]");
    std::string profile, path = "cpu_cpu";
    int device = 0;
    size_t concurrency = 1, max_mib = 256;
    std::vector<size_t> blocks, batches;
    SsdOptions ssd;
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      if ((arg == "--concurrency" || arg == "--max-mib") && i + 1 < argc) {
        (arg == "--concurrency" ? concurrency : max_mib) = Positive(argv[++i]);
      } else if ((arg == "--block-bytes" || arg == "--batches") && i + 1 < argc) {
        (arg == "--block-bytes" ? blocks : batches) = Sizes(argv[++i]);
      } else if (arg == "--path" && i + 1 < argc) {
        path = argv[++i];
      } else if (arg == "--ssd-dir" && i + 1 < argc) {
        ssd.dir = argv[++i];
      } else if (arg == "--ssd-read-prefill-mib" && i + 1 < argc) {
        const std::string value = argv[++i];
        ssd.read_prefill_mib = value == "0" ? 0 : Positive(value);
        if (ssd.read_prefill_mib > std::numeric_limits<size_t>::max() >> 20)
          throw std::invalid_argument("SSD prefill size is too large");
      } else if ((arg == "--ssd-settle-ms" || arg == "--ssd-api-settle-ms") && i + 1 < argc) {
        const std::string value = argv[++i];
        const auto delay = value == "0" ? 0 : Positive(value);
        if (delay > std::numeric_limits<int>::max())
          throw std::invalid_argument("SSD settle delay is too large");
        (arg == "--ssd-settle-ms" ? ssd.settle_ms : ssd.api_settle_ms) = static_cast<int>(delay);
      } else if (arg == "--ssd-io" && i + 1 < argc) {
        const std::string value = argv[++i];
        if (value != "posix" && value != "io_uring")
          throw std::invalid_argument("--ssd-io must be posix or io_uring");
        ssd.config.io.backend = value == "posix" ? UMBPIoBackend::Posix : UMBPIoBackend::IoUring;
      } else if (arg == "--ssd-durability" && i + 1 < argc) {
        const std::string value = argv[++i];
        if (value != "relaxed" && value != "strict")
          throw std::invalid_argument("--ssd-durability must be relaxed or strict");
        ssd.config.durability.mode =
            value == "relaxed" ? UMBPDurabilityMode::Relaxed : UMBPDurabilityMode::Strict;
      } else if (arg == "--ssd-no-crc") {
        ssd.config.verify_crc = false;
      } else if (arg == "--gpu-caller") {
        ssd.gpu_caller = true;
      } else if (arg == "--device" && i + 1 < argc) {
        const std::string value = argv[++i];
        size_t parsed = 0;
        device = std::stoi(value, &parsed);
        if (value.empty() || value.front() < '0' || value.front() > '9' || device < 0 ||
            parsed != value.size())
          throw std::invalid_argument("--device must be a nonnegative GPU ordinal");
      } else if (profile.empty() && arg.rfind("--", 0) != 0) {
        profile = arg;
      } else {
        throw std::invalid_argument("unknown or incomplete option: " + arg);
      }
    }
    if (path != "cpu_cpu" && path != "cpu_gpu" && path != "gpu_cpu" && path != "gpu_gpu" &&
        path != "file_cpu" && path != "cpu_file" && path != "file_gpu" &&
        path != "file_cpu_staged" && path != "file_cpu_staged_no_crc" && path != "cpu_file_staged")
      throw std::invalid_argument("unknown transfer path: " + path);
    if (path == "file_cpu_staged_no_crc" && std::string(argv[1]) == "api")
      throw std::invalid_argument("file_cpu_staged_no_crc is for train/engine probes only");
    if (ssd.gpu_caller &&
        (std::string(argv[1]) != "api" || (path != "file_cpu_staged" && path != "cpu_file_staged")))
      throw std::invalid_argument("--gpu-caller is for staged SSD API validation only");
    if ((concurrency != 1 && concurrency != 2 && concurrency != 4) ||
        (std::string(argv[1]) == "train" && (!profile.empty() || concurrency != 1)))
      throw std::invalid_argument(
          "training requires real single-caller transfers; concurrency is 1/2/4");
    const auto samples = Positive(argv[3]);
    if (samples > std::numeric_limits<int>::max() ||
        max_mib > std::numeric_limits<size_t>::max() / (size_t{2} << 20))
      throw std::invalid_argument("samples or --max-mib is too large");
    Measure(argv[1], std::string(argv[2]) == "cold", static_cast<int>(samples), profile,
            concurrency, max_mib << 20, std::move(blocks), std::move(batches), path, device, ssd);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
