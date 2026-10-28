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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/sim/sim.h"
#include "umbp/umbp_client.h"

namespace {
using namespace mori::umbp;
using Clock = std::chrono::steady_clock;
using Buffer = std::unique_ptr<char, decltype(&std::free)>;

Buffer Allocate(size_t bytes, int value) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 4096, bytes) != 0) throw std::bad_alloc();
  Buffer buffer(static_cast<char*>(ptr), &std::free);
  std::memset(ptr, value, bytes);
  return buffer;
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

// Start callers together; thread creation and wake-up are outside each API timer.
// Each caller owns disjoint keys/buffers but shares the same client/transfer engine.
template <class Fn>
std::vector<int64_t> Concurrent(size_t count, Fn run) {
  if (count == 1) return {run(0)};
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::atomic<size_t> ready{0};
  std::vector<std::future<int64_t>> pending;
  pending.reserve(count);
  try {
    for (size_t i = 0; i < count; ++i) {
      pending.push_back(std::async(std::launch::async, [&, i] {
        ++ready;
        gate.wait();
        return run(i);
      }));
    }
    while (ready.load() != count) std::this_thread::yield();
  } catch (...) {
    start.set_value();  // Drain already-created callers if thread creation failed.
    throw;
  }
  start.set_value();
  std::vector<int64_t> times;
  for (auto& call : pending) times.push_back(call.get());
  return times;
}

void Measure(const std::string& phase, bool cold, int samples, const std::string& profile,
             size_t concurrency, size_t max_bytes, std::vector<size_t> blocks,
             std::vector<size_t> batches) {
  const bool training = phase == "train", sim = !profile.empty();
  const bool custom = !blocks.empty() || !batches.empty();
  if (blocks.empty())
    blocks = training ? std::vector<size_t>{512, 2048, 4096, 65536, 262144, 1048576, 4194304}
                      : std::vector<size_t>{1024, 8192, 32768, 131072, 524288, 2097152, 8388608};
  if (batches.empty())
    batches =
        training ? std::vector<size_t>{1, 8, 32, 64, 128} : std::vector<size_t>{1, 4, 16, 128};
  if (training && !cold && !custom) max_bytes = std::min(max_bytes, size_t{8} << 20);
  size_t cases = 0;
  std::cout << "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns\n";
  for (size_t block : blocks) {
    for (size_t count : batches) {
      // Check the aggregate cap before multiplying user-supplied dimensions.
      if (block > max_bytes / concurrency / count) continue;
      const size_t bytes = block * count, total = bytes * concurrency;
      // A sparse concurrent sweep keeps calibration affordable; never used for fitting.
      if (!custom && concurrency > 1 &&
          ((block != 8192 && block != 524288 && block != 2097152) || count == 4))
        continue;
      ++cases;
      auto src = Allocate(total, 0x5a), dst = Allocate(total, 0);
      std::unique_ptr<TransferEngine> engine;
      std::unique_ptr<IUMBPClient> client;
      if (phase == "api") {
        UMBPConfig config;
        config.page_size = block;
        config.dram.capacity_bytes = std::max(total * 2, size_t{16} << 20);
        config.dram.prefault_threads = 1;
        config.ssd.enabled = false;
        config.sim.enabled = sim;
        config.sim.profile_path = profile;
        config.sim.trace_path = "/dev/null";
        client = CreateUMBPClient(config);
        if (!client->RegisterMemory(reinterpret_cast<uintptr_t>(src.get()), total) ||
            !client->RegisterMemory(reinterpret_cast<uintptr_t>(dst.get()), total))
          throw std::runtime_error("UMBP calibration registration failed");
      } else if (sim) {
        engine = std::make_unique<SimTransferEngine>(profile);
      } else {
        engine = std::make_unique<LocalCopyEngine>();
      }
      struct Request {
        std::vector<std::string> keys;
        std::vector<uintptr_t> srcs, dsts;
        std::vector<TransferItem> items;
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
          std::memcpy(source, &index, std::min(block, sizeof(index)));
          request.keys.push_back(std::to_string(index));
          request.srcs.push_back(reinterpret_cast<uintptr_t>(source));
          request.dsts.push_back(reinterpret_cast<uintptr_t>(target));
          request.items.push_back({TransferRef::HostBytes(source, block), 0,
                                   TransferRef::HostBytes(target, block), 0, block, k});
        }
      }
      auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
      };
      std::vector<std::pair<bool, std::vector<int64_t>>> measurements;
      for (int i = -8; i < samples; ++i) {
        if (client && !client->Clear()) throw std::runtime_error("UMBP calibration clear failed");
        if (cold && !sim) FlushSource(src.get(), total);
        for (bool get : {false, true}) {
          if (engine && get) continue;
          auto times = Concurrent(concurrency, [&](size_t c) {
            const auto& request = requests[c];
            if (engine) {
              auto plan = engine->Plan(request.items);
              if (!plan.rejected_tags.empty()) throw std::runtime_error("rejected transfer");
              const auto start = Clock::now();
              auto handle = engine->Submit(std::move(plan.plans));
              if (!handle) throw std::runtime_error("UMBP calibration transfer failed");
              std::vector<TransferFailure> failures;
              handle->Wait(&failures);
              const auto ns = elapsed(start);
              if (!failures.empty()) throw std::runtime_error("UMBP calibration copy failed");
              return ns;
            }
            const auto start = Clock::now();
            const auto result = get ? client->BatchGet(request.keys, request.dsts, sizes)
                                    : client->BatchPut(request.keys, request.srcs, sizes);
            const auto ns = elapsed(start);
            Check(result, count);
            return ns;
          });
          if (i >= 0) measurements.emplace_back(get, std::move(times));
        }
      }
      if (!sim && std::memcmp(src.get(), dst.get(), total) != 0)
        throw std::runtime_error("UMBP calibration payload mismatch");
      for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& [get, times] = measurements[i];
        for (auto ns : times)
          std::cout << block << ',' << count << ',' << LocalCopyThreads(bytes, count) << ','
                    << concurrency << ','
                    << (engine ? "transfer"
                        : get  ? "get"
                               : "put")
                    << ',' << (engine ? i : i / 2) << ',' << ns << '\n';
      }
    }
  }
  if (cases == 0) throw std::invalid_argument("no workload fits --max-mib for this concurrency");
}

size_t Positive(const std::string& text) {
  size_t parsed = 0;
  const auto value = std::stoull(text, &parsed);
  if (text.front() < '0' || text.front() > '9' || value == 0 || parsed != text.size() ||
      value > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("expected a positive integer");
  return static_cast<size_t>(value);
}

std::vector<size_t> Sizes(const std::string& text) {
  if (text.empty() || text.back() == ',') throw std::invalid_argument("empty size list entry");
  std::istringstream input(text);
  std::string item;
  std::vector<size_t> out;
  while (std::getline(input, item, ',')) out.push_back(Positive(item));
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
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
          "[--concurrency 1|2|4] [--max-mib 256] [--block-bytes B,...] [--batches N,...]");
    std::string profile;
    size_t concurrency = 1, max_mib = 256;
    std::vector<size_t> blocks, batches;
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      if ((arg == "--concurrency" || arg == "--max-mib") && i + 1 < argc) {
        (arg == "--concurrency" ? concurrency : max_mib) = Positive(argv[++i]);
      } else if ((arg == "--block-bytes" || arg == "--batches") && i + 1 < argc) {
        (arg == "--block-bytes" ? blocks : batches) = Sizes(argv[++i]);
      } else if (profile.empty() && arg.rfind("--", 0) != 0) {
        profile = arg;
      } else {
        throw std::invalid_argument("unknown or incomplete option: " + arg);
      }
    }
    if ((concurrency != 1 && concurrency != 2 && concurrency != 4) ||
        (std::string(argv[1]) == "train" && (!profile.empty() || concurrency != 1)))
      throw std::invalid_argument(
          "training requires real single-caller transfers; concurrency is 1/2/4");
    const auto samples = Positive(argv[3]);
    if (samples > std::numeric_limits<int>::max() ||
        max_mib > std::numeric_limits<size_t>::max() / (size_t{2} << 20))
      throw std::invalid_argument("samples or --max-mib is too large");
    Measure(argv[1], std::string(argv[2]) == "cold", static_cast<int>(samples), profile,
            concurrency, max_mib << 20, std::move(blocks), std::move(batches));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
