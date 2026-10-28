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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

void Measure(const std::string& phase, bool cold, int samples, const std::string& profile) {
  const bool training = phase == "train", sim = !profile.empty();
  const std::vector<size_t> blocks = training ? std::vector<size_t>{4, 64, 256, 1024, 4096}
                                              : std::vector<size_t>{8, 128, 512, 2048, 8192};
  const std::vector<size_t> batches =
      training ? std::vector<size_t>{1, 8, 32} : std::vector<size_t>{1, 4, 16};
  std::cout << "block_bytes,batch,effective_threads,api,iteration,elapsed_ns\n";
  for (size_t kib : blocks) {
    for (size_t count : batches) {
      const size_t block = kib * 1024, bytes = block * count;
      if (bytes > ((training && !cold ? 8ULL : 64ULL) << 20)) continue;
      auto src = Allocate(bytes, 0x5a), dst = Allocate(bytes, 0);
      std::unique_ptr<TransferEngine> engine;
      std::unique_ptr<IUMBPClient> client;
      if (phase == "api") {
        UMBPConfig config;
        config.page_size = block;
        config.dram.capacity_bytes = std::max(bytes * 2, size_t{16} << 20);
        config.dram.prefault_threads = 1;
        config.ssd.enabled = false;
        config.sim.enabled = sim;
        config.sim.profile_path = profile;
        config.sim.trace_path = "/dev/null";  // Include formatting and buffered output costs.
        client = CreateUMBPClient(config);
        if (!client->RegisterMemory(reinterpret_cast<uintptr_t>(src.get()), bytes) ||
            !client->RegisterMemory(reinterpret_cast<uintptr_t>(dst.get()), bytes))
          throw std::runtime_error("UMBP calibration registration failed");
      } else if (sim) {
        engine = std::make_unique<SimTransferEngine>(profile);
      } else {
        engine = std::make_unique<LocalCopyEngine>();
      }
      std::vector<std::string> keys;
      std::vector<uintptr_t> srcs, dsts;
      std::vector<size_t> sizes(count, block);
      std::vector<TransferItem> items;
      for (size_t k = 0; k < count; ++k) {
        char* source = src.get() + k * block;
        char* target = dst.get() + k * block;
        keys.push_back(std::to_string(k));
        srcs.push_back(reinterpret_cast<uintptr_t>(source));
        dsts.push_back(reinterpret_cast<uintptr_t>(target));
        items.push_back({TransferRef::HostBytes(source, block), 0,
                         TransferRef::HostBytes(target, block), 0, block, k});
      }
      std::vector<std::pair<int64_t, int64_t>> times;
      auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
      };
      for (int i = -8; i < samples; ++i) {
        if (client && !client->Clear()) throw std::runtime_error("UMBP calibration clear failed");
        if (cold && !sim) FlushSource(src.get(), bytes);
        int64_t first, second = 0;
        if (engine) {
          auto plan = engine->Plan(items);
          const auto start = Clock::now();
          auto handle = engine->Submit(std::move(plan.plans));
          if (!handle) throw std::runtime_error("UMBP calibration transfer failed");
          std::vector<TransferFailure> failures;
          handle->Wait(&failures);
          first = elapsed(start);
          if (!failures.empty()) throw std::runtime_error("UMBP calibration copy failed");
        } else {
          auto start = Clock::now();
          auto result = client->BatchPut(keys, srcs, sizes);
          first = elapsed(start);
          Check(result, count);
          start = Clock::now();
          result = client->BatchGet(keys, dsts, sizes);
          second = elapsed(start);
          Check(result, count);
        }
        if (i >= 0) times.emplace_back(first, second);
      }
      if (!sim && std::memcmp(src.get(), dst.get(), bytes) != 0)
        throw std::runtime_error("UMBP calibration payload mismatch");
      for (size_t i = 0; i < times.size(); ++i) {
        auto emit = [&](const char* api, int64_t ns) {
          std::cout << block << ',' << count << ',' << LocalCopyThreads(bytes, count) << ',' << api
                    << ',' << i << ',' << ns << '\n';
        };
        emit(client ? "put" : "transfer", times[i].first);
        if (client) emit("get", times[i].second);
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5 ||
        (std::string(argv[1]) != "train" && std::string(argv[1]) != "engine" &&
         std::string(argv[1]) != "api") ||
        (std::string(argv[2]) != "cold" && std::string(argv[2]) != "reuse") ||
        (argc == 5 && std::string(argv[1]) == "train"))
      throw std::invalid_argument(
          "usage: umbp_sim_probe train|engine|api cold|reuse samples [profile]");
    size_t parsed = 0;
    const int samples = std::stoi(argv[3], &parsed);
    if (samples <= 0 || parsed != std::strlen(argv[3]))
      throw std::invalid_argument("samples must be a positive integer");
    Measure(argv[1], std::string(argv[2]) == "cold", samples, argc == 5 ? argv[4] : "");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
