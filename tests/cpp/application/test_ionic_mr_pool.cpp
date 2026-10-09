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
// Registers one hugetlb mapping as N equal, non-overlapping MRs to check
// whether a pool larger than the single-MR ceiling (502528 pages,
// test_ionic_mr_limit) can be registered by splitting it. Also tries the whole
// mapping as one MR first, for contrast. On fw 1.117.5-a-77 it cannot: 1536G as
// 2 x 768G registers mr0 and fails mr1 with EINVAL, so the 502528-page limit is
// a budget shared by every MR on the NIC, not a per-MR cap.
//
//   g++ -O2 -std=c++17 -o test_ionic_mr_pool test_ionic_mr_pool.cpp -libverbs
//   ./test_ionic_mr_pool --map 1536G --mrs 2 [--dev ionic_0]   # 2 MiB hugepages
#include <infiniband/verbs.h>
#include <sys/mman.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

namespace {

constexpr uint64_t MiB = 1ULL << 20;
constexpr uint64_t GiB = 1ULL << 30;
constexpr uint64_t kPage = 2 * MiB;
constexpr int kMoriAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

ibv_context* OpenDevice(const std::string& wanted) {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (list == nullptr) return nullptr;
  ibv_context* ctx = nullptr;
  for (int i = 0; i < num && ctx == nullptr; ++i) {
    const char* name = ibv_get_device_name(list[i]);
    if (wanted.empty() ? std::strncmp(name, "ionic", 5) == 0 : wanted == name) {
      ctx = ibv_open_device(list[i]);
    }
  }
  ibv_free_device_list(list);
  return ctx;
}

ibv_mr* Register(ibv_pd* pd, char* base, uint64_t bytes, const char* label) {
  const auto t0 = std::chrono::steady_clock::now();
  ibv_mr* mr = ibv_reg_mr(pd, base, bytes, kMoriAccessFlags);
  const int err = errno;
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %-14s offset=%6lluG size=%6lluG pages=%7llu -> %s (%.2f s)\n", label, 0ULL,
              static_cast<unsigned long long>(bytes / GiB),
              static_cast<unsigned long long>(bytes / kPage),
              mr != nullptr ? "OK" : std::strerror(err), s);
  std::fflush(stdout);
  return mr;
}

}  // namespace

int main(int argc, char** argv) {
  std::string device;
  uint64_t map_gib = 1536;
  uint64_t num_mrs = 2;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--dev")
      device = argv[i + 1];
    else if (arg == "--map")
      map_gib = std::strtoull(argv[i + 1], nullptr, 10);
    else if (arg == "--mrs")
      num_mrs = std::strtoull(argv[i + 1], nullptr, 10);
  }
  const uint64_t map_bytes = map_gib * GiB;
  if (num_mrs == 0 || map_bytes % (num_mrs * kPage) != 0) {
    std::fprintf(stderr, "--map must split into --mrs whole 2 MiB pages\n");
    return 2;
  }

  ibv_context* ctx = OpenDevice(device);
  ibv_pd* pd = ctx != nullptr ? ibv_alloc_pd(ctx) : nullptr;
  if (pd == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", device.empty() ? "ionic*" : device.c_str());
    return 2;
  }

  const auto t0 = std::chrono::steady_clock::now();
  void* addr = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE |
                        (__builtin_ctzll(kPage) << MAP_HUGE_SHIFT),
                    -1, 0);
  if (addr == MAP_FAILED) {
    std::fprintf(stderr, "mmap %lluG of 2 MiB hugepages failed: %s\n",
                 static_cast<unsigned long long>(map_gib), std::strerror(errno));
    return 2;
  }
  char* base = static_cast<char*>(addr);
  std::printf("device=%s mapped %lluG of 2 MiB hugepages in %.1f s\n",
              ibv_get_device_name(ctx->device), static_cast<unsigned long long>(map_gib),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

  std::printf("single MR:\n");
  ibv_mr* whole = Register(pd, base, map_bytes, "whole");
  if (whole != nullptr) ibv_dereg_mr(whole);

  std::printf("%llu MRs:\n", static_cast<unsigned long long>(num_mrs));
  const uint64_t part = map_bytes / num_mrs;
  std::vector<ibv_mr*> mrs;
  for (uint64_t i = 0; i < num_mrs; ++i) {
    const std::string label = "mr" + std::to_string(i);
    ibv_mr* mr = ibv_reg_mr(pd, base + i * part, part, kMoriAccessFlags);
    const int err = errno;
    std::printf("  %-14s offset=%6lluG size=%6lluG pages=%7llu -> %s\n", label.c_str(),
                static_cast<unsigned long long>(i * part / GiB),
                static_cast<unsigned long long>(part / GiB),
                static_cast<unsigned long long>(part / kPage),
                mr != nullptr ? "OK" : std::strerror(err));
    std::fflush(stdout);
    if (mr != nullptr) mrs.push_back(mr);
  }
  const bool all_ok = mrs.size() == num_mrs;
  std::printf("%s: %zu/%llu MRs registered, %lluG total while held together\n",
              all_ok ? "PASSED" : "FAILED", mrs.size(), static_cast<unsigned long long>(num_mrs),
              static_cast<unsigned long long>(mrs.size() * part / GiB));

  for (ibv_mr* mr : mrs) ibv_dereg_mr(mr);
  munmap(addr, map_bytes);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return all_ok ? 0 : 1;
}
