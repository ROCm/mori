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
// Verifies the ionic per-MR size ceilings that aborted the 1536 GiB UMBP DRAM
// pool (ibv_reg_mr -> ENOMEM). An MR needs one 8-byte page-table entry per page,
// and two limits apply to that entry count:
//   - driver: ionic_pgtbl_init() kmallocs the table flat and returns -ENOMEM
//     above KMALLOC_MAX_SIZE (4 MiB on x86_64, MAX_PAGE_ORDER=10), i.e. 524288
//     entries;
//   - firmware: the create-MR admin command fails (EINVAL) above 502528 entries.
//     Measured on fw 1.117.5-a-77 (ionic_rdma 26.03.3.001) with both 4 KiB and
//     2 MiB pages (1005056 MiB registers, 1005058 MiB does not); not documented, so
//     re-measure with --fw-max-entries 0 on other firmware.
// Usable ceiling per MR is therefore 502528 pages: ~1963 MiB of 4 KiB pages,
// ~981.5 GiB of 2 MiB pages.
//
// The largest case is mmapped once with the requested page type and faulted in;
// each case registers a prefix of it with the access flags mori uses (15). The
// predicted outcome comes from the entry count; the test fails on any mismatch.
//
// Standalone on purpose (libibverbs only) so it runs on a bare host:
//   g++ -O2 -std=c++17 -o test_ionic_mr_limit test_ionic_mr_limit.cpp -libverbs
//   ./test_ionic_mr_limit --page 4k --sizes 1G,1963M,1964M,2G+64K,3G
//   ./test_ionic_mr_limit --page 2m --sizes 980G,1005056M,1005058M,1023G   # ~1 TiB hugepages
// run_ionic_mr_limit.sh drives both tiers and the hugepage reservation.
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <unistd.h>

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

constexpr uint64_t KiB = 1ULL << 10;
constexpr uint64_t MiB = 1ULL << 20;
constexpr uint64_t GiB = 1ULL << 30;

// IBV_ACCESS_LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ | REMOTE_ATOMIC, the
// accessFlag:15 in mori's RegisterRdmaMemoryRegionAuto failure line.
constexpr int kMoriAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

enum class Expect { kOk, kEinval, kEnomem, kBoundary };

struct Options {
  std::string device;
  uint64_t page_size = 4 * KiB;
  std::vector<uint64_t> sizes;
  uint64_t kmalloc_max = 4 * MiB;
  // ionic v1 pads the table to BIT(cl_stride - pte_stride) entries; 8 is the
  // usual 64 B line over 8 B PTEs. Only used to flag cases too close to call.
  uint64_t pad_entries = 8;
  // Firmware create-MR cap; 0 disables the EINVAL prediction.
  uint64_t fw_max_entries = 502528;
};

// "1G", "2G-64K", "2G+64K", "1536G": a term or a sum/difference of terms.
bool ParseSize(const std::string& text, uint64_t* out) {
  int64_t total = 0;
  size_t pos = 0;
  int sign = 1;
  while (pos < text.size()) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str() + pos, &end, 10);
    size_t next = static_cast<size_t>(end - text.c_str());
    if (next == pos) return false;
    uint64_t unit = 1;
    if (next < text.size()) {
      switch (text[next]) {
        case 'K':
          unit = KiB;
          ++next;
          break;
        case 'M':
          unit = MiB;
          ++next;
          break;
        case 'G':
          unit = GiB;
          ++next;
          break;
        case 'T':
          unit = 1024 * GiB;
          ++next;
          break;
        default:
          break;
      }
    }
    total += sign * static_cast<int64_t>(value * unit);
    pos = next;
    if (pos < text.size()) {
      if (text[pos] != '+' && text[pos] != '-') return false;
      sign = text[pos] == '+' ? 1 : -1;
      ++pos;
    }
  }
  if (total <= 0) return false;
  *out = static_cast<uint64_t>(total);
  return true;
}

std::string HumanSize(uint64_t bytes) {
  char buf[64];
  if (bytes % GiB == 0) {
    std::snprintf(buf, sizeof(buf), "%lluG", static_cast<unsigned long long>(bytes / GiB));
  } else if (bytes % MiB == 0) {
    std::snprintf(buf, sizeof(buf), "%lluM", static_cast<unsigned long long>(bytes / MiB));
  } else {
    std::snprintf(buf, sizeof(buf), "%lluK", static_cast<unsigned long long>(bytes / KiB));
  }
  return buf;
}

Expect Predict(const Options& opt, uint64_t size, uint64_t* entries) {
  *entries = (size + opt.page_size - 1) / opt.page_size;
  const uint64_t unpadded_bytes = *entries * 8;
  const uint64_t padded_entries =
      1 + ((*entries - 1 + opt.pad_entries - 1) / opt.pad_entries) * opt.pad_entries;
  if (unpadded_bytes > opt.kmalloc_max) return Expect::kEnomem;
  if (padded_entries * 8 > opt.kmalloc_max) return Expect::kBoundary;
  if (opt.fw_max_entries == 0) return Expect::kOk;
  return *entries <= opt.fw_max_entries ? Expect::kOk : Expect::kEinval;
}

void* MapRegion(const Options& opt, uint64_t size) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (opt.page_size != 4 * KiB) {
    flags |= MAP_HUGETLB | MAP_POPULATE | (__builtin_ctzll(opt.page_size) << MAP_HUGE_SHIFT);
  }
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (addr == MAP_FAILED) return nullptr;
  if (opt.page_size == 4 * KiB) {
    // Keep THP from backing the range with 2 MiB pages, which would let the
    // driver pick a larger block size and move the ceiling.
    madvise(addr, size, MADV_NOHUGEPAGE);
    for (uint64_t off = 0; off < size; off += 4 * KiB) static_cast<volatile char*>(addr)[off] = 1;
  }
  return addr;
}

ibv_context* OpenDevice(const std::string& wanted) {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (list == nullptr || num == 0) return nullptr;
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

bool ParseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc) return false;
    const std::string value = argv[++i];
    if (arg == "--dev") {
      opt->device = value;
    } else if (arg == "--page") {
      if (value == "4k")
        opt->page_size = 4 * KiB;
      else if (value == "2m")
        opt->page_size = 2 * MiB;
      else if (value == "1g")
        opt->page_size = GiB;
      else
        return false;
    } else if (arg == "--sizes") {
      size_t start = 0;
      while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string token =
            value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        uint64_t size = 0;
        if (!ParseSize(token, &size)) return false;
        opt->sizes.push_back(size);
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    } else if (arg == "--kmalloc-max") {
      if (!ParseSize(value, &opt->kmalloc_max)) return false;
    } else if (arg == "--fw-max-entries") {
      opt->fw_max_entries = std::strtoull(value.c_str(), nullptr, 10);
    } else if (arg == "--pad-entries") {
      opt->pad_entries = std::strtoull(value.c_str(), nullptr, 10);
      if (opt->pad_entries == 0) return false;
    } else {
      return false;
    }
  }
  return !opt->sizes.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) {
    std::fprintf(stderr,
                 "usage: %s --page 4k|2m|1g --sizes S[,S...] [--dev ionic_N] "
                 "[--kmalloc-max 4M] [--fw-max-entries 502528] [--pad-entries 8]\n",
                 argv[0]);
    return 2;
  }

  ibv_context* ctx = OpenDevice(opt.device);
  if (ctx == nullptr) {
    std::fprintf(stderr, "no RDMA device %s\n", opt.device.empty() ? "ionic*" : opt.device.c_str());
    return 2;
  }
  ibv_pd* pd = ibv_alloc_pd(ctx);
  if (pd == nullptr) {
    std::fprintf(stderr, "ibv_alloc_pd failed: %s\n", std::strerror(errno));
    return 2;
  }

  std::printf(
      "device=%s page=%s driver_max_entries=%llu (ENOMEM above %s) fw_max_entries=%llu "
      "(EINVAL above %s)\n",
      ibv_get_device_name(ctx->device), HumanSize(opt.page_size).c_str(),
      static_cast<unsigned long long>(opt.kmalloc_max / 8),
      HumanSize(opt.kmalloc_max / 8 * opt.page_size).c_str(),
      static_cast<unsigned long long>(opt.fw_max_entries),
      opt.fw_max_entries ? HumanSize(opt.fw_max_entries * opt.page_size).c_str() : "-");
  std::printf("%-10s %10s %10s %-9s %-9s %9s %s\n", "size", "entries", "table", "expect", "got",
              "reg_s", "verdict");

  // Map the largest case once and register prefixes of it: faulting in ~1 TiB
  // of hugetlb memory per case would dominate the run.
  uint64_t map_size = 0;
  for (const uint64_t size : opt.sizes) map_size = size > map_size ? size : map_size;
  const auto map_t0 = std::chrono::steady_clock::now();
  void* addr = MapRegion(opt, map_size);
  if (addr == nullptr) {
    std::fprintf(stderr, "mmap %s of %s pages failed: %s\n", HumanSize(map_size).c_str(),
                 HumanSize(opt.page_size).c_str(), std::strerror(errno));
    return 2;
  }
  std::printf("mapped %s in %.1f s\n", HumanSize(map_size).c_str(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - map_t0).count());

  int failures = 0;
  for (const uint64_t size : opt.sizes) {
    uint64_t entries = 0;
    const Expect expect = Predict(opt, size, &entries);
    const char* expect_str = expect == Expect::kOk       ? "OK"
                             : expect == Expect::kEinval ? "EINVAL"
                             : expect == Expect::kEnomem ? "ENOMEM"
                                                         : "boundary";

    const auto t0 = std::chrono::steady_clock::now();
    ibv_mr* mr = ibv_reg_mr(pd, addr, size, kMoriAccessFlags);
    const int reg_errno = errno;
    const double reg_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const bool ok = mr != nullptr;
    std::string got = "OK";
    if (!ok) {
      got = reg_errno == ENOMEM   ? "ENOMEM"
            : reg_errno == EINVAL ? "EINVAL"
                                  : "errno " + std::to_string(reg_errno);
    }
    const bool pass = expect == Expect::kBoundary || (expect == Expect::kOk && ok) ||
                      (expect == Expect::kEinval && !ok && reg_errno == EINVAL) ||
                      (expect == Expect::kEnomem && !ok && reg_errno == ENOMEM);
    if (!pass) ++failures;
    std::printf("%-10s %10llu %9.2fM %-9s %-9s %9.2f %s\n", HumanSize(size).c_str(),
                static_cast<unsigned long long>(entries), entries * 8.0 / MiB, expect_str,
                got.c_str(), reg_s,
                expect == Expect::kBoundary ? "INFO"
                : pass                      ? "PASS"
                                            : "FAIL");
    std::fflush(stdout);

    if (mr != nullptr) ibv_dereg_mr(mr);
  }
  munmap(addr, map_size);

  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  std::printf("%s: %d failure(s)\n", failures == 0 ? "PASSED" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
