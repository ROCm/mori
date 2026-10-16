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
#include "src/application/transport/rdma/kfd_pin_accounting.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

namespace {

constexpr size_t kMinTrackedBytes = 64ull << 20;
constexpr size_t kProbeBytes = 64ull << 20;
constexpr int kNoise = 2;

std::string AmdgpuVersion() {
  std::ifstream f("/sys/module/amdgpu/version");
  std::string version;
  if (!(f >> version)) version = "in-tree";
  return version;
}

// The VRAM allocation of this process that contains [ptr, ptr + size).
bool FindVramAllocation(void* ptr, size_t size, uintptr_t* base, size_t* allocSize, int* device) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
      info.type != HSA_EXT_POINTER_TYPE_HSA) {
    return false;
  }
  hsa_device_type_t type;
  if (hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
      type != HSA_DEVICE_TYPE_GPU) {
    return false;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(info.agentBaseAddress);
  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  if (p < begin || p + size > begin + info.sizeInBytes) return false;

  hipPointerAttribute_t attr{};
  if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  *base = begin;
  *allocSize = info.sizeInBytes;
  *device = attr.device;
  return true;
}

double FreeVram() {
  size_t freeBytes = 0, total = 0;
  if (hipMemGetInfo(&freeBytes, &total) != hipSuccess) (void)hipGetLastError();
  return static_cast<double>(freeBytes);
}

// How far free VRAM moved, in probe BOs: 0, +1 or -1, or kNoise when it moved
// by anything else, i.e. someone allocated or freed memory meanwhile.
int BoSteps(double delta) {
  const double bo = static_cast<double>(kProbeBytes);
  for (int steps : {0, 1, -1}) {
    if (std::fabs(delta - steps * bo) < bo / 4) return steps;
  }
  return kNoise;
}

const char* StepsName(int steps) {
  switch (steps) {
    case 0:
      return "0";
    case 1:
      return "+1";
    case -1:
      return "-1";
    default:
      return "?";
  }
}

}  // namespace

KfdPinAccounting& KfdPinAccounting::Instance() {
  static KfdPinAccounting* instance = new KfdPinAccounting();
  return *instance;
}

KfdPinAccounting::KfdPinAccounting() {
  std::optional<std::string> value = env::GetString("MORI_KFD_PIN_FIX");
  if (!value.has_value()) return;
  if (env::detail::EqualsIgnoreCase(value->c_str(), "force")) {
    mode = Mode::PerPinCredit;
    MORI_APP_INFO("KFD pin accounting forced on by MORI_KFD_PIN_FIX=force");
    return;
  }
  std::optional<bool> on = env::detail::ParseBool(value->c_str());
  if (!on.has_value()) {
    MORI_APP_WARN("MORI_KFD_PIN_FIX={} not understood; use 1, 0 or force", *value);
    return;
  }
  if (*on) mode = Mode::Unprobed;
}

KfdPinAccounting::Mode KfdPinAccounting::Probe(ibv_pd* pd, int device) {
  int prevDevice = -1;
  if (hipGetDevice(&prevDevice) != hipSuccess) prevDevice = -1;
  void* buf = nullptr;
  if (hipSetDevice(device) != hipSuccess || hipMalloc(&buf, kProbeBytes) != hipSuccess) {
    (void)hipGetLastError();
    if (prevDevice >= 0) (void)hipSetDevice(prevDevice);
    MORI_APP_WARN("KFD pin accounting: probe buffer allocation failed on GPU {}; not adjusting",
                  device);
    return Mode::Unknown;
  }

  // Page-sized MRs still pin the whole probe BO, so each pin moves free VRAM by
  // 0 or one BO. The second probe runs only if something else changed free VRAM
  // during the first.
  const size_t page = static_cast<size_t>(getpagesize());
  Mode result = Mode::Unknown;
  bool peerOk = false;
  int dmabuf = kNoise, first = kNoise, second = kNoise;
  for (int attempt = 0; attempt < 2 && result == Mode::Unknown; ++attempt) {
    double before = FreeVram();
    uint64_t offset = 0;
    int fd = TryExportDmabufFd(buf, page, &offset);
    ibv_mr* viaDmabuf = fd >= 0
                            ? ibv_reg_dmabuf_mr(pd, offset, page, reinterpret_cast<uint64_t>(buf),
                                                fd, IBV_ACCESS_LOCAL_WRITE)
                            : nullptr;
    if (fd >= 0) close(fd);
    dmabuf = viaDmabuf ? BoSteps(FreeVram() - before) : kNoise;
    if (viaDmabuf) ibv_dereg_mr(viaDmabuf);

    before = FreeVram();
    ibv_mr* a = ibv_reg_mr(pd, buf, page, IBV_ACCESS_LOCAL_WRITE);
    peerOk = a != nullptr;
    first = a ? BoSteps(FreeVram() - before) : kNoise;
    ibv_mr* b = a ? ibv_reg_mr(pd, buf, page, IBV_ACCESS_LOCAL_WRITE) : nullptr;
    second = b ? BoSteps(FreeVram() - before) : kNoise;
    if (b) ibv_dereg_mr(b);
    if (a) ibv_dereg_mr(a);

    if (!peerOk) {
      if (dmabuf == -1) result = Mode::Unfixable;
      if (dmabuf == 0) result = Mode::Exact;
    } else if (first == 0 && second == 1 && (dmabuf == -1 || !viaDmabuf)) {
      result = Mode::PerPinCredit;
    } else if (first == 0 && second == 0 && (dmabuf == 0 || !viaDmabuf)) {
      result = Mode::Exact;
    } else if (first == -1 && second == -1) {
      result = Mode::Unfixable;
    }
  }
  (void)hipFree(buf);
  if (prevDevice >= 0) (void)hipSetDevice(prevDevice);

  const std::string version = AmdgpuVersion();
  switch (result) {
    case Mode::PerPinCredit:
      MORI_APP_INFO(
          "KFD pin accounting: amdgpu {} miscounts RDMA pins of VRAM (probe: dma-buf pin {}, "
          "peer-memory pins {} then {} BO); keeping one peer-memory pin per registered "
          "allocation",
          version, StepsName(dmabuf), StepsName(first), StepsName(second));
      break;
    case Mode::Exact:
      MORI_APP_INFO("KFD pin accounting: amdgpu {} counts RDMA pins of VRAM once; nothing to do",
                    version);
      break;
    case Mode::Unfixable:
      MORI_APP_ERROR(
          "KFD pin accounting: amdgpu {} subtracts registered VRAM a second time from "
          "hipMemGetInfo and later allocations (ROCm/mori#721; probe: dma-buf pin {}, "
          "peer-memory pins {} then {} BO){}; leave headroom for registered buffers or install "
          "amdgpu-dkms 31.40.1.1",
          version, StepsName(dmabuf), StepsName(first), StepsName(second),
          peerOk ? "" : ", and peer-memory MRs are unavailable to offset it");
      break;
    default:
      MORI_APP_WARN(
          "KFD pin accounting: probe inconclusive on amdgpu {} (dma-buf pin {}, peer-memory pins "
          "{} then {} BO); not adjusting",
          version, StepsName(dmabuf), StepsName(first), StepsName(second));
      break;
  }
  return result;
}

KfdPinAccounting::Registration KfdPinAccounting::Begin(ibv_pd* pd, void* ptr, size_t size) {
  Registration reg;
  Mode current = mode.load(std::memory_order_acquire);
  if (current != Mode::Unprobed && current != Mode::PerPinCredit) return reg;

  uintptr_t base = 0;
  size_t allocSize = 0;
  int device = -1;
  if (!FindVramAllocation(ptr, size, &base, &allocSize, &device) || allocSize < kMinTrackedBytes) {
    return reg;
  }

  std::unique_lock<std::mutex> lock(mu);
  if (mode.load(std::memory_order_acquire) == Mode::Unprobed) {
    mode.store(Probe(pd, device), std::memory_order_release);
  }
  if (mode.load(std::memory_order_acquire) != Mode::PerPinCredit) return reg;

  bos[base].size = allocSize;
  reg.owner = this;
  reg.lock = std::move(lock);
  reg.bo = base;
  reg.pd = pd;
  return reg;
}

KfdPinAccounting::Registration::Registration(Registration&& other) noexcept
    : owner(std::exchange(other.owner, nullptr)),
      lock(std::move(other.lock)),
      bo(other.bo),
      pd(other.pd),
      committed(other.committed) {}

KfdPinAccounting::Registration::~Registration() {
  if (owner == nullptr || committed) return;
  // Nothing got registered; drop an anchor taken for this attempt unless MRs
  // registered earlier rely on it.
  auto it = owner->bos.find(bo);
  if (it == owner->bos.end() || !it->second.members.empty()) return;
  if (it->second.anchor != nullptr) ibv_dereg_mr(it->second.anchor);
  owner->bos.erase(it);
}

bool KfdPinAccounting::Registration::PreferDmabuf() const {
  return owner != nullptr && HasPin(owner->bos.at(bo));
}

void KfdPinAccounting::Registration::BeforeDmabuf() {
  if (owner == nullptr) return;
  Bo& entry = owner->bos.at(bo);
  if (HasPin(entry)) return;
  entry.anchor = owner->RegisterAnchor(pd, bo);
  if (entry.anchor != nullptr) entry.anchorPd = pd;
}

void KfdPinAccounting::Registration::Commit(ibv_mr* mr, bool viaDmabuf) {
  if (owner == nullptr) return;
  Bo& entry = owner->bos.at(bo);
  const bool pinned = HasPin(entry);
  entry.members.push_back({mr, pd, !viaDmabuf});
  owner->boOfMr[mr] = bo;
  committed = true;
  if (viaDmabuf) return;

  if (entry.anchor != nullptr) {
    // This MR pins the BO through peer-memory itself, which makes the anchor a
    // second pin.
    ibv_dereg_mr(entry.anchor);
    entry.anchor = nullptr;
    entry.anchorPd = nullptr;
  } else if (pinned && !owner->doublePinWarned) {
    owner->doublePinWarned = true;
    MORI_APP_ERROR(
        "KFD pin accounting: VRAM allocation {:#x} ({} MiB) got a second peer-memory MR because "
        "dma-buf registration was not possible; hipMemGetInfo over-reports free VRAM by its size "
        "(ROCm/mori#721)",
        bo, entry.size >> 20);
  }
}

bool KfdPinAccounting::HasPin(const Bo& bo) {
  return bo.anchor != nullptr ||
         std::any_of(bo.members.begin(), bo.members.end(), [](const Member& m) { return m.peer; });
}

ibv_pd* KfdPinAccounting::PdForAnchor(const Bo& bo, ibv_pd* avoid) {
  for (const Member& m : bo.members) {
    if (m.pd != avoid) return m.pd;
  }
  return nullptr;
}

ibv_mr* KfdPinAccounting::RegisterAnchor(ibv_pd* pd, uintptr_t base) {
  ibv_mr* mr = ibv_reg_mr(pd, reinterpret_cast<void*>(base), static_cast<size_t>(getpagesize()),
                          IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr && !anchorWarned) {
    const int err = errno;
    anchorWarned = true;
    MORI_APP_ERROR(
        "KFD pin accounting: anchor MR for VRAM allocation {:#x} failed, errno {} ({}); its "
        "dma-buf MRs reduce hipMemGetInfo by its size (ROCm/mori#721)",
        base, err, strerror(err));
  }
  return mr;
}

int KfdPinAccounting::Deregister(ibv_mr* mr) {
  // MRs are tracked only once a probe or MORI_KFD_PIN_FIX=force selected PerPinCredit.
  if (mode.load(std::memory_order_acquire) != Mode::PerPinCredit) return ibv_dereg_mr(mr);
  std::unique_lock<std::mutex> lock(mu);
  auto owned = boOfMr.find(mr);
  if (owned == boOfMr.end()) {
    lock.unlock();
    return ibv_dereg_mr(mr);
  }
  const uintptr_t base = owned->second;
  boOfMr.erase(owned);
  Bo& entry = bos.at(base);
  auto it = std::find_if(entry.members.begin(), entry.members.end(),
                         [mr](const Member& m) { return m.mr == mr; });
  const Member gone = *it;
  entry.members.erase(it);

  if (gone.peer && !entry.members.empty() && !HasPin(entry)) {
    // `mr` holds the BO's pin; hand it to an anchor before letting go.
    ibv_pd* to = PdForAnchor(entry, gone.pd);
    if (to == nullptr) to = entry.members.front().pd;
    entry.anchor = RegisterAnchor(to, base);
    if (entry.anchor != nullptr) entry.anchorPd = to;
  }
  const int ret = ibv_dereg_mr(mr);
  if (entry.members.empty()) {
    if (entry.anchor != nullptr) ibv_dereg_mr(entry.anchor);
    bos.erase(base);
  }
  return ret;
}

void KfdPinAccounting::ReleasePd(ibv_pd* pd) {
  if (mode.load(std::memory_order_acquire) != Mode::PerPinCredit) return;
  std::lock_guard<std::mutex> lock(mu);
  for (auto it = bos.begin(); it != bos.end();) {
    Bo& entry = it->second;
    if (entry.anchor != nullptr && entry.anchorPd == pd) {
      ibv_pd* to = PdForAnchor(entry, pd);
      ibv_mr* moved = to != nullptr ? RegisterAnchor(to, it->first) : nullptr;
      ibv_dereg_mr(entry.anchor);
      entry.anchor = moved;
      entry.anchorPd = moved != nullptr ? to : nullptr;
    }
    if (entry.members.empty() && entry.anchor == nullptr) {
      it = bos.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace application
}  // namespace mori
