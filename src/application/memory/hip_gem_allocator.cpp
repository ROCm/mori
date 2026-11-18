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
#include "mori/application/memory/hip_gem_allocator.hpp"

#include <fcntl.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "infiniband/verbs.h"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace {

// The few pieces of the kernel UAPI (<drm/drm.h>, <drm/amdgpu_drm.h>) used here.
struct GemCreateIn {
  uint64_t boSize;
  uint64_t alignment;
  uint64_t domains;
  uint64_t domainFlags;
};
struct GemCreateOut {
  uint32_t handle;
  uint32_t pad;
};
union GemCreate {
  GemCreateIn in;
  GemCreateOut out;
};
struct PrimeHandle {
  uint32_t handle;
  uint32_t flags;
  int32_t fd;
};
struct GemClose {
  uint32_t handle;
  uint32_t pad;
};

constexpr unsigned long kIoctlAmdgpuGemCreate = _IOWR('d', 0x40 + 0x00, GemCreate);
constexpr unsigned long kIoctlPrimeHandleToFd = _IOWR('d', 0x2d, PrimeHandle);
constexpr unsigned long kIoctlGemClose = _IOW('d', 0x09, GemClose);
static_assert(kIoctlAmdgpuGemCreate == 0xC0206440UL, "DRM_IOCTL_AMDGPU_GEM_CREATE");
static_assert(kIoctlPrimeHandleToFd == 0xC00C642DUL, "DRM_IOCTL_PRIME_HANDLE_TO_FD");
static_assert(kIoctlGemClose == 0x40086409UL, "DRM_IOCTL_GEM_CLOSE");
constexpr uint64_t kAmdgpuGemDomainVram = 0x4;
constexpr size_t kAlign = size_t{2} << 20;

struct GemEntry {
  size_t size{0};
  int device{-1};
  int renderFd{-1};
  uint32_t handle{0};
  int dmabufFd{-1};
  int importFd{-1};  // the descriptor handed to hipImportExternalMemory
  dev_t dmabufDev{0};
  ino_t dmabufIno{0};
  hipExternalMemory_t ext{nullptr};
  ibv_mr* pin{nullptr};
};

std::mutex gMutex;
std::map<uintptr_t, GemEntry> gBuffers;  // by base address
std::map<int, int> gRenderFds;           // HIP device -> render node fd

// RDMA devices for the one-page pin, opened on first use.
ibv_device** gNics = nullptr;
int gNicCount = 0;
std::vector<ibv_context*> gNicCtx;
std::vector<ibv_pd*> gNicPd;
int gPinNic = -1;  // the device that pinned the last buffer

// Caller holds gMutex.
int RenderFd(int device) {
  auto it = gRenderFds.find(device);
  if (it != gRenderFds.end()) return it->second;
  char bus[64] = {};
  if (hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess) {
    (void)hipGetLastError();
    return -1;
  }
  for (char* c = bus; *c; ++c) {
    if (*c >= 'A' && *c <= 'F') *c = static_cast<char>(*c - 'A' + 'a');
  }
  std::string pattern = std::string("/sys/bus/pci/devices/") + bus + "/drm/renderD*";
  glob_t g{};
  if (glob(pattern.c_str(), 0, nullptr, &g) != 0 || g.gl_pathc < 1) {
    globfree(&g);
    return -1;
  }
  std::string node = std::string("/dev/dri/") + (strrchr(g.gl_pathv[0], '/') + 1);
  globfree(&g);
  int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
  if (fd >= 0) gRenderFds[device] = fd;
  return fd;
}

// Caller holds gMutex.
ibv_pd* NicPd(int i) {
  if (!gNicPd[i]) {
    if (!gNicCtx[i]) gNicCtx[i] = ibv_open_device(gNics[i]);
    if (gNicCtx[i]) gNicPd[i] = ibv_alloc_pd(gNicCtx[i]);
  }
  return gNicPd[i];
}

// Caller holds gMutex. The RDMA devices that may pin buffers, in the order to try them:
// MORI_HIP_GEM_PIN_DEVICE alone if set, else the ones MORI_RDMA_DEVICES selects (a list,
// or "^list" to exclude), else all.
std::vector<int> PinCandidates() {
  std::vector<std::string> listed;
  bool exclude = false;
  if (std::optional<std::string> pin = mori::env::GetString("MORI_HIP_GEM_PIN_DEVICE")) {
    listed.push_back(*pin);
  } else if (std::optional<std::string> sel = mori::env::GetString("MORI_RDMA_DEVICES")) {
    std::string s = *sel;
    if (!s.empty() && s[0] == '^') {
      exclude = true;
      s = s.substr(1);
    }
    size_t pos = 0;
    while (pos <= s.size()) {
      size_t comma = s.find(',', pos);
      std::string name =
          s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      name.erase(0, name.find_first_not_of(" \t"));
      name.erase(name.find_last_not_of(" \t") + 1);
      if (!name.empty()) listed.push_back(name);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  auto index = [&](const std::string& name) {
    for (int i = 0; i < gNicCount; ++i) {
      if (name == gNics[i]->name) return i;
    }
    return -1;
  };
  std::vector<int> order;
  if (!listed.empty() && !exclude) {
    for (const std::string& name : listed) {
      if (int i = index(name); i >= 0) order.push_back(i);
    }
    return order;
  }
  for (int i = 0; i < gNicCount; ++i) {
    bool skip = false;
    for (const std::string& name : listed) skip = skip || name == gNics[i]->name;
    if (!skip) order.push_back(i);
  }
  return order;
}

// Caller holds gMutex. A one-page dma-buf MR at ptr, through the device that pinned the
// last buffer if it is a candidate, then the other candidates in order: a device can
// allocate a protection domain and still reject dma-buf MRs.
ibv_mr* PinPage(void* ptr, int dmabufFd) {
  if (!gNics) {
    gNics = ibv_get_device_list(&gNicCount);
    if (!gNics) return nullptr;
    gNicCtx.assign(gNicCount, nullptr);
    gNicPd.assign(gNicCount, nullptr);
  }
  auto tryNic = [&](int i) -> ibv_mr* {
    ibv_pd* pd = NicPd(i);
    if (!pd) return nullptr;
    return ibv_reg_dmabuf_mr(pd, 0, static_cast<size_t>(getpagesize()),
                             reinterpret_cast<uint64_t>(ptr), dmabufFd, IBV_ACCESS_LOCAL_WRITE);
  };
  errno = ENODEV;
  std::vector<int> candidates = PinCandidates();
  bool lastIsCandidate = false;
  for (int i : candidates) lastIsCandidate = lastIsCandidate || i == gPinNic;
  if (lastIsCandidate) {
    if (ibv_mr* mr = tryNic(gPinNic)) return mr;
  }
  for (int i : candidates) {
    if (i == gPinNic) continue;
    if (ibv_mr* mr = tryNic(i)) {
      if (gPinNic != i) {
        MORI_APP_INFO("HIP GEM allocator: pinning buffers through RDMA device {}", gNics[i]->name);
      }
      gPinNic = i;
      return mr;
    }
  }
  return nullptr;
}

void Destroy(void* ptr, GemEntry& e) {
  if (e.pin) ibv_dereg_mr(e.pin);
  if (ptr) (void)hipFree(ptr);  // unmaps the buffer of the external memory
  if (e.ext) (void)hipDestroyExternalMemory(e.ext);
  // HIP (seen with ROCm 7.2.4) leaves the descriptor it was given open after
  // hipDestroyExternalMemory, which keeps the dma-buf and so the BO alive. Close it,
  // but only while it still names this BO's dma-buf, in case HIP did close it and the
  // number has been reused.
  if (e.importFd >= 0) {
    struct stat st{};
    if (fstat(e.importFd, &st) == 0 && st.st_dev == e.dmabufDev && st.st_ino == e.dmabufIno) {
      close(e.importFd);
    }
  }
  if (e.dmabufFd >= 0) close(e.dmabufFd);
  if (e.handle) {
    GemClose gc{e.handle, 0};
    (void)ioctl(e.renderFd, kIoctlGemClose, &gc);
  }
  (void)hipGetLastError();
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (hipGetDevice(&prev) != hipSuccess) prev = -1;
    (void)hipSetDevice(device);
  }
  ~DeviceGuard() {
    if (prev >= 0) (void)hipSetDevice(prev);
  }

 private:
  int prev{-1};
};

void* Allocate(size_t size, int device) {
  std::lock_guard<std::mutex> lock(gMutex);
  DeviceGuard guard(device);
  GemEntry e;
  e.size = (size + kAlign - 1) & ~(kAlign - 1);
  e.device = device;
  e.renderFd = RenderFd(device);
  if (e.renderFd < 0) {
    MORI_APP_ERROR("HIP GEM allocator: no render node for HIP device {}: {}", device,
                   strerror(errno));
    return nullptr;
  }

  GemCreate gc{};
  gc.in = {e.size, kAlign, kAmdgpuGemDomainVram, 0};
  if (ioctl(e.renderFd, kIoctlAmdgpuGemCreate, &gc) != 0) {
    MORI_APP_ERROR("HIP GEM allocator: DRM_IOCTL_AMDGPU_GEM_CREATE of {} bytes on device {}: {}",
                   e.size, device, strerror(errno));
    return nullptr;
  }
  e.handle = gc.out.handle;

  PrimeHandle ph{e.handle, O_CLOEXEC | O_RDWR, -1};
  if (ioctl(e.renderFd, kIoctlPrimeHandleToFd, &ph) != 0) {
    MORI_APP_ERROR("HIP GEM allocator: DRM_IOCTL_PRIME_HANDLE_TO_FD on device {}: {}", device,
                   strerror(errno));
    Destroy(nullptr, e);
    return nullptr;
  }
  e.dmabufFd = ph.fd;
  struct stat st{};
  if (fstat(e.dmabufFd, &st) == 0) {
    e.dmabufDev = st.st_dev;
    e.dmabufIno = st.st_ino;
  }

  hipExternalMemoryHandleDesc hd{};
  hd.type = hipExternalMemoryHandleTypeOpaqueFd;
  hd.handle.fd = dup(e.dmabufFd);  // HIP's; see Destroy for who closes it
  hd.size = e.size;
  hipError_t err = hipImportExternalMemory(&e.ext, &hd);
  if (err == hipSuccess) e.importFd = hd.handle.fd;
  if (err != hipSuccess) {
    close(hd.handle.fd);
    e.ext = nullptr;
    MORI_APP_ERROR("HIP GEM allocator: hipImportExternalMemory of {} bytes on device {}: {}",
                   e.size, device, hipGetErrorString(err));
    Destroy(nullptr, e);
    return nullptr;
  }
  hipExternalMemoryBufferDesc bd{};
  bd.size = e.size;
  void* ptr = nullptr;
  err = hipExternalMemoryGetMappedBuffer(&ptr, e.ext, &bd);
  if (err != hipSuccess) {
    MORI_APP_ERROR("HIP GEM allocator: hipExternalMemoryGetMappedBuffer on device {}: {}", device,
                   hipGetErrorString(err));
    Destroy(nullptr, e);
    return nullptr;
  }

  e.pin = PinPage(ptr, e.dmabufFd);
  if (!e.pin) {
    MORI_APP_ERROR(
        "HIP GEM allocator: no RDMA device accepted a one-page dma-buf MR on the buffer "
        "(device {}, {} bytes, errno {} ({})); the BO cannot be kept in VRAM without it",
        device, e.size, errno, strerror(errno));
    Destroy(ptr, e);
    return nullptr;
  }
  gBuffers[reinterpret_cast<uintptr_t>(ptr)] = e;
  MORI_APP_DEBUG("HIP GEM allocator: {} bytes at {} on device {}", e.size, ptr, device);
  return ptr;
}

}  // namespace

namespace mori {
namespace application {

bool HipGemLookup(const void* addr, HipGemBuffer* out, bool wantFd) {
  std::lock_guard<std::mutex> lock(gMutex);
  if (gBuffers.empty()) return false;
  const auto a = reinterpret_cast<uintptr_t>(addr);
  auto it = gBuffers.upper_bound(a);
  if (it == gBuffers.begin()) return false;
  --it;
  if (a >= it->first + it->second.size) return false;
  if (out) {
    out->base = reinterpret_cast<void*>(it->first);
    out->size = it->second.size;
    out->device = it->second.device;
    out->dmabufFd = -1;
    if (wantFd) {
      out->dmabufFd = dup(it->second.dmabufFd);
      if (out->dmabufFd < 0) return false;
    }
  }
  return true;
}

}  // namespace application
}  // namespace mori

extern "C" {

void* mori_hip_gem_malloc(ssize_t size, int device, hipStream_t stream) {
  (void)stream;
  if (size <= 0) return nullptr;
  return Allocate(static_cast<size_t>(size), device);
}

void mori_hip_gem_free(void* ptr, ssize_t size, int device, hipStream_t stream) {
  (void)size;
  if (!ptr) return;
  GemEntry e;
  {
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = gBuffers.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == gBuffers.end()) {
      MORI_APP_ERROR("HIP GEM allocator: free of unknown pointer {} on device {}", ptr, device);
      return;
    }
    e = it->second;
    gBuffers.erase(it);
  }
  DeviceGuard guard(e.device);
  if (stream) (void)hipStreamSynchronize(stream);
  (void)hipDeviceSynchronize();
  Destroy(ptr, e);
}

int mori_hip_gem_lookup(const void* addr, void** base, size_t* size, int* dmabuf_fd, int* device) {
  mori::application::HipGemBuffer buf;
  if (!mori::application::HipGemLookup(addr, &buf, dmabuf_fd != nullptr)) return -1;
  if (base) *base = buf.base;
  if (size) *size = buf.size;
  if (device) *device = buf.device;
  if (dmabuf_fd) *dmabuf_fd = buf.dmabufFd;
  return 0;
}

}  // extern "C"
