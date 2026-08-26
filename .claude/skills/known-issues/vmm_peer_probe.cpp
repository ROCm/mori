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
// vmm_peer_probe -- does granting peer access to a HIP VMM allocation evict it from VRAM?
//
// Allocates a VMM buffer on one GPU, grants peer access from another, and watches total
// GPU VRAM usage across the box (via sysfs). On a kernel built without
// CONFIG_DMABUF_MOVE_NOTIFY, amdgpu_dma_buf_pin() strips VRAM from the allowed domains
// and pins the buffer into host system memory, so VRAM usage collapses at that point.
//
//   hipcc -O3 --offload-arch=gfx950 vmm_peer_probe.cpp -o vmm_peer_probe
//   ./vmm_peer_probe [owner_dev] [peer_dev] [size_GB]
//
// See SKILL.md in this directory for the full write-up.
#include <dirent.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x)                                                                        \
  do {                                                                               \
    hipError_t e = (x);                                                              \
    if (e != hipSuccess) {                                                           \
      printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
      return 2;                                                                      \
    }                                                                                \
  } while (0)

// Sum of mem_info_vram_used over every DRM card that exposes it, in bytes.
// Summing avoids having to map HIP device ordinals onto card indices. Scan the directory
// rather than probing card0..cardN -- card numbering is not dense (an 8-GPU box can be
// card0,8,16,...,56).
static long long total_vram_used() {
  DIR* d = opendir("/sys/class/drm");
  if (!d) return -1;
  long long total = 0;
  bool any = false;
  while (struct dirent* e = readdir(d)) {
    if (strncmp(e->d_name, "card", 4) != 0) continue;
    const char* n = e->d_name + 4;
    if (!*n) continue;
    bool digits = true;
    for (const char* c = n; *c; c++)
      if (*c < '0' || *c > '9') digits = false;
    if (!digits) continue;  // skip card0-DP-1 etc.
    std::string p = std::string("/sys/class/drm/") + e->d_name + "/device/mem_info_vram_used";
    FILE* f = fopen(p.c_str(), "r");
    if (!f) continue;
    long long v = 0;
    if (fscanf(f, "%lld", &v) == 1) {
      total += v;
      any = true;
    }
    fclose(f);
  }
  closedir(d);
  return any ? total : -1;
}

static double gb(long long bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

int main(int argc, char** argv) {
  int owner = (argc > 1) ? atoi(argv[1]) : 1;
  int peer = (argc > 2) ? atoi(argv[2]) : 0;
  size_t GB = (argc > 3) ? atoll(argv[3]) : 8;
  size_t bytes = GB << 30;

  int ndev = 0;
  CK(hipGetDeviceCount(&ndev));
  if (owner >= ndev || peer >= ndev || owner == peer) {
    printf("need two distinct devices < %d (got owner=%d peer=%d)\n", ndev, owner, peer);
    return 2;
  }
  if (total_vram_used() < 0) {
    printf(
        "cannot read /sys/class/drm/card*/device/mem_info_vram_used -- "
        "run on the host, not inside a container without /sys access\n");
    return 2;
  }

  printf("# VMM peer placement probe: %zu GB on GPU%d, peer access from GPU%d\n", GB, owner, peer);

  long long base = total_vram_used();

  hipMemAllocationProp prop = {};
  prop.type = (hipMemAllocationType)0x40000000;  // Uncached (fine-grained), what cco uses
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = owner;

  CK(hipSetDevice(owner));
  size_t gran = 0;
  CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
  size_t sz = ((bytes + gran - 1) / gran) * gran;

  hipMemGenericAllocationHandle_t h;
  CK(hipMemCreate(&h, sz, &prop, 0));
  void* va = nullptr;
  CK(hipMemAddressReserve(&va, sz, gran, nullptr, 0));
  CK(hipMemMap(va, sz, 0, h, 0));

  hipMemAccessDesc own{};
  own.location.type = hipMemLocationTypeDevice;
  own.location.id = owner;
  own.flags = hipMemAccessFlagsProtReadWrite;
  CK(hipMemSetAccess(va, sz, &own, 1));
  CK(hipMemset(va, 1, sz));
  CK(hipDeviceSynchronize());

  long long after_alloc = total_vram_used();
  printf("  owner-only        : VRAM in use %+7.2f GB vs baseline\n", gb(after_alloc - base));

  hipMemAccessDesc both[2];
  both[0] = own;
  both[1].location.type = hipMemLocationTypeDevice;
  both[1].location.id = peer;
  both[1].flags = hipMemAccessFlagsProtReadWrite;
  CK(hipMemSetAccess(va, sz, both, 2));
  CK(hipDeviceSynchronize());

  long long after_peer = total_vram_used();
  printf("  peer access granted: VRAM in use %+7.2f GB vs baseline\n", gb(after_peer - base));

  // The allocation should still be resident after the grant. Treat losing more than half
  // of it as eviction; a healthy machine shows no change at all.
  long long lost = after_alloc - after_peer;
  bool affected = lost > (long long)(sz / 2);

  printf("\n");
  if (affected) {
    printf("VERDICT: AFFECTED -- %.2f GB left VRAM when peer access was granted.\n", gb(lost));
    printf(
        "  The buffer was pinned into host system memory; peer \"writes\" now go to\n"
        "  host DRAM over PCIe instead of to peer VRAM over XGMI.\n");
    printf("  Check: grep -E 'CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY)=' /boot/config-$(uname -r)\n");
    printf("  Both must be =y. See SKILL.md (Issue 1) for the fix.\n");
  } else {
    printf("VERDICT: OK -- the allocation stayed in VRAM after peer access was granted.\n");
  }

  CK(hipMemUnmap(va, sz));
  CK(hipMemRelease(h));
  CK(hipMemAddressFree(va, sz));
  return affected ? 1 : 0;
}
