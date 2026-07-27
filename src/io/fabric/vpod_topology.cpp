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
#include "src/io/fabric/vpod_topology.hpp"

#include <hip/hip_runtime_api.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {
namespace fabric {
namespace {

// Read the first line of a sysfs file, trimming trailing whitespace/newlines.
bool ReadSysfsLine(const std::string& path, std::string& out) {
  FILE* f = fopen(path.c_str(), "r");
  if (f == nullptr) return false;
  char buf[128] = {0};
  bool ok = fgets(buf, sizeof(buf), f) != nullptr;
  fclose(f);
  if (!ok) return false;
  out = buf;
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
    out.pop_back();
  }
  return true;
}

// The PCI BDF as sysfs spells it (lowercase). hipDeviceGetPCIBusId may return
// uppercase hex, whereas sysfs directory names are lowercase.
std::string DeviceBdfLower(int hipDevice) {
  char bdf[32] = {0};
  if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), hipDevice) != hipSuccess) return "";
  for (char* p = bdf; *p; ++p) *p = static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
  return std::string(bdf);
}

}  // namespace

bool DeviceSupportsFabric(int hipDevice) {
  // Guard hipDeviceAttributeHandleTypeFabricSupported: it is absent on older
  // ROCm (e.g. 7.2.4), and HIP_FABRIC_API is not defined on 7.14 where the enum
  // does exist. Older ROCm reports fabric unsupported and falls back to XGMI/RDMA.
#if defined(HIP_FABRIC_API) || (HIP_VERSION >= 71400000)
  int vmm = 0;
  int fabric = 0;
  hipError_t err =
      hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, hipDevice);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  err = hipDeviceGetAttribute(&fabric, hipDeviceAttributeHandleTypeFabricSupported, hipDevice);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return vmm != 0 && fabric != 0;
#else
  (void)hipDevice;
  return false;
#endif
}

VpodKey ReadVpodKey(int hipDevice) {
  VpodKey key;

  const char* disable = std::getenv("MORI_IO_FABRIC_DISABLE");
  if (disable != nullptr && std::atoi(disable) != 0) {
    return key;  // forced invalid
  }

  // Manual override: every engine that sets the same value is treated as one
  // vPOD (ppodId stays empty). Useful when sysfs is unavailable but the operator
  // knows the topology.
  const char* manual = std::getenv("MORI_IO_FABRIC_VPOD_ID");
  if (manual != nullptr && manual[0] != '\0') {
    key.valid = true;
    key.vpodId = std::atoi(manual);
    key.vpodSize = 0;
    return key;
  }

  std::string bdf = DeviceBdfLower(hipDevice);
  if (bdf.empty()) return key;
  std::string dir = std::string("/sys/bus/pci/devices/") + bdf + "/ualink";

  std::string link, state, ppod;
  if (!ReadSysfsLine(dir + "/link_type", link)) return key;  // not a UALink GPU
  if (link != "UALoE" && link != "UALLink") return key;
  if (!ReadSysfsLine(dir + "/accel_state", state)) return key;
  if (state != "active" && state != "ready") return key;  // not usable yet
  if (!ReadSysfsLine(dir + "/ppod_id", ppod) || ppod.empty()) return key;

  std::string tmp;
  key.ppodId = ppod;
  key.vpodId = ReadSysfsLine(dir + "/vpod_id", tmp) ? std::atoi(tmp.c_str()) : 0;
  key.vpodSize = ReadSysfsLine(dir + "/vpod_size", tmp) ? std::atoi(tmp.c_str()) : 0;
  key.valid = true;
  MORI_IO_TRACE("FABRIC: device {} vPOD ppod_id={} vpod_id={} vpod_size={}", hipDevice, key.ppodId,
                key.vpodId, key.vpodSize);
  return key;
}

bool VpodKeySame(const VpodKey& a, const VpodKey& b) {
  if (!a.valid || !b.valid) return false;
  return a.vpodId == b.vpodId && a.ppodId == b.ppodId;
}

}  // namespace fabric
}  // namespace io
}  // namespace mori
