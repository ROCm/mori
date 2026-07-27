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
#pragma once

#include <string>

namespace mori {
namespace io {
namespace fabric {

// UALink vPOD (super-node scale-up domain) identity for a single GPU. Two GPUs
// can load/store each other's memory over the fabric iff their keys are valid
// and compare equal. Mirrors mori-cco's CcoLsaKey / RCCL's MNNVL clique model:
// ppod_id (a UUID) makes the key globally unique, since hive_id collides across
// hosts.
struct VpodKey {
  bool valid{false};   // true only when the fabric is present and ACTIVE/READY
  std::string ppodId;  // physical pod UUID (empty in manual-override mode)
  int vpodId{-1};      // virtual pod id within the ppod
  int vpodSize{0};     // number of GPUs in the vPOD (0 if unknown)
};

// Does the device advertise VMM + fabric-handle export capability
// (hipDeviceAttributeVirtualMemoryManagementSupported &&
//  hipDeviceAttributeHandleTypeFabricSupported)?
bool DeviceSupportsFabric(int hipDevice);

// Read the GPU's UALink fabric identity from sysfs (mirrors mori-cco
// CcoReadFabricKey). Returns an invalid key when the device is not on a ready
// UALink fabric. Honors env overrides:
//   MORI_IO_FABRIC_DISABLE=1    -> always invalid (force RDMA/XGMI fallback)
//   MORI_IO_FABRIC_VPOD_ID=<n>  -> manual vPOD id (ppodId left empty; all ranks
//                                  sharing the value are treated as one vPOD)
VpodKey ReadVpodKey(int hipDevice);

// True iff both keys are valid and describe the same scale-up domain.
bool VpodKeySame(const VpodKey& a, const VpodKey& b);

}  // namespace fabric
}  // namespace io
}  // namespace mori
