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
#include "gpu_metrics.h"

#include <algorithm>
#include <cstring>

namespace xgmi {
namespace {

template <typename T>
T Load(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(v));  // little-endian host, as the kernel writes it
  return v;
}

// v1.9 attribute ids (enum amdgpu_metrics_attr_id) and types (enum amdgpu_metrics_attr_type).
constexpr unsigned kIdXgmiLinkWidth = 18;
constexpr unsigned kIdXgmiLinkSpeed = 19;
constexpr unsigned kIdXgmiReadDataAcc = 29;
constexpr unsigned kIdXgmiWriteDataAcc = 30;
constexpr unsigned kIdFirmwareTimestamp = 32;
constexpr unsigned kTypeU16 = 2;
constexpr unsigned kTypeU64 = 6;
constexpr size_t kTypeSize[8] = {1, 1, 2, 2, 4, 4, 8, 8};

bool ParseV19(const uint8_t* raw, size_t n, Layout* out, std::string* err) {
  if (n < 8) {
    *err = "gpu_metrics v1.9: shorter than its header";
    return false;
  }
  const int count = Load<int32_t>(raw + 4);
  struct Want {
    unsigned id;
    unsigned type;
    unsigned count;
    size_t* off;
    const char* name;
    bool found;
  } want[] = {
      {kIdXgmiReadDataAcc, kTypeU64, kNumLinks, &out->rd, "xgmi_read_data_acc", false},
      {kIdXgmiWriteDataAcc, kTypeU64, kNumLinks, &out->wr, "xgmi_write_data_acc", false},
      {kIdFirmwareTimestamp, kTypeU64, 1, &out->fw, "firmware_timestamp", false},
      {kIdXgmiLinkWidth, kTypeU16, 1, &out->width, "xgmi_link_width", false},
      {kIdXgmiLinkSpeed, kTypeU16, 1, &out->speed, "xgmi_link_speed", false},
  };
  size_t off = 8;
  for (int i = 0; i < count; ++i) {
    if (off + 8 > n) {
      *err = "gpu_metrics v1.9: attribute " + std::to_string(i) + " runs past the end";
      return false;
    }
    const uint64_t enc = Load<uint64_t>(raw + off);
    const unsigned type = (enc >> 20) & 0xF;
    const unsigned id = (enc >> 10) & 0x3FF;
    const unsigned inst = enc & 0x3FF;
    off += 8;
    if (type >= 8) {
      *err = "gpu_metrics v1.9: unknown attribute type " + std::to_string(type);
      return false;
    }
    const size_t size = kTypeSize[type] * inst;
    if (off + size > n) {
      *err = "gpu_metrics v1.9: attribute " + std::to_string(id) + " runs past the end";
      return false;
    }
    for (Want& w : want) {
      if (w.id != id) continue;
      if (w.type != type || w.count != inst) {
        *err = std::string("gpu_metrics v1.9: ") + w.name + " has type " + std::to_string(type) +
               " x" + std::to_string(inst) + ", expected type " + std::to_string(w.type) + " x" +
               std::to_string(w.count);
        return false;
      }
      *w.off = off;
      w.found = true;
    }
    off += size;
  }
  for (const Want& w : want) {
    if (!w.found) {
      *err = std::string("gpu_metrics v1.9: no ") + w.name;
      return false;
    }
  }
  return true;
}

}  // namespace

bool ParseLayout(const uint8_t* raw, size_t n, Layout* out, std::string* err) {
  if (n < 4) {
    *err = "gpu_metrics: shorter than its header";
    return false;
  }
  const unsigned frev = raw[2], crev = raw[3];
  Layout l;
  if (frev == 1 && crev == 8) {
    l.rd = 0x88;
    l.wr = 0xC8;
    l.fw = 288;
    l.width = 72;
    l.speed = 74;
  } else if (frev == 1 && crev == 9) {
    if (!ParseV19(raw, n, &l, err)) return false;
  } else {
    *err = "gpu_metrics v" + std::to_string(frev) + "." + std::to_string(crev) +
           " is not supported (known: v1.8, v1.9)";
    return false;
  }
  if (RequiredSize(l) > n) {
    *err = "gpu_metrics: " + std::to_string(n) + " bytes, the fields need " +
           std::to_string(RequiredSize(l));
    return false;
  }
  *out = l;
  return true;
}

size_t RequiredSize(const Layout& l) {
  return std::max({l.rd + 8 * kNumLinks, l.wr + 8 * kNumLinks, l.fw + 8, l.width + 2, l.speed + 2});
}

uint64_t ReadFirmwareTimestamp(const uint8_t* raw, const Layout& l) {
  return Load<uint64_t>(raw + l.fw);
}

void ReadSnapshot(const uint8_t* raw, const Layout& l, Snapshot* out) {
  out->fw = Load<uint64_t>(raw + l.fw);
  for (int i = 0; i < kNumLinks; ++i) {
    out->rd[i] = Load<uint64_t>(raw + l.rd + 8 * i);
    out->wr[i] = Load<uint64_t>(raw + l.wr + 8 * i);
  }
}

double LinkMaxBytesPerSecond(const uint8_t* raw, const Layout& l) {
  return Load<uint16_t>(raw + l.width) * static_cast<double>(Load<uint16_t>(raw + l.speed)) * 1e9 /
         8;
}

}  // namespace xgmi
