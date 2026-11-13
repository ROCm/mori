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
// Parsing of /sys/class/drm/cardN/device/gpu_metrics: the XGMI counters and the PMFW timestamp.
//
// v1.8 (amdgpu 6.16.6): fixed struct gpu_metrics_v1_8.
// v1.9 (amdgpu 6.16.13): header, int attr_count, then packed [u64 encoding, values] per field,
// encoding = unit << 24 | type << 20 | id << 10 | count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace xgmi {

constexpr int kNumLinks = 8;

// Byte offsets of the fields xgmi-netmon uses.
struct Layout {
  size_t rd = 0;     // xgmi_read_data_acc[8], u64, KB
  size_t wr = 0;     // xgmi_write_data_acc[8], u64, KB
  size_t fw = 0;     // firmware_timestamp, u64, 10 ns
  size_t width = 0;  // xgmi_link_width, u16, lanes
  size_t speed = 0;  // xgmi_link_speed, u16, Gbps
};

// Fills *out; on failure returns false and sets *err.
bool ParseLayout(const uint8_t* raw, size_t n, Layout* out, std::string* err);

struct Snapshot {
  uint64_t fw = 0;
  uint64_t rd[kNumLinks] = {};
  uint64_t wr[kNumLinks] = {};
};

// `raw` must hold at least RequiredSize(layout) bytes.
void ReadSnapshot(const uint8_t* raw, const Layout& layout, Snapshot* out);
size_t RequiredSize(const Layout& layout);
uint64_t ReadFirmwareTimestamp(const uint8_t* raw, const Layout& layout);
// Bytes per second per direction: width * Gbps / 8.
double LinkMaxBytesPerSecond(const uint8_t* raw, const Layout& layout);

}  // namespace xgmi
