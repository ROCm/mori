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

#include <cstdio>
#include <cstring>
#include <optional>
#include <tuple>

namespace mori {
namespace application {

// An ionic (AINIC) firmware version: "1.117.5-a-58" is (1, 117, 5, 58), and
// versions compare in that order.
using IonicFwVersion = std::tuple<int, int, int, int>;

// Parse "1.117.5-a-58" or "1.117.5-a58" into (1,117,5,58).
inline std::optional<IonicFwVersion> ParseIonicFwVersion(const char* fw_ver) {
  int major, minor, patch, build;
  char tag;
  if (sscanf(fw_ver, "%d.%d.%d-%c-%d", &major, &minor, &patch, &tag, &build) == 5 ||
      sscanf(fw_ver, "%d.%d.%d-%c%d", &major, &minor, &patch, &tag, &build) == 5) {
    return IonicFwVersion{major, minor, patch, build};
  }
  return std::nullopt;
}

// The firmware version RDMA device `dev_name` reports in sysfs.
inline std::optional<IonicFwVersion> ReadIonicFwVersion(const char* dev_name) {
  char path[256];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/fw_ver", dev_name);

  FILE* f = fopen(path, "r");
  if (!f) return std::nullopt;

  char buf[64] = {};
  fgets(buf, sizeof(buf), f);
  fclose(f);

  // Strip trailing newline.
  buf[strcspn(buf, "\n")] = '\0';
  return ParseIonicFwVersion(buf);
}

}  // namespace application
}  // namespace mori
