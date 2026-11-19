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
// A counter source: one (NIC, kind) pair, queried by exactly one sampler thread.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "util.h"

namespace netmon {

class Source {
 public:
  virtual ~Source() = default;
  virtual const char* kind() const = 0;  // "rdma" | "ipc" | "ethtool"
  virtual const std::vector<std::string>& names() const = 0;
  // Where in [t0, t1] the device snapshot is taken (evidence 02); written to meta.json.
  virtual double snapshot_fraction() const = 0;
  // Startup probe, run on the sampler thread (it may block in the driver). Default: one query.
  virtual Status Probe(std::string* detail);
  // Fill names().size() values. Must not allocate on the hot path.
  virtual Status Query(uint64_t* out) = 0;
};

inline Status Source::Probe(std::string* detail) {
  std::vector<uint64_t> tmp(names().size());
  Status st = Query(tmp.data());
  if (st.ok()) *detail = "ok";
  return st;
}

}  // namespace netmon
