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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <google/protobuf/message_lite.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "umbp_workload.pb.h"

namespace mori::umbp::benchmark {

inline constexpr uint32_t kWorkloadTraceSchemaVersion = 1;

// Errors caused by I/O, a malformed trace, or an unsupported version.
class TraceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TraceWriter {
 public:
  TraceWriter(const std::string& path, const ::umbp::benchmark::WorkloadTraceHeader& header);
  ~TraceWriter();

  TraceWriter(const TraceWriter&) = delete;
  TraceWriter& operator=(const TraceWriter&) = delete;

  void Write(const ::umbp::benchmark::WorkloadEvent& event);
  void Close();

 private:
  void WriteRecord(const google::protobuf::MessageLite& record, size_t max_bytes);

  std::ofstream stream_;
  bool closed_ = false;
};

class TraceReader {
 public:
  explicit TraceReader(const std::string& path);

  const ::umbp::benchmark::WorkloadTraceHeader& header() const { return header_; }

  // Returns false only at a clean record boundary at end-of-file.
  bool ReadNext(::umbp::benchmark::WorkloadEvent* event);

 private:
  bool ReadRecord(google::protobuf::MessageLite* record, size_t max_bytes, bool allow_eof);

  std::ifstream stream_;
  ::umbp::benchmark::WorkloadTraceHeader header_;
};

}  // namespace mori::umbp::benchmark
