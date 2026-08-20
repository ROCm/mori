// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include <google/protobuf/message_lite.h>

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
