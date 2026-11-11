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
// Command-line parsing for the `record` subcommand (getopt_long, validated in one place).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace netmon {

struct RecordArgs {
  std::string out;
  std::string host_id;
  std::vector<std::string> nics;
  bool sources_auto = true;
  bool want_rdma = true, want_ipc = true, want_ethtool = true;
  bool ipc_unverified = false;
  uint64_t interval_rdma_ns = 2000000ull;
  uint64_t interval_ipc_ns = 2000000ull;
  uint64_t interval_ethtool_ns = 1000000000ull;
  uint64_t duration_ns = 0;
  std::map<std::string, std::string> tags;
  std::vector<int> cpus;
  std::string owner;
  bool sudo = false;
  bool quiet = false;
};

constexpr uint64_t kMaxIntervalNs = 60ull * 1000000000ull;

// argv[0] is the subcommand name ("record"). Returns false and sets *error on bad input.
bool ParseRecordArgs(int argc, char** argv, RecordArgs* out, std::string* error);
bool ParseIntervals(const std::string& spec, RecordArgs* out, std::string* error);
bool ParseCpuList(const std::string& spec, std::vector<int>* out);
// Replace every "{host}" with host_id.
std::string ExpandHost(const std::string& path, const std::string& host_id);

const char* RecordUsage();

}  // namespace netmon
