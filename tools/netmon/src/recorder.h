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
// Orchestrates a recording: writer startup, probe phase, sampling, and the stop state machine
// (NETMON_CPP_DESIGN.md §6.4).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "discover.h"
#include "sampler.h"

namespace netmon {

enum ExitCode : int {
  kExitOk = 0,
  kExitSetup = 1,  // argument/environment error, or startup aborted
  kExitAllStreamsFailed = 2,
  kExitWriteFailed = 3,
  kExitAbandoned = 4,
};

struct Degraded {
  std::string netdev, source, reason;
};

struct RecordOptions {
  std::string out_dir;
  uint64_t duration_ns = 0;  // 0: until SIGINT/SIGTERM
  bool quiet = false;
  uint64_t probe_timeout_ns = 5000000000ull;
  uint64_t stop_timeout_ns = 3000000000ull;
  uint64_t io_timeout_ns = 30000000000ull;

  // Context for meta.json.
  std::string host_id;
  std::map<std::string, std::string> tags;
  std::vector<std::string> command;
  std::vector<Nic> nics;
  std::vector<Degraded> degraded;

  // Live push (NETMON_LIVE_PUSH_DESIGN); disabled when push_url is empty.
  std::string push_url, experiment, grafana;
  uint64_t push_interval_ns = 1000000000ull;
  bool push_all = false;
};

// Runs the recording. Must be called before any other thread exists (it sets the signal
// mask that all threads inherit). Returns an ExitCode, or calls _exit() when threads or I/O
// are stuck (objects must stay alive until the process ends).
int RunRecording(std::vector<std::unique_ptr<Stream>>* streams, RecordOptions* opt);

}  // namespace netmon
