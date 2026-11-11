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
};

// Runs the recording. Must be called before any other thread exists (it sets the signal
// mask that all threads inherit). Returns an ExitCode, or calls _exit() when threads or I/O
// are stuck (objects must stay alive until the process ends).
int RunRecording(std::vector<std::unique_ptr<Stream>>* streams, RecordOptions* opt);

}  // namespace netmon
