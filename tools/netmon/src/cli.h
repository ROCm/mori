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
