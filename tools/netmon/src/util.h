// Common helpers: Status, clocks, string utilities.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace netmon {

// Error handling is uniform across threads: every stream keeps an error count and its last
// error message for meta.json, and exceptions cannot cross thread boundaries.
class Status {
 public:
  Status() = default;
  static Status Ok() { return Status(); }
  static Status Error(std::string msg) { return Status(std::move(msg)); }
  static Status Errno(const std::string& what, int err);  // "what: strerror(err)"

  bool ok() const { return !failed_; }
  const std::string& message() const { return msg_; }

 private:
  explicit Status(std::string msg) : failed_(true), msg_(std::move(msg)) {}
  bool failed_ = false;
  std::string msg_;
};

uint64_t MonoNs();
uint64_t RealNs();

std::string Trim(const std::string& s);
std::vector<std::string> Split(const std::string& s, char sep);
bool StartsWith(const std::string& s, const std::string& prefix);
// Read a small sysfs/proc text file; returns false if it cannot be read.
bool ReadTextFile(const std::string& path, std::string* out);
std::string Hostname();
// Parse "2ms", "500us", "1s", "1.5m", "0". Max 1 day. Returns false on error.
bool ParseDurationNs(const std::string& s, uint64_t* ns);
std::string FormatDurationNs(uint64_t ns);

}  // namespace netmon
