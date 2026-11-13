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
// Python's os.path.abspath, lexically (no filesystem access): join with cwd, drop "." and
// empty parts, resolve "..".
std::string LexicalAbsPath(const std::string& path, const std::string& cwd);
// Name of the parent directory of `path` (the importer's default experiment name):
// os.path.basename(os.path.dirname(os.path.abspath(path))).
std::string ParentDirName(const std::string& path, const std::string& cwd);
// Whether this process runs in a container (Docker, containerd, Kubernetes, Podman), judged
// from the usual markers under `root` ("" in production; tests point it at a fake tree).
bool InContainer(const std::string& root = "");

}  // namespace netmon
