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

#include "mori/jit/v2/util.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace mori {
namespace jit {
namespace v2 {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Subprocess
// ---------------------------------------------------------------------------

CommandResult RunProgram(const std::vector<std::string>& argv) {
  CommandResult res;
  if (argv.empty()) return res;

  int fds[2];
  if (pipe(fds) != 0) {
    res.output = "pipe() failed: ";
    res.output += std::strerror(errno);
    return res;
  }

  pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    res.output = "fork() failed: ";
    res.output += std::strerror(errno);
    return res;
  }

  if (pid == 0) {
    // Child: stdout and stderr both into the pipe.
    close(fds[0]);
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);

    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    execvp(cargv[0], cargv.data());
    _exit(127);  // execvp only returns on failure
  }

  close(fds[1]);
  std::array<char, 4096> buf;
  ssize_t n;
  while ((n = read(fds[0], buf.data(), buf.size())) > 0) res.output.append(buf.data(), size_t(n));
  close(fds[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  // A child killed by a signal (Ctrl-C during a long hipcc) must not read as success.
  res.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  return res;
}

CommandResult RunCommand(const std::string& commandLine) {
  return RunProgram({"/bin/sh", "-c", commandLine + " 2>&1"});
}

std::string Quote(const std::string& s) {
  if (s.find_first_of(" \t\n\"'\\$`|&;<>()") == std::string::npos) return s;
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

std::string JoinArgv(const std::vector<std::string>& argv) {
  std::string out;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i) out += ' ';
    out += Quote(argv[i]);
  }
  return out;
}

std::vector<std::string> SplitWhitespace(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) out.push_back(tok);
  return out;
}

// ---------------------------------------------------------------------------
// Filesystem
// ---------------------------------------------------------------------------

bool MakeDirs(const std::string& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  return fs::is_directory(path);
}

void FsyncPath(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  }
}

bool WriteFileSynced(const std::string& path, const std::string& data) {
  {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) return false;
  }
  FsyncPath(path);
  return true;
}

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

void FsyncDirRecursive(const std::string& dir) {
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    if (entry.is_directory(ec))
      FsyncDirRecursive(entry.path().string());
    else if (entry.is_regular_file(ec))
      FsyncPath(entry.path().string());
  }
  FsyncPath(dir);
}

std::string MakeUniqueTempDir(const std::string& root, const std::string& prefix) {
  if (!MakeDirs(root)) return "";
  static std::atomic<uint64_t> counter{0};
  for (int attempt = 0; attempt < 64; ++attempt) {
    std::ostringstream name;
    name << prefix << '.' << ::getpid() << '.' << counter.fetch_add(1, std::memory_order_relaxed)
         << '.' << attempt;
    fs::path cand = fs::path(root) / name.str();
    std::error_code ec;
    if (fs::create_directory(cand, ec) && !ec) return cand.string();
  }
  return "";
}

void SafeRemoveAll(const std::string& path) {
  std::error_code ec;
  fs::remove_all(path, ec);
}

bool PublishDir(const std::string& tmpDir, const std::string& finalDir) {
  std::error_code ec;
  fs::create_directories(fs::path(finalDir).parent_path(), ec);

  FsyncDirRecursive(tmpDir);

  ec.clear();
  fs::rename(tmpDir, finalDir, ec);
  if (!ec) {
    // Make the new directory entry itself durable.
    FsyncPath(fs::path(finalDir).parent_path().string());
    return true;
  }

  // Another rank published first (or the rename genuinely failed). Their copy is
  // byte-equivalent -- same content hash -- so drop ours either way.
  SafeRemoveAll(tmpDir);
  return false;
}

}  // namespace v2
}  // namespace jit
}  // namespace mori
