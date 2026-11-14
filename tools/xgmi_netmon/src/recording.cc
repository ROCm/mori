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
#include "recording.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#include "topology.h"

namespace xgmi {
namespace {

template <size_t N>
void CopyString(char (&dst)[N], const std::string& src) {
  std::memset(dst, 0, N);
  std::memcpy(dst, src.data(), std::min(src.size(), N - 1));
}

}  // namespace

FileHeader MakeHeader(const Gpu& gpu, const std::string& host, const std::string& experiment,
                      uint64_t period_ns) {
  FileHeader h;
  std::memset(&h, 0, sizeof(h));
  std::memcpy(h.magic, kMagic, sizeof(h.magic));
  h.version = kFormatVersion;
  h.record_bytes = sizeof(Record);
  h.gpu_idx = gpu.idx;
  for (int i = 0; i < kNumLinks; ++i) h.peer[i] = gpu.peer[i];
  h.link_max = gpu.link_max;
  h.period_ns = period_ns;
  CopyString(h.card, gpu.card);
  CopyString(h.pci, gpu.pci);
  CopyString(h.host, host);
  CopyString(h.experiment, experiment);
  return h;
}

bool WriteAll(int fd, const void* data, size_t n, std::string* err) {
  const char* p = static_cast<const char*>(data);
  while (n > 0) {
    const ssize_t w = write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      *err = std::string("write: ") + std::strerror(errno);
      return false;
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

bool ReadRecording(const std::string& path, FileHeader* header, std::vector<Record>* records,
                   size_t* partial_bytes, std::string* err) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    *err = path + ": " + std::strerror(errno);
    return false;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    *err = path + ": " + std::strerror(errno);
    close(fd);
    return false;
  }
  std::vector<char> buf(static_cast<size_t>(st.st_size));
  size_t got = 0;
  while (got < buf.size()) {
    const ssize_t r = read(fd, buf.data() + got, buf.size() - got);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  close(fd);
  if (got < sizeof(FileHeader)) {
    *err = path + ": shorter than the header";
    return false;
  }
  std::memcpy(header, buf.data(), sizeof(FileHeader));
  if (std::memcmp(header->magic, kMagic, sizeof(kMagic)) != 0) {
    *err = path + ": not an xgmi-netmon recording";
    return false;
  }
  if (header->version != kFormatVersion || header->record_bytes != sizeof(Record)) {
    *err = path + ": format version " + std::to_string(header->version) + " / record size " +
           std::to_string(header->record_bytes) + " not supported";
    return false;
  }
  const size_t body = got - sizeof(FileHeader);
  const size_t n = body / sizeof(Record);
  *partial_bytes = body % sizeof(Record);
  records->resize(n);
  if (n) std::memcpy(records->data(), buf.data() + sizeof(FileHeader), n * sizeof(Record));
  return true;
}

}  // namespace xgmi
