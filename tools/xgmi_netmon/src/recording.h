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
// On-disk recording: one file per GPU, a 512-byte header followed by fixed-size records.
//
//   DIR/meta.json    status, host, experiment, GPUs, statistics (for people; tools use the headers)
//   DIR/gpu<N>.bin   FileHeader, then Record... (little-endian)
//
// A record is written only when firmware_timestamp changed, i.e. once per PMFW snapshot that was
// read. A recording killed mid-write may end with a partial record; ReadRecording drops it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_metrics.h"

namespace xgmi {

struct Record {
  uint64_t t0_mono_ns;   // CLOCK_MONOTONIC just before the read
  uint64_t t1_mono_ns;   // CLOCK_MONOTONIC just after the read
  uint64_t real_ns;      // CLOCK_REALTIME read just before t0
  uint64_t real_err_ns;  // time taken to read CLOCK_REALTIME (wall-clock error bound)
  Snapshot s;  // firmware_timestamp (10 ns), xgmi_read_data_acc[8], xgmi_write_data_acc[8] (KB)
};
static_assert(sizeof(Record) == 21 * 8, "Record must be 21 packed u64");

constexpr char kMagic[8] = {'X', 'G', 'M', 'I', 'N', 'M', '0', '1'};
constexpr uint32_t kFormatVersion = 1;

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t record_bytes;
  int32_t gpu_idx;
  int32_t peer[kNumLinks];  // port -> peer GPU idx, -1 if not connected
  uint32_t reserved0;
  double link_max;  // bytes/s per direction
  uint64_t period_ns;
  char card[16];
  char pci[32];
  char host[64];
  char experiment[128];
  uint8_t reserved[200];
};
static_assert(sizeof(FileHeader) == 512, "FileHeader must be 512 bytes");

constexpr size_t kMaxHostLen = sizeof(FileHeader::host) - 1;
constexpr size_t kMaxExperimentLen = sizeof(FileHeader::experiment) - 1;

struct Gpu;
FileHeader MakeHeader(const Gpu& gpu, const std::string& host, const std::string& experiment,
                      uint64_t period_ns);

// write() until done; false with *err on failure.
bool WriteAll(int fd, const void* data, size_t n, std::string* err);

// Reads a whole gpu<N>.bin. *partial_bytes is the size of a trailing partial record (dropped).
bool ReadRecording(const std::string& path, FileHeader* header, std::vector<Record>* records,
                   size_t* partial_bytes, std::string* err);

}  // namespace xgmi
