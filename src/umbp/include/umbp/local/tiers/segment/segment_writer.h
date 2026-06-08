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
#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/segment/segment_index.h"
#include "umbp/storage/io/storage_io_driver.h"

namespace mori::umbp::segment {

struct PreparedRecord {
  std::vector<char> record;
  WriteReservation reservation;
};

class Writer {
 public:
  explicit Writer(StorageIoDriver& io_driver) : io_driver_(io_driver) {}

  // Phase 1 (caller holds mu_): prepare record buffer and reserve index space.
  // Returns false if capacity is exhausted.
  bool Prepare(const std::string& key, const void* data, size_t size, Meta* segment_meta,
               Index& index, PreparedRecord* out) const;

  // Phase 2 (caller holds io_mu_ only): write the prepared record to disk.
  IoStatus WriteRecord(int fd, const PreparedRecord& pr, bool should_sync) const;

  // Phase 2 batch variant: write multiple prepared records to disk.
  IoStatus WriteRecords(int fd, const std::vector<PreparedRecord>& records, bool should_sync) const;

 private:
  StorageIoDriver& io_driver_;
};

}  // namespace mori::umbp::segment
