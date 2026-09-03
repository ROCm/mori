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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/benchmark/workload_trace.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace mori::umbp::benchmark {
namespace {

constexpr std::array<char, 8> kMagic = {'U', 'M', 'B', 'P', 'T', 'R', 'C', 'E'};
constexpr uint32_t kEnvelopeVersion = 1;

// Ceilings that stop a truncated or hostile trace from being read into
// unbounded memory. Fixed on purpose: a per-instance limit was threaded through
// every reader and writer and no caller ever set anything but the defaults.
constexpr size_t kMaxHeaderBytes = 1 << 20;
constexpr size_t kMaxEventBytes = 2 << 20;
constexpr size_t kMaxKeyBytes = 1 << 20;
constexpr uint64_t kMaxValueBytes = uint64_t{1} << 40;

void WriteU32(std::ostream* stream, uint32_t value) {
  std::array<char, 4> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
  stream->write(bytes.data(), bytes.size());
}

uint32_t DecodeU32(const std::array<char, 4>& bytes) {
  uint32_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i])) << (i * 8);
  }
  return value;
}

void ValidateHeader(const ::umbp::benchmark::WorkloadTraceHeader& header) {
  if (header.schema_version() != kWorkloadTraceSchemaVersion) {
    throw TraceError("unsupported workload trace schema version: " +
                     std::to_string(header.schema_version()));
  }
  if (header.time_unit() != ::umbp::benchmark::WorkloadTraceHeader::TIME_UNIT_NANOSECONDS) {
    throw TraceError("unsupported workload trace time unit");
  }
}

void ValidateEvent(const ::umbp::benchmark::WorkloadEvent& event) {
  if (event.operation() != ::umbp::benchmark::WorkloadEvent::PUT &&
      event.operation() != ::umbp::benchmark::WorkloadEvent::GET) {
    throw TraceError("workload event has an invalid operation");
  }
  if (event.key().empty()) throw TraceError("workload event has an empty key");
  if (event.key().size() > kMaxKeyBytes) {
    throw TraceError("workload event key exceeds configured limit");
  }
  if (event.value_size() > kMaxValueBytes) {
    throw TraceError("workload event value exceeds configured limit");
  }
}

}  // namespace

TraceWriter::TraceWriter(const std::string& path,
                         const ::umbp::benchmark::WorkloadTraceHeader& header)
    : stream_(path, std::ios::binary | std::ios::trunc) {
  if (!stream_) throw TraceError("failed to open workload trace for writing: " + path);
  ValidateHeader(header);
  stream_.write(kMagic.data(), kMagic.size());
  WriteU32(&stream_, kEnvelopeVersion);
  if (!stream_) throw TraceError("failed to write workload trace envelope");
  WriteRecord(header, kMaxHeaderBytes);
}

TraceWriter::~TraceWriter() {
  if (!closed_) {
    stream_.flush();
    stream_.close();
  }
}

void TraceWriter::WriteRecord(const google::protobuf::MessageLite& record, size_t max_bytes) {
  const size_t size = record.ByteSizeLong();
  if (size > max_bytes || size > std::numeric_limits<uint32_t>::max()) {
    throw TraceError("workload trace record exceeds configured limit");
  }
  std::string bytes;
  if (!record.SerializeToString(&bytes)) {
    throw TraceError("failed to serialize workload trace record");
  }
  WriteU32(&stream_, static_cast<uint32_t>(bytes.size()));
  stream_.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream_) throw TraceError("failed to write workload trace record");
}

void TraceWriter::Write(const ::umbp::benchmark::WorkloadEvent& event) {
  if (closed_) throw TraceError("cannot write to a closed workload trace");
  ValidateEvent(event);
  WriteRecord(event, kMaxEventBytes);
}

void TraceWriter::Close() {
  if (closed_) return;
  stream_.flush();
  if (!stream_) throw TraceError("failed to flush workload trace");
  stream_.close();
  if (stream_.fail()) throw TraceError("failed to close workload trace");
  closed_ = true;
}

TraceReader::TraceReader(const std::string& path) : stream_(path, std::ios::binary) {
  if (!stream_) throw TraceError("failed to open workload trace for reading: " + path);

  std::array<char, kMagic.size()> magic{};
  stream_.read(magic.data(), magic.size());
  if (stream_.gcount() != static_cast<std::streamsize>(magic.size()) || magic != kMagic) {
    throw TraceError("invalid or truncated workload trace magic");
  }
  std::array<char, 4> version{};
  stream_.read(version.data(), version.size());
  if (stream_.gcount() != static_cast<std::streamsize>(version.size())) {
    throw TraceError("truncated workload trace envelope");
  }
  const uint32_t envelope_version = DecodeU32(version);
  if (envelope_version != kEnvelopeVersion) {
    throw TraceError("unsupported workload trace envelope version: " +
                     std::to_string(envelope_version));
  }
  ReadRecord(&header_, kMaxHeaderBytes, false);
  ValidateHeader(header_);
}

bool TraceReader::ReadRecord(google::protobuf::MessageLite* record, size_t max_bytes,
                             bool allow_eof) {
  std::array<char, 4> length{};
  stream_.read(length.data(), length.size());
  const std::streamsize length_read = stream_.gcount();
  if (length_read == 0 && allow_eof && stream_.eof()) return false;
  if (length_read != static_cast<std::streamsize>(length.size())) {
    throw TraceError("truncated workload trace record length");
  }
  const uint32_t size = DecodeU32(length);
  if (size > max_bytes) throw TraceError("workload trace record exceeds configured limit");

  std::string bytes(size, '\0');
  stream_.read(bytes.data(), static_cast<std::streamsize>(size));
  if (stream_.gcount() != static_cast<std::streamsize>(size)) {
    throw TraceError("truncated workload trace record");
  }
  if (!record->ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
    throw TraceError("malformed workload trace protobuf record");
  }
  return true;
}

bool TraceReader::ReadNext(::umbp::benchmark::WorkloadEvent* event) {
  if (event == nullptr) throw std::invalid_argument("event must not be null");
  event->Clear();
  if (!ReadRecord(event, kMaxEventBytes, true)) return false;
  ValidateEvent(*event);
  return true;
}

}  // namespace mori::umbp::benchmark
