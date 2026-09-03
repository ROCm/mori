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
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/benchmark/payload.h"
#include "umbp/distributed/benchmark/workload_source.h"
#include "umbp/distributed/benchmark/workload_trace.h"
#include "umbp/distributed/benchmark/workload_trace_recorder.h"

namespace mori::umbp::benchmark {
namespace {

std::string TempPath(const std::string& suffix) {
  static std::atomic<uint64_t> sequence{0};
  const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
  return (std::filesystem::temp_directory_path() /
          ("umbp-workload-" + std::to_string(getpid()) + "-" + std::to_string(id) + "-" +
           std::to_string(sequence.fetch_add(1)) + suffix))
      .string();
}

::umbp::benchmark::WorkloadTraceHeader Header() {
  ::umbp::benchmark::WorkloadTraceHeader header;
  header.set_schema_version(kWorkloadTraceSchemaVersion);
  header.set_time_unit(::umbp::benchmark::WorkloadTraceHeader::TIME_UNIT_NANOSECONDS);
  header.set_seed(42);
  (*header.mutable_metadata())["profile"] = "test";
  return header;
}

::umbp::benchmark::WorkloadEvent Event() {
  ::umbp::benchmark::WorkloadEvent event;
  event.set_relative_timestamp_ns(1234);
  event.set_client_id(3);
  event.set_operation_id(9);
  event.set_operation(::umbp::benchmark::WorkloadEvent::PUT);
  event.set_key("key");
  event.set_value_size(257);
  event.set_batch_id(7);
  return event;
}

std::vector<char> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void WriteBytes(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

TEST(WorkloadTraceTest, RoundTripsHeaderAndEvents) {
  const std::string path = TempPath(".trace");
  {
    TraceWriter writer(path, Header());
    writer.Write(Event());
    writer.Close();
  }

  TraceReader reader(path);
  EXPECT_EQ(reader.header().seed(), 42);
  EXPECT_EQ(reader.header().metadata().at("profile"), "test");
  ::umbp::benchmark::WorkloadEvent actual;
  ASSERT_TRUE(reader.ReadNext(&actual));
  EXPECT_EQ(actual.SerializeAsString(), Event().SerializeAsString());
  EXPECT_FALSE(reader.ReadNext(&actual));
  std::filesystem::remove(path);
}

TEST(WorkloadTraceTest, RejectsTruncationCorruptionAndUnsupportedEnvelope) {
  const std::string valid = TempPath("-valid.trace");
  {
    TraceWriter writer(valid, Header());
    writer.Write(Event());
    writer.Close();
  }
  const std::vector<char> bytes = ReadBytes(valid);

  const std::string truncated = TempPath("-truncated.trace");
  WriteBytes(truncated, std::vector<char>(bytes.begin(), bytes.end() - 1));
  TraceReader truncated_reader(truncated);
  ::umbp::benchmark::WorkloadEvent event;
  EXPECT_THROW(truncated_reader.ReadNext(&event), TraceError);

  const std::string corrupt = TempPath("-corrupt.trace");
  auto corrupt_bytes = bytes;
  corrupt_bytes.back() = static_cast<char>(0xff);
  WriteBytes(corrupt, corrupt_bytes);
  TraceReader corrupt_reader(corrupt);
  EXPECT_THROW(corrupt_reader.ReadNext(&event), TraceError);

  const std::string unsupported = TempPath("-unsupported.trace");
  auto unsupported_bytes = bytes;
  unsupported_bytes[8] = 2;
  WriteBytes(unsupported, unsupported_bytes);
  EXPECT_THROW(TraceReader ignored(unsupported), TraceError);

  const std::string unsupported_schema = TempPath("-unsupported-schema.trace");
  auto unsupported_schema_bytes = bytes;
  ASSERT_GT(unsupported_schema_bytes.size(), 17);
  ASSERT_EQ(static_cast<unsigned char>(unsupported_schema_bytes[16]), 0x08);
  unsupported_schema_bytes[17] = 2;
  WriteBytes(unsupported_schema, unsupported_schema_bytes);
  EXPECT_THROW(TraceReader ignored(unsupported_schema), TraceError);

  std::filesystem::remove(valid);
  std::filesystem::remove(truncated);
  std::filesystem::remove(corrupt);
  std::filesystem::remove(unsupported);
  std::filesystem::remove(unsupported_schema);
}

TEST(WorkloadTraceTest, PayloadIsDeterministicAndValidatable) {
  const auto first = GenerateDeterministicPayload("alpha", 17, 99, 37);
  const auto second = GenerateDeterministicPayload("alpha", 17, 99, 37);
  EXPECT_EQ(first, second);
  EXPECT_NE(first, GenerateDeterministicPayload("alpha", 18, 99, 37));
  EXPECT_TRUE(ValidateDeterministicPayload("alpha", 17, 99, first.data(), first.size()));
  auto damaged = first;
  damaged[13] ^= 1;
  EXPECT_FALSE(ValidateDeterministicPayload("alpha", 17, 99, damaged.data(), damaged.size()));
}

TEST(WorkloadTraceTest, SyntheticGenerationIsDeterministicAcrossProfiles) {
  for (SyntheticProfile profile : {SyntheticProfile::kSequential, SyntheticProfile::kUniform,
                                   SyntheticProfile::kHotsetZipf, SyntheticProfile::kReadAfterWrite,
                                   SyntheticProfile::kMixed, SyntheticProfile::kCapacityPressure}) {
    SyntheticWorkloadConfig config;
    config.profile = profile;
    config.seed = 123;
    config.operation_count = 32;
    config.key_count = 8;
    config.min_value_size = 8;
    config.max_value_size = 64;
    config.value_size_distribution = ValueSizeDistribution::kUniform;
    config.client_count = 3;
    config.batch_size = 2;
    config.qps = 1000;

    SyntheticWorkloadSource first(config);
    SyntheticWorkloadSource second(config);
    ::umbp::benchmark::WorkloadEvent left;
    ::umbp::benchmark::WorkloadEvent right;
    for (uint64_t i = 0; i < config.operation_count; ++i) {
      ASSERT_TRUE(first.Next(&left));
      ASSERT_TRUE(second.Next(&right));
      EXPECT_EQ(left.SerializeAsString(), right.SerializeAsString());
    }
    EXPECT_FALSE(first.Next(&left));
  }
}

TEST(WorkloadTraceTest, SyntheticKeysAreImmutableAndDependenciesStayOnOneClient) {
  SyntheticWorkloadConfig config;
  config.profile = SyntheticProfile::kMixed;
  config.seed = 9;
  config.operation_count = 64;
  config.key_count = 4;
  config.client_count = 3;
  config.batch_size = 2;
  config.qps = 1000;
  config.read_ratio = 0.5;

  SyntheticWorkloadSource source(config);
  std::map<std::string, uint64_t> put_ids;
  ::umbp::benchmark::WorkloadEvent event;
  while (source.Next(&event)) {
    const size_t version_separator = event.key().find("-v", config.key_prefix.size());
    ASSERT_NE(version_separator, std::string::npos);
    const uint64_t base_key = std::stoull(
        event.key().substr(config.key_prefix.size(), version_separator - config.key_prefix.size()));
    EXPECT_EQ(event.client_id(), base_key % config.client_count);
    if (event.operation() == ::umbp::benchmark::WorkloadEvent::PUT) {
      EXPECT_TRUE(put_ids.emplace(event.key(), event.operation_id()).second);
    } else {
      ASSERT_TRUE(put_ids.count(event.key()));
      EXPECT_EQ(event.operation_id(), put_ids[event.key()]);
    }
  }
  EXPECT_GE(put_ids.size(), config.key_count);
}

TEST(WorkloadTraceTest, CapacityPressureUsesUniqueWritesAndBatchTimestamps) {
  SyntheticWorkloadConfig config;
  config.profile = SyntheticProfile::kCapacityPressure;
  config.operation_count = 12;
  config.key_count = 2;
  config.client_count = 2;
  config.batch_size = 3;
  config.qps = 600;
  config.read_ratio = 1.0;  // pressure overrides this to remain write-only

  SyntheticWorkloadSource source(config);
  std::set<std::string> keys;
  std::map<uint64_t, uint64_t> timestamp_by_batch;
  ::umbp::benchmark::WorkloadEvent event;
  while (source.Next(&event)) {
    EXPECT_EQ(event.operation(), ::umbp::benchmark::WorkloadEvent::PUT);
    EXPECT_TRUE(keys.insert(event.key()).second);
    const auto [it, inserted] =
        timestamp_by_batch.emplace(event.batch_id(), event.relative_timestamp_ns());
    if (!inserted) EXPECT_EQ(event.relative_timestamp_ns(), it->second);
  }
  EXPECT_EQ(keys.size(), config.operation_count);
}

TEST(WorkloadTraceRecorderTest, KeepsKeysVerbatimAndRecordsEveryOutcome) {
  const std::string path = TempPath(".trace");
  {
    WorkloadTraceRecorderOptions options;
    options.path = path;
    options.client_id = 7;
    options.seed = 42;
    WorkloadTraceRecorder recorder(std::move(options));
    recorder.RecordBatchPut({"key"}, {64}, {WorkloadTraceOutcome::kSuccess});
    recorder.RecordBatchPut({"key"}, {64}, {WorkloadTraceOutcome::kSuccess});
    recorder.RecordBatchPut({"full"}, {64}, {WorkloadTraceOutcome::kFailure});
    recorder.RecordBatchGet({"key", "missing"}, {64, 64},
                            {WorkloadTraceOutcome::kSuccess, WorkloadTraceOutcome::kMiss});
    EXPECT_EQ(recorder.event_count(), 5u);
    recorder.Close();
  }

  TraceReader reader(path);
  std::vector<::umbp::benchmark::WorkloadEvent> events;
  ::umbp::benchmark::WorkloadEvent event;
  while (reader.ReadNext(&event)) events.push_back(event);
  ASSERT_EQ(events.size(), 5u);
  // The overwrite is two writes of one key, which is what makes it an
  // overwrite; a versioned key would replay as two unrelated objects.
  EXPECT_EQ(events[0].key(), "key");
  EXPECT_EQ(events[1].key(), "key");
  EXPECT_EQ(events[2].key(), "full");
  EXPECT_EQ(events[2].outcome(), ::umbp::benchmark::WorkloadEvent::OUTCOME_FAILURE);
  EXPECT_EQ(events[3].key(), "key");
  EXPECT_EQ(events[3].outcome(), ::umbp::benchmark::WorkloadEvent::OUTCOME_SUCCESS);
  // A GET for a key this client never wrote is still traffic the tiers saw.
  EXPECT_EQ(events[4].key(), "missing");
  EXPECT_EQ(events[4].outcome(), ::umbp::benchmark::WorkloadEvent::OUTCOME_MISS);
  EXPECT_EQ(events[3].batch_id(), events[4].batch_id());
  EXPECT_LE(events[0].relative_timestamp_ns(), events[1].relative_timestamp_ns());
  EXPECT_LE(events[1].relative_timestamp_ns(), events[3].relative_timestamp_ns());
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace mori::umbp::benchmark
