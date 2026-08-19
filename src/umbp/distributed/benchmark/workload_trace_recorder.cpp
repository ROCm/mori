// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/benchmark/workload_trace_recorder.h"

#include <stdexcept>
#include <utility>

namespace mori::umbp::benchmark {
namespace {

::umbp::benchmark::WorkloadTraceHeader MakeHeader(
    const WorkloadTraceRecorderOptions& options) {
  ::umbp::benchmark::WorkloadTraceHeader header;
  header.set_schema_version(kWorkloadTraceSchemaVersion);
  header.set_time_unit(
      ::umbp::benchmark::WorkloadTraceHeader::TIME_UNIT_NANOSECONDS);
  header.set_seed(options.seed);
  auto& metadata = *header.mutable_metadata();
  metadata["source"] = "production";
  metadata["payload_mode"] = "external";
  metadata["node_id"] = options.node_id;
  metadata["backend_policy"] = options.backend_policy;
  return header;
}

void ValidateBatch(const std::vector<std::string>& keys,
                   const std::vector<size_t>& sizes,
                   const std::vector<bool>& successes) {
  if (keys.size() != sizes.size() || keys.size() != successes.size()) {
    throw std::invalid_argument(
        "workload trace batch keys, sizes, and successes must have equal lengths");
  }
}

}  // namespace

WorkloadTraceRecorder::WorkloadTraceRecorder(Options options)
    : client_id_(options.client_id),
      writer_(options.path, MakeHeader(options)),
      start_(Clock::now()) {}

WorkloadTraceRecorder::~WorkloadTraceRecorder() {
  try {
    Close();
  } catch (...) {
  }
}

uint64_t WorkloadTraceRecorder::RelativeTimestampNs() const {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_)
          .count());
}

void WorkloadTraceRecorder::RecordBatchPut(
    const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
    const std::vector<bool>& successes) {
  ValidateBatch(keys, sizes, successes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (failed_) return;
  const uint64_t batch_id = ++next_batch_id_;
  const uint64_t timestamp_ns = RelativeTimestampNs();

  for (size_t i = 0; i < keys.size(); ++i) {
    if (!successes[i]) continue;

    const uint64_t operation_id = ++next_operation_id_;
    ::umbp::benchmark::WorkloadEvent event;
    event.set_relative_timestamp_ns(timestamp_ns);
    event.set_client_id(client_id_);
    event.set_operation_id(operation_id);
    event.set_operation(::umbp::benchmark::WorkloadEvent::PUT);
    const std::string recorded_key =
        keys[i] + "#umbp-v" + std::to_string(operation_id);
    event.set_key(recorded_key);
    event.set_value_size(static_cast<uint64_t>(sizes[i]));
    event.set_batch_id(batch_id);
    try {
      writer_.Write(event);
    } catch (const std::exception& exception) {
      failed_ = true;
      error_ = exception.what();
      return;
    } catch (...) {
      failed_ = true;
      error_ = "unknown workload trace write error";
      return;
    }
    ++event_count_;
    if (successes[i]) {
      last_put_ids_[keys[i]] = operation_id;
      recorded_keys_[keys[i]] = recorded_key;
    }
  }
}

void WorkloadTraceRecorder::RecordBatchGet(
    const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
    const std::vector<bool>& successes) {
  ValidateBatch(keys, sizes, successes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (failed_) return;
  const uint64_t batch_id = ++next_batch_id_;
  const uint64_t timestamp_ns = RelativeTimestampNs();

  for (size_t i = 0; i < keys.size(); ++i) {
    if (!successes[i]) continue;

    const auto put = last_put_ids_.find(keys[i]);
    if (put == last_put_ids_.end()) continue;
    const uint64_t operation_id = put->second;
    ::umbp::benchmark::WorkloadEvent event;
    event.set_relative_timestamp_ns(timestamp_ns);
    event.set_client_id(client_id_);
    event.set_operation_id(operation_id);
    event.set_operation(::umbp::benchmark::WorkloadEvent::GET);
    const auto recorded = recorded_keys_.find(keys[i]);
    event.set_key(recorded == recorded_keys_.end() ? keys[i] : recorded->second);
    event.set_value_size(static_cast<uint64_t>(sizes[i]));
    event.set_batch_id(batch_id);
    try {
      writer_.Write(event);
    } catch (const std::exception& exception) {
      failed_ = true;
      error_ = exception.what();
      return;
    } catch (...) {
      failed_ = true;
      error_ = "unknown workload trace write error";
      return;
    }
    ++event_count_;
  }
}

void WorkloadTraceRecorder::Close() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    writer_.Close();
  } catch (const std::exception& exception) {
    failed_ = true;
    if (error_.empty()) error_ = exception.what();
  } catch (...) {
    failed_ = true;
    if (error_.empty()) error_ = "unknown workload trace close error";
  }
  if (failed_) throw TraceError(error_);
}

uint64_t WorkloadTraceRecorder::event_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return event_count_;
}

}  // namespace mori::umbp::benchmark
