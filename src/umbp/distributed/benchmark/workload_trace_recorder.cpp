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
                   const std::vector<WorkloadTraceOutcome>& outcomes) {
  if (keys.size() != sizes.size() || keys.size() != outcomes.size()) {
    throw std::invalid_argument(
        "workload trace batch keys, sizes, and outcomes must have equal lengths");
  }
}

::umbp::benchmark::WorkloadEvent::Outcome ToProtoOutcome(WorkloadTraceOutcome outcome) {
  switch (outcome) {
    case WorkloadTraceOutcome::kSuccess:
      return ::umbp::benchmark::WorkloadEvent::OUTCOME_SUCCESS;
    case WorkloadTraceOutcome::kMiss:
      return ::umbp::benchmark::WorkloadEvent::OUTCOME_MISS;
    case WorkloadTraceOutcome::kFailure:
      return ::umbp::benchmark::WorkloadEvent::OUTCOME_FAILURE;
  }
  return ::umbp::benchmark::WorkloadEvent::OUTCOME_UNSPECIFIED;
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

bool WorkloadTraceRecorder::WriteLocked(const ::umbp::benchmark::WorkloadEvent& event) {
  try {
    writer_.Write(event);
  } catch (const std::exception& exception) {
    failed_ = true;
    error_ = exception.what();
    return false;
  } catch (...) {
    failed_ = true;
    error_ = "unknown workload trace write error";
    return false;
  }
  ++event_count_;
  return true;
}

void WorkloadTraceRecorder::RecordBatch(::umbp::benchmark::WorkloadEvent::Operation operation,
                                       const std::vector<std::string>& keys,
                                       const std::vector<size_t>& sizes,
                                       const std::vector<WorkloadTraceOutcome>& outcomes) {
  ValidateBatch(keys, sizes, outcomes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (failed_) return;
  const uint64_t batch_id = ++next_batch_id_;
  const uint64_t timestamp_ns = RelativeTimestampNs();

  for (size_t i = 0; i < keys.size(); ++i) {
    ::umbp::benchmark::WorkloadEvent event;
    event.set_relative_timestamp_ns(timestamp_ns);
    event.set_client_id(client_id_);
    event.set_operation_id(++next_operation_id_);
    event.set_operation(operation);
    event.set_key(keys[i]);
    event.set_value_size(static_cast<uint64_t>(sizes[i]));
    event.set_batch_id(batch_id);
    event.set_outcome(ToProtoOutcome(outcomes[i]));
    if (!WriteLocked(event)) return;
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
