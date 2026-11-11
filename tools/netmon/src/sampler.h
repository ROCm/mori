// One sampler thread per (NIC, source): probe phase, then absolute-deadline sampling.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "batch_queue.h"
#include "source.h"

namespace netmon {

// Record layout (u64 little-endian): t0_mono, t1_mono, real, real_err, values...
constexpr size_t kRecordHeaderWords = 4;
constexpr uint64_t kSubmitPeriodNs = 50000000ull;  // 50 ms
constexpr size_t kMaxBatchRecords = 256;
constexpr size_t kBackToBackBatchRecords = 64;
constexpr size_t kQueueQuotaPerStream = 200;  // ~10 s at 50 ms per batch
constexpr uint64_t kMaxConsecutiveErrors = 100;
constexpr size_t kMaxDroppedSpans = 1000;

size_t BatchRecordsFor(uint64_t interval_ns);

struct StreamStatsSnapshot {
  uint64_t samples = 0, missed_slots = 0, dropped = 0, errors = 0, max_rtt_ns = 0;
  std::string last_error;
  std::vector<std::pair<uint64_t, uint64_t>> dropped_spans;
  uint64_t dropped_spans_overflow = 0;
  bool stopped_by_errors = false;
};

struct StreamStats {
  std::atomic<uint64_t> samples{0}, missed_slots{0}, dropped{0}, errors{0}, max_rtt_ns{0};
  std::atomic<bool> stopped_by_errors{false};
  std::mutex mu;  // guards the fields below
  std::string last_error;
  std::vector<std::pair<uint64_t, uint64_t>> dropped_spans;
  uint64_t dropped_spans_overflow = 0;

  StreamStatsSnapshot Snapshot();
};

enum class ProbeState { kPending, kOk, kFailed };
enum class Command { kWait, kGo, kAbandon };

// Shared between the main thread and all samplers.
struct Control {
  std::mutex mu;
  std::condition_variable cv;
  Command cmd = Command::kWait;  // guarded by mu
  uint64_t go_mono = 0;          // set with kGo; reference for the sampling grid
  uint64_t go_real = 0;
  std::atomic<bool> stop{false};
};

struct Stream {
  int id = 0;
  std::string netdev, ibdev, pci;
  std::unique_ptr<Source> src;
  uint64_t interval_ns = 0;
  bool required = false;        // explicitly requested: probe failure aborts the run
  bool ipc_unverified = false;  // ipc enabled on a firmware outside the verified list
  int cpu = -1;

  std::string file() const { return netdev + "." + src->kind() + ".bin"; }
  size_t record_words() const { return kRecordHeaderWords + src->names().size(); }

  // Probe result, guarded by Control::mu.
  ProbeState probe = ProbeState::kPending;
  std::string probe_detail;

  StreamStats stats;
  std::atomic<bool> exited{false};
  std::thread thread;
};

// Thread body. Never throws.
void SamplerMain(Stream* s, Control* ctl, BatchQueue* queue);

// Installs the (empty) SIGUSR1 handler used to interrupt blocking calls on stop.
void InstallWakeSignalHandler();
constexpr int kWakeSignal = 10;  // SIGUSR1

}  // namespace netmon
