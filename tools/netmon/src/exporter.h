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

// Live push to VictoriaMetrics (NETMON_LIVE_PUSH_DESIGN). The writer offers every written batch
// (Offer: lock-free, allocation-free, drops when the stream's ring is full); one exporter
// thread turns samples into intervals (LiveRates) and POSTs them. Nothing here can block the
// samplers or the writer; push failures never change the recording's exit code.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "batch_queue.h"
#include "http_client.h"
#include "writer.h"

namespace netmon {

struct PushOptions {
  std::string url;
  std::string experiment;
  std::string host;
  uint64_t interval_ns = 1000000000ull;
  bool all_counters = false;
  bool quiet = false;
  uint64_t max_body_bytes = 8ull << 20;           // per request (uncompressed); smaller in tests
  std::string grafana = "http://127.0.0.1:3000";  // only for the printed URLs
};

struct ExportStream {
  std::string netdev, source;
  uint64_t interval_ns = 0;
  double snapshot_fraction = 0.5;
  size_t record_words = 0;   // full record width in the recording
  size_t batch_records = 0;  // the stream's batch capacity
  std::vector<std::string> names;
  int64_t speed_mbps = -1;
};

struct PushStats {
  uint64_t points_sent = 0, requests = 0, failed_points = 0, dropped_samples = 0;
  uint64_t unconfirmed_dropped = 0, clock_back_intervals = 0, log_dropped = 0;
  bool abandoned = false;
  std::string last_error;
};

// Counters pushed for a source: the importer's KEY_COUNTERS (always) and, from their first
// change on, the optional ones (PFC; every other counter with all=true). Returns column
// indices into `names` and sets optional[i] for each returned column.
std::vector<size_t> SelectPushCounters(const std::string& source,
                                       const std::vector<std::string>& names, bool all,
                                       std::vector<bool>* optional);
extern const char* const kKeyCountersRdma[];
extern const char* const kKeyCountersIpc[];

// Bounded, non-blocking stderr messages from a single detached thread (design §6 item 6).
class AsyncLog {
 public:
  AsyncLog();
  void Post(std::string msg);  // drops (and counts) when the 8 slots are full
  bool Idle();                 // nothing queued and nothing being written
  uint64_t dropped() const { return dropped_.load(); }

 private:
  void Main();
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::string> q_;
  bool writing_ = false;
  std::atomic<uint64_t> dropped_{0};
};

class Exporter : public ExportHook {
 public:
  enum class Phase { kPreflight, kReady, kFailed, kRunning, kDone };

  explicit Exporter(PushOptions opt);

  // Starts the exporter thread, which runs the preflight (design §4.4) and then waits for
  // Configure() or Abort().
  void Start();
  Phase phase();
  std::string preflight_error();

  // After the writer created the files, before sampling starts (main thread). Allocates the
  // rings (fails if the byte bound is over the limit) and lets the thread run.
  bool Configure(std::vector<ExportStream> streams, uint64_t start_real_ns, std::string* error);
  std::string Describe() const;  // startup line: rings, rate estimate
  void Abort();                  // the recording did not start

  // ---- writer thread ----
  void Offer(const Batch& b) override;
  void CloseInput();                      // after the last Offer
  bool WaitDone(uint64_t deadline_mono);  // exporter finished its last push

  PushStats Snapshot();  // never waits for the exporter

  // Test hook: makes the exporter thread throw (an allocation failure stand-in).
  static std::atomic<bool> test_throw;
  bool LogIdle() { return log_.Idle(); }
  std::vector<std::string> DashboardUrls(const std::string& grafana) const;

 private:
  struct Ring {
    size_t slots = 0, slot_records = 0, width = 0;
    std::vector<uint64_t> data;
    std::vector<size_t> counts;
    std::vector<size_t> cols;  // record columns copied (after the 4 header words)
    std::atomic<uint64_t> head{0}, tail{0};
    std::atomic<uint64_t> dropped{0};
  };
  struct Series {
    std::string metric;  // JSON object text
    std::vector<int64_t> ts;
    std::vector<double> v;
  };
  struct StreamState;
  class Sink;

  void Main();
  bool ConfigureImpl(std::vector<ExportStream> streams, uint64_t start_real_ns, std::string* error);
  bool Preflight(std::string* error);
  void Run();
  void ProcessRing(size_t i);
  void Flush(bool final_push);
  void Send(std::string* body, uint64_t points, bool with_meta, bool final_push);
  void MetaLines(std::string* body);
  void UpdateStep();
  void SetError(const std::string& e);
  void Warn(const std::string& e);

  PushOptions opt_;
  std::string url_query_prefix_;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  Phase phase_ = Phase::kPreflight;
  bool abort_ = false;
  std::string preflight_error_;

  std::vector<std::unique_ptr<Ring>> rings_;
  std::vector<std::unique_ptr<StreamState>> streams_;
  std::vector<Series> series_;
  std::vector<ExportStream> cfg_;
  uint64_t pending_points_ = 0;
  int64_t anchor_ms_ = 0;
  // netmon_step_ms: the preliminary value is written at the anchor together with the other
  // metadata (so any zoomed-in view finds a step); later increases at the time they happen.
  int64_t step_prelim_ = 0, step_published_ = 0, step_pending_ = 0, step_pending_ts_ = 0;
  bool meta_published_ = false;
  uint64_t backoff_until_ = 0, backoff_ns_ = 0, last_warn_ = 0;
  std::atomic<bool> input_closed_{false};
  std::atomic<bool> crashed_{false};  // an exception stopped the exporter (push only)

  std::atomic<uint64_t> points_sent_{0}, requests_{0}, failed_points_{0};
  std::atomic<uint64_t> unconfirmed_{0}, clock_back_{0};
  std::mutex err_mu_;  // guards last_error_ only; held for an assignment at most
  std::string last_error_;

  AsyncLog log_;
  HttpClient http_;
  int lock_fd_ = -1;  // abstract-namespace socket held for the process lifetime (§4.4)
};

}  // namespace netmon
