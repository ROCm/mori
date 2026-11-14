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
#include "sampler.h"

#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <time.h>

#include <algorithm>
#include <cerrno>

namespace netmon {

static_assert(kWakeSignal == SIGUSR1, "kWakeSignal must be SIGUSR1");

size_t BatchRecordsFor(uint64_t interval_ns) {
  if (interval_ns == 0) return kBackToBackBatchRecords;
  uint64_t n = kSubmitPeriodNs / interval_ns;
  return static_cast<size_t>(std::clamp<uint64_t>(n, 1, kMaxBatchRecords));
}

StreamStatsSnapshot StreamStats::Snapshot() {
  StreamStatsSnapshot s;
  s.samples = samples.load();
  s.missed_slots = missed_slots.load();
  s.dropped = dropped.load();
  s.errors = errors.load();
  s.max_rtt_ns = max_rtt_ns.load();
  s.stopped_by_errors = stopped_by_errors.load();
  std::lock_guard<std::mutex> lk(mu);
  s.last_error = last_error;
  s.dropped_spans = dropped_spans;
  s.dropped_spans_overflow = dropped_spans_overflow;
  return s;
}

static void OnWake(int) {}

void InstallWakeSignalHandler() {
  struct sigaction sa{};
  sa.sa_handler = OnWake;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: blocking calls return EINTR
  sigaction(SIGUSR1, &sa, nullptr);
}

namespace {

void SleepUntilMono(uint64_t t, const std::atomic<bool>& stop) {
  timespec ts{static_cast<time_t>(t / 1000000000ull), static_cast<long>(t % 1000000000ull)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
    if (stop.load()) return;
  }
}

void AtomicMax(std::atomic<uint64_t>& a, uint64_t v) {
  uint64_t cur = a.load();
  while (v > cur && !a.compare_exchange_weak(cur, v)) {
  }
}

class Submitter {
 public:
  Submitter(Stream* s, BatchQueue* q)
      : s_(s), q_(q), records_(BatchRecordsFor(s->interval_ns)), width_(s->record_words()) {
    batch_ = std::make_unique<Batch>(s->id, width_, records_);
  }
  Batch* batch() { return batch_.get(); }
  bool closed() const { return closed_; }

  // Hand the current batch to the writer. Never blocks and never allocates on the full path.
  void Submit() {
    if (!batch_ || batch_->count == 0) return;
    const uint64_t first = batch_->Record(0)[0];
    const uint64_t last = batch_->Record(batch_->count - 1)[0];
    const size_t n = batch_->count;
    switch (q_->Push(batch_)) {
      case BatchQueue::PushResult::kOk:
        batch_ = std::make_unique<Batch>(s_->id, width_, records_);
        last_dropped_ = false;
        break;
      case BatchQueue::PushResult::kFull:
        s_->stats.dropped.fetch_add(n);
        RecordDroppedSpan(first, last);
        batch_->count = 0;  // reuse the same batch
        last_dropped_ = true;
        break;
      case BatchQueue::PushResult::kClosed:
        batch_->count = 0;  // too late: the writer has been finalised
        closed_ = true;
        break;
    }
  }

 private:
  void RecordDroppedSpan(uint64_t first, uint64_t last) {
    std::lock_guard<std::mutex> lk(s_->stats.mu);
    auto& spans = s_->stats.dropped_spans;
    if (last_dropped_ && !spans.empty()) {
      spans.back().second = last;  // contiguous with the previous drop
    } else if (spans.size() < kMaxDroppedSpans) {
      spans.emplace_back(first, last);
    } else {
      s_->stats.dropped_spans_overflow++;
    }
  }

  Stream* s_;
  BatchQueue* q_;
  size_t records_;
  size_t width_;
  std::unique_ptr<Batch> batch_;
  bool last_dropped_ = false;
  bool closed_ = false;
};

}  // namespace

void SamplerMain(Stream* s, Control* ctl, BatchQueue* queue) {
  // Receive the wake signal on this thread only (SIGINT/SIGTERM stay blocked, inherited).
  sigset_t wake;
  sigemptyset(&wake);
  sigaddset(&wake, SIGUSR1);
  pthread_sigmask(SIG_UNBLOCK, &wake, nullptr);
  if (s->cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(s->cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }

  // ---- probe phase ----
  std::string detail;
  Status st = s->src->Probe(&detail);
  Command cmd;
  {
    std::unique_lock<std::mutex> lk(ctl->mu);
    s->probe = st.ok() ? ProbeState::kOk : ProbeState::kFailed;
    s->probe_detail = st.ok() ? detail : st.message();
    ctl->cv.notify_all();
    if (st.ok()) ctl->cv.wait(lk, [&] { return ctl->cmd != Command::kWait; });
    cmd = st.ok() ? ctl->cmd : Command::kAbandon;
  }
  if (cmd != Command::kGo) {
    s->exited.store(true);
    return;
  }

  // ---- sampling phase ----
  const uint64_t iv = s->interval_ns;
  const uint64_t lead = 20000000ull;  // let every thread reach its first deadline
  uint64_t next = ctl->go_mono + lead;
  if (iv > 0) next += (iv - (ctl->go_real + lead) % iv) % iv;  // wall-clock grid alignment

  Submitter sub(s, queue);
  uint64_t last_submit = MonoNs();
  uint64_t consecutive_err = 0;
  while (!ctl->stop.load() && !sub.closed()) {
    SleepUntilMono(next, ctl->stop);
    if (ctl->stop.load()) break;

    Batch* b = sub.batch();
    uint64_t* rec = b->Slot();
    const uint64_t mono_a = MonoNs();
    const uint64_t real = RealNs();
    const uint64_t t0 = MonoNs();
    st = s->src->Query(rec + kRecordHeaderWords);
    const uint64_t t1 = MonoNs();
    if (st.ok()) {
      rec[0] = t0;
      rec[1] = t1;
      rec[2] = real;
      rec[3] = t0 - mono_a;
      b->count++;
      s->stats.samples.fetch_add(1);
      AtomicMax(s->stats.max_rtt_ns, t1 - t0);
      consecutive_err = 0;
    } else if (!ctl->stop.load()) {  // errors caused by the stop signal are not counted
      s->stats.errors.fetch_add(1);
      {
        std::lock_guard<std::mutex> lk(s->stats.mu);
        s->stats.last_error = st.message();
      }
      if (++consecutive_err >= kMaxConsecutiveErrors) {
        s->stats.stopped_by_errors.store(true);
        break;
      }
    }

    const uint64_t now = MonoNs();
    if (iv == 0) {
      next = now;
    } else {
      next += iv;
      if (next <= now) {  // no catch-up bursts: skip to the next future slot
        const uint64_t behind = (now - next) / iv + 1;
        s->stats.missed_slots.fetch_add(behind);
        next += behind * iv;
      }
    }
    if (b->Full() || now - last_submit >= kSubmitPeriodNs) {
      sub.Submit();
      last_submit = now;
    }
  }
  sub.Submit();  // tail batch (rejected if the queue is already closed)
  s->exited.store(true);
}

}  // namespace netmon
