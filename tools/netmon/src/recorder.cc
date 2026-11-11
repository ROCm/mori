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
#include "recorder.h"

#include <pthread.h>
#include <signal.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <thread>

#include "meta_json.h"
#include "writer.h"

namespace netmon {

namespace {

struct StreamInfo {
  std::string file, netdev, ibdev, pci, source;
  uint64_t interval_ns = 0;
  size_t record_words = 0;
  double snapshot_fraction = 0;
  bool ipc_unverified = false;
  std::vector<std::string> names;
};

// Everything meta.json needs, captured by value so the writer can build it on its thread.
struct MetaContext {
  std::string host_id, hostname, kernel, out_dir;
  std::map<std::string, std::string> tags;
  std::vector<std::string> command;
  std::vector<Nic> nics;
  std::vector<Degraded> degraded;
  std::vector<StreamInfo> streams;
  uint64_t start_real = 0, start_mono = 0, end_real = 0, end_mono = 0;
  std::vector<StreamStatsSnapshot> stats;  // empty for the initial meta
  std::vector<std::string> abandoned;
  std::string stop_reason;
};

std::string Kernel() {
  utsname u{};
  if (uname(&u) != 0) return "";
  return std::string(u.release);
}

std::string BuildMeta(const MetaContext& c, const std::string& status,
                      const std::string& write_error) {
  JsonWriter j;
  j.BeginObject();
  j.Key("format_version").Uint(2);
  j.Key("status").String(status);
  if (!write_error.empty()) j.Key("error").String(write_error);
  if (!c.stop_reason.empty()) j.Key("stop_reason").String(c.stop_reason);
  j.Key("tool").BeginObject().Key("name").String("ainic-netmon");
  j.Key("version").String(NETMON_VERSION).EndObject();
  j.Key("host_id").String(c.host_id);
  j.Key("hostname").String(c.hostname);
  j.Key("kernel").String(c.kernel);
  j.Key("tags").BeginObject();
  for (const auto& [k, v] : c.tags) j.Key(k).String(v);
  j.EndObject();
  j.Key("start").BeginObject().Key("realtime_ns").Uint(c.start_real);
  j.Key("monotonic_ns").Uint(c.start_mono).EndObject();
  j.Key("end");
  if (c.end_mono) {
    j.BeginObject().Key("realtime_ns").Uint(c.end_real);
    j.Key("monotonic_ns").Uint(c.end_mono).EndObject();
  } else {
    j.Null();
  }
  j.Key("command").StringArray(c.command);
  j.Key("record_layout").String("u64le: t0_mono_ns, t1_mono_ns, real_ns, real_err_ns, values...");
  j.Key("nics").BeginArray();
  for (const auto& n : c.nics) {
    j.BeginObject().Key("netdev").String(n.netdev).Key("pci").String(n.pci);
    j.Key("card").String(n.card).Key("ibdev").String(n.ibdev).Key("ipc_dev").String(n.ipc_dev);
    j.Key("driver_version").String(n.driver_version).Key("fw_version").String(n.fw_version);
    j.Key("speed_mbps").Int(n.speed_mbps).EndObject();
  }
  j.EndArray();
  j.Key("streams").BeginArray();
  for (const auto& s : c.streams) {
    j.BeginObject().Key("file").String(s.file).Key("netdev").String(s.netdev);
    j.Key("ibdev").String(s.ibdev).Key("pci").String(s.pci).Key("source").String(s.source);
    j.Key("interval_ns").Uint(s.interval_ns).Key("record_words").Uint(s.record_words);
    j.Key("snapshot_fraction").Double(s.snapshot_fraction);
    j.Key("ipc_unverified").Bool(s.ipc_unverified);
    j.Key("names").StringArray(s.names).EndObject();
  }
  j.EndArray();
  j.Key("degraded").BeginArray();
  for (const auto& d : c.degraded) {
    j.BeginObject().Key("netdev").String(d.netdev).Key("source").String(d.source);
    j.Key("reason").String(d.reason).EndObject();
  }
  j.EndArray();
  j.Key("abandoned").StringArray(c.abandoned);
  j.Key("run_stats");
  if (c.stats.empty()) {
    j.Null();
  } else {
    j.BeginObject();
    for (size_t i = 0; i < c.streams.size() && i < c.stats.size(); i++) {
      const auto& st = c.stats[i];
      j.Key(c.streams[i].file).BeginObject();
      j.Key("samples").Uint(st.samples).Key("missed_slots").Uint(st.missed_slots);
      j.Key("dropped").Uint(st.dropped).Key("errors").Uint(st.errors);
      j.Key("max_rtt_us").Double(static_cast<double>(st.max_rtt_ns) / 1e3);
      j.Key("stopped_by_errors").Bool(st.stopped_by_errors);
      j.Key("last_error");
      if (st.last_error.empty())
        j.Null();
      else
        j.String(st.last_error);
      j.Key("dropped_spans_mono_ns").BeginArray();
      for (const auto& [a, b] : st.dropped_spans) j.BeginArray().Uint(a).Uint(b).EndArray();
      j.EndArray();
      j.Key("dropped_spans_overflow").Uint(st.dropped_spans_overflow);
      j.EndObject();
    }
    j.EndObject();
  }
  j.EndObject();
  return j.str() + "\n";
}

// Sleep that is not ended by the (blocked) SIGINT/SIGTERM and swallows nothing.
void SleepNs(uint64_t ns) {
  timespec ts{static_cast<time_t>(ns / 1000000000ull), static_cast<long>(ns % 1000000000ull)};
  while (nanosleep(&ts, &ts) != 0) {
  }
}

// Waits up to timeout for SIGINT/SIGTERM; returns the signal or 0.
int WaitSignal(const sigset_t& set, uint64_t timeout_ns) {
  timespec ts{static_cast<time_t>(timeout_ns / 1000000000ull),
              static_cast<long>(timeout_ns % 1000000000ull)};
  int sig = sigtimedwait(&set, nullptr, &ts);
  return sig > 0 ? sig : 0;
}

// Message on a path that must finish in bounded time. stderr may be a full pipe or a file on a
// hung filesystem, and another thread may hold the stderr FILE lock, so even a raw write can
// block: it runs on a detached thread and we wait at most `wait_ns`. If it does not finish,
// the message is lost (the details are in meta.json) and _exit ends that thread.
void BoundedErr(const std::string& msg, uint64_t wait_ns = 100000000ull) {
  struct Job {
    std::string text;
    std::atomic<bool> done{false};
  };
  auto* job = new Job{msg};  // intentionally leaked: the thread may outlive this call
  std::thread([job] {
    size_t off = 0;
    while (off < job->text.size()) {
      ssize_t w = write(STDERR_FILENO, job->text.data() + off, job->text.size() - off);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) break;
      off += static_cast<size_t>(w);
    }
    job->done.store(true);
  }).detach();
  const uint64_t until = MonoNs() + wait_ns;
  while (!job->done.load() && MonoNs() < until) SleepNs(1000000ull);
}

// Exit while other threads may still be stuck inside driver or filesystem calls: no
// destructors and no stdio (a flush could wait on stderr backpressure or a held lock).
[[noreturn]] void HardExit(int code, const std::string& msg = "") {
  if (!msg.empty()) BoundedErr(msg);
  _exit(code);
}

}  // namespace

int RunRecording(std::vector<std::unique_ptr<Stream>>* streams_in, RecordOptions* opt) {
  auto& all = *streams_in;

  // ---- signals: set before any thread exists, so every thread inherits the mask ----
  sigset_t stop_sigs;
  sigemptyset(&stop_sigs);
  sigaddset(&stop_sigs, SIGINT);
  sigaddset(&stop_sigs, SIGTERM);
  sigset_t block = stop_sigs;
  sigaddset(&block, SIGUSR1);  // samplers unblock it for themselves
  pthread_sigmask(SIG_BLOCK, &block, nullptr);
  InstallWakeSignalHandler();
  signal(SIGPIPE, SIG_IGN);

  // Heap objects that stuck threads may still touch are intentionally never freed on the
  // _exit paths.
  auto* queue = new BatchQueue(all.size(), kQueueQuotaPerStream);
  auto* ctl = new Control();
  auto* writer = new Writer(queue, opt->out_dir, opt->quiet);
  Status st;

  // ---- output directory and lock (writer thread) ----
  writer->RequestPrepare();
  if (!writer->Wait(MonoNs() + opt->io_timeout_ns, &st)) {
    HardExit(kExitSetup,
             "error: preparing " + opt->out_dir + " timed out (filesystem not responding)\n");
  }
  auto abort_writer = [&](const char* why) {
    if (why) fprintf(stderr, "error: %s\n", why);
    writer->RequestAbort();
    Status ignored;
    if (!writer->Wait(MonoNs() + opt->io_timeout_ns, &ignored)) HardExit(kExitSetup);
    delete writer;
    return kExitSetup;
  };
  if (!st.ok()) {
    fprintf(stderr, "error: %s\n", st.message().c_str());
    return abort_writer(nullptr);
  }

  // ---- probe phase: every source is queried once on its own thread ----
  for (size_t i = 0; i < all.size(); i++) {
    all[i]->id = static_cast<int>(i);
    Stream* s = all[i].get();
    s->thread = std::thread(SamplerMain, s, ctl, queue);
  }
  const uint64_t probe_deadline = MonoNs() + opt->probe_timeout_ns;
  bool interrupted = false;
  for (;;) {
    bool pending = false;
    {
      std::lock_guard<std::mutex> lk(ctl->mu);
      for (auto& s : all) pending |= s->probe == ProbeState::kPending;
    }
    if (!pending) break;
    if (WaitSignal(stop_sigs, 100000000ull)) {
      interrupted = true;
      break;
    }
    if (MonoNs() >= probe_deadline) break;
  }
  if (interrupted) {
    // Cancel: ready probes give up now; probes still running keep their original deadline,
    // so a probe stuck in an uninterruptible call cannot turn the cancel into an endless join.
    {
      std::lock_guard<std::mutex> lk(ctl->mu);
      ctl->cmd = Command::kAbandon;
      ctl->cv.notify_all();
    }
    for (;;) {
      bool pending = false;
      {
        std::lock_guard<std::mutex> lk(ctl->mu);
        for (auto& s : all) pending |= s->probe == ProbeState::kPending;
      }
      if (!pending || MonoNs() >= probe_deadline) break;
      SleepNs(50000000ull);
    }
  }

  std::vector<Stream*> final_streams;
  std::string abort_reason;
  std::string probe_report;  // printed after the lock is released (stdio may block)
  {
    std::lock_guard<std::mutex> lk(ctl->mu);
    for (auto& s : all) {
      const std::string name = s->netdev + "." + s->src->kind();
      if (s->probe == ProbeState::kPending) {
        probe_report += "error: " + name + " did not answer its startup probe within " +
                        FormatDurationNs(opt->probe_timeout_ns) + "\n";
        abort_reason = "stuck";
      } else if (s->probe == ProbeState::kFailed) {
        if (s->required) {
          probe_report += "error: " + name + ": " + s->probe_detail + "\n";
          if (abort_reason.empty()) abort_reason = "required source failed";
        } else {
          probe_report += "skip " + name + ": " + s->probe_detail + "\n";
          opt->degraded.push_back({s->netdev, s->src->kind(), s->probe_detail});
        }
      } else {
        if (s->src->kind() == std::string("ipc")) {
          probe_report += name + ": " + s->probe_detail + "\n";
        }
        final_streams.push_back(s.get());
      }
    }
  }
  if (abort_reason != "stuck") fputs(probe_report.c_str(), stderr);
  if (abort_reason == "stuck") {
    // A thread is inside a driver call we cannot interrupt: clean up and leave without
    // destroying anything it may still use.
    {
      std::lock_guard<std::mutex> lk(ctl->mu);
      ctl->cmd = Command::kAbandon;  // let the probes that did answer leave
      ctl->cv.notify_all();
    }
    writer->RequestAbort();
    Status ignored;
    if (writer->Wait(MonoNs() + 5000000000ull, &ignored)) writer->Join();
    for (auto& s : all) {  // probes that answered have left by now; only stuck ones remain
      if (s->exited.load() && s->thread.joinable()) s->thread.join();
    }
    HardExit(kExitSetup, probe_report);  // bounded: stderr may be full
  }
  if (interrupted) abort_reason = "interrupted during startup";
  if (abort_reason.empty() && final_streams.empty()) abort_reason = "no usable stream";

  auto abandon_all = [&]() {
    {
      std::lock_guard<std::mutex> lk(ctl->mu);
      ctl->cmd = Command::kAbandon;
      ctl->cv.notify_all();
    }
    for (auto& s : all) {
      if (s->thread.joinable()) s->thread.join();
    }
  };
  if (!abort_reason.empty()) {
    abandon_all();
    return abort_writer(abort_reason.c_str());
  }

  // ---- create files and the initial meta (writer thread) ----
  MetaContext mc;
  mc.host_id = opt->host_id;
  mc.hostname = Hostname();
  mc.kernel = Kernel();
  mc.out_dir = opt->out_dir;
  mc.tags = opt->tags;
  mc.command = opt->command;
  mc.nics = opt->nics;
  mc.degraded = opt->degraded;
  std::vector<WriterStream> ws;
  for (size_t i = 0; i < final_streams.size(); i++) {
    Stream* s = final_streams[i];
    s->id = static_cast<int>(i);  // dense ids for the queue and the writer
    StreamInfo si;
    si.file = s->file();
    si.netdev = s->netdev;
    si.ibdev = s->ibdev;
    si.pci = s->pci;
    si.source = s->src->kind();
    si.interval_ns = s->interval_ns;
    si.record_words = s->record_words();
    si.snapshot_fraction = s->src->snapshot_fraction();
    si.ipc_unverified = s->ipc_unverified;
    si.names = s->src->names();
    mc.streams.push_back(si);
    WriterStream w;
    w.file = si.file;
    w.label = s->netdev + "." + si.source;
    w.width = si.record_words;
    auto idx = [&](const char* name) {
      for (size_t k = 0; k < si.names.size(); k++) {
        if (si.names[k] == name) return static_cast<int>(k);
      }
      return -1;
    };
    if (si.source == "rdma") {
      w.key_tx = idx("tx_rdma_ucast_bytes");
      w.key_rx = idx("rx_rdma_ucast_bytes");
    } else if (si.source == "ipc") {
      w.key_tx = idx("octets_tx_ok");
      w.key_rx = idx("octets_rx_ok");
    }
    ws.push_back(w);
  }
  mc.start_real = RealNs();
  mc.start_mono = MonoNs();
  writer->RequestCreate(ws, BuildMeta(mc, "running", ""));
  if (!writer->Wait(MonoNs() + opt->io_timeout_ns, &st)) {
    HardExit(kExitSetup, "error: creating output files timed out (filesystem not responding)\n");
  }
  if (!st.ok()) {
    abandon_all();
    return abort_writer(st.message().c_str());
  }

  // ---- go ----
  size_t q_bytes = 0, max_b = 0;
  for (auto* s : final_streams) {
    size_t b = BatchRecordsFor(s->interval_ns) * s->record_words() * 8;
    q_bytes += (kQueueQuotaPerStream + 1) * b;
    max_b = std::max(max_b, b);
  }
  fprintf(stderr, "recording %zu streams to %s (%s); batch payload bound %.1f MiB\n",
          final_streams.size(), opt->out_dir.c_str(),
          opt->duration_ns ? FormatDurationNs(opt->duration_ns).c_str() : "until Ctrl-C",
          static_cast<double>(q_bytes + max_b) / (1 << 20));
  {
    std::lock_guard<std::mutex> lk(ctl->mu);
    ctl->go_mono = MonoNs();
    ctl->go_real = RealNs();
    ctl->cmd = Command::kGo;
    ctl->cv.notify_all();
  }
  for (auto& s : all) {  // streams that failed their probe have already returned
    bool in_final = false;
    for (auto* f : final_streams) in_final |= f == s.get();
    if (!in_final && s->thread.joinable()) s->thread.join();
  }

  // ---- run until a stop condition ----
  const uint64_t end_at = opt->duration_ns ? ctl->go_mono + 20000000ull + opt->duration_ns : 0;
  std::string reason;
  for (;;) {
    if (WaitSignal(stop_sigs, 100000000ull)) {
      reason = "interrupted";
      break;
    }
    if (end_at && MonoNs() >= end_at) {
      reason = "complete";
      break;
    }
    if (writer->failed()) {
      reason = "write failed";
      break;
    }
    bool alive = false;
    for (auto* s : final_streams) alive |= !s->exited.load();
    if (!alive) {
      reason = "all streams stopped";
      break;
    }
  }

  // ---- stop ----
  ctl->stop.store(true);
  const uint64_t stop_deadline = MonoNs() + opt->stop_timeout_ns;
  for (;;) {
    bool alive = false;
    for (auto* s : final_streams) {
      if (!s->exited.load()) {
        alive = true;
        pthread_kill(s->thread.native_handle(), SIGUSR1);
      }
    }
    if (!alive || MonoNs() >= stop_deadline) break;
    SleepNs(100000000ull);
  }
  queue->Close();  // from here on, every submission is rejected
  for (auto* s : final_streams) {
    if (s->exited.load()) {
      s->thread.join();
    } else {
      mc.abandoned.push_back(s->file());
      // The writer may be printing (and stderr may be a full pipe): no stdio before it is done.
      BoundedErr("warning: " + s->netdev + "." + s->src->kind() + " did not stop within " +
                 FormatDurationNs(opt->stop_timeout_ns) + " (stuck in a driver call)\n");
    }
  }
  bool all_errored = true;
  for (auto* s : final_streams) {
    mc.stats.push_back(s->stats.Snapshot());
    all_errored &= mc.stats.back().stopped_by_errors;
  }
  mc.end_real = RealNs();
  mc.end_mono = MonoNs();
  mc.stop_reason = reason;
  std::string status = reason == "complete"      ? "complete"
                       : reason == "interrupted" ? "interrupted"
                                                 : "failed";
  if (reason == "all streams stopped" && all_errored)
    mc.stop_reason = "all streams stopped by errors";

  const bool drain = !writer->failed();
  writer->RequestFinish(drain, status, [mc](const std::string& s, const std::string& err) {
    return BuildMeta(mc, s, err);
  });
  const bool finished = writer->Wait(MonoNs() + opt->io_timeout_ns, &st);

  if (!finished) {
    // The writer is stuck (e.g. in an NFS write or its console output): touch no stdio.
    HardExit(kExitWriteFailed, "error: finalising the recording timed out\n");
  }
  writer->Join();  // done: its stdio use is over

  std::string summary;
  for (size_t i = 0; i < final_streams.size(); i++) {
    const auto& x = mc.stats[i];
    char buf[512];
    snprintf(buf, sizeof(buf),
             "%s: samples %llu, missed %llu, dropped %llu, errors %llu, max rtt %.1f us%s\n",
             final_streams[i]->file().c_str(), static_cast<unsigned long long>(x.samples),
             static_cast<unsigned long long>(x.missed_slots),
             static_cast<unsigned long long>(x.dropped), static_cast<unsigned long long>(x.errors),
             static_cast<double>(x.max_rtt_ns) / 1e3,
             x.stopped_by_errors ? " (stopped by errors)" : "");
    summary += buf;
    if (!x.last_error.empty()) summary += "  last error: " + x.last_error + "\n";
  }

  int code = kExitOk;
  if (writer->failed()) {
    code = kExitWriteFailed;
    summary += "error: " + writer->error() + "\n";
  } else if (!mc.abandoned.empty()) {
    code = kExitAbandoned;
  } else if (all_errored) {
    code = kExitAllStreamsFailed;
  }
  if (!mc.abandoned.empty()) {
    // Stuck samplers remain: the exit must stay bounded, so the summary is best effort.
    HardExit(code, summary);
  }
  fputs(summary.c_str(), stderr);  // normal exit: an ordinary, possibly blocking, print
  delete writer;
  delete ctl;
  delete queue;
  return code;
}

}  // namespace netmon
