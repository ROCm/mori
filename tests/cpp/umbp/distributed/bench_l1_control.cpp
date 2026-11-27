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

// What the L1 pool's control calls cost an engine, over the socket and over
// shared memory. CPU only: pools are host memory and a copy is a sleep.
//
//   --mode=serve    the L1 pool service, as umbp_standalone_server hosts it
//   --mode=rank     one engine rank: the scheduler's block bookkeeping (a port
//                   of ATOM's UmbpL1Pool), driven by a session schedule every
//                   rank derives from the same seed, in lockstep steps
//   --mode=percall  one client, one call shape at a time
//   --mode=manager  no transport: what the server's own bookkeeping costs
//
// A rank does what an engine's scheduler thread does to the pool each step:
// finish requests (release their blocks), publish the blocks the last forward
// filled, admit requests (claim the prefix it holds, ask other ranks' pools
// for the rest, allocate what is new), grow decoding requests a block at a
// time, and flush. A second thread reports device progress (Commit) when the
// step's forward would have finished; four more run the copying fetches.
// Each call is timed where the engine would wait for it.

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/service_type.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/l1/l1_pool.h"
#include "umbp/l1/l1_service.h"

using namespace mori::umbp::l1;
using Clock = std::chrono::steady_clock;

namespace {

// ------------------------------- options -----------------------------------

struct Options {
  std::string mode = "rank";
  std::string address = "unix:///tmp/umbp_l1_bench.sock";
  std::string transport = "grpc";
  std::string out;
  int rank = 0;
  int ranks = 4;
  uint32_t slots = 777000;
  uint32_t sessions = 1024;
  uint32_t steps = 720;
  uint32_t warmup_steps = 120;
  double step_ms = 83.0;
  double forward_ms = 70.0;
  // Requests admitted a second on one rank, and how long one runs.
  double admit_per_s = 3.3;
  double run_s = 22.0;
  // Context length in blocks: log-normal, by its median and sigma.
  double prefix_median = 545;
  double prefix_sigma = 1.154;
  uint32_t prefix_max = 6665;
  double new_blocks = 40;   // prompt blocks nobody holds yet, mean
  double out_blocks = 6;    // blocks a request decodes, mean
  double same_rank = 0.64;  // a session's next turn lands on the rank its last one ran on
  uint64_t seed = 1;
  uint64_t start_ns = 0;     // CLOCK_MONOTONIC instant of step 0; 0: now + 1 s
  bool keyless_tail = true;  // a finished request's last, partial block has no key
  bool busy = true;          // the scheduler thread never sleeps, like an engine's
  int cpu = -1;              // pin the scheduler thread
  double copy_us_per_slot = 5.0;
  uint32_t percall_iters = 2000;
  uint32_t percall_gap_us = 0;
};

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
      std::fprintf(stderr, "bad argument %s (want --name=value)\n", arg.c_str());
      std::exit(2);
    }
    const std::string name = arg.substr(2, eq - 2), value = arg.substr(eq + 1);
    auto num = [&] { return std::strtod(value.c_str(), nullptr); };
    if (name == "mode")
      o.mode = value;
    else if (name == "address")
      o.address = value;
    else if (name == "transport")
      o.transport = value;
    else if (name == "out")
      o.out = value;
    else if (name == "rank")
      o.rank = static_cast<int>(num());
    else if (name == "ranks")
      o.ranks = static_cast<int>(num());
    else if (name == "slots")
      o.slots = static_cast<uint32_t>(num());
    else if (name == "sessions")
      o.sessions = static_cast<uint32_t>(num());
    else if (name == "steps")
      o.steps = static_cast<uint32_t>(num());
    else if (name == "warmup_steps")
      o.warmup_steps = static_cast<uint32_t>(num());
    else if (name == "step_ms")
      o.step_ms = num();
    else if (name == "forward_ms")
      o.forward_ms = num();
    else if (name == "admit_per_s")
      o.admit_per_s = num();
    else if (name == "run_s")
      o.run_s = num();
    else if (name == "prefix_median")
      o.prefix_median = num();
    else if (name == "prefix_sigma")
      o.prefix_sigma = num();
    else if (name == "prefix_max")
      o.prefix_max = static_cast<uint32_t>(num());
    else if (name == "new_blocks")
      o.new_blocks = num();
    else if (name == "out_blocks")
      o.out_blocks = num();
    else if (name == "same_rank")
      o.same_rank = num();
    else if (name == "seed")
      o.seed = std::strtoull(value.c_str(), nullptr, 10);
    else if (name == "start_ns")
      o.start_ns = std::strtoull(value.c_str(), nullptr, 10);
    else if (name == "busy")
      o.busy = num() != 0;
    else if (name == "keyless_tail")
      o.keyless_tail = num() != 0;
    else if (name == "cpu")
      o.cpu = static_cast<int>(num());
    else if (name == "copy_us_per_slot")
      o.copy_us_per_slot = num();
    else if (name == "percall_iters")
      o.percall_iters = static_cast<uint32_t>(num());
    else if (name == "percall_gap_us")
      o.percall_gap_us = static_cast<uint32_t>(num());
    else {
      std::fprintf(stderr, "unknown option --%s\n", name.c_str());
      std::exit(2);
    }
  }
  return o;
}

uint64_t NowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void WaitUntilNs(uint64_t when, bool busy) {
  if (!busy) {
    const uint64_t now = NowNs();
    if (when > now + 200000) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(when - now - 100000));
    }
  }
  while (NowNs() < when) {
  }
}

void Pin(int cpu) {
  if (cpu < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

[[noreturn]] void Die(const std::string& what) {
  std::fprintf(stderr, "bench_l1_control: %s\n", what.c_str());
  std::exit(1);
}

void Must(const Status& status, const char* what) {
  if (!status.ok) Die(std::string(what) + ": " + status.error);
}

constexpr uint64_t kSlotBytes = 16;

// ------------------------------- server ------------------------------------

// Host memory whose copies take the time a GPU copy of that many slots takes.
// Copies into one device queue behind each other, as on the real server.
class TimedCopyApi final : public MemoryApi {
 public:
  explicit TimedCopyApi(double us_per_slot) : us_per_slot_(us_per_slot) {}
  bool Allocate(int device, uint64_t bytes, void** base, ExportedMemory* exported,
                std::string* error) override {
    return host_->Allocate(device, bytes, base, exported, error);
  }
  void Free(int device, void* base, uint64_t bytes) override { host_->Free(device, base, bytes); }
  bool Import(int device, const std::string& handle, uint64_t offset, void** base, void** mapped,
              std::string* error) override {
    return host_->Import(device, handle, offset, base, mapped, error);
  }
  void Unimport(int device, void* mapped) override { host_->Unimport(device, mapped); }
  bool Copy(int, void*, int, const void*, uint64_t, std::string*) override { return true; }
  bool Synchronize(int, std::string*) override { return true; }
  bool CopyBatch(int dst_device, const std::vector<Fragment>& fragments, std::string*) override {
    uint64_t bytes = 0;
    for (const Fragment& f : fragments) bytes += f.bytes;
    std::mutex* device_mutex;
    {
      std::lock_guard<std::mutex> lock(table_mutex_);
      device_mutex = &device_mutex_[dst_device];
    }
    std::lock_guard<std::mutex> lock(*device_mutex);
    std::this_thread::sleep_for(std::chrono::nanoseconds(
        static_cast<int64_t>(static_cast<double>(bytes / kSlotBytes) * us_per_slot_ * 1000.0)));
    return true;
  }

 private:
  double us_per_slot_;
  std::shared_ptr<MemoryApi> host_ = MakeHostMemoryApi();
  std::mutex table_mutex_;
  std::map<int, std::mutex> device_mutex_;
};

std::atomic<bool> g_stop{false};
void OnSignal(int) { g_stop.store(true); }

int Serve(const Options& o) {
  auto manager = std::make_shared<PoolManager>(std::make_shared<TimedCopyApi>(o.copy_us_per_slot));
  std::unique_ptr<grpc::Service> service = MakeL1PoolService(manager);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(o.address, grpc::InsecureServerCredentials());
  // As umbp_standalone_server sets them.
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 4);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 32);
  builder.SetMaxReceiveMessageSize(-1);
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) Die("cannot listen on " + o.address);
  std::signal(SIGTERM, OnSignal);
  std::signal(SIGINT, OnSignal);
  std::printf("serving %s\n", o.address.c_str());
  std::fflush(stdout);
  while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  StopL1PoolService(service.get());
  server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
  return 0;
}

// ------------------------------ schedule -----------------------------------

uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// The key of block `index` of a conversation. A conversation that starts
// over (`generation`) shares nothing with its past.
std::string Key(uint32_t session, uint32_t generation, uint32_t index) {
  const uint64_t h = Mix((uint64_t{session} << 40) ^ (uint64_t{generation} << 24) ^ index) | 1;
  return std::string(reinterpret_cast<const char*>(&h), sizeof(h));
}

struct Turn {
  uint32_t session = 0;
  uint32_t generation = 0;
  uint32_t start = 0;  // step it arrives
  uint32_t end = 0;    // step it finishes, given it starts on time
  int rank = 0;
  uint32_t prefix = 0;  // blocks of context it arrives with
  uint32_t fresh = 0;   // prompt blocks nobody holds
  uint32_t out = 0;     // blocks it decodes
};

struct SessionStart {
  uint32_t generation = 0;
  uint32_t blocks = 0;
  int rank = 0;
};

// Every rank derives the same schedule from the seed and plays its own part.
struct Schedule {
  std::vector<SessionStart> initial;
  std::vector<Turn> turns;
};

Schedule MakeSchedule(const Options& o) {
  Schedule schedule;
  std::mt19937_64 rng(o.seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::lognormal_distribution<double> context(std::log(o.prefix_median), o.prefix_sigma);
  auto draw_context = [&] {
    return static_cast<uint32_t>(std::clamp(context(rng), 40.0, static_cast<double>(o.prefix_max)));
  };
  struct Session {
    uint32_t generation = 0;
    uint32_t blocks = 0;
    int rank = 0;
    uint32_t free_at = 0;
  };
  std::vector<Session> sessions(o.sessions);
  for (Session& s : sessions) {
    s.blocks = draw_context();
    s.rank = static_cast<int>(rng() % static_cast<uint64_t>(o.ranks));
    schedule.initial.push_back(SessionStart{0, s.blocks, s.rank});
  }
  const double per_step = o.admit_per_s * o.ranks * o.step_ms / 1000.0;
  const double run_steps = o.run_s * 1000.0 / o.step_ms;
  std::poisson_distribution<int> arrivals(per_step);
  std::exponential_distribution<double> duration(1.0 / run_steps);
  std::exponential_distribution<double> fresh(1.0 / o.new_blocks);
  std::poisson_distribution<int> out(o.out_blocks);
  uint32_t next = 0;
  for (uint32_t step = 0; step < o.steps; ++step) {
    const int n = arrivals(rng);
    for (int a = 0; a < n; ++a) {
      // The session that has waited longest.
      uint32_t tried = 0;
      while (tried < o.sessions && sessions[next].free_at > step) {
        next = (next + 1) % o.sessions;
        ++tried;
      }
      if (tried == o.sessions) break;
      Session& s = sessions[next];
      const uint32_t id = next;
      next = (next + 1) % o.sessions;
      Turn turn;
      turn.session = id;
      turn.generation = s.generation;
      turn.start = step;
      turn.end = step + std::max<uint32_t>(4, static_cast<uint32_t>(duration(rng)));
      turn.rank = s.rank;
      if (o.ranks > 1 && uniform(rng) >= o.same_rank) {
        turn.rank = static_cast<int>(
            (static_cast<uint64_t>(s.rank) + 1 + rng() % static_cast<uint64_t>(o.ranks - 1)) %
            static_cast<uint64_t>(o.ranks));
      }
      turn.prefix = s.blocks;
      turn.fresh = std::clamp<uint32_t>(static_cast<uint32_t>(fresh(rng)) + 1, 1, 512);
      turn.out = static_cast<uint32_t>(out(rng));
      schedule.turns.push_back(turn);
      s.rank = turn.rank;
      s.blocks += turn.fresh + turn.out;
      s.free_at = turn.end + 2;
      if (s.blocks > o.prefix_max) {
        // The conversation ends; the session starts another.
        ++s.generation;
        s.blocks = std::max<uint32_t>(40, draw_context() / 4);
      }
    }
  }
  return schedule;
}

// ------------------------------- timing ------------------------------------

struct Series {
  std::vector<uint32_t> nanos;
  std::vector<uint32_t> size;
};

struct Recorder {
  std::mutex mutex;
  std::map<std::string, Series> calls;
  bool on = false;

  void Add(const char* name, uint64_t nanos, size_t size) {
    if (!on) return;
    std::lock_guard<std::mutex> lock(mutex);
    Series& s = calls[name];
    s.nanos.push_back(static_cast<uint32_t>(std::min<uint64_t>(nanos, UINT32_MAX)));
    s.size.push_back(static_cast<uint32_t>(size));
  }
};

template <typename T>
void JsonArray(std::ofstream& out, const std::vector<T>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
  out << "]";
}

// ----------------------- the scheduler's view of a pool --------------------

// ATOM's UmbpL1Pool: an exact local view of the pool, with the transitions
// UMBP has not been told yet batched until the next flush.
class LocalPool {
 public:
  LocalPool(L1Client* client, std::string id, uint32_t slots, Recorder* recorder)
      : client_(client), id_(std::move(id)), hash_(slots), ref_(slots, 0), recorder_(recorder) {}

  uint64_t scheduler_nanos = 0;  // time the scheduler thread spent inside calls
  uint64_t evicted = 0;

  const std::string& id() const { return id_; }
  bool Holds(const std::string& key) const { return index_.count(key) != 0; }
  uint32_t SlotOf(const std::string& key) const { return index_.at(key); }
  size_t Room() const { return ref_.size() - used_ - stash_.size(); }

  void Publish(uint32_t slot, const std::string& key) {
    std::string& old = hash_[slot];
    if (!old.empty() && old != key) {
      auto it = index_.find(old);
      if (it != index_.end() && it->second == slot) index_.erase(it);
    }
    old = key;
    auto prev = index_.find(key);
    if (prev != index_.end() && prev->second == slot) return;
    index_[key] = slot;
    ops_.push_back(Op{OpKind::kPublish, slot, key});
  }

  // The pool already holds `key` in `slot`, cached: only this view learns it.
  void Adopt(uint32_t slot, const std::string& key) {
    hash_[slot] = key;
    index_[key] = slot;
  }

  void Claim(uint32_t slot) {
    if (ref_[slot]++ != 0) return;
    ++used_;
    ops_.push_back(Op{OpKind::kHold, slot, ""});
  }

  void Free(uint32_t slot) {
    if (--ref_[slot] != 0) return;
    --used_;
    ops_.push_back(Op{OpKind::kRelease, slot, ""});
  }

  void Flush() {
    if (ops_.empty()) return;
    const uint64_t t0 = NowNs();
    Must(client_->Apply(id_, ops_, t0), "apply");
    Timed("apply", t0, ops_.size());
    ops_.clear();
  }

  void Prepare(size_t count) {
    if (count > stash_.size()) Refill(count - stash_.size());
  }

  // A block for new content, held by the caller.
  uint32_t Pop() {
    if (stash_.empty() && !Refill(1)) Die("pool " + id_ + " has no free block");
    const uint32_t slot = stash_.front();
    stash_.pop_front();
    ref_[slot] = 1;
    ++used_;
    return slot;
  }

  // How many leading keys the group could supply.
  uint32_t Probe(const std::vector<std::string>& keys) {
    FetchOptions options;
    options.probe_only = true;
    const uint64_t t0 = NowNs();
    FetchResult result = client_->Fetch(id_, keys, options);
    Must(result.status, "probe");
    Timed("probe", t0, keys.size());
    return result.available;
  }

 private:
  void Timed(const char* name, uint64_t t0, size_t size) {
    const uint64_t dt = NowNs() - t0;
    scheduler_nanos += dt;
    recorder_->Add(name, dt, size);
  }

  bool Refill(size_t need) {
    const size_t available = Room();
    need = std::min(need, available);
    if (need == 0) return false;
    Flush();
    const uint64_t t0 = NowNs();
    AllocateResult got = client_->Allocate(
        id_, static_cast<uint32_t>(std::min(std::max<size_t>(need, 32), available)),
        static_cast<uint32_t>(need));
    Must(got.status, "allocate");
    Timed("allocate", t0, got.slots.size());
    // What the grant destroyed must be what this view says those slots held.
    size_t gone = 0;
    for (uint32_t slot : got.slots) {
      std::string& key = hash_[slot];
      if (!key.empty()) {
        auto it = index_.find(key);
        if (it != index_.end() && it->second == slot) {
          index_.erase(it);
          ++gone;
        }
        key.clear();
      }
      stash_.push_back(slot);
    }
    if (gone != got.evicted_keys.size()) {
      Die("pool " + id_ + ": local view diverged (evicted here " + std::to_string(gone) +
          ", by UMBP " + std::to_string(got.evicted_keys.size()) + ")");
    }
    evicted += gone;
    return !got.slots.empty();
  }

  L1Client* client_;
  std::string id_;
  std::vector<std::string> hash_;
  std::vector<uint32_t> ref_;
  std::unordered_map<std::string, uint32_t> index_;
  size_t used_ = 0;
  std::deque<uint32_t> stash_;
  std::vector<Op> ops_;
  Recorder* recorder_;
};

// --------------------------------- rank ------------------------------------

struct Copy {
  std::vector<std::string> keys;
  std::vector<uint32_t> slots;
  uint32_t landed = 0;
  // Set, after `landed`, by the thread that ran the copy.
  std::shared_ptr<bool> done_flag;
};

struct Request {
  Turn turn;
  std::vector<uint32_t> table;
  std::shared_ptr<Copy> copy;
  bool admitted = false;
  uint32_t end = 0;
  uint32_t grown = 0;
  uint32_t started = 0;
  // 1 + the index of the block decode is writing, the table's last; 0: none.
  uint32_t open_key = 0;
};

class Workers {
 public:
  explicit Workers(int n) {
    for (int i = 0; i < n; ++i) threads_.emplace_back([this] { Run(); });
  }
  ~Workers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : threads_) t.join();
  }
  void Submit(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(std::move(job));
    }
    wake_.notify_one();
  }

 private:
  void Run() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      job();
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::function<void()>> jobs_;
  bool stop_ = false;
  std::vector<std::thread> threads_;
};

L1ClientOptions ClientOptions(const Options& o) {
  L1ClientOptions options = L1ClientOptions::FromEnv();
  options.shm = o.transport == "shm";
  return options;
}

// Fill the pool: the conversations whose last turn ran here, then content
// nobody will ask for again. Everything cached and settled.
void Populate(L1Client* client, LocalPool* pool, const Options& o, const Schedule& schedule) {
  uint32_t filled = 0;
  uint64_t filler = 0;
  std::vector<Op> ops;
  auto flush = [&] {
    if (ops.empty()) return;
    Must(client->Apply(pool->id(), ops, 0), "populate");
    ops.clear();
  };
  auto put = [&](const std::string& key) {
    ops.push_back(Op{OpKind::kHold, filled, ""});
    ops.push_back(Op{OpKind::kPublish, filled, key});
    ops.push_back(Op{OpKind::kRelease, filled, ""});
    pool->Adopt(filled, key);
    ++filled;
    if (ops.size() >= 6000) flush();
  };
  for (size_t s = 0; s < schedule.initial.size() && filled < o.slots; ++s) {
    const SessionStart& start = schedule.initial[s];
    if (start.rank != o.rank) continue;
    for (uint32_t b = 0; b < start.blocks && filled < o.slots; ++b) {
      put(Key(static_cast<uint32_t>(s), start.generation, b));
    }
  }
  while (filled < o.slots) {
    // Even, where a conversation's keys are odd: never asked for.
    const uint64_t h =
        Mix((uint64_t{0xf111} << 48) ^ (static_cast<uint64_t>(o.rank) << 40) ^ filler++) & ~1ull;
    put(std::string(reinterpret_cast<const char*>(&h), sizeof(h)));
  }
  flush();
  // A synchronous call: everything above has been applied when it returns.
  std::vector<PoolStats> stats;
  Must(client->Stats(&stats), "stats");
  for (const PoolStats& pool_stats : stats) {
    if (pool_stats.pool_id == pool->id() && pool_stats.cached != o.slots) {
      Die("populate left " + std::to_string(pool_stats.cached) + " cached slots, not " +
          std::to_string(o.slots));
    }
  }
}

void WriteResult(const Options& o, Recorder* recorder, const std::vector<uint32_t>& step_nanos,
                 const std::map<std::string, uint64_t>& counters) {
  if (o.out.empty()) return;
  std::ofstream out(o.out);
  out << "{\"mode\":\"" << o.mode << "\",\"rank\":" << o.rank << ",\"transport\":\"" << o.transport
      << "\",\"step_ms\":" << o.step_ms << ",\"measured_steps\":" << step_nanos.size();
  for (const auto& counter : counters) out << ",\"" << counter.first << "\":" << counter.second;
  out << ",\"step_nanos\":";
  JsonArray(out, step_nanos);
  out << ",\"calls\":{";
  bool first = true;
  for (const auto& call : recorder->calls) {
    out << (first ? "" : ",") << "\"" << call.first << "\":{\"nanos\":";
    JsonArray(out, call.second.nanos);
    out << ",\"size\":";
    JsonArray(out, call.second.size);
    out << "}";
    first = false;
  }
  out << "}}\n";
}

int RunRank(const Options& o) {
  const Schedule schedule = MakeSchedule(o);
  std::vector<std::vector<Turn>> arrivals(o.steps);
  for (const Turn& turn : schedule.turns) {
    if (turn.rank == o.rank) arrivals[turn.start].push_back(turn);
  }

  L1Client client(o.address, ClientOptions(o));
  if ((o.transport == "shm") != client.shm_active())
    Die("transport " + o.transport + " not active");
  const std::string id = "rank" + std::to_string(o.rank);
  Must(client.Attach(id, "bench", o.rank, uint64_t{o.slots} * kSlotBytes).status, "attach");
  Layout layout;
  layout.num_slots = o.slots;
  layout.regions = {Region{0, kSlotBytes}};
  Must(client.BindLayout(id, layout), "bind");
  Must(client.Open(id).status, "open");

  Recorder recorder;
  LocalPool pool(&client, id, o.slots, &recorder);
  Populate(&client, &pool, o, schedule);
  std::printf("rank %d populated\n", o.rank);
  std::fflush(stdout);

  const uint64_t period = static_cast<uint64_t>(o.step_ms * 1e6);
  const uint64_t forward = static_cast<uint64_t>(o.forward_ms * 1e6);
  const uint64_t start = o.start_ns != 0 ? o.start_ns : NowNs() + 1000000000ull;
  if (NowNs() > start) Die("step 0 is already in the past; start the ranks with a later start_ns");

  // Device progress: the runner's watcher thread, one Commit a forward. The
  // runner is another process than the scheduler, so it has its own client.
  L1Client runner(o.address, ClientOptions(o));
  std::mutex commit_mutex;
  std::condition_variable commit_wake;
  std::deque<std::pair<uint64_t, uint64_t>> commits;  // (stamp, when the forward is done)
  bool commit_stop = false;
  std::thread committer([&] {
    for (;;) {
      std::pair<uint64_t, uint64_t> next;
      {
        std::unique_lock<std::mutex> lock(commit_mutex);
        commit_wake.wait(lock, [&] { return commit_stop || !commits.empty(); });
        if (commits.empty()) return;
        next = commits.front();
        commits.pop_front();
      }
      WaitUntilNs(next.second, false);
      const uint64_t t0 = NowNs();
      Must(runner.Commit(id, next.first), "commit");
      recorder.Add("commit", NowNs() - t0, 1);
    }
  });

  std::map<std::string, uint64_t> counters;
  std::vector<uint32_t> step_nanos;
  {
    Workers copiers(4);
    std::vector<Request> running, waiting;
    std::vector<std::pair<uint32_t, std::string>> filled;  // blocks the last forward wrote

    auto admit = [&](Request request, uint32_t step) {
      const Turn& t = request.turn;
      uint32_t hits = 0;
      while (hits < t.prefix) {
        const std::string key = Key(t.session, t.generation, hits);
        if (!pool.Holds(key)) break;
        const uint32_t slot = pool.SlotOf(key);
        pool.Claim(slot);
        request.table.push_back(slot);
        ++hits;
      }
      const uint32_t compute = t.prefix - hits + t.fresh;
      pool.Prepare(compute);
      for (uint32_t i = 0; i < compute; ++i) {
        const uint32_t slot = pool.Pop();
        request.table.push_back(slot);
        filled.emplace_back(slot, Key(t.session, t.generation, hits + i));
      }
      counters["prefix_blocks"] += t.prefix;
      counters["prefix_hit_blocks"] += hits;
      counters["computed_blocks"] += compute;
      request.admitted = true;
      request.started = step;
      request.end = step + (t.end - t.start);
      running.push_back(std::move(request));
    };

    auto arrive = [&](const Turn& t, uint32_t step) {
      Request request;
      request.turn = t;
      ++counters["requests"];
      uint32_t local = 0;
      while (local < t.prefix && pool.Holds(Key(t.session, t.generation, local))) ++local;
      if (local == t.prefix) {
        admit(std::move(request), step);
        return;
      }
      // What this rank holds of the prefix was just used: a hold and a
      // release each, so the copy below does not push it out.
      for (uint32_t i = 0; i < local; ++i) pool.Claim(pool.SlotOf(Key(t.session, t.generation, i)));
      for (uint32_t i = 0; i < local; ++i) pool.Free(pool.SlotOf(Key(t.session, t.generation, i)));
      std::vector<std::string> rest;
      rest.reserve(t.prefix - local);
      for (uint32_t i = local; i < t.prefix; ++i) rest.push_back(Key(t.session, t.generation, i));
      const uint32_t available = pool.Probe(rest);
      size_t n = std::min<size_t>(available, pool.Room());
      if (n == 0) {
        admit(std::move(request), step);
        return;
      }
      pool.Prepare(n);
      auto copy = std::make_shared<Copy>();
      copy->keys.assign(rest.begin(), rest.begin() + static_cast<long>(n));
      for (size_t i = 0; i < n; ++i) copy->slots.push_back(pool.Pop());
      ++counters["fetches"];
      counters["fetch_blocks"] += n;
      copiers.Submit([&client, &recorder, &id, copy] {
        FetchOptions options;
        options.dst_slots = copy->slots;
        const uint64_t t0 = NowNs();
        FetchResult result = client.Fetch(id, copy->keys, options);
        Must(result.status, "fetch");
        recorder.Add("fetch_copy", NowNs() - t0, copy->keys.size());
        copy->landed = static_cast<uint32_t>(result.slots.size());
        std::atomic_store(&copy->done_flag, std::make_shared<bool>(true));
      });
      request.copy = std::move(copy);
      waiting.push_back(std::move(request));
    };

    for (uint32_t step = 0; step < o.steps; ++step) {
      WaitUntilNs(start + step * period, o.busy);
      recorder.on = step >= o.warmup_steps;
      const uint64_t before = pool.scheduler_nanos;

      // The last forward's outputs: blocks it filled, requests it finished.
      for (const auto& block : filled) pool.Publish(block.first, block.second);
      filled.clear();
      for (size_t i = 0; i < running.size();) {
        if (running[i].end > step) {
          ++i;
          continue;
        }
        // The block decode was still writing is not full: with `keyless_tail`
        // it is released without a key, as the engine does, and is a vacant
        // slot the pool hands out again first.
        if (running[i].open_key != 0 && !o.keyless_tail) {
          const Turn& t = running[i].turn;
          pool.Publish(running[i].table.back(),
                       Key(t.session, t.generation, running[i].open_key - 1));
        }
        for (uint32_t slot : running[i].table) pool.Free(slot);
        running[i] = std::move(running.back());
        running.pop_back();
      }
      pool.Flush();

      // Schedule the next one. Copies that landed are published and released
      // like blocks a forward filled; their requests are admitted.
      for (size_t i = 0; i < waiting.size();) {
        Copy& copy = *waiting[i].copy;
        if (std::atomic_load(&copy.done_flag) == nullptr) {
          ++i;
          continue;
        }
        for (size_t k = 0; k < copy.slots.size(); ++k) {
          if (k < copy.landed && !pool.Holds(copy.keys[k]))
            pool.Publish(copy.slots[k], copy.keys[k]);
          pool.Free(copy.slots[k]);
        }
        counters["fetch_landed_blocks"] += copy.landed;
        Request request = std::move(waiting[i]);
        waiting[i] = std::move(waiting.back());
        waiting.pop_back();
        request.copy.reset();
        admit(std::move(request), step);
      }
      for (const Turn& t : arrivals[step]) arrive(t, step);
      for (Request& request : running) {
        const Turn& t = request.turn;
        const uint32_t length = request.end - request.started;
        while (request.grown < t.out &&
               step >= request.started + uint64_t{request.grown + 1} * length / (t.out + 1)) {
          // The block being decoded into is full: publish it, take the next.
          // With nothing in the stash that is an Apply, then an Allocate of 1.
          if (request.open_key != 0) {
            pool.Publish(request.table.back(), Key(t.session, t.generation, request.open_key - 1));
          }
          const uint32_t slot = pool.Pop();
          request.table.push_back(slot);
          request.open_key = t.prefix + t.fresh + request.grown + 1;
          ++request.grown;
        }
      }
      pool.Flush();

      // The forward is issued now; its writes land when it is done.
      {
        std::lock_guard<std::mutex> lock(commit_mutex);
        commits.emplace_back(NowNs(), start + step * period + forward);
      }
      commit_wake.notify_one();
      if (recorder.on) {
        step_nanos.push_back(
            static_cast<uint32_t>(std::min<uint64_t>(pool.scheduler_nanos - before, UINT32_MAX)));
        counters["running_sum"] += running.size();
      }
    }
    recorder.on = false;
  }
  {
    std::lock_guard<std::mutex> lock(commit_mutex);
    commit_stop = true;
  }
  commit_wake.notify_all();
  committer.join();
  counters["evicted"] = pool.evicted;
  WriteResult(o, &recorder, step_nanos, counters);
  std::printf("rank %d done\n", o.rank);
  return 0;
}

// -------------------------------- per call ---------------------------------

// One call shape at a time against a full pool, from one client.
int RunPerCall(const Options& o) {
  Pin(o.cpu);
  L1Client client(o.address, ClientOptions(o));
  if ((o.transport == "shm") != client.shm_active())
    Die("transport " + o.transport + " not active");
  Layout layout;
  layout.num_slots = o.slots;
  layout.regions = {Region{0, kSlotBytes}};
  for (const char* id : {"mine", "peer"}) {
    Must(client.Attach(id, "percall", id[0] == 'm' ? 0 : 1, uint64_t{o.slots} * kSlotBytes).status,
         "attach");
    Must(client.BindLayout(id, layout), "bind");
    Must(client.Open(id).status, "open");
  }
  auto key = [](uint64_t space, uint64_t i) {
    const uint64_t h = Mix((space << 56) ^ i);
    return std::string(reinterpret_cast<const char*>(&h), sizeof(h));
  };
  // Both pools full of cached content.
  for (const char* id : {"mine", "peer"}) {
    std::vector<Op> ops;
    for (uint32_t slot = 0; slot < o.slots; ++slot) {
      ops.push_back(Op{OpKind::kHold, slot, ""});
      ops.push_back(Op{OpKind::kPublish, slot, key(id[0] == 'm' ? 1 : 2, slot)});
      ops.push_back(Op{OpKind::kRelease, slot, ""});
      if (ops.size() >= 6000 || slot + 1 == o.slots) {
        Must(client.Apply(id, ops, 0), "populate");
        ops.clear();
      }
    }
  }
  std::vector<PoolStats> stats;
  Must(client.Stats(&stats), "stats");

  Recorder recorder;
  recorder.on = true;
  const uint64_t gap = uint64_t{o.percall_gap_us} * 1000;
  uint64_t next = NowNs();
  // `call` is timed; `after` restores the pool for the next round, untimed.
  auto run = [&](const std::string& name, size_t size, uint32_t iters,
                 const std::function<void(uint32_t)>& call,
                 const std::function<void(uint32_t)>& after) {
    for (uint32_t i = 0; i < iters + 20; ++i) {
      next = std::max(next + gap, NowNs());
      WaitUntilNs(next, o.busy);
      const uint64_t t0 = NowNs();
      call(i);
      const uint64_t dt = NowNs() - t0;
      if (i >= 20) recorder.Add(name.c_str(), dt, size);
      if (after) after(i);
    }
    // A call that returns a reply: whatever was queued has been applied.
    Must(client.Stats(&stats), "stats");
  };

  uint64_t stamp = 1;
  for (uint32_t n : {4u, 64u, 1000u, 4000u}) {
    if (n > o.slots / 2) continue;
    // A request arriving (holds of its prefix) and leaving (releases).
    std::vector<Op> hold, release;
    for (uint32_t slot = 0; slot < n; ++slot) {
      hold.push_back(Op{OpKind::kHold, slot, ""});
      release.push_back(Op{OpKind::kRelease, slot, ""});
    }
    run(
        "apply/" + std::to_string(n), n, o.percall_iters,
        [&](uint32_t i) { Must(client.Apply("mine", (i & 1) ? release : hold, 0), "apply"); },
        nullptr);
    if ((o.percall_iters + 20) & 1) Must(client.Apply("mine", release, 0), "apply");
  }
  run(
      "commit", 1, o.percall_iters,
      [&](uint32_t) { Must(client.Commit("mine", stamp++), "commit"); }, nullptr);
  uint64_t fresh = uint64_t{1} << 40;
  for (uint32_t n : {1u, 2u, 40u, 128u, 545u}) {
    if (n > o.slots / 2) continue;
    AllocateResult got;
    run(
        "allocate/" + std::to_string(n), n, o.percall_iters / (n > 100 ? 4 : 1),
        [&](uint32_t) {
          got = client.Allocate("mine", std::max(n, 32u), n);
          Must(got.status, "allocate");
          if (got.slots.size() != n || got.evicted_keys.size() != n)
            Die("allocate did not evict n");
        },
        [&](uint32_t) {
          // The forward fills them; the request later leaves.
          std::vector<Op> ops;
          for (uint32_t slot : got.slots)
            ops.push_back(Op{OpKind::kPublish, slot, key(1, fresh++)});
          for (uint32_t slot : got.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
          Must(client.Apply("mine", ops, 0), "refill");
        });
  }
  for (uint32_t n : {545u, 2393u}) {
    if (n > o.slots) continue;
    std::vector<std::string> keys;
    for (uint32_t i = 0; i < n; ++i) keys.push_back(key(2, i));
    FetchOptions probe;
    probe.probe_only = true;
    run(
        "probe/" + std::to_string(n), n, o.percall_iters / 4,
        [&](uint32_t) {
          FetchResult found = client.Fetch("mine", keys, probe);
          Must(found.status, "probe");
          if (found.available != n) Die("probe found " + std::to_string(found.available));
        },
        nullptr);
  }
  WriteResult(o, &recorder, {}, {});
  return 0;
}

// -------------------------------- manager ----------------------------------

// The server's own work, with no transport: PoolManager in this process, at
// the engine's pool size, every pool full. One thread first (cost per op),
// then one thread a pool at once (what the manager's lock costs).
int RunManager(const Options& o) {
  Pin(o.cpu);
  PoolManager manager(MakeHostMemoryApi());
  Layout layout;
  layout.num_slots = o.slots;
  layout.regions = {Region{0, kSlotBytes}};
  auto key = [](uint64_t space, uint64_t i) {
    const uint64_t h = Mix((space << 56) ^ i);
    return std::string(reinterpret_cast<const char*>(&h), sizeof(h));
  };
  auto name = [](int pool) { return "p" + std::to_string(pool); };
  for (int pool = 0; pool < o.ranks; ++pool) {
    Must(manager.Attach(name(pool), "g", pool, uint64_t{o.slots} * kSlotBytes).status, "attach");
    Must(manager.BindLayout(name(pool), layout), "bind");
    Must(manager.Open(name(pool)).status, "open");
    // Conversations: runs of 512 slots. Two in five are also held by the next
    // pool (a prefix one rank copied from another), the rest by this one only.
    std::vector<Op> ops;
    for (uint32_t slot = 0; slot < o.slots; ++slot) {
      const uint32_t run = slot / 512;
      const bool from_previous = run % 5 == 1;
      const uint64_t space =
          static_cast<uint64_t>(from_previous ? (pool + o.ranks - 1) % o.ranks : pool) + 1;
      const uint64_t id = from_previous ? uint64_t{slot} - 512 : slot;
      ops.push_back(Op{OpKind::kHold, slot, ""});
      ops.push_back(Op{OpKind::kPublish, slot, key(space, id)});
      ops.push_back(Op{OpKind::kRelease, slot, ""});
      if (ops.size() >= 6000 || slot + 1 == o.slots) {
        Must(manager.Apply(name(pool), ops, 0), "populate");
        ops.clear();
      }
    }
  }

  for (const PoolStats& stats : manager.Stats()) {
    if (stats.cached != o.slots) {
      Die("pool " + stats.pool_id + " has " + std::to_string(stats.cached) + " cached and " +
          std::to_string(stats.vacant) + " vacant slots after it was filled");
    }
  }

  struct Result {
    std::string what;
    uint64_t items = 0;
    uint64_t nanos = 0;
  };
  std::vector<uint64_t> next_fresh;
  for (int pool = 0; pool < o.ranks; ++pool) {
    next_fresh.push_back((uint64_t{1} << 40) + (static_cast<uint64_t>(pool) << 36));
  }
  // One pool's share of the scheduler's work, `rounds` times over.
  auto work = [&](int pool, uint32_t rounds, std::vector<Result>* results) {
    const std::string id = name(pool);
    std::map<std::string, Result> acc;
    auto timed = [&](const char* what, uint64_t items, const std::function<void()>& call) {
      const uint64_t t0 = NowNs();
      call();
      Result& r = acc[what];
      r.what = what;
      r.items += items;
      r.nanos += NowNs() - t0;
    };
    // New content, never a key this pool has published before.
    uint64_t& fresh = next_fresh[static_cast<size_t>(pool)];
    std::mt19937_64 rng(o.seed + static_cast<uint64_t>(pool));
    for (uint32_t round = 0; round < rounds; ++round) {
      // A request arrives holding a prefix this pool has, and later leaves.
      const uint32_t n = 545;
      const uint32_t first = static_cast<uint32_t>(rng() % (o.slots - n));
      std::vector<Op> hold, release;
      for (uint32_t slot = first; slot < first + n; ++slot) {
        hold.push_back(Op{OpKind::kHold, slot, ""});
        release.push_back(Op{OpKind::kRelease, slot, ""});
      }
      timed("apply: hold a cached slot", n, [&] { Must(manager.Apply(id, hold, 0), "hold"); });
      timed("apply: release a slot", n, [&] { Must(manager.Apply(id, release, 0), "release"); });
      // It needs blocks for new content: one at a time (decode), and a batch.
      for (uint32_t want : {1u, 1u, 1u, 1u, 128u}) {
        AllocateResult got;
        timed(want == 1 ? "allocate 1: evict a slot" : "allocate 128: evict a slot", want,
              [&] { got = manager.Allocate(id, std::max(want, 32u), want); });
        Must(got.status, "allocate");
        if (got.slots.size() != want) {
          Die("allocate of " + std::to_string(want) + " returned " +
              std::to_string(got.slots.size()));
        }
        std::vector<Op> publish, done;
        for (uint32_t slot : got.slots) {
          publish.push_back(Op{OpKind::kPublish, slot, key(9, fresh++)});
          done.push_back(Op{OpKind::kRelease, slot, ""});
        }
        timed("apply: publish a slot", want,
              [&] { Must(manager.Apply(id, publish, 0), "publish"); });
        Must(manager.Apply(id, done, 0), "release");
      }
      // And asks what another pool holds of a prefix.
      std::vector<std::string> keys;
      const int peer = (pool + 1) % o.ranks;
      const uint64_t base = (rng() % (o.slots / 512 - 2)) * 512;
      for (uint32_t i = 0; i < n; ++i)
        keys.push_back(key(static_cast<uint64_t>(peer) + 1, base + i));
      FetchOptions probe;
      probe.probe_only = true;
      timed("probe: a key", n, [&] { Must(manager.Fetch(id, keys, probe).status, "probe"); });
    }
    for (auto& entry : acc) results->push_back(entry.second);
  };

  const uint32_t rounds = std::max<uint32_t>(50, o.percall_iters / 4);
  std::vector<Result> alone;
  work(0, 20, &alone);  // warm
  alone.clear();
  const uint64_t t0 = NowNs();
  work(0, rounds, &alone);
  const double alone_s = static_cast<double>(NowNs() - t0) / 1e9;
  std::printf(
      "manager alone: %d pools x %u slots, full; one thread on one pool, %u rounds in %.3f s\n",
      o.ranks, o.slots, rounds, alone_s);
  std::printf("  %-28s %12s %10s\n", "operation", "items", "ns/item");
  for (const Result& r : alone) {
    std::printf("  %-28s %12llu %10.1f\n", r.what.c_str(), static_cast<unsigned long long>(r.items),
                static_cast<double>(r.nanos) / static_cast<double>(r.items));
  }
  // The same work on every pool at once: with one lock it takes as long as
  // doing the pools one after another.
  std::vector<std::vector<Result>> each(static_cast<size_t>(o.ranks));
  std::vector<std::thread> threads;
  const uint64_t t1 = NowNs();
  for (int pool = 0; pool < o.ranks; ++pool) {
    threads.emplace_back([&, pool] { work(pool, rounds, &each[static_cast<size_t>(pool)]); });
  }
  for (std::thread& t : threads) t.join();
  const double together_s = static_cast<double>(NowNs() - t1) / 1e9;
  std::printf(
      "%d threads, one a pool, the same rounds each: %.3f s = %.2fx one thread's time "
      "(1.00x: no contention; %d.00x: fully serialised)\n",
      o.ranks, together_s, together_s / alone_s, o.ranks);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = Parse(argc, argv);
  if (o.mode == "serve") return Serve(o);
  if (o.mode == "rank") return RunRank(o);
  if (o.mode == "percall") return RunPerCall(o);
  if (o.mode == "manager") return RunManager(o);
  Die("unknown mode " + o.mode);
}
