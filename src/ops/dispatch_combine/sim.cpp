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
#include "mori/ops/dispatch_combine/sim.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "mori/utils/env_utils.hpp"
#include "mori/utils/flat_yaml.hpp"

namespace mori {
namespace moe {
namespace sim {

namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kTimeout = std::chrono::seconds(300);

double LatencyUs(const char* name) {
  const char* raw = std::getenv(name);
  char* end = nullptr;
  const double us = raw ? std::strtod(raw, &end) : -1;
  if (raw == nullptr || end == raw || *end != '\0' || !(us >= 0)) {
    throw std::runtime_error(std::string("MORI_EP_SIM_ENABLE=1 needs ") + name +
                             " set to a latency in microseconds");
  }
  return us;
}

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}

uint64_t SplitMix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// MORI_EP_SIM_ROUTING: one entry per MoE layer, in the order the layers dispatch.
//   - alpha: 1.2        expert popularity ~ 1 / rank^alpha; 0 (or no file) = uniform
//     shuffle_seed: 7   optional: which experts are the hot ones; omitted = expert 0 is hottest
std::vector<std::vector<double>> LoadRoutingWeights(int numExperts) {
  const char* path = std::getenv("MORI_EP_SIM_ROUTING");
  if (path == nullptr || *path == '\0') return {std::vector<double>(numExperts, 1.0)};
  const auto entries = yaml::LoadFlatList(path);
  if (entries.empty()) throw std::runtime_error(std::string(path) + ": no layers");
  std::vector<std::vector<double>> layers;
  for (size_t i = 0; i < entries.size(); ++i) {
    const std::string where = std::string(path) + ": layer " + std::to_string(i);
    yaml::CheckKeys(entries[i], {"alpha", "shuffle_seed"}, where);
    const double alpha = yaml::GetDouble(entries[i], "alpha", 0, where);
    if (!(alpha >= 0)) throw std::runtime_error(where + ": alpha must be >= 0");
    std::vector<int> order(numExperts);
    std::iota(order.begin(), order.end(), 0);
    if (entries[i].count("shuffle_seed") != 0) {
      std::mt19937_64 rng(yaml::GetInt(entries[i], "shuffle_seed", 0, where));
      std::shuffle(order.begin(), order.end(), rng);
    }
    std::vector<double> w(numExperts);
    for (int r = 0; r < numExperts; ++r) w[order[r]] = std::pow(r + 1.0, -alpha);
    layers.push_back(std::move(w));
  }
  return layers;
}

// One per rank, its own cache line.
struct alignas(64) RankState {
  std::atomic<uint64_t> published;
};

void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

// The default 50 us timer slack would dominate a 30 us op.
void FineTimerSlack() {
  thread_local const bool done = (prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL), true);
  (void)done;
}

}  // namespace

// Rank r's entry for op seq lives in slot seq % 2: a rank reaches op seq + 2 only after every rank
// published seq + 1, which each does only after it finished reading seq.
struct EpSim::Slot {
  int64_t deadlineNs;
  int32_t phase;
  int32_t numTokens;
  // Followed by, as the sender's dispatch buckets them: ids [maxTokens, topk]; toCount [world],
  // how many of its tokens go to each rank; hits [world * expertsPerRank], per expert; and
  // toList [world, maxTokens], which tokens go to each rank.
  int32_t* Ints() { return reinterpret_cast<int32_t*>(this + 1); }
  const int32_t* Ints() const { return reinterpret_cast<const int32_t*>(this + 1); }
};

// An expert distribution with Vose's alias table, for O(1) draws.
struct EpSim::Layer {
  std::vector<double> weight, prob;
  std::vector<int> alias;

  explicit Layer(std::vector<double> w)
      : weight(std::move(w)), prob(weight.size(), 1.0), alias(weight.size()) {
    const int n = static_cast<int>(weight.size());
    const double sum = std::accumulate(weight.begin(), weight.end(), 0.0);
    std::vector<double> p(n);
    std::vector<int> small, large;
    for (int i = 0; i < n; ++i) {
      p[i] = weight[i] * n / sum;
      (p[i] < 1 ? small : large).push_back(i);
      alias[i] = i;
    }
    while (!small.empty() && !large.empty()) {
      const int s = small.back(), l = large.back();
      small.pop_back();
      prob[s] = p[s];
      alias[s] = l;
      p[l] += p[s] - 1;
      if (p[l] < 1) {
        large.pop_back();
        small.push_back(l);
      }
    }
  }
};

bool IsEnabled() { return env::IsEnvVarEnabled("MORI_EP_SIM_ENABLE"); }

EpSim::EpSim(const std::string& group, int rank, int worldSize, int maxTokens, int topk,
             int expertsPerRank, int maxRecv, bool asyncSend)
    : rank_(rank),
      world_(worldSize),
      maxTokens_(maxTokens),
      topk_(topk),
      expertsPerRank_(expertsPerRank),
      maxRecv_(maxRecv),
      asyncSend_(asyncSend),
      dispatchUs_(LatencyUs("MORI_EP_SIM_DISPATCH_US")),
      combineUs_(LatencyUs("MORI_EP_SIM_COMBINE_US")) {
  if (topk > worldSize * expertsPerRank) throw std::invalid_argument("EP sim: topk > experts");
  for (auto& w : LoadRoutingWeights(worldSize * expertsPerRank)) layers_.emplace_back(std::move(w));

  const size_t ints = static_cast<size_t>(maxTokens) * topk + worldSize +
                      static_cast<size_t>(worldSize) * expertsPerRank +
                      static_cast<size_t>(worldSize) * maxTokens;
  scratch_.resize(ints);
  slotBytes_ = (sizeof(Slot) + sizeof(int32_t) * ints + 63) / 64 * 64;
  bytes_ = sizeof(RankState) * world_ + 2 * world_ * slotBytes_;
  if (world_ == 1) {
    base_ = static_cast<char*>(std::calloc(1, bytes_));
    return;
  }
  if (group.empty()) throw std::invalid_argument("EP sim with world_size > 1 needs a group name");
  name_ = "/mori_ep_sim." + group;
  const int fd = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) throw std::system_error(errno, std::generic_category(), "shm_open " + name_);
  struct stat st{};
  // Ranks race to size it; any of them may win, and all ask for the same size.
  const bool ok =
      fstat(fd, &st) == 0 && (static_cast<size_t>(st.st_size) == bytes_ ||
                              (st.st_size == 0 && ftruncate(fd, static_cast<off_t>(bytes_)) == 0));
  void* p = ok ? mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
  close(fd);
  if (p == MAP_FAILED) {
    throw std::runtime_error("EP sim: cannot map " + name_ +
                             " (ranks disagree on the EP config, or a stale segment)");
  }
  base_ = static_cast<char*>(p);
}

EpSim::~EpSim() {
  if (world_ == 1) {
    std::free(base_);
  } else if (base_ != nullptr) {
    munmap(base_, bytes_);
  }
}

EpSim::Slot* EpSim::SlotOf(uint64_t seq, int rank) const {
  return reinterpret_cast<Slot*>(base_ + sizeof(RankState) * world_ +
                                 ((seq % 2) * world_ + rank) * slotBytes_);
}

void EpSim::Arrive() { arrivalNs_ = NowNs(); }

EpSim::Slot* EpSim::Begin(int phase, double latencyUs, int numTokens) {
  if (numTokens > maxTokens_) {
    throw std::invalid_argument("EP sim: " + std::to_string(numTokens) +
                                " tokens exceed max_num_inp_token_per_rank " +
                                std::to_string(maxTokens_));
  }
  Slot* mine = SlotOf(++seq_, rank_);
  mine->deadlineNs =
      (arrivalNs_ != 0 ? arrivalNs_ : NowNs()) + static_cast<int64_t>(latencyUs * 1e3);
  arrivalNs_ = 0;
  mine->phase = phase;
  mine->numTokens = numTokens;
  return mine;
}

void EpSim::Publish(const Slot* mine) {
  auto* state = reinterpret_cast<RankState*>(base_);
  state[rank_].published.store(seq_, std::memory_order_release);
  FineTimerSlack();
  const auto start = Clock::now();
  deadlineNs_ = 0;
  for (int r = 0; r < world_; ++r) {
    while (state[r].published.load(std::memory_order_acquire) < seq_) {
      // Spin through a decode-sized wait; a sleep would wake up late for it.
      const auto waited = Clock::now() - start;
      if (waited > std::chrono::microseconds(50)) {
        std::this_thread::sleep_for(std::chrono::microseconds(2));
      } else {
        CpuRelax();
      }
      if (waited > kTimeout) {
        throw std::runtime_error("EP sim rank " + std::to_string(rank_) + ": rank " +
                                 std::to_string(r) + " never reached op " + std::to_string(seq_));
      }
    }
    const Slot* other = SlotOf(seq_, r);
    if (other->phase != mine->phase) {
      throw std::runtime_error("EP sim op " + std::to_string(seq_) + ": ranks " +
                               std::to_string(rank_) + " and " + std::to_string(r) +
                               " disagree on dispatch vs combine");
    }
    deadlineNs_ = std::max(deadlineNs_, other->deadlineNs);
  }
  // Every rank has the segment mapped by now; drop its name so nothing outlives the job.
  if (seq_ == 1 && rank_ == 0 && !name_.empty()) shm_unlink(name_.c_str());
}

void EpSim::Route(int numTokens) {
  const Layer& layer = layers_[dispatches_++ % layers_.size()];
  const int n = static_cast<int>(layer.weight.size());
  int32_t* ids = scratch_.data();
  int32_t* toCount = ids + static_cast<size_t>(maxTokens_) * topk_;
  int32_t* hits = toCount + world_;
  int32_t* toList = hits + n;
  std::fill(toCount, toCount + world_ + n, 0);
  std::vector<int> lastToken(world_, -1);
  uint64_t rng = (static_cast<uint64_t>(rank_) << 40) ^ seq_;  // reproducible, per rank and op
  for (int t = 0; t < numTokens; ++t) {
    int32_t* row = ids + static_cast<size_t>(t) * topk_;
    for (int j = 0; j < topk_; ++j) {
      auto taken = [&](int e) { return std::find(row, row + j, e) != row + j; };
      int e = -1;
      for (int tries = 0; tries < 64 && e < 0; ++tries) {
        const uint64_t r = SplitMix64(rng);
        const int i = static_cast<int>(((r >> 32) * n) >> 32);
        const int pick = (r & 0xffffffffu) * 0x1p-32 < layer.prob[i] ? i : layer.alias[i];
        if (!taken(pick)) e = pick;
      }
      if (e < 0) {
        // Heavy skew keeps redrawing the experts already picked: draw from the rest directly.
        double rest = 0;
        for (int c = 0; c < n; ++c) rest += taken(c) ? 0 : layer.weight[c];
        double u = (SplitMix64(rng) >> 11) * 0x1p-53 * rest;
        for (int c = 0; c < n && (e < 0 || u >= 0); ++c) {
          if (taken(c)) continue;
          e = c;
          u -= layer.weight[c];
        }
      }
      row[j] = e;
      ++hits[e];
      const int to = e / expertsPerRank_;
      if (lastToken[to] != t) {  // once per destination rank
        lastToken[to] = t;
        toList[static_cast<size_t>(to) * maxTokens_ + toCount[to]++] = t;
      }
    }
  }
}

int EpSim::Dispatch(int numTokens, int32_t* recvIds, int32_t* expertCount) {
  const int numExperts = world_ * expertsPerRank_;
  const size_t idsLen = static_cast<size_t>(maxTokens_) * topk_;
  Slot* mine = Begin(0, dispatchUs_, numTokens);

  // Send side: draw the routing and bucket the tokens by destination, so a receiver touches only
  // its own. Built in private memory first: writing the slot piecemeal would pull each of its
  // cache lines back from the ranks that read it last time, several times slower.
  Route(numTokens);
  const int32_t* toCount = scratch_.data() + idsLen;
  int32_t* slot = mine->Ints();
  std::memcpy(slot, scratch_.data(), sizeof(int32_t) * numTokens * topk_);
  std::memcpy(slot + idsLen, toCount, sizeof(int32_t) * (world_ + numExperts));
  for (int r = 0; r < world_; ++r) {
    const size_t at = idsLen + world_ + numExperts + static_cast<size_t>(r) * maxTokens_;
    std::memcpy(slot + at, scratch_.data() + at, sizeof(int32_t) * toCount[r]);
  }
  Publish(mine);

  // Receive side.
  const int lo = rank_ * expertsPerRank_;
  std::fill(expertCount, expertCount + expertsPerRank_, 0);
  int recv = 0;
  for (int r = 0; r < world_; ++r) {
    const int32_t* src = SlotOf(seq_, r)->Ints();
    const int32_t* srcToCount = src + idsLen;
    const int32_t* srcHits = srcToCount + world_;
    const int32_t* srcList = srcHits + numExperts + static_cast<size_t>(rank_) * maxTokens_;
    const int n = srcToCount[rank_];
    if (recv + n > maxRecv_) {
      throw std::runtime_error("EP sim rank " + std::to_string(rank_) +
                               " receives more tokens than max_num_tokens_to_recv " +
                               std::to_string(maxRecv_));
    }
    for (int e = 0; e < expertsPerRank_; ++e) expertCount[e] += srcHits[lo + e];
    for (int i = 0; i < n; ++i) {
      std::memcpy(recvIds + static_cast<size_t>(recv++) * topk_, src + srcList[i] * topk_,
                  sizeof(int32_t) * topk_);
    }
  }
  lastTokens_ = numTokens;
  return recv;
}

int EpSim::Combine() {
  Publish(Begin(1, combineUs_, 0));
  return lastTokens_;
}

void EpSim::Wait() const {
  // Spin the last 20 us: the scheduler's wake-up delay would dominate a short op.
  FineTimerSlack();
  constexpr auto kSpinTail = std::chrono::microseconds(20);
  const Clock::time_point end{std::chrono::nanoseconds(deadlineNs_)};
  if (end - Clock::now() > kSpinTail) std::this_thread::sleep_until(end - kSpinTail);
  while (Clock::now() < end) {
  }
}

}  // namespace sim
}  // namespace moe
}  // namespace mori
