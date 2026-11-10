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
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>

#include "mori/utils/env_utils.hpp"
#include "mori/utils/flat_yaml.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace moe {
namespace sim {

namespace {

using Clock = std::chrono::steady_clock;

std::optional<double> ParseNonNegativeDouble(const char* raw) {
  errno = 0;
  char* end = nullptr;
  double v = std::strtod(raw, &end);
  if (errno != 0 || end == raw || *end != '\0' || !std::isfinite(v) || v < 0) return std::nullopt;
  return v;
}

std::optional<double> ParsePositiveDouble(const char* raw) {
  auto v = ParseNonNegativeDouble(raw);
  if (!v.has_value() || *v == 0) return std::nullopt;
  return v;
}

template <typename Target, typename V>
void Override(const char* name, Target& target, std::optional<V> (*parse)(const char*)) {
  const char* raw = mori::env::Get(name);
  if (raw == nullptr || raw[0] == '\0') return;
  if (auto value = parse(raw); value.has_value()) {
    target = static_cast<Target>(*value);
  } else {
    MORI_OPS_WARN("Ignore invalid env {}={}", name, raw);
  }
}

const char* KernelName(KernelType t) {
  switch (t) {
    case KernelType::IntraNode:
      return "IntraNode";
    case KernelType::InterNode:
      return "InterNode";
    case KernelType::InterNodeV1:
      return "InterNodeV1";
    case KernelType::InterNodeV1LL:
      return "InterNodeV1LL";
    case KernelType::AsyncLL:
      return "AsyncLL";
    case KernelType::IntraNodeLL:
      return "IntraNodeLL";
  }
  return "Unknown";
}

const char* QuantName(QuantType q) {
  switch (q) {
    case QuantType::None:
      return "none";
    case QuantType::Fp8DirectCast:
      return "fp8_direct_cast";
    case QuantType::Fp8BlockwiseQuant:
      return "fp8_blockwise";
    case QuantType::Fp4BlockwiseQuant:
      return "fp4_blockwise";
  }
  return "unknown";
}

const char* PhaseName(Phase p) { return p == Phase::Dispatch ? "dispatch" : "combine"; }

// Catalog dtype names, as in the tuning configs.
std::string DtypeName(hipDataType t) {
  switch (t) {
    case HIP_R_32F:
      return "f32";
    case HIP_R_16BF:
      return "bf16";
    case HIP_R_16F:
      return "f16";
    case HIP_R_8F_E4M3:
      return "fp8_e4m3";
    case HIP_R_8F_E4M3_FNUZ:
      return "fp8_e4m3_fnuz";
#if __has_include(<hip/hip_ext_ocp.h>)
    case HIP_R_4F_E2M1:
      return "fp4";
#endif
    default:
      return "dtype" + std::to_string(static_cast<int>(t));
  }
}

// Bytes per element of the tensor the caller passes; fp4 tensors hold two values per element.
double ElemBytes(const std::string& dtype) {
  if (dtype == "f32") return 4;
  if (dtype == "bf16" || dtype == "f16") return 2;
  return 1;
}

int CombineScaleDim() { return env::GetPositiveIntOr("MORI_FP8_COMBINE_SCALE_DIM", 56); }

// Bytes one (token, destination rank) pair puts on the wire.
double PairBytes(Phase phase, const std::string& dtype, int hidden, int topk,
                 const std::string& quant) {
  if (phase == Phase::Dispatch) return hidden * ElemBytes(dtype) + topk * 8.0 + 4.0;
  if (quant == "fp8_direct_cast") return hidden;
  if (quant == "fp8_blockwise") return hidden + CombineScaleDim() * 4.0;
  if (quant == "fp4_blockwise") return (hidden + 1) / 2 + CombineScaleDim() * 4.0;
  return hidden * ElemBytes(dtype);
}

// Probability that one rank holds at least one of a token's `topk` distinct experts, drawn
// uniformly from `numExperts` (the routing the catalog was measured with).
double RankHitProbability(int numExperts, int expertsPerRank, int topk) {
  if (topk <= 0 || numExperts <= 0) return 0;
  double miss = 1.0;
  for (int i = 0; i < topk; ++i) {
    double num = static_cast<double>(numExperts - expertsPerRank - i);
    if (num <= 0) return 1.0;
    miss *= num / (numExperts - i);
  }
  return 1.0 - miss;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                             Catalog                                            */
/* ---------------------------------------------------------------------------------------------- */
struct Curve {
  std::string gpu, kernel, phase, dtype, quant;
  int epSize{0}, hiddenDim{0}, topk{0}, numExperts{0}, nodeSize{0};
  bool zeroCopy{false};
  std::vector<double> tokens, latencyUs;
  std::string source;
};

const std::set<std::string> kCatalogKeys = {"gpu", "kernel", "ep_size", "gpu_per_node", "phase",
                                            "dtype", "quant", "zero_copy", "hidden_dim", "topk",
                                            "num_experts", "tokens", "latency_us",
                                            // Free text.
                                            "hardware", "measured", "notes"};

std::vector<double> ParseList(const std::string& raw, const std::string& where, const char* field) {
  std::vector<double> out;
  std::stringstream ss(raw);
  for (std::string item; std::getline(ss, item, ',');) {
    item.erase(0, item.find_first_not_of(' '));
    item.erase(item.find_last_not_of(' ') + 1);
    auto v = ParsePositiveDouble(item.c_str());
    if (!v)
      throw std::runtime_error(where + ": " + field + " has a non-positive value '" + item + "'");
    out.push_back(*v);
  }
  return out;
}

using CurveKey = std::tuple<std::string, std::string, int, std::string, std::string, int, int,
                            std::string, bool>;

CurveKey KeyOf(const Curve& c) {
  return {c.gpu, c.kernel, c.epSize, c.phase, c.dtype, c.hiddenDim, c.topk, c.quant, c.zeroCopy};
}

void AddToCatalog(std::map<CurveKey, Curve>& catalog, const std::vector<yaml::FlatMap>& entries,
                  const std::string& source) {
  std::set<CurveKey> seen;
  for (size_t i = 0; i < entries.size(); ++i) {
    const yaml::FlatMap& e = entries[i];
    const std::string where = source + ": entry " + std::to_string(i + 1);
    yaml::CheckKeys(e, kCatalogKeys, where);
    for (const char* f : {"gpu", "kernel", "ep_size", "phase", "dtype", "hidden_dim", "topk",
                          "num_experts", "tokens", "latency_us"}) {
      if (e.count(f) == 0) throw std::runtime_error(where + ": missing " + f);
    }
    Curve c;
    c.gpu = yaml::GetString(e, "gpu");
    c.kernel = yaml::GetString(e, "kernel");
    c.phase = yaml::GetString(e, "phase");
    c.dtype = yaml::GetString(e, "dtype");
    c.quant = yaml::GetString(e, "quant", "none");
    c.zeroCopy = yaml::GetBool(e, "zero_copy", false, where);
    c.epSize = static_cast<int>(yaml::GetInt(e, "ep_size", 0, where));
    c.hiddenDim = static_cast<int>(yaml::GetInt(e, "hidden_dim", 0, where));
    c.topk = static_cast<int>(yaml::GetInt(e, "topk", 0, where));
    c.numExperts = static_cast<int>(yaml::GetInt(e, "num_experts", 0, where));
    c.nodeSize = std::min(c.epSize, static_cast<int>(yaml::GetInt(e, "gpu_per_node", 8, where)));
    c.tokens = ParseList(yaml::GetString(e, "tokens"), where, "tokens");
    c.latencyUs = ParseList(yaml::GetString(e, "latency_us"), where, "latency_us");
    c.source = where;
    if (c.phase != "dispatch" && c.phase != "combine") {
      throw std::runtime_error(where + ": phase must be dispatch or combine");
    }
    if (c.epSize <= 0 || c.hiddenDim <= 0 || c.topk <= 0 || c.numExperts < c.epSize ||
        c.numExperts % c.epSize != 0 || c.nodeSize <= 0) {
      throw std::runtime_error(where +
                               ": ep_size, hidden_dim, topk, gpu_per_node must be "
                               "positive and num_experts a multiple of ep_size");
    }
    if (c.tokens.empty() || c.tokens.size() != c.latencyUs.size()) {
      throw std::runtime_error(where + ": tokens and latency_us need the same number of values");
    }
    for (size_t j = 1; j < c.tokens.size(); ++j) {
      if (c.tokens[j] <= c.tokens[j - 1]) {
        throw std::runtime_error(where + ": tokens must increase");
      }
    }
    if (!seen.insert(KeyOf(c)).second) {
      throw std::runtime_error(where + ": duplicate curve in the same file");
    }
    catalog[KeyOf(c)] = std::move(c);
  }
}

double Interp(const Curve& c, double n) {
  const auto& t = c.tokens;
  const auto& l = c.latencyUs;
  if (t.size() == 1) return n <= t[0] ? l[0] : l[0] * n / t[0];
  if (n <= t[0]) {
    // Down to the fixed cost: extend the first segment, never below half the first point.
    double slope = (l[1] - l[0]) / (t[1] - t[0]);
    if (slope <= 0) return l[0];
    return std::max(l[0] - slope * (t[0] - n), 0.5 * l[0]);
  }
  size_t i = std::upper_bound(t.begin(), t.end(), n) - t.begin();
  if (i >= t.size()) {
    size_t a = t.size() - 2, b = t.size() - 1;
    double slope = std::max(0.0, (l[b] - l[a]) / (t[b] - t[a]));
    return l[b] + slope * (n - t[b]);
  }
  double f = (n - t[i - 1]) / (t[i] - t[i - 1]);
  return l[i - 1] + f * (l[i] - l[i - 1]);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Shared-memory rendezvous                                 */
/* ---------------------------------------------------------------------------------------------- */
constexpr uint64_t kShmMagic = 0x4d4f524945505331ull;  // "MORIEPS1"
constexpr uint32_t kRingDepth = 4;

struct SegmentHeader {
  uint64_t magic;
  uint32_t world;
  uint32_t numExperts;
  uint32_t depth;
  uint32_t entryBytes;
};

struct alignas(64) RankHeader {
  std::atomic<uint64_t> joined;
  std::atomic<uint64_t> published;
  std::atomic<uint64_t> consumed;
};
static_assert(std::atomic<uint64_t>::is_always_lock_free, "shared atomics must be lock-free");

struct EntryHeader {
  int64_t arrivalNs;
  int32_t phase;
  int32_t numTokens;
};

size_t RoundUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

std::string ShmName(const std::string& group) {
  std::string name = "/mori_ep_sim.";
  for (char ch : group.substr(0, 200)) {
    name += (std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-') ? ch : '_';
  }
  return name;
}

void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

void EnsureFineTimerSlack() {
  // The default 50 us slack would dominate a 30 us op.
  thread_local bool done = (prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL), true);
  (void)done;
}

template <typename Ready, typename Describe>
void WaitFor(Ready ready, std::chrono::seconds timeout, Describe describe) {
  EnsureFineTimerSlack();
  const auto start = Clock::now();
  for (int spins = 0; !ready(); ++spins) {
    if (spins < 64) {
      CpuRelax();
      continue;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(spins < 4096 ? 2 : 50));
    if (Clock::now() - start > timeout) throw std::runtime_error(describe());
  }
}

void SleepUntil(Clock::time_point end, std::chrono::nanoseconds spinTail) {
  EnsureFineTimerSlack();
  if (end - Clock::now() > spinTail) std::this_thread::sleep_until(end - spinTail);
  while (Clock::now() < end) CpuRelax();
}

int64_t ToNs(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

std::atomic<int> gEnvGroupHandles{0};

}  // namespace

bool IsEnabled() { return mori::env::IsEnvVarEnabled("MORI_EP_SIM_ENABLE"); }

/* ---------------------------------------------------------------------------------------------- */
/*                                              Model                                             */
/* ---------------------------------------------------------------------------------------------- */
struct EpSim::Model {
  std::string gpu;
  std::string catalogFiles;
  std::vector<Curve> curves;  // this GPU, kernel and EP size
  bool internode{false};
  double latencyUs{25}, xgmiGbps{350}, rdmaGbps{45};

  struct Pick {
    const Curve* curve{nullptr};
    double entryFanIntra{0}, entryFanInter{0};
    double byteRatio{1};
    double pairBytes{0};
    std::string source;
  };
  mutable std::map<std::tuple<int, std::string, int, bool>, Pick> picks;
  mutable std::set<std::string> warned;

  const Pick& Choose(const OpDesc& op, const EpDispatchCombineConfig& cfg) const {
    const std::string dtype = DtypeName(op.dtype);
    const std::string quant = op.phase == Phase::Combine ? QuantName(cfg.quantType) : "none";
    const bool zeroCopy = op.phase == Phase::Combine && op.zeroCopy;
    auto key = std::make_tuple(static_cast<int>(op.phase), dtype, op.hiddenDim, zeroCopy);
    if (auto it = picks.find(key); it != picks.end()) return it->second;

    Pick p;
    p.pairBytes = PairBytes(op.phase, dtype, op.hiddenDim, cfg.numExpertPerToken, quant);
    int best = -1, bestTopkGap = 0;
    for (const Curve& c : curves) {
      if (c.phase != PhaseName(op.phase)) continue;
      int score = (c.quant == quant) * 16 + (c.zeroCopy == zeroCopy) * 8 + (c.dtype == dtype) * 4 +
                  (c.hiddenDim == op.hiddenDim) * 2 + (c.topk == cfg.numExpertPerToken);
      int topkGap = std::abs(c.topk - cfg.numExpertPerToken);
      if (score > best || (score == best && topkGap < bestTopkGap)) {
        best = score;
        bestTopkGap = topkGap;
        p.curve = &c;
      }
    }
    if (p.curve != nullptr) {
      const Curve& c = *p.curve;
      double hit = RankHitProbability(c.numExperts, c.numExperts / c.epSize, c.topk);
      p.entryFanIntra = c.nodeSize * hit;
      p.entryFanInter = (c.epSize - c.nodeSize) * hit;
      p.byteRatio = p.pairBytes / PairBytes(op.phase, c.dtype, c.hiddenDim, c.topk, c.quant);
      p.source = c.source;
      if (best != 31) {
        std::ostringstream ss;
        ss << c.source << " scaled by bytes x" << p.byteRatio << " (catalog " << c.dtype << "/"
           << c.hiddenDim << "/topk" << c.topk << "/" << c.quant << (c.zeroCopy ? "/zc" : "")
           << ", call " << dtype << "/" << op.hiddenDim << "/topk" << cfg.numExpertPerToken << "/"
           << quant << (zeroCopy ? "/zc" : "") << ")";
        p.source = ss.str();
      }
    } else {
      std::ostringstream ss;
      ss << "analytic (latency " << latencyUs << " us, xgmi " << xgmiGbps << " GB/s, rdma "
         << rdmaGbps << " GB/s): ";
      if (gpu.empty()) {
        ss << "MORI_EP_SIM_GPU is not set";
      } else {
        ss << "catalog [" << catalogFiles << "] has no " << PhaseName(op.phase) << " curve for "
           << gpu << " " << KernelName(cfg.kernelType) << " ep" << cfg.worldSize;
      }
      p.source = ss.str();
    }
    if (p.curve == nullptr || best != 31) {
      if (warned.insert(p.source).second && cfg.rank == 0) {
        MORI_OPS_WARN("EP sim {}: {}", PhaseName(op.phase), p.source);
      }
    }
    return picks.emplace(key, std::move(p)).first->second;
  }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                             Shared                                             */
/* ---------------------------------------------------------------------------------------------- */
struct EpSim::Shared {
  std::string name;
  int world{0}, rank{0}, numExperts{0};
  size_t bytes{0}, entryBytes{0};
  void* base{nullptr};

  ~Shared() {
    if (base != nullptr) munmap(base, bytes);
  }

  SegmentHeader* Header() const { return static_cast<SegmentHeader*>(base); }
  RankHeader* Rank(int r) const {
    return reinterpret_cast<RankHeader*>(static_cast<char*>(base) + 64) + r;
  }
  char* Entry(int r, uint64_t seq) const {
    char* entries = static_cast<char*>(base) + 64 + RoundUp(sizeof(RankHeader) * world, 64);
    return entries + (static_cast<size_t>(r) * kRingDepth + seq % kRingDepth) * entryBytes;
  }
  double* RankLoad(char* entry) const {
    return reinterpret_cast<double*>(entry + sizeof(EntryHeader));
  }
  double* ExpertLoad(char* entry) const { return RankLoad(entry) + world; }

  std::string Missing(const std::function<bool(int)>& arrived) const {
    std::string list;
    for (int r = 0; r < world; ++r) {
      if (!arrived(r)) list += (list.empty() ? "" : ",") + std::to_string(r);
    }
    return list;
  }

  void Join(const std::string& group, int world_, int rank_, int numExperts_,
            std::chrono::seconds timeout) {
    world = world_;
    rank = rank_;
    numExperts = numExperts_;
    name = ShmName(group);
    entryBytes = RoundUp(sizeof(EntryHeader) + sizeof(double) * (world + numExperts), 64);
    bytes = 64 + RoundUp(sizeof(RankHeader) * world, 64) + world * kRingDepth * entryBytes;

    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "shm_open " + name);
    struct stat st{};
    if (fstat(fd, &st) != 0 || (st.st_size != 0 && static_cast<size_t>(st.st_size) != bytes) ||
        (st.st_size == 0 && ftruncate(fd, static_cast<off_t>(bytes)) != 0)) {
      int err = errno;
      close(fd);
      throw std::runtime_error("EP sim segment " + name + " exists with another size (" +
                               std::to_string(st.st_size) + " vs " + std::to_string(bytes) +
                               "), a different EP config or a stale run: " + std::strerror(err));
    }
    base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
      base = nullptr;
      throw std::system_error(errno, std::generic_category(), "mmap " + name);
    }
    // Every rank writes the same header; a mismatch after the join means the configs differ.
    SegmentHeader want{kShmMagic, static_cast<uint32_t>(world), static_cast<uint32_t>(numExperts),
                       kRingDepth, static_cast<uint32_t>(entryBytes)};
    std::memcpy(Header(), &want, sizeof(want));
    Rank(rank)->joined.store(1, std::memory_order_release);
    WaitFor(
        [&] {
          for (int r = 0; r < world; ++r) {
            if (Rank(r)->joined.load(std::memory_order_acquire) == 0) return false;
          }
          return true;
        },
        timeout,
        [&] {
          return "EP sim rank " + std::to_string(rank) + ": ranks [" +
                 Missing([&](int r) { return Rank(r)->joined.load() != 0; }) + "] of " +
                 std::to_string(world) + " never joined " + name;
        });
    if (std::memcmp(Header(), &want, sizeof(want)) != 0) {
      throw std::runtime_error("EP sim ranks joined " + name + " with different EP configs");
    }
    // Everyone has it mapped; drop the name so nothing is left behind, even after a crash.
    if (rank == 0) shm_unlink(name.c_str());
  }

  void Publish(uint64_t seq, int64_t arrivalNs, Phase phase, int numTokens,
               const std::vector<double>& rankLoad, const std::vector<double>& expertLoad,
               std::chrono::seconds timeout) {
    if (seq > kRingDepth) {
      WaitFor(
          [&] {
            for (int r = 0; r < world; ++r) {
              if (Rank(r)->consumed.load(std::memory_order_acquire) < seq - kRingDepth) {
                return false;
              }
            }
            return true;
          },
          timeout, [&] { return "EP sim: peers stopped consuming ops in " + name; });
    }
    char* e = Entry(rank, seq);
    EntryHeader h{arrivalNs, static_cast<int32_t>(phase), numTokens};
    std::memcpy(e, &h, sizeof(h));
    std::memcpy(RankLoad(e), rankLoad.data(), sizeof(double) * world);
    std::memcpy(ExpertLoad(e), expertLoad.data(), sizeof(double) * numExperts);
    Rank(rank)->published.store(seq, std::memory_order_release);
  }

  void AwaitAll(uint64_t seq, Phase phase, std::chrono::seconds timeout) {
    WaitFor(
        [&] {
          for (int r = 0; r < world; ++r) {
            if (Rank(r)->published.load(std::memory_order_acquire) < seq) return false;
          }
          return true;
        },
        timeout,
        [&] {
          return "EP sim rank " + std::to_string(rank) + ": op " + std::to_string(seq) + " (" +
                 PhaseName(phase) + ") still waits for ranks [" +
                 Missing([&](int r) { return Rank(r)->published.load() >= seq; }) +
                 "]; a rank skipped or reordered a dispatch/combine";
        });
    for (int r = 0; r < world; ++r) {
      EntryHeader h;
      std::memcpy(&h, Entry(r, seq), sizeof(h));
      if (h.phase != static_cast<int32_t>(phase)) {
        throw std::runtime_error("EP sim op " + std::to_string(seq) + ": rank " +
                                 std::to_string(rank) + " runs " + PhaseName(phase) + " but rank " +
                                 std::to_string(r) + " runs " +
                                 PhaseName(static_cast<Phase>(h.phase)));
      }
    }
  }

  void Consumed(uint64_t seq) { Rank(rank)->consumed.store(seq, std::memory_order_release); }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                              EpSim                                             */
/* ---------------------------------------------------------------------------------------------- */
EpSim::EpSim(const EpDispatchCombineConfig& config)
    : config_(config),
      numExperts_(config.worldSize * config.numExpertPerRank),
      nodeSize_(std::max(1, std::min(config.gpuPerNode, config.worldSize))) {
  if (config_.worldSize <= 0 || config_.numExpertPerRank <= 0 || config_.numExpertPerToken <= 0) {
    throw std::runtime_error(
        "EP sim needs world_size, num_experts_per_rank and "
        "num_experts_per_token > 0");
  }
  if (env::GetString("MORI_EP_SIM_SYNC")) sync_ = env::IsEnvVarEnabled("MORI_EP_SIM_SYNC");
  if (auto routing = env::GetString("MORI_EP_SIM_ROUTING")) {
    if (*routing == "uniform") {
      useIndices_ = false;
    } else if (*routing != "indices") {
      throw std::runtime_error("MORI_EP_SIM_ROUTING must be indices or uniform, got " + *routing);
    }
  }
  Override("MORI_EP_SIM_TIME_SCALE", timeScale_, ParsePositiveDouble);
  double spinUs = 20;
  Override("MORI_EP_SIM_SPIN_US", spinUs, ParseNonNegativeDouble);
  spinTail_ = std::chrono::nanoseconds(static_cast<int64_t>(spinUs * 1000));
  double timeoutS = 300;
  Override("MORI_EP_SIM_TIMEOUT_S", timeoutS, ParsePositiveDouble);
  timeout_ = std::chrono::seconds(static_cast<int64_t>(std::ceil(timeoutS)));
  localExpertCount_.assign(config_.numExpertPerRank, 0);
}

EpSim::~EpSim() {
  if (trace_ != nullptr) std::fclose(trace_);
}

void EpSim::Start(const std::string& group, const std::string& defaultCatalog, bool join) {
  if (started_) throw std::runtime_error("EP sim already started");
  auto model = std::make_unique<Model>();
  model->internode = config_.worldSize > config_.gpuPerNode;
  model->latencyUs = model->internode ? 60 : 25;
  model->gpu = env::GetString("MORI_EP_SIM_GPU").value_or("");
  model->catalogFiles = env::GetString("MORI_EP_SIM_CATALOG").value_or(defaultCatalog);
  std::map<CurveKey, Curve> catalog;
  std::stringstream paths(model->catalogFiles);
  for (std::string path; std::getline(paths, path, ',');) {
    if (!path.empty()) AddToCatalog(catalog, yaml::LoadFlatList(path), path);
  }
  if (!model->gpu.empty()) {
    bool knownGpu = false;
    for (auto& item : catalog) {
      const Curve& c = item.second;
      knownGpu |= c.gpu == model->gpu;
      if (c.gpu == model->gpu && c.kernel == KernelName(config_.kernelType) &&
          c.epSize == config_.worldSize) {
        model->curves.push_back(c);
      }
    }
    if (!knownGpu) {
      std::set<std::string> gpus;
      for (auto& item : catalog) gpus.insert(item.second.gpu);
      std::string list;
      for (auto& g : gpus) list += (list.empty() ? "" : ", ") + g;
      throw std::runtime_error("MORI_EP_SIM_GPU=" + model->gpu + " is not in the EP sim catalog [" +
                               model->catalogFiles + "], which has " +
                               (list.empty() ? std::string("none") : list));
    }
  }
  Override("MORI_EP_SIM_LATENCY_US", model->latencyUs, ParseNonNegativeDouble);
  Override("MORI_EP_SIM_XGMI_GBPS", model->xgmiGbps, ParsePositiveDouble);
  Override("MORI_EP_SIM_RDMA_GBPS", model->rdmaGbps, ParsePositiveDouble);
  model_ = std::move(model);

  std::string name = group;
  if (join && sync_ && config_.worldSize > 1) {
    if (name.empty()) {
      auto base = env::GetString("MORI_EP_SIM_GROUP");
      if (!base) {
        throw std::runtime_error(
            "EP sim with world_size > 1 needs a rendezvous name: set MORI_EP_SIM_GROUP (same on "
            "every rank, unique per job) or MORI_EP_SIM_SYNC=0");
      }
      name = *base + ".h" + std::to_string(gEnvGroupHandles.fetch_add(1));
    }
    shared_ = std::make_unique<Shared>();
    shared_->Join(name, config_.worldSize, config_.rank, numExperts_, timeout_);
  }

  if (join) {
    if (auto path = env::GetString("MORI_EP_SIM_TRACE")) {
      std::string p = *path;
      if (auto pos = p.find("{rank}"); pos != std::string::npos) {
        p.replace(pos, 6, std::to_string(config_.rank));
      } else {
        p += ".rank" + std::to_string(config_.rank);
      }
      trace_ = std::fopen(p.c_str(), "a");
      if (trace_ == nullptr) {
        MORI_OPS_WARN("EP sim: cannot open MORI_EP_SIM_TRACE file {}: {}", p, std::strerror(errno));
      } else {
        std::setvbuf(trace_, nullptr, _IOLBF, 0);
      }
    }
    if (config_.rank == 0) {
      MORI_OPS_INFO(
          "EP sim: {} ep{} gpu '{}' ({} curves from [{}]), rendezvous {}, routing {}, "
          "time_scale {}",
          KernelName(config_.kernelType), config_.worldSize, model_->gpu, model_->curves.size(),
          model_->catalogFiles, shared_ ? shared_->name : std::string("none (local only)"),
          useIndices_ ? "indices" : "uniform", timeScale_);
    }
  }
  started_ = true;
}

void EpSim::EnsureStarted() {
  if (!started_) Start("", "");
}

void EpSim::CountLoad(const OpDesc& op, std::vector<double>* rankLoad,
                      std::vector<double>* expertLoad) const {
  const int world = config_.worldSize;
  const int perRank = config_.numExpertPerRank;
  const int topk = config_.numExpertPerToken;
  rankLoad->assign(world, 0.0);
  expertLoad->assign(numExperts_, 0.0);
  if (!useIndices_ || op.indices == nullptr) {
    const double hit = RankHitProbability(numExperts_, perRank, topk);
    for (double& v : *rankLoad) v = op.numTokens * hit;
    for (double& v : *expertLoad) v = static_cast<double>(op.numTokens) * topk / numExperts_;
    return;
  }
  std::vector<int> lastToken(world, -1);
  for (int t = 0; t < op.numTokens; ++t) {
    for (int j = 0; j < topk; ++j) {
      int32_t expert = op.indices[static_cast<size_t>(t) * topk + j];
      if (expert < 0) continue;
      if (expert >= numExperts_) {
        throw std::runtime_error(
            "EP sim: token " + std::to_string(t) + " routes to expert " + std::to_string(expert) +
            " of " + std::to_string(numExperts_) +
            "; if the routing is not real (faked forward), set MORI_EP_SIM_ROUTING=uniform");
      }
      (*expertLoad)[expert] += 1;
      int dest = expert / perRank;
      if (lastToken[dest] != t) {
        lastToken[dest] = t;
        (*rankLoad)[dest] += 1;
      }
    }
  }
}

Estimate EpSim::Duration(const OpDesc& op, double sendIntra, double sendInter, double recvIntra,
                         double recvInter) const {
  const Model::Pick& p = model_->Choose(op, config_);
  const double intra = std::max(sendIntra, recvIntra);
  const double inter = std::max(sendInter, recvInter);
  Estimate e;
  e.source = p.source;
  if (p.curve != nullptr) {
    double n = p.entryFanIntra > 0 ? intra / p.entryFanIntra : 0;
    if (p.entryFanInter > 0) n = std::max(n, inter / p.entryFanInter);
    e.tokens = n * p.byteRatio;
    e.latencyUs = Interp(*p.curve, e.tokens);
  } else {
    // GB/s is bytes per ns.
    double wireNs =
        std::max(intra * p.pairBytes / model_->xgmiGbps, inter * p.pairBytes / model_->rdmaGbps);
    e.tokens = op.numTokens;
    e.latencyUs = model_->latencyUs + wireNs / 1000.0;
  }
  return e;
}

Estimate EpSim::EstimateUniform(const OpDesc& op) const {
  if (!model_) throw std::runtime_error("EP sim: Start() before EstimateUniform()");
  const double hit =
      RankHitProbability(numExperts_, config_.numExpertPerRank, config_.numExpertPerToken);
  const double intra = op.numTokens * nodeSize_ * hit;
  const double inter = op.numTokens * (config_.worldSize - nodeSize_) * hit;
  return Duration(op, intra, inter, intra, inter);
}

void EpSim::Post(const OpDesc& op) {
  EnsureStarted();
  if (pending_) {
    throw std::runtime_error(std::string("EP sim: ") + PhaseName(op.phase) +
                             " posted while the previous " + PhaseName(posted_.phase) +
                             " was not received (missing dispatch_recv/combine_recv)");
  }
  posted_.seq = ++seq_;
  posted_.phase = op.phase;
  posted_.op = op;
  posted_.op.indices = nullptr;  // the caller's buffer is only valid during this call
  posted_.arrival = Clock::now();
  CountLoad(op, &posted_.rankLoad, &posted_.expertLoad);
  if (op.phase == Phase::Dispatch) lastDispatchTokens_ = op.numTokens;
  if (shared_) {
    shared_->Publish(posted_.seq, ToNs(posted_.arrival), op.phase, op.numTokens, posted_.rankLoad,
                     posted_.expertLoad, timeout_);
  }
  pending_ = true;
}

int EpSim::Wait() {
  if (!pending_) throw std::runtime_error("EP sim: receive without a posted send");
  pending_ = false;
  const Posted& p = posted_;
  const int world = config_.worldSize;
  const int me = config_.rank;
  auto nodeOf = [&](int r) { return r / nodeSize_; };
  auto scaled = [&](double us) {
    return std::chrono::nanoseconds(static_cast<int64_t>(std::llround(us * 1000.0 / timeScale_)));
  };

  OpStats s;
  s.seq = p.seq;
  s.phase = p.phase;
  Clock::time_point end;
  double recv = 0;
  if (shared_) {
    shared_->AwaitAll(p.seq, p.phase, timeout_);
    // load[src][dst] in (token, rank) pairs, plus each rank's arrival.
    std::vector<double> load(static_cast<size_t>(world) * world);
    std::vector<int64_t> arrival(world);
    std::vector<double> localExperts(config_.numExpertPerRank, 0.0);
    for (int r = 0; r < world; ++r) {
      char* e = shared_->Entry(r, p.seq);
      EntryHeader h;
      std::memcpy(&h, e, sizeof(h));
      arrival[r] = h.arrivalNs;
      std::memcpy(&load[static_cast<size_t>(r) * world], shared_->RankLoad(e),
                  sizeof(double) * world);
      const double* experts = shared_->ExpertLoad(e) + me * config_.numExpertPerRank;
      for (int x = 0; x < config_.numExpertPerRank; ++x) localExperts[x] += experts[x];
    }
    shared_->Consumed(p.seq);

    int64_t endNs = 0;
    for (int r = 0; r < world; ++r) {
      double sendIntra = 0, sendInter = 0, recvIntra = 0, recvInter = 0;
      for (int o = 0; o < world; ++o) {
        bool sameNode = nodeOf(o) == nodeOf(r);
        double out = load[static_cast<size_t>(r) * world + o];
        double in = load[static_cast<size_t>(o) * world + r];
        (sameNode ? sendIntra : sendInter) += out;
        (sameNode ? recvIntra : recvInter) += in;
      }
      Estimate est = Duration(p.op, sendIntra, sendInter, recvIntra, recvInter);
      endNs = std::max(endNs, arrival[r] + scaled(est.latencyUs).count());
      if (r == me) {
        s.modelUs = est.latencyUs;
        s.loadTokens = est.tokens;
        s.source = est.source;
        recv = recvIntra + recvInter;
      }
    }
    end = Clock::time_point(
        std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(endNs)));
    if (p.phase == Phase::Dispatch) {
      for (int x = 0; x < config_.numExpertPerRank; ++x) {
        localExpertCount_[x] = static_cast<int>(std::llround(localExperts[x]));
      }
    }
  } else {
    // No peers to ask: assume the traffic coming in mirrors what goes out.
    double sendIntra = 0, sendInter = 0;
    for (int o = 0; o < world; ++o) {
      (nodeOf(o) == nodeOf(me) ? sendIntra : sendInter) += p.rankLoad[o];
    }
    Estimate est = Duration(p.op, sendIntra, sendInter, sendIntra, sendInter);
    s.modelUs = est.latencyUs;
    s.loadTokens = est.tokens;
    s.source = est.source;
    end = p.arrival + scaled(est.latencyUs);
    recv = sendIntra + sendInter;
    if (p.phase == Phase::Dispatch) {
      for (int x = 0; x < config_.numExpertPerRank; ++x) {
        localExpertCount_[x] =
            static_cast<int>(std::llround(p.expertLoad[me * config_.numExpertPerRank + x]));
      }
    }
  }

  SleepUntil(end, spinTail_);
  s.elapsedUs =
      std::chrono::duration<double, std::micro>(Clock::now() - p.arrival).count() * timeScale_;
  s.recvTokens = p.phase == Phase::Dispatch ? static_cast<int>(std::llround(recv)) : p.op.numTokens;
  last_ = s;
  WriteTrace(s, ToNs(p.arrival), ToNs(end));
  return s.recvTokens;
}

void EpSim::WriteTrace(const OpStats& s, int64_t arrivalNs, int64_t endNs) {
  if (trace_ == nullptr) return;
  std::fprintf(trace_,
               "{\"seq\":%llu,\"rank\":%d,\"phase\":\"%s\",\"tokens\":%d,\"load_tokens\":%.2f,"
               "\"model_us\":%.3f,\"elapsed_us\":%.3f,\"recv\":%d,\"arrival_ns\":%lld,"
               "\"end_ns\":%lld}\n",
               static_cast<unsigned long long>(s.seq), config_.rank, PhaseName(s.phase),
               posted_.op.numTokens, s.loadTokens, s.modelUs, s.elapsedUs, s.recvTokens,
               static_cast<long long>(arrivalNs), static_cast<long long>(endNs));
}

namespace {

int SimOp(EpDispatchCombineHandle& handle, OpDesc op) {
  if (!handle.sim)
    throw std::runtime_error("EP sim op on a handle built without MORI_EP_SIM_ENABLE");
  if (op.hiddenDim <= 0) op.hiddenDim = handle.config.hiddenDim;
  handle.sim->Post(op);
  handle.curHiddenDim = op.hiddenDim;
  if (op.phase == Phase::Dispatch) handle.curRankNumToken = op.numTokens;
  return handle.config.kernelType == KernelType::AsyncLL ? -1 : handle.sim->Wait();
}

}  // namespace

int SimDispatch(EpDispatchCombineHandle& handle, const int32_t* indices, int numTokens,
                hipDataType dtype, int hiddenDim, bool hasScales) {
  OpDesc op;
  op.phase = Phase::Dispatch;
  op.dtype = dtype;
  op.hiddenDim = hiddenDim;
  op.numTokens = numTokens;
  op.scaleBytes = hasScales ? static_cast<int>(handle.config.ScaleBytes()) : 0;
  op.indices = indices;
  return SimOp(handle, op);
}

int SimCombine(EpDispatchCombineHandle& handle, const int32_t* indices, int numTokens,
               hipDataType dtype, int hiddenDim, bool zeroCopy) {
  OpDesc op;
  op.phase = Phase::Combine;
  op.dtype = dtype;
  op.hiddenDim = hiddenDim;
  op.numTokens = numTokens;
  op.zeroCopy = zeroCopy;
  op.indices = indices;
  return SimOp(handle, op);
}

}  // namespace sim
}  // namespace moe
}  // namespace mori
