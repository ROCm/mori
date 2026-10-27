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

// Runs a fault catalog: by default tests/cpp/io/fault_catalog/ibverbs.yaml, one
// entry per real-world fault on the ibverbs data path (MORI-IO fault simulation,
// part 1/N). The YAML file explains each field and the four checks every entry
// is held to (ends, honest, recovers, alive); this file only runs them.
//
// Processes, all on one node:
//   runner     runs each entry in its own initiator process, so a crash or a
//              hang in one does not stop the rest, and tabulates the results
//   initiator  GPU 0 and its closest NIC: issues the transfers
//   target     GPU 1 and its closest NIC: owns the remote buffer and answers the
//              initiator over a socketpair (fill, check, notified?, arm, ...)
// With two NICs the data crosses the wire and the switch, as between two
// nodes. On a single-GPU machine both ends share GPU 0 and its NIC.
//
// Usage: test_fault_injection [id-substring]
//   MORI_IO_FAULT_CATALOG=<file.yaml>  runs another catalog
//   MORI_IO_FAULT_TARGET_GPU=<n>       puts the target on GPU n (0: same NIC as the initiator)
//   MORI_IO_FAULT_REPORT=<file.md>     appends a markdown results table (CI job summary)
// Exits non-zero only on a FAIL (a check failed that the entry does not expect)
// or an XPASS (a known_bug entry now passes a check it is expected to fail).
// Needs -DENABLE_IO_FAULT_INJECTION=ON, an active RDMA NIC and a GPU (two of each
// for the cross-NIC setup).

#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/application/topology/system.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"
#include "mori/utils/flat_yaml.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/fault_injector.hpp"

using namespace mori::io;

namespace {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Fault catalog                                         */
/* ---------------------------------------------------------------------------------------------- */

enum class Op {
  Write,  // one 1 MiB RDMA write
  Read,   // one 1 MiB RDMA read
  Batch,  // BatchWrite of 2 non-adjacent segments, one per QP (qpPerTransfer=2)
};

enum class Side { Initiator, Target };  // which process the fault is armed in

// One catalog entry; the YAML file documents the fields.
struct Entry {
  std::string id;
  std::string realWorld;
  std::string rule;  // empty = no fault
  Op op{Op::Write};
  bool notify{false};
  int warmup{0};
  Side side{Side::Initiator};
  std::string peerFailure;      // empty = unchecked, none = expect no event
  std::string during;           // empty, reject, succeed
  std::string env;              // KEY=VAL[,KEY=VAL]
  std::string knownBug;         // empty = expected to pass every check
  std::set<std::string> fails;  // with knownBug: the checks expected to fail today
};

const std::set<std::string> kChecks = {"ends", "honest", "recovers", "alive", "reported", "during"};

std::string CatalogPath() {
  const char* path = std::getenv("MORI_IO_FAULT_CATALOG");
  if (path == nullptr || path[0] == '\0') return MORI_IO_FAULT_CATALOG_DEFAULT;
  return std::strchr(path, '/') ? path : std::string(MORI_IO_FAULT_CATALOG_DIR) + "/" + path;
}

// Loads and validates a catalog; throws with the file and entry on any mistake,
// so a typo never silently drops or weakens an entry.
std::vector<Entry> LoadCatalog(const std::string& path) {
  const std::set<std::string> kFields = {"id",     "real_world",   "rule",   "transfer",
                                         "notify", "warmup",       "side",   "known_bug",
                                         "fails",  "peer_failure", "during", "env"};
  std::vector<Entry> entries;
  std::set<std::string> ids;
  for (const mori::yaml::FlatMap& raw : mori::yaml::LoadFlatList(path)) {
    std::string where = path + ": entry " + std::to_string(entries.size() + 1);
    mori::yaml::CheckKeys(raw, kFields, where);

    Entry e;
    e.id = mori::yaml::GetString(raw, "id");
    if (e.id.empty()) throw std::runtime_error(where + ": missing 'id'");
    where += " (" + e.id + ")";
    if (!ids.insert(e.id).second) throw std::runtime_error(where + ": duplicate id");
    e.realWorld = mori::yaml::GetString(raw, "real_world");
    e.rule = mori::yaml::GetString(raw, "rule");
    std::string transfer = mori::yaml::GetString(raw, "transfer", "write");
    if (transfer == "write") {
      e.op = Op::Write;
    } else if (transfer == "read") {
      e.op = Op::Read;
    } else if (transfer == "batch") {
      e.op = Op::Batch;
    } else {
      throw std::runtime_error(where + ": transfer must be write, read or batch, got '" + transfer +
                               "'");
    }
    e.notify = mori::yaml::GetBool(raw, "notify", false, where);
    long warmup = mori::yaml::GetInt(raw, "warmup", 0, where);
    if (warmup < 0) throw std::runtime_error(where + ": warmup must not be negative");
    e.warmup = static_cast<int>(warmup);
    std::string side = mori::yaml::GetString(raw, "side", "initiator");
    if (side == "initiator") {
      e.side = Side::Initiator;
    } else if (side == "target") {
      e.side = Side::Target;
    } else {
      throw std::runtime_error(where + ": side must be initiator or target, got '" + side + "'");
    }
    e.peerFailure = mori::yaml::GetString(raw, "peer_failure");
    const std::set<std::string> reasons = {"",
                                           "none",
                                           "LOCAL_QP_ERROR",
                                           "LOCAL_CQ_ERROR",
                                           "LOCAL_DEVICE_FATAL",
                                           "LOCAL_PORT_DOWN",
                                           "PEER_UNREACHABLE"};
    if (!reasons.count(e.peerFailure)) throw std::runtime_error(where + ": bad peer_failure");
    e.during = mori::yaml::GetString(raw, "during");
    if (!e.during.empty() && e.during != "reject" && e.during != "succeed" && e.during != "fail") {
      throw std::runtime_error(where + ": during must be reject, succeed or fail");
    }
    e.env = mori::yaml::GetString(raw, "env");
    std::istringstream vars(e.env);
    for (std::string var; std::getline(vars, var, ',');) {
      size_t eq = var.find('=');
      if (eq == std::string::npos || eq == 0 ||
          var.substr(0, eq).find_first_not_of(
              "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") !=
              std::string::npos) {
        throw std::runtime_error(where + ": env requires KEY=VAL[,KEY=VAL]");
      }
    }
    e.knownBug = mori::yaml::GetString(raw, "known_bug");
    std::istringstream fails(mori::yaml::GetString(raw, "fails"));
    for (std::string check; std::getline(fails, check, ',');) {
      check.erase(0, check.find_first_not_of(' '));
      check.erase(check.find_last_not_of(' ') + 1);
      if (kChecks.count(check) == 0) {
        throw std::runtime_error(where + ": fails lists '" + check +
                                 "'; checks are ends, honest, recovers, alive, reported, during");
      }
      e.fails.insert(check);
    }
    if ((e.fails.count("reported") && e.peerFailure.empty()) ||
        (e.fails.count("during") && e.during.empty())) {
      throw std::runtime_error(where + ": fails names an unchecked field");
    }
    if (e.knownBug.empty() != e.fails.empty()) {
      throw std::runtime_error(where + ": known_bug and fails go together");
    }
    if (!e.rule.empty()) {
      try {
        ParseFaultRule(e.rule);
      } catch (const std::invalid_argument& ex) {
        throw std::runtime_error(where + ": " + ex.what());
      }
    }
    entries.push_back(std::move(e));
  }
  return entries;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Shared by both processes                                   */
/* ---------------------------------------------------------------------------------------------- */

constexpr size_t kBufBytes = 1 << 20;
constexpr size_t kSegBytes = 256 << 10;  // Batch: [0, 256K) and [512K, 768K)
constexpr size_t kSeg2Offset = 512 << 10;
constexpr int kDeadlineMs = 5000;
constexpr int kSetupMs = 30000;  // target start-up: HIP init, engine, buffer
constexpr int kRpcMs = 10000;
constexpr int kChildTimeoutMs = 90000;
constexpr int kInitiatorGpu = 0;
constexpr const char* kInitiatorKey = "fault_initiator";
constexpr const char* kTargetKey = "fault_target";

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;

struct TestSkip : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

const char* CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::SUCCESS:
      return "SUCCESS";
    case StatusCode::INIT:
      return "INIT";
    case StatusCode::IN_PROGRESS:
      return "IN_PROGRESS";
    case StatusCode::ERR_INVALID_ARGS:
      return "ERR_INVALID_ARGS";
    case StatusCode::ERR_NOT_FOUND:
      return "ERR_NOT_FOUND";
    case StatusCode::ERR_RDMA_OP:
      return "ERR_RDMA_OP";
    case StatusCode::ERR_BAD_STATE:
      return "ERR_BAD_STATE";
    case StatusCode::ERR_GPU_OP:
      return "ERR_GPU_OP";
    default:
      return "?";
  }
}

// The target uses a second GPU, and so a second NIC, when there is one.
// MORI_IO_FAULT_TARGET_GPU overrides it (e.g. 0 to keep both ends on one NIC).
int TargetGpu() {
  if (const char* env = std::getenv("MORI_IO_FAULT_TARGET_GPU")) return std::atoi(env);
  int gpus = 0;
  if (hipGetDeviceCount(&gpus) != hipSuccess) return kInitiatorGpu;
  return gpus >= 2 ? 1 : kInitiatorGpu;
}

// The NIC MORI picks for a buffer on `gpu` (the same topology match it uses).
std::string NicFor(int gpu) { return mori::application::TopoSystem().MatchGpuAndNic(gpu); }

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  socklen_t len = sizeof(addr);
  int port = -1;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
      getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
    port = ntohs(addr.sin_port);
  }
  close(fd);
  return port;
}

std::unique_ptr<IOEngine> MakeEngine(const std::string& key, const Entry& e) {
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";  // control plane only; data goes over RDMA
  cfg.port = GetFreePort();
  if (cfg.port <= 0) throw std::runtime_error("failed to allocate a tcp port");
  auto engine = std::make_unique<IOEngine>(key, cfg);
  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.enableNotification = e.notify;
  rdmaCfg.notifPerQp = 64;
  rdmaCfg.qpPerTransfer = e.op == Op::Batch ? 2 : 1;
  engine->CreateBackend(BackendType::RDMA, rdmaCfg);
  return engine;
}

struct GpuBuffer {
  MemoryDesc desc{};
  void* ptr{nullptr};

  GpuBuffer(IOEngine* engine, int gpu) {
    HIP_RUNTIME_CHECK(hipSetDevice(gpu));
    HIP_RUNTIME_CHECK(hipMalloc(&ptr, kBufBytes));
    desc = engine->RegisterMemory(ptr, kBufBytes, gpu, MemoryLocationType::GPU);
  }

  void Fill(int pattern) {
    HIP_RUNTIME_CHECK(hipMemset(ptr, pattern, kBufBytes));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }

  // The bytes a transfer of `op` writes all hold `pattern`.
  bool Holds(Op op, int pattern) {
    std::vector<unsigned char> host(kBufBytes);
    HIP_RUNTIME_CHECK(hipMemcpy(host.data(), ptr, kBufBytes, hipMemcpyDeviceToHost));
    auto filled = [&](size_t off, size_t len) {
      for (size_t i = off; i < off + len; ++i) {
        if (host[i] != static_cast<unsigned char>(pattern)) return false;
      }
      return true;
    };
    if (op == Op::Batch) return filled(0, kSegBytes) && filled(kSeg2Offset, kSegBytes);
    return filled(0, kBufBytes);
  }
};

// Length-prefixed messages over the initiator/target socketpair.
bool SendMsg(int fd, const std::string& msg) {
  uint32_t len = static_cast<uint32_t>(msg.size());
  std::string buf(reinterpret_cast<const char*>(&len), sizeof(len));
  buf += msg;
  for (size_t off = 0; off < buf.size();) {
    ssize_t n = send(fd, buf.data() + off, buf.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

// False on EOF (the peer exited), an error or the timeout (< 0 waits forever).
bool RecvMsg(int fd, std::string* msg, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  auto readExact = [&](char* dst, size_t len) {
    for (size_t off = 0; off < len;) {
      int waitMs = -1;
      if (timeoutMs >= 0) {
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - std::chrono::steady_clock::now())
                        .count();
        if (left <= 0) return false;
        waitMs = static_cast<int>(left);
      }
      pollfd p{fd, POLLIN, 0};
      int r = poll(&p, 1, waitMs);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) return false;
      ssize_t n = read(fd, dst + off, len - off);
      if (n <= 0) return false;
      off += static_cast<size_t>(n);
    }
    return true;
  };
  uint32_t len = 0;
  if (!readExact(reinterpret_cast<char*>(&len), sizeof(len))) return false;
  msg->assign(len, '\0');
  return len == 0 || readExact(&(*msg)[0], len);
}

template <typename T>
std::string Pack(const T& value) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, value);
  return std::string(buf.data(), buf.size());
}

template <typename T>
T Unpack(const std::string& bytes) {
  msgpack::object_handle handle = msgpack::unpack(bytes.data(), bytes.size());
  return handle.get().as<T>();
}

const char* FailureName(PeerFailureReason reason) {
  switch (reason) {
    case PeerFailureReason::LOCAL_QP_ERROR:
      return "LOCAL_QP_ERROR";
    case PeerFailureReason::LOCAL_CQ_ERROR:
      return "LOCAL_CQ_ERROR";
    case PeerFailureReason::LOCAL_DEVICE_FATAL:
      return "LOCAL_DEVICE_FATAL";
    case PeerFailureReason::LOCAL_PORT_DOWN:
      return "LOCAL_PORT_DOWN";
    case PeerFailureReason::PEER_UNREACHABLE:
      return "PEER_UNREACHABLE";
    default:
      return "unknown";
  }
}

std::string CollectFailures(IOEngine* engine, int ms, const std::string& expected) {
  std::string reasons;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  do {
    PeerFailureEvent event;
    while (engine->PopPeerFailure(&event)) {
      std::string reason = FailureName(event.reason);
      reasons += reason + " ";
      if (reason == expected) return reasons;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return reasons;
}

void ApplyEnv(const Entry& e) {
  std::istringstream vars(e.env);
  for (std::string var; std::getline(vars, var, ',');) {
    size_t eq = var.find('=');
    if (setenv(var.substr(0, eq).c_str(), var.substr(eq + 1).c_str(), 1) != 0) {
      throw std::runtime_error("could not set catalog environment");
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Target process                                          */
/* ---------------------------------------------------------------------------------------------- */

// Owns the remote engine and buffer and answers the initiator until it says
// "quit" or goes away. Requests: fill <pattern> | check <op> <pattern> |
// told <transfer id> <ms> | arm <rule> | disarm | fired | ping.
int ServeTarget(const Entry& e, int fd) {
  const int gpu = TargetGpu();
  auto engine = MakeEngine(kTargetKey, e);
  GpuBuffer buf(engine.get(), gpu);
  std::string msg;
  if (!SendMsg(fd, Pack(engine->GetEngineDesc())) || !SendMsg(fd, Pack(buf.desc)) ||
      !RecvMsg(fd, &msg, kSetupMs)) {
    return kExitFail;
  }
  engine->RegisterRemoteEngine(Unpack<EngineDesc>(msg));
  if (!SendMsg(fd, NicFor(gpu) + " " + std::to_string(gpu))) return kExitFail;

  while (RecvMsg(fd, &msg, -1)) {
    std::istringstream in(msg);
    std::string cmd;
    in >> cmd;
    std::string reply = "ok";
    if (cmd == "fill") {
      int pattern = 0;
      in >> pattern;
      buf.Fill(pattern);
    } else if (cmd == "check") {
      int op = 0, pattern = 0;
      in >> op >> pattern;
      reply = buf.Holds(static_cast<Op>(op), pattern) ? "ok" : "no";
    } else if (cmd == "told") {
      TransferUniqueId id = 0;
      int ms = 0;
      in >> id >> ms;
      reply = "no";
      auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
      do {
        TransferStatus inbound;
        if (engine->PopInboundTransferStatus(kInitiatorKey, id, &inbound)) {
          reply = "ok";
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (std::chrono::steady_clock::now() < deadline);
    } else if (cmd == "failures") {
      int ms = 0;
      std::string expected;
      in >> ms >> expected;
      reply = CollectFailures(engine.get(), ms, expected);
    } else if (cmd == "arm") {
      std::string rule;
      in >> rule;
      FaultInjector::Instance().Arm(ParseFaultRule(rule));
    } else if (cmd == "disarm") {
      FaultInjector::Instance().Disarm();
    } else if (cmd == "fired") {
      reply = std::to_string(FaultInjector::Instance().Fired());
    } else if (cmd == "injection_failed") {
      reply = FaultInjector::Instance().Failed() ? "yes" : "no";
    } else if (cmd == "quit") {
      break;
    } else if (cmd != "ping") {
      reply = "unknown request '" + cmd + "'";
    }
    if (!SendMsg(fd, reply)) break;
  }
  return kExitPass;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Initiator process                                        */
/* ---------------------------------------------------------------------------------------------- */

struct Transfer {
  TransferStatus status;
  TransferUniqueId id{0};
  Op op{Op::Write};
  int pattern{0};
};

// The initiator's engine and buffer, plus the target process it started. The
// initiator exits with _Exit, so nothing here is torn down: a faulted engine may
// not shut down cleanly, and that is not what these checks are about. The
// target is killed with the initiator (PR_SET_PDEATHSIG).
class PeerRig {
 public:
  explicit PeerRig(const Entry& e) : entry_(e) {
    if (!RdmaBackend::HasActiveDevices()) throw TestSkip("requires an active RDMA device");
    int gpus = 0;
    if (hipGetDeviceCount(&gpus) != hipSuccess || gpus == 0) throw TestSkip("requires a GPU");
    StartTarget();

    engine_ = MakeEngine(kInitiatorKey, e);
    local_ = std::make_unique<GpuBuffer>(engine_.get(), kInitiatorGpu);
    std::string engineDesc, memDesc, hello;
    if (!RecvMsg(fd_, &engineDesc, kSetupMs) || !RecvMsg(fd_, &memDesc, kSetupMs)) {
      throw std::runtime_error("the target process did not start");
    }
    engine_->RegisterRemoteEngine(Unpack<EngineDesc>(engineDesc));
    remote_ = Unpack<MemoryDesc>(memDesc);
    if (!SendMsg(fd_, Pack(engine_->GetEngineDesc())) || !RecvMsg(fd_, &hello, kSetupMs)) {
      throw std::runtime_error("the target process did not finish setup");
    }
    std::string targetNic, targetGpu;
    std::istringstream(hello) >> targetNic >> targetGpu;
    std::string initiatorNic = NicFor(kInitiatorGpu);
    std::printf("    topology: initiator GPU %d %s -> target pid %d GPU %s %s (%s)\n",
                kInitiatorGpu, initiatorNic.c_str(), static_cast<int>(pid_), targetGpu.c_str(),
                targetNic.c_str(), initiatorNic == targetNic ? "same NIC" : "cross-NIC");
  }

  // Starts one transfer with a fresh data pattern in its source and zeros in its destination.
  Transfer* Start(Op op) {
    transfers_.push_back(std::make_unique<Transfer>());
    Transfer* t = transfers_.back().get();
    t->op = op;
    t->id = engine_->AllocateTransferUniqueId();
    t->pattern = static_cast<int>(transfers_.size() % 250 + 1);
    if (op == Op::Read) {
      Ask("fill " + std::to_string(t->pattern));
      local_->Fill(0);
    } else {
      local_->Fill(t->pattern);
      Ask("fill 0");
    }

    switch (op) {
      case Op::Write:
        engine_->Write(local_->desc, 0, remote_, 0, kBufBytes, &t->status, t->id);
        break;
      case Op::Read:
        engine_->Read(local_->desc, 0, remote_, 0, kBufBytes, &t->status, t->id);
        break;
      case Op::Batch: {
        MemDescVec local{local_->desc};
        MemDescVec remote{remote_};
        BatchSizeVec offsets{{0, kSeg2Offset}};
        BatchSizeVec sizes{{kSegBytes, kSegBytes}};
        TransferStatusPtrVec statuses{&t->status};
        TransferUniqueIdVec ids{t->id};
        engine_->BatchWrite(local, offsets, remote, offsets, sizes, statuses, ids);
        break;
      }
    }
    return t;
  }

  // The transfer's bytes are in its destination.
  bool DataLanded(const Transfer& t) {
    if (t.op == Op::Read) return local_->Holds(t.op, t.pattern);
    auto reply =
        Ask("check " + std::to_string(static_cast<int>(t.op)) + " " + std::to_string(t.pattern));
    return reply && *reply == "ok";
  }

  // With notifications off there is nothing to tell the target, so this holds.
  bool TargetTold(const Transfer& t, int timeoutMs) {
    if (!entry_.notify) return true;
    auto reply =
        Ask("told " + std::to_string(t.id) + " " + std::to_string(timeoutMs), timeoutMs + kRpcMs);
    return reply && *reply == "ok";
  }

  // SUCCESS within the deadline, data landed, target told.
  bool Completes(Transfer* t, std::string* why) {
    StatusCode code = t->status.WaitFor(kDeadlineMs);
    if (code != StatusCode::SUCCESS) {
      *why = std::string(CodeName(code)) + " " + t->status.Message();
      return false;
    }
    if (!DataLanded(*t)) {
      *why = "SUCCESS but data did not land";
      return false;
    }
    if (!TargetTold(*t, kDeadlineMs)) {
      *why = "SUCCESS but the target was never notified";
      return false;
    }
    return true;
  }

  void Arm(const std::string& rule) {
    if (entry_.side == Side::Initiator) {
      FaultInjector::Instance().Arm(ParseFaultRule(rule));
    } else if (!Ask("arm " + rule)) {
      throw std::runtime_error("could not arm the fault in the target process");
    }
  }

  void Disarm() {
    FaultInjector::Instance().Disarm();
    Ask("disarm");
  }

  // How often the armed side fired; nullopt if that side is gone and cannot say.
  std::optional<uint64_t> Fired() {
    if (entry_.side == Side::Initiator) return FaultInjector::Instance().Fired();
    auto reply = Ask("fired");
    if (!reply) return std::nullopt;
    return std::stoull(*reply);
  }

  bool InjectionFailed() {
    if (entry_.side == Side::Initiator) return FaultInjector::Instance().Failed();
    auto reply = Ask("injection_failed");
    return reply && *reply == "yes";
  }

  bool Reported() {
    const int ms = entry_.peerFailure == "none" ? 500 : kDeadlineMs;
    std::string reasons;
    if (entry_.side == Side::Target) {
      auto reply = Ask("failures " + std::to_string(ms) + " " + entry_.peerFailure, ms + kRpcMs);
      if (!reply) return false;
      reasons = *reply;
    } else {
      reasons = CollectFailures(engine_.get(), ms, entry_.peerFailure);
    }
    std::printf("    peer failures: %s\n", reasons.empty() ? "none" : reasons.c_str());
    if (entry_.peerFailure == "none") return reasons.empty();
    std::istringstream in(reasons);
    for (std::string reason; in >> reason;) {
      if (reason == entry_.peerFailure) return true;
    }
    return false;
  }

  bool SessionAlive(bool wait) {
    if (!session_) session_ = engine_->CreateSession(local_->desc, remote_);
    if (!session_) return false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kDeadlineMs);
    do {
      if (session_->Alive()) return true;
      if (!wait) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  // Whether the target process is still up and answering; `why` says what happened if not.
  bool TargetAlive(std::string* why) {
    if (Ask("ping")) return true;
    int status = 0;
    pid_t done = 0;
    for (int i = 0; i < 20 && done == 0; ++i) {  // it may still be on its way out
      done = waitpid(pid_, &status, WNOHANG);
      if (done == 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (done == pid_ && WIFEXITED(status)) {
      *why = "exited with code " + std::to_string(WEXITSTATUS(status));
    } else if (done == pid_ && WIFSIGNALED(status)) {
      *why = std::string("crashed: ") + strsignal(WTERMSIG(status));
    } else {
      *why = "stopped answering";
    }
    return false;
  }

 private:
  void StartTarget() {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw std::runtime_error("socketpair failed");
    std::string fdArg = std::to_string(sv[1]);  // built before fork: the child only execs
    pid_ = fork();
    if (pid_ < 0) throw std::runtime_error("fork failed");
    if (pid_ == 0) {
      close(sv[0]);
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      execl("/proc/self/exe", "test_fault_injection", "--target", entry_.id.c_str(), fdArg.c_str(),
            static_cast<char*>(nullptr));
      _exit(127);
    }
    close(sv[1]);
    fd_ = sv[0];
  }

  // One request to the target; nullopt once it is gone or does not answer in time.
  // After a miss the channel is out of step, so every later request misses too.
  std::optional<std::string> Ask(const std::string& request, int timeoutMs = kRpcMs) {
    std::string reply;
    if (!targetGone_ && SendMsg(fd_, request) && RecvMsg(fd_, &reply, timeoutMs)) return reply;
    targetGone_ = true;
    return std::nullopt;
  }

  const Entry& entry_;
  pid_t pid_{-1};
  int fd_{-1};
  bool targetGone_{false};
  std::unique_ptr<IOEngine> engine_;
  std::unique_ptr<GpuBuffer> local_;
  MemoryDesc remote_{};
  std::optional<IOEngineSession> session_;
  std::vector<std::unique_ptr<Transfer>> transfers_;  // MORI keeps pointers to their statuses
};

// Prints "RESULT ends=<0|1> honest=<0|1> recovers=<0|1> target=<0|1>" for the runner.
int RunEntry(const Entry& e) {
  ApplyEnv(e);
  PeerRig rig(e);
  std::string why;
  for (int i = 0; i < e.warmup; ++i) {
    if (!rig.Completes(rig.Start(e.op), &why)) {
      std::printf("    warmup transfer %d failed: %s\n", i, why.c_str());
      return kExitFail;
    }
  }

  const bool faulted = !e.rule.empty();
  if (faulted) rig.Arm(e.rule);

  Transfer* t = rig.Start(e.op);
  StatusCode code = t->status.WaitFor(kDeadlineMs);
  bool ends = code != StatusCode::IN_PROGRESS;
  bool honest = true;
  if (code == StatusCode::SUCCESS) {
    honest = rig.DataLanded(*t) && rig.TargetTold(*t, kDeadlineMs);
  }
  std::printf("    faulted transfer: %s %s\n", CodeName(code), t->status.Message().c_str());

  std::optional<uint64_t> fired = faulted ? rig.Fired() : std::optional<uint64_t>(0);
  int reported = e.peerFailure.empty() ? -1 : static_cast<int>(rig.Reported());
  int during = -1;
  if (!e.during.empty()) {
    if (e.peerFailure.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Transfer* next = rig.Start(e.op);
    if (e.during == "reject") {
      during = next->status.Code() == StatusCode::ERR_RDMA_OP;
    } else if (e.during == "fail") {
      next->status.WaitFor(kDeadlineMs);
      during = next->status.Failed();
    } else {
      during = rig.Completes(next, &why);
    }
    std::printf("    during fault: %s %s\n", CodeName(next->status.Code()),
                next->status.Message().c_str());
  }
  rig.Disarm();
  if (faulted && rig.InjectionFailed()) {
    std::printf("    the injection failed; this is a harness error, not a known bug\n");
    return kExitFail;
  }
  if (faulted && fired && *fired == 0) {
    std::printf("    the fault never fired: the entry does not exercise its rule\n");
    return kExitFail;
  }

  if (e.peerFailure == "LOCAL_PORT_DOWN") rig.SessionAlive(true);
  bool recovers = rig.Completes(rig.Start(e.op), &why) && rig.SessionAlive(false);
  if (!recovers) std::printf("    after the fault cleared: %s\n", why.c_str());

  bool targetAlive = rig.TargetAlive(&why);
  if (!targetAlive) std::printf("    target process: %s\n", why.c_str());

  if (faulted && rig.InjectionFailed()) {
    std::printf("    the injection failed; this is a harness error, not a known bug\n");
    return kExitFail;
  }
  std::printf("RESULT ends=%d honest=%d recovers=%d target=%d reported=%d during=%d\n", ends,
              honest, recovers, targetAlive, reported, during);
  return ends && honest && recovers && targetAlive && reported != 0 && during != 0 ? kExitPass
                                                                                   : kExitFail;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Runner (parent process)                                  */
/* ---------------------------------------------------------------------------------------------- */

// What one entry's run showed.
struct Outcome {
  bool skipped{false};
  bool hasResult{false};  // the initiator printed its RESULT line
  // 1 = passed, 0 = failed, -1 = not reached (a process died before reporting)
  int ends{-1}, honest{-1}, recovers{-1}, target{-1}, reported{-1}, during{-1};
  bool alive{true};
  std::string note;      // how a process died, if one did
  std::string topology;  // "initiator GPU 0 ionic_0 -> target ... (cross-NIC)"
};

// Runs one entry in an initiator process, echoing its output, and reads back its result.
Outcome RunChild(const char* self, const Entry& e) {
  Outcome out;
  int fds[2];
  if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    dup2(fds[1], STDOUT_FILENO);
    close(fds[0]);
    close(fds[1]);
    execl(self, self, "--entry", e.id.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  close(fds[1]);

  std::string text;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kChildTimeoutMs);
  bool timedOut = false;
  while (true) {
    pollfd p{fds[0], POLLIN, 0};
    if (poll(&p, 1, 100) > 0) {
      char buf[4096];
      ssize_t n = read(fds[0], buf, sizeof(buf));
      if (n <= 0) break;  // both processes closed stdout: exited
      text.append(buf, n);
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);  // the target follows (PR_SET_PDEATHSIG)
      timedOut = true;
      break;
    }
  }
  close(fds[0]);
  int wstatus = 0;
  waitpid(pid, &wstatus, 0);

  size_t start = 0;
  while (start < text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(start, end - start);
    if (line.rfind("RESULT ", 0) == 0) {
      out.hasResult =
          std::sscanf(
              line.c_str(), "RESULT ends=%d honest=%d recovers=%d target=%d reported=%d during=%d",
              &out.ends, &out.honest, &out.recovers, &out.target, &out.reported, &out.during) == 6;
    } else if (!line.empty()) {
      size_t topo = line.find("topology: ");
      if (topo != std::string::npos) out.topology = line.substr(topo + 10);
      std::printf("%s\n", line.c_str());
    }
    start = end + 1;
  }

  if (timedOut) {
    out.alive = false;
    out.note = "killed after " + std::to_string(kChildTimeoutMs / 1000) + " s";
  } else if (WIFSIGNALED(wstatus)) {
    out.alive = false;
    out.note = std::string("initiator crashed: ") + strsignal(WTERMSIG(wstatus));
  } else if (WEXITSTATUS(wstatus) == kExitSkip) {
    out.skipped = true;
  } else if (WEXITSTATUS(wstatus) != kExitPass && WEXITSTATUS(wstatus) != kExitFail) {
    out.alive = false;
    out.note = "initiator exited with code " + std::to_string(WEXITSTATUS(wstatus));
  } else if (out.target == 0) {
    out.alive = false;
    out.note = "target process died";
  }
  return out;
}

const char* Mark(int v) { return v < 0 ? "-" : v ? "ok" : "FAIL"; }

std::string Join(const std::set<std::string>& items) {
  std::string out;
  for (const std::string& item : items) out += (out.empty() ? "" : ",") + item;
  return out;
}

// How a run compares with what the catalog expects of the entry.
struct Verdict {
  enum Kind { Pass, XFail, Fail, XPass, Skip, kKinds } kind{Fail};
  std::string why;
};

const char* const kVerdictNames[] = {"PASS", "XFAIL", "FAIL", "XPASS", "SKIP"};

// PASS:  every check passed and none was expected to fail.
// XFAIL: exactly the `fails` checks of a known_bug entry failed.
// FAIL:  a check failed that the entry does not expect (or the entry produced no result).
// XPASS: a check a known_bug entry expects to fail passed: the bug looks fixed.
// A check that was not reached (a process died first) counts neither way.
Verdict Judge(const Entry& e, const Outcome& o) {
  if (o.skipped) return {Verdict::Skip, ""};
  if (!o.hasResult && o.alive) return {Verdict::Fail, "no result; see the output above"};
  const std::pair<const char*, int> checks[] = {{"ends", o.ends},         {"honest", o.honest},
                                                {"recovers", o.recovers}, {"alive", o.alive},
                                                {"reported", o.reported}, {"during", o.during}};
  std::set<std::string> unexpected, fixed;
  for (const auto& [name, result] : checks) {
    bool expectedToFail = e.fails.count(name) > 0;
    if (result == 0 && !expectedToFail) unexpected.insert(name);
    if (result == 1 && expectedToFail) fixed.insert(name);
  }
  if (!unexpected.empty()) {
    return {Verdict::Fail,
            "unexpected: " + Join(unexpected) + (o.note.empty() ? "" : " (" + o.note + ")")};
  }
  if (!fixed.empty()) {
    return {Verdict::XPass, "now passes: " + Join(fixed) + "; remove known_bug " + e.knownBug +
                                " from the catalog"};
  }
  if (!e.knownBug.empty()) return {Verdict::XFail, "known bug: " + e.knownBug};
  return {Verdict::Pass, ""};
}

void CheckParser() {
  for (const char* bad :
       {"no_such_kind", "cqe_drop:skip", "cqe_drop:bogus=1", "qp_error:skip=x", "async_event",
        "async_event:value=typo", "async_event:value=999", "async_event:value=srq_err",
        "async_event:value=srq_limit_reached", "async_event:value=14", "async_event:value=15",
        "port_down:count=0", "port_flap:count=2", "port_flap:value=0", "port_flap:value=-1"}) {
    bool threw = false;
    try {
      ParseFaultRule(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected a parse error for '") + bad + "'");
  }
  auto event = ParseFaultRule("async_event:value=device_fatal:op=write:skip=1");
  if (event.kind != FaultKind::AsyncEvent || event.value != IBV_EVENT_DEVICE_FATAL ||
      event.skip != 1) {
    throw std::runtime_error("async_event spec parsed wrong");
  }
  if (ParseFaultRule("port_flap").value != 200 ||
      ParseFaultRule("port_down").kind != FaultKind::PortDown) {
    throw std::runtime_error("port defaults parsed wrong");
  }
  FaultRule r = ParseFaultRule("cqe_drop:op=write:skip=3:count=2:qpn=17");
  if (r.kind != FaultKind::CqeDrop || r.wcOpcode != IBV_WC_RDMA_WRITE || r.skip != 3 ||
      r.count != 2 || r.qpn != 17) {
    throw std::runtime_error("cqe_drop spec parsed wrong");
  }
}

// No GPU or NIC access: COMM_EST needs no provider state change or port query.
// Exercise the real injector hooks with only the post_send op replaced by a no-op.
void CheckAsyncSources() {
  ibv_context ctxA{}, ctxB{};
  ctxA.ops.post_send = ctxB.ops.post_send = [](ibv_qp*, ibv_send_wr*, ibv_send_wr**) { return 0; };
  ibv_qp qpA{}, qpB{};
  qpA.context = &ctxA;
  qpB.context = &ctxB;
  qpA.qp_num = qpB.qp_num = 17;  // same number, different contexts
  ibv_send_wr wr{};
  wr.opcode = IBV_WR_SEND;
  ibv_send_wr* bad = nullptr;
  auto& injector = FaultInjector::Instance();
  int fdA = injector.WatchAsyncContext(&ctxA);
  int fdB = injector.WatchAsyncContext(&ctxB);
  auto readable = [](int fd) {
    pollfd p{fd, POLLIN, 0};
    return poll(&p, 1, 0) == 1 && (p.revents & POLLIN);
  };
  injector.Arm(ParseFaultRule("async_event:value=comm_est:op=send:count=0"));
  for (int i = 0; i < 33; ++i) injector.PostSend(&qpA, &wr, &bad);
  if (injector.Fired() != 33 || injector.Failed()) {
    throw std::runtime_error("successful async injections were not counted");
  }
  if (!readable(fdA) || readable(fdB)) throw std::runtime_error("async context isolation failed");
  for (int i = 0; i < 33; ++i) {
    ibv_async_event event{};
    if (injector.GetAsyncEvent(&ctxA, &event) != 0 || event.element.qp != &qpA ||
        event.event_type != IBV_EVENT_COMM_EST) {
      throw std::runtime_error("synthetic event was not delivered");
    }
    injector.AckAsyncEvent(&event);
    if (readable(fdA) != (i < 32)) throw std::runtime_error("async source lost level readiness");
  }
  if (qpA.events_completed != 0) throw std::runtime_error("synthetic ack reached the provider");
  injector.PostSend(&qpB, &wr, &bad);
  if (readable(fdA) || !readable(fdB)) throw std::runtime_error("second context isolation failed");
  injector.UnwatchAsyncContext(&ctxB);  // pending events must not survive address reuse
  fdB = injector.WatchAsyncContext(&ctxB);
  if (readable(fdB)) throw std::runtime_error("unwatch left a stale event");
  injector.Disarm();
  injector.UnwatchAsyncContext(&ctxA);
  injector.UnwatchAsyncContext(&ctxB);

  // A syscall failure must neither throw through a worker nor count as injected.
  fdA = injector.WatchAsyncContext(&ctxA);
  injector.Arm(ParseFaultRule("async_event:value=comm_est:op=send"));
  close(fdA);
  if (injector.PostSend(&qpA, &wr, &bad) != 0 || !injector.Failed() || injector.Fired() != 0) {
    throw std::runtime_error("failed async source was not recorded correctly");
  }
  injector.UnwatchAsyncContext(&ctxA);
  injector.Arm(ParseFaultRule("async_event:value=comm_est:op=send"));
  fdA = injector.WatchAsyncContext(&ctxA);
  injector.PostSend(&qpA, &wr, &bad);
  close(fdA);
  ibv_async_event event{};
  if (injector.GetAsyncEvent(&ctxA, &event) == 0 || !injector.Failed() || errno != EIO) {
    throw std::runtime_error("failed eventfd read was not reported without throwing");
  }
  injector.UnwatchAsyncContext(&ctxA);
  injector.Disarm();
}

// Runs `body` as a child role: line-buffered output, no teardown on the way out.
[[noreturn]] void RunRole(const std::function<int()>& body) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // keep what was printed if the process crashes
  int rc = kExitFail;
  try {
    rc = body();
  } catch (const TestSkip& ex) {
    std::printf("    skipped: %s\n", ex.what());
    rc = kExitSkip;
  } catch (const std::exception& ex) {
    std::printf("    error: %s\n", ex.what());
  }
  std::fflush(stdout);
  std::_Exit(rc);  // skip teardown: a faulted engine may never shut down
}

// Prints the results table and summary. On GitHub Actions it also emits one
// warning per known bug and one error per FAIL/XPASS, and with
// MORI_IO_FAULT_REPORT=<file> it appends a markdown table to that file (for the
// job summary). Returns the exit code: non-zero only for a FAIL or an XPASS.
int Report(const std::vector<std::pair<const Entry*, Outcome>>& results,
           const std::string& catalogPath) {
  int counts[Verdict::kKinds] = {};
  std::map<std::string, std::vector<std::string>> knownBugs;  // bug -> entries failing on it
  std::string topology, rows;
  const bool github = std::getenv("GITHUB_ACTIONS") != nullptr;

  std::printf("\n%-36s %-6s %-6s %-8s %-6s %-7s %s\n", "entry", "ends", "honest", "recovers",
              "alive", "verdict", "note");
  for (const auto& [e, o] : results) {
    Verdict v = Judge(*e, o);
    counts[v.kind]++;
    if (v.kind == Verdict::XFail) knownBugs[e->knownBug].push_back(e->id);
    if (topology.empty()) topology = o.topology;
    const char* alive = o.skipped ? "-" : o.alive ? "ok" : "FAIL";
    std::printf("%-36s %-6s %-6s %-8s %-6s %-7s %s\n", e->id.c_str(), Mark(o.ends), Mark(o.honest),
                Mark(o.recovers), alive, kVerdictNames[v.kind], v.why.c_str());
    std::printf("    reported=%s during=%s\n", Mark(o.reported), Mark(o.during));
    rows += "| " + e->id + " | " + Mark(o.ends) + " | " + Mark(o.honest) + " | " +
            Mark(o.recovers) + " | " + alive + " | " + Mark(o.reported) + " | " + Mark(o.during) +
            " | " + kVerdictNames[v.kind] + " | " + v.why + " |\n";
    if (github && (v.kind == Verdict::Fail || v.kind == Verdict::XPass)) {
      std::printf("::error title=Fault catalog %s::%s %s\n", kVerdictNames[v.kind], e->id.c_str(),
                  v.why.c_str());
    }
  }

  std::string bugs;
  for (const auto& [bug, ids] : knownBugs) {
    bugs += (bugs.empty() ? "" : ", ") + bug + " x" + std::to_string(ids.size());
    if (github) {
      std::string list;
      for (const std::string& id : ids) list += (list.empty() ? "" : ", ") + id;
      std::printf("::warning title=Known bug %s::%zu fault catalog entries expected to fail: %s\n",
                  bug.c_str(), ids.size(), list.c_str());
    }
  }
  std::string summary = std::to_string(counts[Verdict::Pass]) + " pass, " +
                        std::to_string(counts[Verdict::XFail]) + " known bugs (xfail), " +
                        std::to_string(counts[Verdict::Fail]) + " new failures, " +
                        std::to_string(counts[Verdict::XPass]) + " fixed (xpass), " +
                        std::to_string(counts[Verdict::Skip]) + " skipped";
  if (!bugs.empty()) std::printf("known bugs: %s\n", bugs.c_str());
  std::printf("==== test_fault_injection: %zu entries: %s ====\n", results.size(), summary.c_str());

  if (const char* report = std::getenv("MORI_IO_FAULT_REPORT")) {
    std::ofstream md(report, std::ios::app);
    md << "### MORI-IO fault catalog: " << catalogPath.substr(catalogPath.find_last_of('/') + 1)
       << "\n\n**" << summary << "**";
    if (!topology.empty()) md << "  \nTopology: " << topology;
    if (!bugs.empty()) md << "  \nKnown bugs: " << bugs;
    md << "\n\n| entry | ends | honest | recovers | alive | reported | during | verdict | note |\n"
       << "|---|---|---|---|---|---|---|---|---|\n"
       << rows << "\n";
  }
  return counts[Verdict::Fail] + counts[Verdict::XPass] == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Engines re-read the IO level from the environment; keep the injector's
  // "armed"/"injected" warnings visible unless the caller chose a level.
  setenv("MORI_IO_LOG_LEVEL", "warn", /*overwrite=*/0);
  SetLogLevel("warn");
  // Both ends are on one node; keep every transfer on RDMA, never GPU-to-GPU XGMI.
  setenv("MORI_DISABLE_AUTO_XGMI", "1", /*overwrite=*/1);

  const std::string path = CatalogPath();
  std::vector<Entry> catalog;
  try {
    catalog = LoadCatalog(path);
  } catch (const std::exception& ex) {
    std::printf("[FAIL] catalog: %s\n", ex.what());
    return 1;
  }
  auto find = [&](const char* id) -> const Entry* {
    for (const Entry& e : catalog) {
      if (e.id == id) return &e;
    }
    return nullptr;
  };

  if (argc == 3 && std::strcmp(argv[1], "--entry") == 0) {
    const Entry* e = find(argv[2]);
    RunRole([&] { return e != nullptr ? RunEntry(*e) : kExitFail; });
  }
  if (argc == 4 && std::strcmp(argv[1], "--target") == 0) {
    const Entry* e = find(argv[2]);
    int fd = std::atoi(argv[3]);
    RunRole([&] { return e != nullptr ? ServeTarget(*e, fd) : kExitFail; });
  }

  const std::string filter = argc > 1 ? argv[1] : "";
  try {
    CheckParser();
    std::printf("[PASS] parser rejects malformed rules\n");
  } catch (const std::exception& ex) {
    std::printf("[FAIL] parser: %s\n", ex.what());
    return 1;
  }
  std::printf("catalog: %s (%zu entries)\n", path.c_str(), catalog.size());
  if (filter == "--check-catalog") return 0;
  if (filter == "--check-async-sources") {
    try {
      CheckAsyncSources();
      std::printf(
          "[PASS] async sources: readiness, context isolation, ack, cleanup, syscall failures\n");
    } catch (const std::exception& ex) {
      std::printf("[FAIL] async sources: %s\n", ex.what());
      return 1;
    }
    return 0;
  }

  std::vector<std::pair<const Entry*, Outcome>> results;
  for (const Entry& e : catalog) {
    if (e.id.find(filter) == std::string::npos) continue;
    std::printf("---- %s: %s [%s%s]\n", e.id.c_str(), e.realWorld.c_str(),
                e.rule.empty() ? "no fault" : e.rule.c_str(),
                e.side == Side::Target ? ", in target" : "");
    std::fflush(stdout);
    results.emplace_back(&e, RunChild("/proc/self/exe", e));
  }

  return Report(results, path);
}
