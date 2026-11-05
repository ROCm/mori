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
#include "src/ras/runtime.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "mori/ras/ras.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/ras/logging.hpp"

namespace mori {
namespace ras {
namespace {

std::atomic<bool> g_logEnabled{true};

// The process-wide runtime. Its core is never freed.
std::mutex g_runtimeMu;
std::atomic<RasCore*> g_core{nullptr};
std::atomic<RasRuntime*> g_published{nullptr};
RasRuntime g_facade;
bool g_atexitRegistered = false;
RuntimeStatus& StatusLocked() {
  static auto* status = new RuntimeStatus();
  return *status;
}

// The process that first asked for the runtime. RAS doesn't follow a fork: in a child of that
// process, the runtime is off. Checked before any lock, because a lock held at fork time stays
// held in the child, which has no thread to release it.
std::atomic<pid_t> g_ownerPid{0};
bool InForkedChild() {
  pid_t owner = g_ownerPid.load(std::memory_order_acquire);
  return owner != 0 && owner != getpid();
}

void StopAtExit() {
  // A forked child inherits this handler, but not the RAS thread it would stop.
  if (InForkedChild()) return;
  DisableLoggingForExit();
  RasCore* core;
  {
    std::lock_guard<std::mutex> lock(g_runtimeMu);
    core = g_core.load(std::memory_order_acquire);
    g_published.store(nullptr, std::memory_order_release);
    if (core != nullptr) StatusLocked().state = RunState::kStopped;
  }
  if (core != nullptr) core->Stop();
}

// The first IPv4 address of `ifname`, else its first global IPv6 address.
std::string InterfaceAddress(const std::string& ifname) {
  ifaddrs* list = nullptr;
  if (getifaddrs(&list) != 0) return "";
  std::string v4, v6;
  for (ifaddrs* ifa = list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifname != ifa->ifa_name) continue;
    char buf[INET6_ADDRSTRLEN] = {0};
    if (ifa->ifa_addr->sa_family == AF_INET && v4.empty()) {
      inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr, buf,
                sizeof(buf));
      v4 = buf;
    } else if (ifa->ifa_addr->sa_family == AF_INET6 && v6.empty()) {
      const in6_addr& a = reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr)->sin6_addr;
      if (IN6_IS_ADDR_LINKLOCAL(&a)) continue;
      inet_ntop(AF_INET6, &a, buf, sizeof(buf));
      v6 = buf;
    }
  }
  freeifaddrs(list);
  return v4.empty() ? v6 : v4;
}

}  // namespace

spdlog::logger* RasLogger() {
  if (!g_logEnabled.load(std::memory_order_acquire)) return nullptr;
  // A pointer, deliberately leaked: no destructor runs at exit, so the logger outlives
  // spdlog's registry and the ModuleLogger that hand it out.
  static auto* logger =
      new std::shared_ptr<spdlog::logger>(ModuleLogger::GetInstance().GetLogger(modules::RAS));
  return logger->get();
}

void DisableLoggingForExit() { g_logEnabled.store(false, std::memory_order_release); }

RasCore::RasCore(CoreParams params) : params_(std::move(params)) {
  uuid_ = NewRuntimeUuid();
  token_ = NewToken();
}

RasCore::~RasCore() {
  Stop();
  if (thread_ != nullptr) {
    // Abandoned by Stop: it may still be running, so it keeps the link manager and queue.
    (void)queue_.release();
    (void)links_.release();
  }
}

bool RasCore::Start(std::string* error) {
  std::string scratch;
  if (error == nullptr) error = &scratch;
  if (running_.load()) return true;
  if (stopped_.load()) {
    *error = "the runtime was stopped";
    return false;
  }
  if (uuid_.IsZero() || token_.empty()) {
    *error = "getrandom gave no random bytes for the runtime UUID and token";
    return false;
  }
  std::string bindHost = params_.host;
  advertisedHost_ = params_.host;
  if (!params_.config.ifname.empty()) {
    bindHost = InterfaceAddress(params_.config.ifname);
    if (bindHost.empty()) {
      *error = "MORI_RAS_IFNAME=" + params_.config.ifname + " has no usable address";
      return false;
    }
    advertisedHost_ = bindHost;
  } else if (bindHost == "0.0.0.0") {
    bindHost.clear();
  }

  auto queue = std::make_unique<EventQueue>(params_.eventCapacity);
  if (!queue->HasWakeFd()) {
    *error = std::string("eventfd: ") + std::strerror(errno);
    return false;
  }
  auto links = std::make_unique<LinkManager>(
      LinkManager::Options{params_.config, uuid_, token_, queue.get()});
  if (!links->Listen(bindHost, error) || !links->InitLoop(error)) return false;
  queue_ = std::move(queue);
  links_ = std::move(links);
  try {
    thread_ = new std::thread([this] { ThreadMain(); });
  } catch (const std::exception& e) {
    *error = std::string("the RAS thread didn't start: ") + e.what();
    links_.reset();
    queue_.reset();
    return false;
  }
  running_.store(true, std::memory_order_release);
  return true;
}

void RasCore::ThreadMain() {
  // Priority and affinity before the name, so a thread visible as mori-ras is configured.
  setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 19);
  if (!params_.config.cpuAffinity.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : params_.config.cpuAffinity) {
      if (cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    }
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
      RAS_LOG_WARN("MORI_RAS_CPU_AFFINITY can't be applied ({}); the RAS thread isn't pinned",
                   std::strerror(rc));
    }
  }
  pthread_setname_np(pthread_self(), "mori-ras");

  links_->SetIdentity(
      CollectIdentity(params_.nodeId, params_.hostname, params_.host, params_.pid, uuid_));
  while (!links_->StopRequested()) links_->RunOnce(1000);
  links_->Shutdown();
  {
    std::lock_guard<std::mutex> lock(doneMu_);
    threadDone_ = true;
  }
  doneCv_.notify_all();
}

void RasCore::Stop() {
  bool expected = false;
  if (!stopped_.compare_exchange_strong(expected, true)) return;
  if (!running_.load()) return;
  links_->RequestStop();
  bool done;
  {
    std::unique_lock<std::mutex> lock(doneMu_);
    done = doneCv_.wait_for(lock, std::chrono::seconds(2), [this] { return threadDone_; });
  }
  if (done) {
    thread_->join();
    delete thread_;
    thread_ = nullptr;
  } else {
    thread_->detach();  // still running; ~RasCore leaks what it uses
  }
  running_.store(false, std::memory_order_release);
}

bool RasCore::Enqueue(RasEvent&& ev) {
  if (!running_.load(std::memory_order_acquire) || stopped_.load(std::memory_order_acquire)) {
    return false;
  }
  return queue_->Enqueue(std::move(ev));
}

RasEndpoint RasCore::Endpoint() const {
  if (!running_.load(std::memory_order_acquire)) return RasEndpoint{};
  return RasEndpoint{advertisedHost_, links_->Port(), token_, uuid_};
}

RuntimeSnapshot RasCore::Snapshot() const {
  return links_ ? links_->Snapshot() : RuntimeSnapshot{};
}

void RasCore::AddDegraded(uint32_t reasons) {
  if (links_) links_->AddDegraded(reasons);
}

RasRuntime* RasRuntime::Get() {
  if (InForkedChild()) return nullptr;
  return g_published.load(std::memory_order_acquire);
}

RasRuntime* RasRuntime::EnsureStarted(const StartParams& params) {
  if (RasRuntime* r = Get()) return r;
  if (InForkedChild()) return nullptr;
  // Set before taking the lock, so a child forked while this thread holds it sees RAS as off
  // instead of waiting on it forever.
  g_ownerPid.store(getpid(), std::memory_order_release);
  std::lock_guard<std::mutex> lock(g_runtimeMu);
  RuntimeStatus& status = StatusLocked();
  if (status.state == RunState::kRunning) return &g_facade;
  // Disabled, failed, and stopped all stick for the life of the process.
  if (status.state != RunState::kNotStarted) return nullptr;

  RasConfig config = ConfigFromEnv();
  for (const std::string& w : config.warnings) RAS_LOG_WARN("{}", w);
  if (!config.enable) {
    status = RuntimeStatus{RunState::kDisabled, 0, "MORI_RAS_ENABLE is not set"};
    return nullptr;
  }
  CoreParams cp;
  cp.config = config;
  cp.nodeId = params.nodeId;
  cp.hostname = params.hostname;
  cp.host = params.host;
  cp.pid = params.pid;
  auto* core = new RasCore(std::move(cp));
  std::string error;
  if (!core->Start(&error)) {
    status = RuntimeStatus{RunState::kFailed, kDegradedInitFailed, error};
    RAS_LOG_WARN("RAS is off for this process: {}", error);
    delete core;
    return nullptr;
  }
  g_core.store(core, std::memory_order_release);
  status = RuntimeStatus{RunState::kRunning, 0, ""};
  if (!g_atexitRegistered) {
    // Exit runs atexit handlers and static destructors in reverse order of registration, so
    // building the logging statics first makes StopAtExit run before they are destroyed.
    (void)RasLogger();
    std::atexit(StopAtExit);
    g_atexitRegistered = true;
  }
  RasEndpoint ep = core->Endpoint();
  RAS_LOG_INFO("RAS runtime {} listening on {}:{}", ep.uuid.ToString(), ep.host, ep.port);
  g_published.store(&g_facade, std::memory_order_release);
  return &g_facade;
}

RuntimeStatus RasRuntime::Status() {
  if (InForkedChild()) return RuntimeStatus{RunState::kDisabled, 0, "RAS doesn't follow a fork"};
  std::lock_guard<std::mutex> lock(g_runtimeMu);
  RuntimeStatus status = StatusLocked();
  if (RasCore* core = g_core.load(std::memory_order_acquire)) {
    status.degraded |= core->Snapshot().degraded;
  }
  return status;
}

RasEndpoint RasRuntime::Endpoint() const {
  RasCore* core = g_core.load(std::memory_order_acquire);
  return core != nullptr ? core->Endpoint() : RasEndpoint{};
}

bool RasRuntime::Enqueue(RasEvent ev) {
  RasCore* core = g_core.load(std::memory_order_acquire);
  return core != nullptr && core->Enqueue(std::move(ev));
}

RuntimeSnapshot RasRuntime::Snapshot() const {
  RasCore* core = g_core.load(std::memory_order_acquire);
  return core != nullptr ? core->Snapshot() : RuntimeSnapshot{};
}

}  // namespace ras
}  // namespace mori
