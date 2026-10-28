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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mori/ras/ras.hpp"
#include "src/ras/fd_registry.hpp"
#include "src/ras/logging.hpp"

namespace mori {
namespace ras {
namespace {

std::atomic<bool> g_logEnabled{true};

// Every live core, so a forked child can orphan them. Heap-allocated and never freed, so no
// static destructor runs while another thread may still use it at exit.
std::mutex g_coresMu;
std::vector<RasCore*>& Cores() {
  static auto* cores = new std::vector<RasCore*>();
  return *cores;
}

// The process-wide runtime. Its core is never freed either.
std::mutex g_runtimeMu;
std::atomic<RasCore*> g_core{nullptr};
std::atomic<RasRuntime*> g_published{nullptr};
RasRuntime g_facade;
bool g_atexitRegistered = false;
RuntimeStatus& StatusLocked() {
  static auto* status = new RuntimeStatus();
  return *status;
}

void RuntimePrepareFork() {
  g_runtimeMu.lock();
  g_coresMu.lock();
}

void RuntimeParentFork() {
  g_coresMu.unlock();
  g_runtimeMu.unlock();
}

// Runs after the fd registry's child handler has closed every RAS fd.
void RuntimeChildFork() {
  for (RasCore* core : Cores()) core->OrphanAfterFork();
  Cores().clear();
  g_core.store(nullptr, std::memory_order_release);
  g_published.store(nullptr, std::memory_order_release);
  StatusLocked() = RuntimeStatus{};
  g_coresMu.unlock();
  g_runtimeMu.unlock();
}

void InstallRuntimeForkHandlers() {
  static std::once_flag once;
  std::call_once(once, [] {
    // The registry's handlers first: prepare handlers run in reverse order of registration and
    // child handlers in order, so the locks nest the same way as in normal operation and the
    // child closes the fds before orphaning the cores.
    fdreg::InstallForkHandlers();
    pthread_atfork(RuntimePrepareFork, RuntimeParentFork, RuntimeChildFork);
  });
}

void StopAtExit() {
  DisableLoggingForExit();
  RasCore* core;
  {
    std::lock_guard<std::mutex> lock(g_runtimeMu);
    core = g_core.load(std::memory_order_acquire);
    g_published.store(nullptr, std::memory_order_release);
    if (core != nullptr) StatusLocked().state = RunState::kStopped;
  }
  if (core != nullptr) core->Stop(/*sendLeave=*/true);
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

bool LogEnabled() { return g_logEnabled.load(std::memory_order_acquire); }
void DisableLoggingForExit() { g_logEnabled.store(false, std::memory_order_release); }

RasCore::RasCore(CoreParams params) : params_(std::move(params)) {
  InstallRuntimeForkHandlers();
  uuid_ = NewRuntimeUuid();
  token_ = NewToken();
  std::lock_guard<std::mutex> lock(g_coresMu);
  Cores().push_back(this);
}

RasCore::~RasCore() {
  if (orphaned_.load(std::memory_order_acquire)) {
    // A forked child's copy: nothing it points to may be touched, so leak it all.
    (void)queue_.release();
    (void)links_.release();
    return;
  }
  Stop(/*sendLeave=*/true);
  {
    std::lock_guard<std::mutex> lock(g_coresMu);
    auto& cores = Cores();
    cores.erase(std::remove(cores.begin(), cores.end(), this), cores.end());
  }
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
  if (stopped_.load() || orphaned_.load()) {
    *error = "the runtime was stopped";
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

  links_->SetIdentity(CollectIdentity(params_.nodeId, params_.hostname, params_.host, params_.pid,
                                      uuid_, params_.identity));
  while (!links_->StopRequested()) links_->RunOnce(1000);
  links_->Shutdown(sendLeave_.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lock(doneMu_);
    threadDone_ = true;
  }
  doneCv_.notify_all();
}

void RasCore::Stop(bool sendLeave) {
  if (orphaned_.load()) return;
  bool expected = false;
  if (!stopped_.compare_exchange_strong(expected, true)) return;
  if (!running_.load()) return;
  sendLeave_.store(sendLeave, std::memory_order_release);
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

bool RasCore::SetLabels(const Labels& labels) {
  return running_.load(std::memory_order_acquire) && links_->SetLabels(labels);
}

void RasCore::AddDegraded(uint32_t reasons) {
  if (links_) links_->AddDegraded(reasons);
}

void RasCore::OrphanAfterFork() {
  orphaned_.store(true, std::memory_order_release);
  running_.store(false, std::memory_order_release);
  if (queue_) queue_->ForgetWakeFd();
  if (links_) links_->ForgetFds();
}

RasRuntime* RasRuntime::Get() { return g_published.load(std::memory_order_acquire); }

RasRuntime* RasRuntime::EnsureStarted(const StartParams& params) {
  if (RasRuntime* r = Get()) return r;
  InstallRuntimeForkHandlers();
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
    RAS_LOG_DEBUG("RAS is off for this process: {}", error);
    delete core;
    return nullptr;
  }
  g_core.store(core, std::memory_order_release);
  status = RuntimeStatus{RunState::kRunning, 0, ""};
  if (!g_atexitRegistered) {
    std::atexit(StopAtExit);
    g_atexitRegistered = true;
  }
  RasEndpoint ep = core->Endpoint();
  RAS_LOG_DEBUG("RAS runtime {} listening on {}:{}", ep.uuid.ToString(), ep.host, ep.port);
  g_published.store(&g_facade, std::memory_order_release);
  return &g_facade;
}

RuntimeStatus RasRuntime::Status() {
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

bool RasRuntime::SetLabels(const Labels& labels) {
  RasCore* core = g_core.load(std::memory_order_acquire);
  return core != nullptr && core->SetLabels(labels);
}

}  // namespace ras
}  // namespace mori
