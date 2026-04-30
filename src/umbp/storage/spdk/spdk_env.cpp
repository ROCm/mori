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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SPDK environment — migrated to umbp namespace.

#include "umbp/storage/spdk/spdk_env.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "umbp/common/log.h"

extern "C" {
#include "spdk/bdev.h"
#include "spdk/cpuset.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/log.h"
#include "spdk/thread.h"
}

#define SPDK_THREAD(p) (static_cast<struct spdk_thread*>(p))
#define SPDK_DESC(p) (static_cast<struct spdk_bdev_desc*>(p))
#define SPDK_CHAN(p) (static_cast<struct spdk_io_channel*>(p))

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------
void umbp_bdev_init_complete_cb(void*, int) {}

static void bdev_event_cb(enum spdk_bdev_event_type type, struct spdk_bdev*, void*) {
  UMBP_LOG_INFO("SpdkEnv: bdev event type=%d", static_cast<int>(type));
}

static void submit_single_io(umbp::SpdkIoRequest* req, void* bdev_desc) {
  spdk_bdev_io_completion_cb done = [](struct spdk_bdev_io* bio, bool ok, void* arg) {
    auto* r = static_cast<umbp::SpdkIoRequest*>(arg);
    spdk_bdev_free_io(bio);
    r->success = ok;
    if (ok && r->op == umbp::SpdkIoRequest::READ && r->dst_iov && r->dst_iovcnt > 0) {
      const char* src = static_cast<const char*>(r->buf) + r->dst_skip;
      for (int i = 0; i < r->dst_iovcnt; ++i) {
        std::memcpy(r->dst_iov[i].iov_base, src, r->dst_iov[i].iov_len);
        src += r->dst_iov[i].iov_len;
      }
    }
    r->completed.store(true, std::memory_order_release);
  };

  if (req->op == umbp::SpdkIoRequest::WRITE) {
    if (req->src_iov && req->src_iovcnt > 0) {
      char* dst = static_cast<char*>(req->buf);
      size_t copied = 0;
      for (int i = 0; i < req->src_iovcnt; ++i) {
        std::memcpy(dst, req->src_iov[i].iov_base, req->src_iov[i].iov_len);
        dst += req->src_iov[i].iov_len;
        copied += req->src_iov[i].iov_len;
      }
      if (req->nbytes > copied) std::memset(dst, 0, req->nbytes - copied);
    } else if (req->src_data) {
      std::memcpy(req->buf, req->src_data, req->src_len);
      if (req->nbytes > req->src_len)
        std::memset(static_cast<char*>(req->buf) + req->src_len, 0, req->nbytes - req->src_len);
    }
  }

  auto* ch = SPDK_CHAN(req->_io_channel);
  int rc;
  if (req->op == umbp::SpdkIoRequest::WRITE) {
    rc = spdk_bdev_write(SPDK_DESC(bdev_desc), ch, req->buf, req->offset, req->nbytes, done, req);
  } else {
    rc = spdk_bdev_read(SPDK_DESC(bdev_desc), ch, req->buf, req->offset, req->nbytes, done, req);
  }
  if (rc != 0) {
    UMBP_LOG_ERROR("SpdkEnv: bdev I/O submit failed rc=%d", rc);
    req->success = false;
    req->completed.store(true, std::memory_order_release);
  }
}

void umbp_execute_io_cb(void* ctx) {
  auto& env = umbp::SpdkEnv::Instance();
  submit_single_io(static_cast<umbp::SpdkIoRequest*>(ctx), env.bdev_desc_);
}

void umbp_execute_io_batch_cb(void* ctx) {
  auto& env = umbp::SpdkEnv::Instance();
  auto* req = static_cast<umbp::SpdkIoRequest*>(ctx);

  static constexpr size_t kYieldAfterWriteBytes = 4ULL * 1024 * 1024;
  size_t write_bytes = 0;

  while (req) {
    auto* next = req->_next_batch;

    bool has_write_copy = (req->op == umbp::SpdkIoRequest::WRITE) &&
                          (req->src_data || (req->src_iov && req->src_iovcnt > 0));

    submit_single_io(req, env.bdev_desc_);

    if (has_write_copy) {
      write_bytes += req->nbytes;
      if (write_bytes >= kYieldAfterWriteBytes && next) {
        spdk_thread_send_msg(spdk_get_thread(), umbp_execute_io_batch_cb, next);
        return;
      }
    }
    req = next;
  }
}

void umbp_init_reactor_cb(void* arg1, void* arg2) {
  auto* env = static_cast<umbp::SpdkEnv*>(arg1);
  int idx = static_cast<int>(reinterpret_cast<intptr_t>(arg2));

  struct spdk_cpuset cpumask;
  spdk_cpuset_zero(&cpumask);
  spdk_cpuset_set_cpu(&cpumask, spdk_env_get_current_core(), true);

  char name[32];
  std::snprintf(name, sizeof(name), "umbp_io_%d", idx);
  struct spdk_thread* th = spdk_thread_create(name, &cpumask);
  if (!th) {
    UMBP_LOG_ERROR("SpdkEnv: spdk_thread_create failed for reactor %d", idx);
    umbp_signal_init_done(env, -1);
    return;
  }
  spdk_set_thread(th);

  auto* ch = spdk_bdev_get_io_channel(SPDK_DESC(env->bdev_desc_));
  if (!ch) {
    UMBP_LOG_ERROR("SpdkEnv: io_channel failed for reactor %d", idx);
    umbp_signal_init_done(env, -1);
    return;
  }

  env->reactors_[idx].spdk_thread = th;
  env->reactors_[idx].io_channel = ch;
  env->reactors_[idx].core_id = spdk_env_get_current_core();

  UMBP_LOG_INFO("SpdkEnv: reactor %d ready on core %u", idx, spdk_env_get_current_core());

  int ready = env->reactors_ready_.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (ready == env->num_reactors_) {
    umbp_signal_init_done(env, 0);
  }
}

void umbp_cleanup_reactor_cb(void* arg1, void* arg2) {
  auto* env = static_cast<umbp::SpdkEnv*>(arg1);
  int idx = static_cast<int>(reinterpret_cast<intptr_t>(arg2));

  auto& r = env->reactors_[idx];
  if (r.io_channel) {
    spdk_set_thread(SPDK_THREAD(r.spdk_thread));
    spdk_put_io_channel(SPDK_CHAN(r.io_channel));
    r.io_channel = nullptr;
  }
  if (idx > 0 && r.spdk_thread) {
    spdk_set_thread(SPDK_THREAD(r.spdk_thread));
    spdk_thread_exit(SPDK_THREAD(r.spdk_thread));
    r.spdk_thread = nullptr;
  }

  int done = env->reactors_ready_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (done == 0) {
    if (env->bdev_desc_) {
      spdk_bdev_close(SPDK_DESC(env->bdev_desc_));
      env->bdev_desc_ = nullptr;
    }
    spdk_app_stop(0);
  }
}

void umbp_signal_init_done(umbp::SpdkEnv* env, int rc) {
  env->init_result_ = rc;
  {
    std::lock_guard<std::mutex> lk(env->init_mutex_);
    env->init_complete_ = true;
  }
  env->init_cv_.notify_one();
  if (rc != 0) spdk_app_stop(-1);
}

void umbp_app_start_cb(void* ctx) {
  auto* env = static_cast<umbp::SpdkEnv*>(ctx);
  auto& cfg = env->config_;

  struct spdk_bdev_desc* desc = nullptr;
  int rc = spdk_bdev_open_ext(cfg.bdev_name.c_str(), true, bdev_event_cb, nullptr, &desc);
  if (rc != 0) {
    UMBP_LOG_ERROR("SpdkEnv: spdk_bdev_open_ext('%s') failed rc=%d", cfg.bdev_name.c_str(), rc);
    umbp_signal_init_done(env, rc);
    return;
  }
  env->bdev_desc_ = desc;

  struct spdk_bdev* bd = spdk_bdev_desc_get_bdev(desc);
  env->bdev_ = bd;
  env->block_size_ = spdk_bdev_get_block_size(bd);
  env->bdev_size_ = static_cast<uint64_t>(spdk_bdev_get_num_blocks(bd)) * env->block_size_;

  UMBP_LOG_INFO("SpdkEnv: bdev '%s' opened — block_size=%u total_size=%lu", cfg.bdev_name.c_str(),
                env->block_size_, static_cast<unsigned long>(env->bdev_size_));

  std::vector<uint32_t> cores;
  uint32_t core;
  SPDK_ENV_FOREACH_CORE(core) { cores.push_back(core); }
  env->num_reactors_ = static_cast<int>(cores.size());
  env->reactors_.resize(env->num_reactors_);
  env->reactors_ready_.store(0, std::memory_order_relaxed);

  UMBP_LOG_INFO("SpdkEnv: %d reactor core(s) available", env->num_reactors_);

  if (env->num_reactors_ == 0) {
    umbp_signal_init_done(env, -1);
    return;
  }

  {
    struct spdk_io_channel* ch = spdk_bdev_get_io_channel(desc);
    if (!ch) {
      UMBP_LOG_ERROR("SpdkEnv: io_channel failed for master reactor");
      umbp_signal_init_done(env, -1);
      return;
    }
    env->reactors_[0].spdk_thread = spdk_get_thread();
    env->reactors_[0].io_channel = ch;
    env->reactors_[0].core_id = cores[0];

    UMBP_LOG_INFO("SpdkEnv: reactor 0 ready on core %u", cores[0]);

    int ready = env->reactors_ready_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (ready == env->num_reactors_) {
      umbp_signal_init_done(env, 0);
      return;
    }
  }

  for (int i = 1; i < env->num_reactors_; ++i) {
    struct spdk_event* ev = spdk_event_allocate(cores[i], umbp_init_reactor_cb, env,
                                                reinterpret_cast<void*>(static_cast<intptr_t>(i)));
    spdk_event_call(ev);
  }
}

void umbp_open_bdev_cb(void*) { /* unused, kept for link compatibility */ }

// ===========================================================================
namespace umbp {

SpdkEnv& SpdkEnv::Instance() {
  static SpdkEnv instance;
  return instance;
}

SpdkEnv::~SpdkEnv() {
  if (initialized_.load(std::memory_order_acquire)) {
    Shutdown();
  }
}

int SpdkEnv::Init(const SpdkEnvConfig& config) {
  if (initialized_.load(std::memory_order_acquire)) {
    UMBP_LOG_WARN("SpdkEnv: already initialized");
    return 0;
  }

  config_ = config;

  // Parse comma-separated PCI addresses for multi-disk support.
  std::vector<std::string> pci_addrs;
  if (!config_.nvme_pci_addr.empty()) {
    std::istringstream ss(config_.nvme_pci_addr);
    std::string addr;
    while (std::getline(ss, addr, ',')) {
      if (!addr.empty()) pci_addrs.push_back(addr);
    }
  }

  bool use_raid = pci_addrs.size() > 1;
  if (config_.bdev_name.empty()) {
    if (use_raid)
      config_.bdev_name = "Raid0";
    else if (!pci_addrs.empty())
      config_.bdev_name = config_.nvme_ctrl_name + "n1";
  }

  init_complete_ = false;
  init_result_ = -1;

  std::string json_path;
  if (config_.use_malloc_bdev) {
    json_path = "/tmp/umbp_spdk_bdev.json";
    FILE* f = fopen(json_path.c_str(), "w");
    if (f) {
      fprintf(f,
              "{\n"
              "  \"subsystems\": [{\n"
              "    \"subsystem\": \"bdev\",\n"
              "    \"config\": [{\n"
              "      \"method\": \"bdev_malloc_create\",\n"
              "      \"params\": {\n"
              "        \"name\": \"%s\",\n"
              "        \"num_blocks\": %" PRIu64
              ",\n"
              "        \"block_size\": %u\n"
              "      }\n"
              "    }]\n"
              "  }]\n"
              "}\n",
              config_.bdev_name.c_str(), config_.malloc_num_blocks, config_.malloc_block_size);
      fclose(f);
    }
  } else if (!pci_addrs.empty()) {
    json_path = "/tmp/umbp_spdk_nvme_bdev.json";
    FILE* f = fopen(json_path.c_str(), "w");
    if (f) {
      fprintf(f, "{\n  \"subsystems\": [{\n    \"subsystem\": \"bdev\",\n    \"config\": [\n");

      // Attach each NVMe controller.
      for (size_t i = 0; i < pci_addrs.size(); ++i) {
        std::string ctrl = "NVMe" + std::to_string(i);
        fprintf(f,
                "      {\n"
                "        \"method\": \"bdev_nvme_attach_controller\",\n"
                "        \"params\": {\n"
                "          \"name\": \"%s\",\n"
                "          \"trtype\": \"PCIe\",\n"
                "          \"traddr\": \"%s\"\n"
                "        }\n"
                "      }",
                ctrl.c_str(), pci_addrs[i].c_str());
        fprintf(f, ",\n");
        UMBP_LOG_INFO("SpdkEnv: NVMe bdev JSON — ctrl=%s traddr=%s", ctrl.c_str(),
                      pci_addrs[i].c_str());
      }

      if (use_raid) {
        // UMBP_SPDK_RAID_STRIP_KB: strip size in KB (default 128).
        // 128KB balances per-disk parallelism (both disks active per I/O)
        // with manageable sub-I/O count (16 per 2MB DMA chunk).
        const char* strip_env = std::getenv("UMBP_SPDK_RAID_STRIP_KB");
        int strip_kb = strip_env ? std::atoi(strip_env) : 128;
        if (strip_kb < 4) strip_kb = 4;

        fprintf(f,
                "      {\n"
                "        \"method\": \"bdev_raid_create\",\n"
                "        \"params\": {\n"
                "          \"name\": \"Raid0\",\n"
                "          \"strip_size_kb\": %d,\n"
                "          \"raid_level\": \"raid0\",\n"
                "          \"base_bdevs\": [",
                strip_kb);
        for (size_t i = 0; i < pci_addrs.size(); ++i) {
          if (i > 0) fprintf(f, ", ");
          fprintf(f, "\"NVMe%zun1\"", i);
        }
        fprintf(f,
                "]\n"
                "        }\n"
                "      }\n");
        UMBP_LOG_INFO("SpdkEnv: RAID0 bdev — %zu disks, strip_size=%dKB", pci_addrs.size(),
                      strip_kb);
      } else {
        // Remove trailing comma for single-disk case: rewind past ",\n"
        fseek(f, -2, SEEK_CUR);
        fprintf(f, "\n");
      }

      fprintf(f, "    ]\n  }]\n}\n");
      fclose(f);
    }
  }

  reactor_thread_ = std::thread([this, json_path]() {
    struct spdk_app_opts opts = {};
    spdk_app_opts_init(&opts, sizeof(opts));
    opts.name = config_.name.c_str();
    opts.shutdown_cb = nullptr;
    opts.reactor_mask = config_.reactor_mask.c_str();
    if (config_.mem_size_mb > 0) {
      opts.mem_size = config_.mem_size_mb;
    }
    if (!json_path.empty()) {
      opts.json_config_file = json_path.c_str();
    }

    int rc = spdk_app_start(&opts, umbp_app_start_cb, this);
    if (rc != 0) {
      UMBP_LOG_ERROR("SpdkEnv: spdk_app_start returned rc=%d", rc);
    }
    spdk_app_fini();

    if (!init_complete_) {
      std::lock_guard<std::mutex> lk(init_mutex_);
      init_result_ = rc;
      init_complete_ = true;
      init_cv_.notify_one();
    }
  });

  {
    std::unique_lock<std::mutex> lk(init_mutex_);
    init_cv_.wait(lk, [this] { return init_complete_; });
  }

  if (init_result_ != 0) {
    UMBP_LOG_ERROR("SpdkEnv: init failed rc=%d", init_result_);
    if (reactor_thread_.joinable()) reactor_thread_.join();
    return init_result_;
  }

  initialized_.store(true, std::memory_order_release);
  UMBP_LOG_INFO("SpdkEnv: initialization complete — %d reactor(s)", num_reactors_);
  return 0;
}

void SpdkEnv::CleanupOnSpdkThread(void* ctx) {
  auto* self = static_cast<SpdkEnv*>(ctx);
  if (self->bdev_desc_) {
    spdk_bdev_close(SPDK_DESC(self->bdev_desc_));
    self->bdev_desc_ = nullptr;
  }
  spdk_app_stop(0);
}

void SpdkEnv::Shutdown() {
  if (!initialized_.load(std::memory_order_acquire)) return;
  initialized_.store(false, std::memory_order_release);

  std::vector<uint32_t> cores(num_reactors_);
  for (int i = 0; i < num_reactors_; ++i) cores[i] = reactors_[i].core_id;

  reactors_ready_.store(num_reactors_, std::memory_order_release);

  for (int i = 0; i < num_reactors_; ++i) {
    struct spdk_event* ev = spdk_event_allocate(cores[i], umbp_cleanup_reactor_cb, this,
                                                reinterpret_cast<void*>(static_cast<intptr_t>(i)));
    spdk_event_call(ev);
  }

  if (reactor_thread_.joinable()) reactor_thread_.join();

  {
    std::lock_guard<std::mutex> lk(dma_pool_mutex_);
    for (auto& e : dma_pool_) spdk_dma_free(e.buf);
    dma_pool_.clear();
  }

  UMBP_LOG_INFO("SpdkEnv: shutdown complete");
}

void SpdkEnv::CleanupThreadLocalCtx() { /* no per-thread state */ }

// ---------------------------------------------------------------------------
// I/O submission — per-IO round-robin across reactors.
// ---------------------------------------------------------------------------
int SpdkEnv::SubmitIoAsync(SpdkIoRequest* req) {
  if (!initialized_.load(std::memory_order_acquire) || num_reactors_ == 0) return -EINVAL;

  int rid = static_cast<int>(next_reactor_.fetch_add(1, std::memory_order_relaxed) %
                             static_cast<uint64_t>(num_reactors_));
  auto& r = reactors_[rid];

  req->_io_channel = r.io_channel;
  req->completed.store(false, std::memory_order_release);
  req->success = false;

  return spdk_thread_send_msg(SPDK_THREAD(r.spdk_thread), umbp_execute_io_cb, req);
}

int SpdkEnv::SubmitIoBatchAsync(SpdkIoRequest** reqs, int count) {
  if (!initialized_.load(std::memory_order_acquire) || num_reactors_ == 0) return -EINVAL;
  if (count <= 0) return 0;

  constexpr int kMaxReactors = 64;
  SpdkIoRequest* heads[kMaxReactors] = {};
  SpdkIoRequest* tails[kMaxReactors] = {};

  uint64_t base = next_reactor_.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);

  for (int i = 0; i < count; ++i) {
    int rid =
        static_cast<int>((base + static_cast<uint64_t>(i)) % static_cast<uint64_t>(num_reactors_));
    auto* req = reqs[i];
    req->_io_channel = reactors_[rid].io_channel;
    req->completed.store(false, std::memory_order_release);
    req->success = false;
    req->_next_batch = nullptr;

    if (!heads[rid]) {
      heads[rid] = tails[rid] = req;
    } else {
      tails[rid]->_next_batch = req;
      tails[rid] = req;
    }
  }

  for (int r = 0; r < num_reactors_; ++r) {
    if (!heads[r]) continue;
    spdk_thread_send_msg(SPDK_THREAD(reactors_[r].spdk_thread), umbp_execute_io_batch_cb, heads[r]);
  }
  return 0;
}

int SpdkEnv::PollIo() { return 0; }

void SpdkEnv::SubmitIo(SpdkIoRequest* req) {
  int rc = SubmitIoAsync(req);
  if (rc != 0) {
    req->success = false;
    req->completed.store(true, std::memory_order_release);
    return;
  }
  while (!req->completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
  }
}

// ---------------------------------------------------------------------------
// Reactor accessors
// ---------------------------------------------------------------------------
void* SpdkEnv::GetReactorChannel(int idx) const {
  if (idx < 0 || idx >= num_reactors_) return nullptr;
  return reactors_[idx].io_channel;
}

void* SpdkEnv::GetReactorThread(int idx) const {
  if (idx < 0 || idx >= num_reactors_) return nullptr;
  return reactors_[idx].spdk_thread;
}

void SpdkEnv::SendMsgToReactor(int idx, void (*fn)(void*), void* arg) {
  if (idx < 0 || idx >= num_reactors_) return;
  spdk_thread_send_msg(SPDK_THREAD(reactors_[idx].spdk_thread), fn, arg);
}

// ---------------------------------------------------------------------------
// DMA buffer helpers
// ---------------------------------------------------------------------------
void* SpdkEnv::DmaMalloc(size_t size, size_t align) {
  return spdk_dma_malloc(size, align, nullptr);
}

void SpdkEnv::DmaFree(void* buf) { spdk_dma_free(buf); }

void* SpdkEnv::DmaPoolAlloc(size_t needed, size_t align) {
  {
    std::lock_guard<std::mutex> lk(dma_pool_mutex_);
    int best = -1;
    size_t best_size = SIZE_MAX;
    for (int i = 0; i < static_cast<int>(dma_pool_.size()); ++i) {
      if (dma_pool_[i].size >= needed && dma_pool_[i].size < best_size) {
        best = i;
        best_size = dma_pool_[i].size;
      }
    }
    if (best >= 0) {
      void* buf = dma_pool_[best].buf;
      dma_pool_[best] = dma_pool_.back();
      dma_pool_.pop_back();
      return buf;
    }
  }
  return spdk_dma_malloc(needed, align, nullptr);
}

void SpdkEnv::DmaPoolFree(void* buf, size_t size) {
  std::lock_guard<std::mutex> lk(dma_pool_mutex_);
  dma_pool_.push_back({buf, size});
}

int SpdkEnv::DmaPoolAllocBatch(void** out_bufs, size_t needed, int count, size_t align) {
  int got = 0;
  {
    std::lock_guard<std::mutex> lk(dma_pool_mutex_);
    for (int i = static_cast<int>(dma_pool_.size()) - 1; i >= 0 && got < count; --i) {
      if (dma_pool_[i].size >= needed) {
        out_bufs[got++] = dma_pool_[i].buf;
        dma_pool_[i] = dma_pool_.back();
        dma_pool_.pop_back();
      }
    }
  }
  for (int i = got; i < count; ++i) {
    out_bufs[i] = spdk_dma_malloc(needed, align, nullptr);
    if (!out_bufs[i]) {
      UMBP_LOG_ERROR("DmaPoolAllocBatch FAIL: malloc_fail_at=%d needed=%zu count=%d", i, needed,
                     count);
      return i;
    }
    ++got;
  }
  return got;
}

void SpdkEnv::DmaPoolFreeBatch(void* const* bufs, size_t size, int count) {
  std::lock_guard<std::mutex> lk(dma_pool_mutex_);
  dma_pool_.reserve(dma_pool_.size() + count);
  for (int i = 0; i < count; ++i) dma_pool_.push_back({bufs[i], size});
}

void SpdkEnv::DmaPoolPrewarm(size_t buf_size, int count, size_t align) {
  auto bufs = std::make_unique<void*[]>(count);
  int ok = 0;
  for (int i = 0; i < count; ++i) {
    bufs[i] = spdk_dma_malloc(buf_size, align, nullptr);
    if (!bufs[i]) break;
    ++ok;
  }
  if (ok > 0) DmaPoolFreeBatch(bufs.get(), buf_size, ok);
  if (ok < count) {
    UMBP_LOG_ERROR("SpdkEnv: DMA pool pre-warm PARTIAL %d/%d × %zu bytes", ok, count, buf_size);
  } else {
    UMBP_LOG_INFO("SpdkEnv: DMA pool pre-warmed %d × %zu bytes", ok, buf_size);
  }
}

void SpdkEnv::DmaPoolDrain() {
  std::lock_guard<std::mutex> lk(dma_pool_mutex_);
  for (auto& e : dma_pool_) {
    spdk_dma_free(e.buf);
  }
  dma_pool_.clear();
}

}  // namespace umbp
