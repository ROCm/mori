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

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"

namespace mori {
namespace io {

class IOEngineConfig;

/* ---------------------------------------------------------------------------------------------- */
/*                                          BackendConfig                                         */
/* ---------------------------------------------------------------------------------------------- */
struct BackendConfig {
  BackendConfig(BackendType t) : type(t) {}
  ~BackendConfig() = default;

  BackendType Type() const { return type; }

 private:
  BackendType type;
};

struct RdmaBackendConfig : public BackendConfig {
  RdmaBackendConfig() : BackendConfig(BackendType::RDMA) {}
  RdmaBackendConfig(int qpPerTransfer_, int postBatchSize_, int numWorkerThreads_,
                    PollCqMode pollCqMode_)
      : BackendConfig(BackendType::RDMA),
        qpPerTransfer(qpPerTransfer_),
        postBatchSize(postBatchSize_),
        numWorkerThreads(numWorkerThreads_),
        pollCqMode(pollCqMode_) {}

  int qpPerTransfer{1};
  int postBatchSize{-1};
  int numWorkerThreads{1};
  PollCqMode pollCqMode{PollCqMode::POLLING};
};

inline std::ostream& operator<<(std::ostream& os, const RdmaBackendConfig& c) {
  return os << "qpPerTransfer[" << c.qpPerTransfer << "] postBatchSize[" << c.postBatchSize
            << "] numWorkerThreads[" << c.numWorkerThreads << "]";
}

struct TcpBackendConfig : public BackendConfig {
  TcpBackendConfig() : BackendConfig(BackendType::TCP) {}
  // For future extension (e.g., parallelism, buffer sizing)
  int numWorkerThreads{4};
  bool preconnect{true};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         BackendSession                                         */
/* ---------------------------------------------------------------------------------------------- */
class BackendSession {
 public:
  BackendSession() = default;
  virtual ~BackendSession() = default;

  virtual void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                         TransferStatus* status, TransferUniqueId id, bool isRead) = 0;
  inline void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) {
    ReadWrite(localOffset, remoteOffset, size, status, id, false);
  }
  inline void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                   TransferUniqueId id) {
    ReadWrite(localOffset, remoteOffset, size, status, id, true);
  }

  virtual void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                              const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                              bool isRead) = 0;
  inline void BatchWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, false);
  }
  inline void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, true);
  }
  virtual bool Alive() const = 0;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                             Backend                                            */
/* ---------------------------------------------------------------------------------------------- */
class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;

  virtual void RegisterRemoteEngine(const EngineDesc&) = 0;
  virtual void DeregisterRemoteEngine(const EngineDesc&) = 0;

  virtual void RegisterMemory(const MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(const MemoryDesc& desc) = 0;

  virtual void ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                         const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                         TransferStatus* status, TransferUniqueId id, bool isRead) = 0;
  inline void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
    ReadWrite(localSrc, localOffset, remoteDest, remoteOffset, size, status, id, false);
  }
  inline void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                   size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
    ReadWrite(localDest, localOffset, remoteSrc, remoteOffset, size, status, id, true);
  }

  virtual void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                              const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                              const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                              bool isRead) = 0;
  inline void BatchWrite(const MemoryDesc& localSrc, const SizeVec& localOffsets,
                         const MemoryDesc& remoteDest, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localSrc, localOffsets, remoteDest, remoteOffsets, sizes, status, id, false);
  }
  inline void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                        const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
    BatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id, true);
  }

  virtual BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) = 0;

  // Take the transfer status of an inbound op
  virtual bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) = 0;
  // Health querying (default: Healthy if backend doesn't override)
  enum class Health : uint8_t { Healthy = 0, Degraded = 1, Failed = 2 };
  enum class ErrorCode : uint8_t {
    Ok = 0,
    Transient,
    Timeout,
    Protection,
    RemoteDisconnect,
    CQPollFailed,
    Internal
  };
  enum class Severity : uint8_t { Info = 0, Recoverable = 1, Fatal = 2 };
  struct ErrorRecord {
    ErrorCode code{ErrorCode::Ok};
    Severity severity{Severity::Info};
    int vendor_err{0};
    const char* msg{nullptr};
  };
  virtual Health health() const noexcept { return health_.load(std::memory_order_acquire); }

 protected:
  void report_error(const ErrorRecord& rec) noexcept {
    // if fatal -> Failed; if recoverable and previously healthy -> Degraded.
    if (rec.severity == Severity::Fatal) {
      health_.store(Health::Failed, std::memory_order_release);
    } else if (rec.severity == Severity::Recoverable) {
      auto now_ms = []() -> int64_t {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
      }();
      const auto& th = thresholds();
      Health cur = health_.load(std::memory_order_acquire);
      if (cur == Health::Healthy) {
        health_.compare_exchange_strong(cur, Health::Degraded, std::memory_order_acq_rel);
      }
      int64_t start = recoverable_window_start_ms_.load(std::memory_order_acquire);
      if (start == 0 || now_ms - start > static_cast<int64_t>(th.recoverable_window_ms)) {
        recoverable_window_start_ms_.store(now_ms, std::memory_order_release);
        recoverable_window_count_.store(1, std::memory_order_release);
      } else {
        uint32_t cnt = recoverable_window_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (cnt >= th.recoverable_fail_threshold) {
          health_.store(Health::Failed, std::memory_order_release);
        }
      }
    }
    on_error(rec);
  }

  // Optional override point (placed after existing virtuals to reduce ABI disturbance risk)
  virtual void on_error(const ErrorRecord&) {}

 private:
  std::atomic<Health> health_{Health::Healthy};
  // Recoverable error escalation window tracking
  std::atomic<uint32_t> recoverable_window_count_{0};
  std::atomic<int64_t> recoverable_window_start_ms_{0};

  struct Thresholds {
    uint32_t recoverable_fail_threshold{50};
    uint32_t recoverable_window_ms{30000};
  };
  static void ConfigureErrorThresholds(uint32_t failThreshold, uint32_t windowMs) {
    auto& t = thresholds();
    t.recoverable_fail_threshold = failThreshold;
    t.recoverable_window_ms = windowMs;
  }
  static Thresholds& thresholds() {
    static Thresholds t{};
    return t;
  }
};

}  // namespace io
}  // namespace mori
