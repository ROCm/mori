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
#include <atomic>
#include <memory>

#include "mori/io/backend.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

// FallbackBackend wraps two concrete backends (primary e.g. RDMA, secondary e.g. TCP) and
// transparently switches to the secondary when the primary transitions to Failed.
class FallbackBackend : public Backend {
 public:
  FallbackBackend(std::unique_ptr<Backend> primary, std::unique_ptr<Backend> secondary)
      : primary_(std::move(primary)), secondary_(std::move(secondary)) {
    active_.store(primary_.get(), std::memory_order_release);
  }
  ~FallbackBackend() override = default;

  void RegisterRemoteEngine(const EngineDesc& d) override {
    delegate(&Backend::RegisterRemoteEngine, d);
  }
  void DeregisterRemoteEngine(const EngineDesc& d) override {
    delegate(&Backend::DeregisterRemoteEngine, d);
  }
  void RegisterMemory(const MemoryDesc& desc) override { delegate(&Backend::RegisterMemory, desc); }
  void DeregisterMemory(const MemoryDesc& desc) override {
    delegate(&Backend::DeregisterMemory, desc);
  }

  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead) override {
    maybe_switch();
    active_.load(std::memory_order_acquire)
        ->ReadWrite(localDest, localOffset, remoteSrc, remoteOffset, size, status, id, isRead);
  }
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override {
    maybe_switch();
    active_.load(std::memory_order_acquire)
        ->BatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id,
                         isRead);
  }
  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) override {
    maybe_switch();
    return active_.load(std::memory_order_acquire)->CreateSession(local, remote);
  }
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                TransferStatus* status) override {
    maybe_switch();
    return active_.load(std::memory_order_acquire)->PopInboundTransferStatus(remote, id, status);
  }

  Health health() const noexcept override {
    return active_.load(std::memory_order_acquire)->health();
  }

 private:
  template <typename Method, typename Arg>
  void delegate(Method m, const Arg& a) {
    maybe_switch();
    (active_.load(std::memory_order_acquire)->*m)(a);
  }

  void maybe_switch() {
    Backend* cur = active_.load(std::memory_order_acquire);
    if (cur == primary_.get()) {
      if (primary_->health() == Health::Failed) {
        Backend* expected = primary_.get();
        if (active_.compare_exchange_strong(expected, secondary_.get(),
                                            std::memory_order_acq_rel)) {
          MORI_IO_WARN(
              "FallbackBackend: switching from primary to secondary backend due to failure");
        }
      }
    }
  }

  std::unique_ptr<Backend> primary_{};
  std::unique_ptr<Backend> secondary_{};
  std::atomic<Backend*> active_{nullptr};
};

}  // namespace io
}  // namespace mori
