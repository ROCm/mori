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
#include <cstdint>
#include <memory>

namespace mori {
namespace io {
namespace internal {

enum class IoFailureKind : uint8_t {
  None = 0,
  FlushCascade,
  RootCause,
};

struct IoCallDiagnostics {
  void Initialize(const char* callSiteLabel) { label = callSiteLabel; }

  void MarkFlushCascade() {
    IoFailureKind current = failureKind.load(std::memory_order_relaxed);
    while (current == IoFailureKind::None &&
           !failureKind.compare_exchange_weak(current, IoFailureKind::FlushCascade,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
    }
  }

  void MarkRootCause() { failureKind.store(IoFailureKind::RootCause, std::memory_order_release); }

  IoFailureKind CurrentFailureKind() const { return failureKind.load(std::memory_order_acquire); }

  const char* Label() const { return label; }

  bool TryMarkLogged(IoFailureKind kind) {
    if (kind == IoFailureKind::None) return false;

    IoFailureKind current = loggedKind.load(std::memory_order_acquire);
    while (true) {
      if (kind == IoFailureKind::FlushCascade) {
        if (current != IoFailureKind::None) return false;
      } else if (current == IoFailureKind::RootCause) {
        return false;
      }

      if (loggedKind.compare_exchange_weak(current, kind, std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
        return true;
      }
    }
  }

 private:
  const char* label{nullptr};
  std::atomic<IoFailureKind> failureKind{IoFailureKind::None};
  std::atomic<IoFailureKind> loggedKind{IoFailureKind::None};
};

struct IoCallDiagnosticsCapture {
  std::shared_ptr<IoCallDiagnostics>* diagnostics{nullptr};
  const char* label{nullptr};
};

inline thread_local IoCallDiagnosticsCapture* currentIoDiagnosticsCapture = nullptr;

class ScopedIoCallDiagnosticsCapture {
 public:
  ScopedIoCallDiagnosticsCapture(std::shared_ptr<IoCallDiagnostics>* diagnostics, const char* label)
      : state_{diagnostics, label}, prev_(currentIoDiagnosticsCapture) {
    currentIoDiagnosticsCapture = &state_;
  }

  ~ScopedIoCallDiagnosticsCapture() { currentIoDiagnosticsCapture = prev_; }

  ScopedIoCallDiagnosticsCapture(const ScopedIoCallDiagnosticsCapture&) = delete;
  ScopedIoCallDiagnosticsCapture& operator=(const ScopedIoCallDiagnosticsCapture&) = delete;

 private:
  IoCallDiagnosticsCapture state_{};
  IoCallDiagnosticsCapture* prev_{nullptr};
};

template <typename Meta>
inline void PublishCurrentIoCallDiagnostics(const std::shared_ptr<Meta>& meta) {
  if (currentIoDiagnosticsCapture == nullptr ||
      currentIoDiagnosticsCapture->diagnostics == nullptr ||
      *currentIoDiagnosticsCapture->diagnostics) {
    return;
  }
  meta->diagnostics.Initialize(currentIoDiagnosticsCapture->label);
  *currentIoDiagnosticsCapture->diagnostics =
      std::shared_ptr<IoCallDiagnostics>(meta, &meta->diagnostics);
}

}  // namespace internal
}  // namespace io
}  // namespace mori
