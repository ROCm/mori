#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "mori/io/common.hpp"

namespace mori::io::fabric::internal {

// A failed device fence is not proof that remote DMA has stopped. Keep the
// references for the process lifetime; recovery/fencing must happen above IO.
struct QuarantineEntry {
  QuarantineEntry() = default;
  explicit QuarantineEntry(std::function<void()> references) : references(std::move(references)) {}
  std::function<void()> references;
  std::shared_ptr<void> region;
  std::shared_ptr<void> staging;
  QuarantineEntry* next{nullptr};
};

// Prepare the entry before GPU submission. Retaining it after a failed drain
// must also work when the failure was caused by host allocation pressure.
inline void QuarantineResources(std::unique_ptr<QuarantineEntry> entry) noexcept {
  static std::mutex mutex;
  static QuarantineEntry* held = nullptr;
  std::lock_guard<std::mutex> lock(mutex);
  auto* resource = entry.release();
  if (resource == nullptr) return;
  resource->next = held;
  held = resource;
}

inline void QuarantineResources(std::function<void()> references) {
  QuarantineResources(std::make_unique<QuarantineEntry>(std::move(references)));
}

inline void SettleSubmissionError(TransferStatus* status, const std::string& message,
                                  bool drainConfirmed, std::function<void()> references) {
  if (!drainConfirmed) {
    QuarantineResources(std::move(references));
    status->SetResourcesQuarantined();
  }
  status->Update(StatusCode::ERR_GPU_OP,
                 message + (drainConfirmed ? "" : "; DRAIN_UNCONFIRMED: resources quarantined"));
}

}  // namespace mori::io::fabric::internal
