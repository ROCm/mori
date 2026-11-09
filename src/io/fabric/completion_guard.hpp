#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "mori/io/common.hpp"

namespace mori::io::fabric::internal {

// A failed device fence is not proof that remote DMA has stopped. Keep the
// references for the process lifetime; recovery/fencing must happen above IO.
inline void QuarantineResources(std::function<void()> references) {
  static auto* mutex = new std::mutex;
  static auto* held = new std::vector<std::function<void()>>;
  std::lock_guard<std::mutex> lock(*mutex);
  held->push_back(std::move(references));
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
