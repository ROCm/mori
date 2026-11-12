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
#include "umbp/distributed/pool/fabric_read_push.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

#include "umbp/common/host_registration.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/range_map.h"
#include "umbp/distributed/transfer/coalesce.h"

namespace mori::umbp {
FabricReadPush::FabricReadPush(TransferEngine* engine, PeerDirectory* peers,
                               BackendRegistry* backends, int device, HostDeviceAddress address,
                               std::function<void()> quarantine)
    : engine_(engine),
      peers_(peers),
      backends_(backends),
      device_(device),
      address_(std::move(address)),
      quarantine_(std::move(quarantine)) {}

FabricReadPush::~FabricReadPush() {
  if (quarantined_) return;
  for (const auto& [key, ref] : aliases_) {
    try {
      engine_->Deregister(ref);
    } catch (...) {
      quarantine_();
    }
  }
}

FabricReadPushResult FabricReadPush::Execute(std::vector<PoolResolvedEntry>* entries,
                                             const FabricReadPushRequest& request) {
  FabricReadPushResult result;
  if (quarantined_) return result;
  if (!engine_ || !peers_ || !backends_ || entries->size() != request.ranges.size()) return result;
  // The pull protocol retries whole BUSY batches. Preserve that behavior:
  // no successful prefix may write before the reader falls back and retries.
  if (std::any_of(entries->begin(), entries->end(), [](const auto& entry) {
        return EffectiveResolveOutcome(entry.resolved) == ResolveOutcome::kBusy;
      }))
    return result;
  if (!peers_->EnsureRemoteEngine("push:" + request.reader_id, request.engine_desc)) return result;
  struct Pending {
    TransferEngine* engine;
    std::function<void()> quarantine;
    std::vector<TransferRef> registrations;
    std::vector<std::shared_ptr<const void>> holds;
    std::unique_ptr<TransferHandle> handle;
    ~Pending() {
      for (const auto& ref : registrations) {
        try {
          engine->Deregister(ref);
        } catch (...) {
          quarantine();
        }
      }
    }
  };
  auto pending = std::make_unique<Pending>();
  pending->engine = engine_;
  pending->quarantine = quarantine_;
  std::vector<TransferItem> transfers;
  std::map<std::tuple<uint32_t, uint32_t, uint64_t>, TransferRef> aliases;
  for (size_t i = 0; i < entries->size(); ++i) {
    const auto& entry = (*entries)[i];
    const auto& resolved = entry.resolved;
    if (!resolved.found) continue;
    if (!resolved.hold || resolved.pages.empty() || !resolved.page_size) return result;
    auto* backend = backends_->Get(entry.backend_id);
    if (!backend) return result;
    pending->holds.push_back(resolved.hold);
    const uint64_t covered = resolved.covered_size ? resolved.covered_size : resolved.size;
    for (const auto& range : request.ranges[i]) {
      size_t offset = range.source_offset;
      if (!range.size || range.destination.loc != mori::io::MemoryLocationType::GPU ||
          !range.destination.HasMemoryDesc() || range.destination.HasHostPtr() ||
          range.destination_offset > range.destination.size ||
          range.size > range.destination.size - range.destination_offset ||
          (resolved.covered_size &&
           !RebaseRangeOntoWindow(resolved.covered_offset, resolved.covered_size, &offset,
                                  range.size)))
        return result;
      const bool walked = ForEachRangePageFragment(
          resolved.pages, resolved.page_size, covered, offset, range.size,
          [&](size_t page, uint64_t at, size_t bytes, size_t copied) {
            const auto buffer_index = resolved.pages[page].buffer_index;
            auto source = backend->BufferRef(buffer_index);
            if (!source.HasHostPtr() || at > source.size || bytes > source.size - at) return false;
            while (bytes) {
              size_t part = bytes;
              uint64_t source_offset = at;
              TransferRef view = source;
              if (source.loc == mori::io::MemoryLocationType::CPU) {
                const uintptr_t base = reinterpret_cast<uintptr_t>(source.host_ptr);
                const size_t chunk = HostRegistrationChunkBytes();
                const uintptr_t begin = chunk ? std::max(base, (base + at) / chunk * chunk) : base;
                const uint64_t chunk_offset = begin - base;
                const size_t chunk_size =
                    chunk ? std::min<uint64_t>(source.size - chunk_offset, chunk - begin % chunk)
                          : source.size;
                part = std::min<uint64_t>(part, chunk_size - (at - chunk_offset));
                const auto key = std::make_tuple(entry.backend_id, buffer_index, chunk_offset);
                auto old = aliases.find(key);
                if (old == aliases.end()) {
                  void* alias = address_(reinterpret_cast<void*>(begin), chunk_size, device_);
                  if (!alias) return false;
                  // This is the GPU address-space view of pinned host pages,
                  // used only as a local kernel input, never exported as HBM.
                  std::lock_guard<std::mutex> lock(aliases_mutex_);
                  const auto identity =
                      std::make_pair(reinterpret_cast<uintptr_t>(alias), chunk_size);
                  const auto cached = aliases_.find(identity);
                  if (cached != aliases_.end()) {
                    view = cached->second;
                  } else {
                    // Allocate ownership before registration so a later map
                    // allocation failure still deregisters the new view.
                    pending->registrations.emplace_back();
                    pending->registrations.back() = engine_->RegisterMemoryWithBackends(
                        alias, chunk_size, mori::io::MemoryLocationType::GPU, device_,
                        mori::io::BackendBit(mori::io::BackendType::FABRIC));
                    view = pending->registrations.back();
                    if (!view.HasMemoryDesc()) return false;
                    if (aliases_.size() < 64) {
                      aliases_.emplace(identity, view);
                      pending->registrations.pop_back();
                    }
                  }
                  old = aliases.emplace(key, view).first;
                }
                view = old->second;
                source_offset = at - chunk_offset;
              }
              if (!engine_->CanHandle(view, range.destination)) return false;
              if (transfers.empty() ||
                  !ExtendRegisteredGpuTransfer(&transfers.back(), view, source_offset,
                                               range.destination, range.destination_offset + copied,
                                               part, i)) {
                transfers.push_back({view, source_offset, range.destination,
                                     range.destination_offset + copied, part, i});
              }
              at += part;
              copied += part;
              bytes -= part;
            }
            return true;
          });
      if (!walked) return result;
    }
  }
  auto planned = engine_->Plan(transfers);
  if (!planned.rejected_tags.empty()) return result;
  result.supported = true;
  if (planned.plans.empty()) return result;
  for (const auto& item : transfers) result.submitted_bytes += item.size;
  result.submitted_plans = planned.plans.size();
  auto fail = [&] {
    for (auto& entry : *entries) {
      entry.resolved.found = false;
      entry.resolved.outcome = ResolveOutcome::kFailed;
    }
  };
  try {
    pending->handle = engine_->Submit(std::move(planned.plans));
    if (!pending->handle) {
      fail();
      return result;
    }
    std::vector<TransferFailure> failures;
    pending->handle->Wait(&failures);
    for (const auto& failure : failures) {
      if (failure.resources_quarantined) result.drained = false;
      for (size_t tag : failure.tags)
        if (tag < entries->size()) {
          (*entries)[tag].resolved.found = false;
          (*entries)[tag].resolved.outcome = ResolveOutcome::kFailed;
        }
    }
  } catch (...) {
    result.drained = false;
  }
  if (!result.drained) {
    fail();
    quarantined_ = true;
    quarantine_();
    (void)pending.release();
  }
  return result;
}
}  // namespace mori::umbp
