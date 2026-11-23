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

#include <algorithm>
#include <msgpack.hpp>

#include "mori/utils/mori_log.hpp"
#include "umbp/sim/sim.h"

namespace mori::umbp {

TransferRef SimTransferEngine::RegisterMemory(void* base, size_t size,
                                              mori::io::MemoryLocationType loc, int device) {
  auto ref = TransferRef::HostBytes(base, size, loc, device);
  if (!ref.HasHostPtr()) return ref;
  ref.mem.engineKey = engine_key_;
  ref.mem.id = next_memory_id_.fetch_add(1);
  ref.mem.data = reinterpret_cast<uintptr_t>(base);
  ref.mem.size = size;
  ref.mem.loc = loc;
  ref.mem.deviceId = device;
  return ref;
}

std::vector<uint8_t> SimTransferEngine::PackedLocalEngineDesc() const {
  // A msgpack string, deliberately NOT a mori-io EngineDesc array. Both engines
  // reject the other format during GetPeerInfo, before any data-plane access.
  msgpack::sbuffer packed;
  msgpack::pack(packed, engine_key_);
  return {packed.data(), packed.data() + packed.size()};
}

bool SimTransferEngine::EnsureRemoteEngine(const std::string& node_id, const std::string& packed) {
  if (packed.empty()) return HasRemoteEngine(node_id);
  try {
    const auto key = msgpack::unpack(packed.data(), packed.size()).get().as<std::string>();
    if (key.rfind("umbp-sim/", 0) != 0 || key == engine_key_) return false;
    std::lock_guard<std::mutex> lock(remotes_mutex_);
    auto& remote = remotes_[node_id];
    if (remote.engine_key != key) remote = RemoteNode{};
    remote.engine_key = key;
    return true;
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[UMBP sim] peer '{}' is not a compatible simulator: {}", node_id, e.what());
    return false;
  }
}

bool SimTransferEngine::HasRemoteEngine(const std::string& node_id) const {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  return remotes_.count(node_id) != 0;
}

void SimTransferEngine::ForgetRemote(const std::string& node_id) {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  remotes_.erase(node_id);
}

void SimTransferEngine::CacheRemoteBuffers(const std::string& node_id,
                                           const std::vector<BufferMemoryDescBytes>& descs) {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto remote = remotes_.find(node_id);
  if (remote == remotes_.end()) return;
  for (const auto& desc : descs) {
    if (desc.backend_id >= kMaxBackendsPerPeer || desc.desc_bytes.empty()) continue;
    try {
      const auto mem = msgpack::unpack(reinterpret_cast<const char*>(desc.desc_bytes.data()),
                                       desc.desc_bytes.size())
                           .get()
                           .as<mori::io::MemoryDesc>();
      if (mem.engineKey != remote->second.engine_key || mem.size == 0 || mem.data == 0 ||
          (mem.loc != mori::io::MemoryLocationType::CPU &&
           mem.loc != mori::io::MemoryLocationType::GPU))
        continue;
      auto& shelf = remote->second.buffers[desc.backend_id];
      if (shelf.size() <= desc.buffer_index) shelf.resize(size_t{desc.buffer_index} + 1);
      if (shelf[desc.buffer_index].Valid()) {
        if (!(shelf[desc.buffer_index].mem == mem))
          MORI_UMBP_ERROR("[UMBP sim] conflicting buffer descriptor from '{}'", node_id);
      } else {
        shelf[desc.buffer_index] = TransferRef::Remote(mem);
      }
    } catch (const std::exception& e) {
      MORI_UMBP_ERROR("[UMBP sim] bad buffer descriptor from '{}': {}", node_id, e.what());
    }
  }
}

bool SimTransferEngine::HasRemoteBuffers(const std::string& node_id) const {
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto remote = remotes_.find(node_id);
  if (remote == remotes_.end()) return false;
  for (const auto& shelf : remote->second.buffers)
    if (std::any_of(shelf.begin(), shelf.end(), [](const auto& ref) { return ref.Valid(); }))
      return true;
  return false;
}

std::vector<TransferRef> SimTransferEngine::RemoteBufferSnapshot(const std::string& node_id,
                                                                 uint32_t backend_id) const {
  if (backend_id >= kMaxBackendsPerPeer) return {};
  std::lock_guard<std::mutex> lock(remotes_mutex_);
  auto remote = remotes_.find(node_id);
  return remote == remotes_.end() ? std::vector<TransferRef>{} : remote->second.buffers[backend_id];
}

}  // namespace mori::umbp
