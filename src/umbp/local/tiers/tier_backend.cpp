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
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

bool TierBackend::WriteFromPtr(const std::string& key, uintptr_t src_ptr, size_t size) {
  return Write(key, reinterpret_cast<const void*>(src_ptr), size);
}

std::vector<char> TierBackend::Read(const std::string& key) { return {}; }

TierCapabilities TierBackend::Capabilities() const { return {}; }

const void* TierBackend::ReadPtr(const std::string& key, size_t* out_size) { return nullptr; }

bool TierBackend::WriteBatch(const std::vector<std::string>& keys,
                             const std::vector<const void*>& data_ptrs,
                             const std::vector<size_t>& sizes) {
  return false;
}

std::vector<bool> TierBackend::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                                const std::vector<uintptr_t>& dst_ptrs,
                                                const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = ReadIntoPtr(keys[i], dst_ptrs[i], sizes[i]);
  }
  return results;
}

std::vector<bool> TierBackend::BatchWrite(const std::vector<std::string>& keys,
                                          const std::vector<const void*>& data_ptrs,
                                          const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Write(keys[i], data_ptrs[i], sizes[i]);
  }
  return results;
}

std::vector<bool> TierBackend::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                                const std::vector<uintptr_t>& dst_ptrs,
                                                const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = ReadIntoPtr(keys[i], dst_ptrs[i], sizes[i]);
  }
  return results;
}

std::string TierBackend::GetLRUKey() const { return ""; }

std::vector<std::string> TierBackend::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::string lru = GetLRUKey();
  if (lru.empty()) return {};
  return {lru};
}

std::optional<std::string> TierBackend::GetLocationId(const std::string& key) const {
  (void)key;
  return std::nullopt;
}

}  // namespace mori::umbp
