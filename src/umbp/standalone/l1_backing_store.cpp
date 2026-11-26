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
#include "umbp/standalone/l1_backing_store.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mori::umbp::standalone {
namespace {

class ClientL1BackingStore final : public l1::BackingStore {
 public:
  ClientL1BackingStore(IUMBPClient* client, mori::io::MemoryLocationType location)
      : client_(client), location_(location) {}

  bool AddMemory(void* base, uint64_t bytes, int device, std::string* error) override {
    std::lock_guard<std::mutex> lock(mutex_);
    // Two pools over one allocation register it once and keep it until both
    // have gone: the store holds one region per base address.
    auto known = regions_.find(base);
    if (known != regions_.end()) {
      if (bytes > known->second.bytes) {
        *error = "the allocation is already registered with the store, with fewer bytes";
        return false;
      }
      ++known->second.users;
      return true;
    }
    const bool gpu = location_ == mori::io::MemoryLocationType::GPU;
    if (!client_->RegisterMemory(reinterpret_cast<uintptr_t>(base), static_cast<size_t>(bytes),
                                 location_, gpu ? device : -1,
                                 MemoryRegistration::kLocalCopyOnly)) {
      *error = "the store refused the registration";
      return false;
    }
    regions_.emplace(base, Region{bytes, 1});
    return true;
  }

  void RemoveMemory(void* base) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto known = regions_.find(base);
    if (known == regions_.end() || --known->second.users != 0) return;
    regions_.erase(known);
    client_->DeregisterMemory(reinterpret_cast<uintptr_t>(base));
  }

  std::vector<bool> Write(const std::string& space, const std::vector<Block>& blocks) override {
    Ranges ranges(space, blocks);
    return client_->BatchPutRanges(ranges.keys, ranges.object_sizes, ranges.addresses, ranges.sizes,
                                   ranges.offsets);
  }

  std::vector<bool> Read(const std::string& space, const std::vector<Block>& blocks) override {
    Ranges ranges(space, blocks);
    return client_->BatchGetRanges(ranges.keys, ranges.addresses, ranges.sizes, ranges.offsets);
  }

  std::vector<bool> Contains(const std::string& space,
                             const std::vector<std::string>& keys) override {
    std::vector<std::string> store_keys;
    store_keys.reserve(keys.size());
    for (const std::string& key : keys) store_keys.push_back(L1StoreKey(space, key));
    return client_->BatchExists(store_keys);
  }

 private:
  // The blocks as the store's ranged calls take them: each block one object,
  // its fragments the object's ranges, end to end from offset 0.
  struct Ranges {
    Ranges(const std::string& space, const std::vector<Block>& blocks) {
      keys.reserve(blocks.size());
      object_sizes.reserve(blocks.size());
      addresses.resize(blocks.size());
      sizes.resize(blocks.size());
      offsets.resize(blocks.size());
      for (size_t i = 0; i < blocks.size(); ++i) {
        const std::vector<Fragment>& fragments = blocks[i].fragments;
        keys.push_back(L1StoreKey(space, blocks[i].key));
        addresses[i].reserve(fragments.size());
        sizes[i].reserve(fragments.size());
        offsets[i].reserve(fragments.size());
        size_t offset = 0;
        for (const Fragment& fragment : fragments) {
          addresses[i].push_back(reinterpret_cast<uintptr_t>(fragment.address));
          sizes[i].push_back(static_cast<size_t>(fragment.bytes));
          offsets[i].push_back(offset);
          offset += static_cast<size_t>(fragment.bytes);
        }
        object_sizes.push_back(offset);
      }
    }

    std::vector<std::string> keys;
    std::vector<size_t> object_sizes;
    std::vector<std::vector<uintptr_t>> addresses;
    std::vector<std::vector<size_t>> sizes;
    std::vector<std::vector<size_t>> offsets;
  };

  struct Region {
    uint64_t bytes = 0;
    uint32_t users = 0;
  };

  IUMBPClient* const client_;
  const mori::io::MemoryLocationType location_;
  std::mutex mutex_;
  std::map<void*, Region> regions_;
};

}  // namespace

std::string L1StoreKey(const std::string& space, const std::string& key) {
  bool printable = true;
  for (const char c : key) {
    if (c < 0x21 || c > 0x7E) {
      printable = false;
      break;
    }
  }
  std::string out;
  out.reserve(space.size() + 2 * key.size() + 8);
  out += "l1/";
  out += space;
  // The two spellings cannot name the same key: they differ in this letter.
  out += printable ? "/s/" : "/x/";
  if (printable) {
    out += key;
    return out;
  }
  static const char kHex[] = "0123456789abcdef";
  for (const char c : key) {
    const auto byte = static_cast<unsigned char>(c);
    out += kHex[byte >> 4];
    out += kHex[byte & 0xF];
  }
  return out;
}

std::shared_ptr<l1::BackingStore> MakeL1BackingStore(IUMBPClient* client,
                                                     mori::io::MemoryLocationType location) {
  return std::make_shared<ClientL1BackingStore>(client, location);
}

}  // namespace mori::umbp::standalone
