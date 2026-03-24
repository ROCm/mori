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

#include <cstddef>
#include <list>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "umbp/common/storage_tier.h"

namespace mori::umbp {

class LocalBlockIndex {
 public:
  // Index operations
  bool MayExist(const std::string& key) const;
  std::optional<LocalLocation> Lookup(const std::string& key) const;
  void Insert(const std::string& key, const LocalLocation& loc);
  std::optional<LocalLocation> Remove(const std::string& key);
  bool UpdateTier(const std::string& key, StorageTier new_tier);
  size_t Count() const;
  void Clear();

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, LocalLocation> index_;
};

}  // namespace mori::umbp
