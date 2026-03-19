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

#include <memory>
#include <optional>
#include <string>

#include "umbp/block_index/global_block_index.h"
#include "umbp/client_registry.h"
#include "umbp/route_get_strategy.h"
#include "umbp/route_put_strategy.h"

namespace mori::umbp {

class Router {
 public:
  /// Construct with injected strategy objects.
  /// If either strategy is nullptr, a default is created:
  ///   RouteGet  -> RandomRouteGetStrategy
  ///   RoutePut  -> TierAwareMostAvailableStrategy
  Router(GlobalBlockIndex& index, ClientRegistry& registry,
         std::unique_ptr<RouteGetStrategy> get_strategy = nullptr,
         std::unique_ptr<RoutePutStrategy> put_strategy = nullptr);
  ~Router() = default;

  Router(const Router&) = delete;
  Router& operator=(const Router&) = delete;

  /// Pick an existing replica to read from.
  /// Returns nullopt if the key is not in the index.
  std::optional<Location> RouteGet(const std::string& key, const std::string& node_id);

  /// Pick a target node and allocate capacity for a write.
  /// Internally retries with different candidates if allocation fails.
  /// Returns nullopt if no suitable node exists.
  std::optional<RoutePutResult> RoutePut(const std::string& key, const std::string& node_id,
                                         uint64_t block_size);

 private:
  GlobalBlockIndex& index_;
  ClientRegistry& registry_;
  std::unique_ptr<RouteGetStrategy> get_strategy_;
  std::unique_ptr<RoutePutStrategy> put_strategy_;
};

}  // namespace mori::umbp
