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

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/master_metadata_store.h"
#include "umbp/distributed/routing/route_get_strategy.h"
#include "umbp/distributed/routing/route_put_strategy.h"

namespace mori::umbp {

// Result of RouteGet, populated from the master's index projection.
// The reader follows up with peer.ResolveKey to fetch pages/descs.
struct RouteGetResolution {
  Location location;
  std::string peer_address;
};

class Router {
 public:
  Router(IMasterMetadataStore& store, std::unique_ptr<RouteGetStrategy> get_strategy = nullptr,
         std::unique_ptr<RoutePutStrategy> put_strategy = nullptr);
  ~Router() = default;

  Router(const Router&) = delete;
  Router& operator=(const Router&) = delete;

  // Pick a replica to read from.  Returns nullopt if the key is not in
  // the index, or if every replica's owning node is in `exclude_nodes`.
  std::optional<RouteGetResolution> RouteGet(const std::string& key, const std::string& node_id,
                                             const std::unordered_set<std::string>& exclude_nodes);

  // Pick a target node to write to.  Master holds no per-Put state in
  // the new design: the writer follows up with peer.AllocateSlot to
  // actually reserve capacity, and on ENOSPC at the peer the writer
  // retries RoutePut with the failed node added to `exclude_nodes`.
  std::optional<RoutePutResult> RoutePut(const std::string& key, const std::string& node_id,
                                         uint64_t block_size,
                                         const std::unordered_set<std::string>& exclude_nodes);

  std::vector<std::optional<RoutePutResult>> BatchRoutePut(
      const std::vector<std::string>& keys, const std::string& node_id,
      const std::vector<uint64_t>& block_sizes,
      const std::unordered_set<std::string>& exclude_nodes);

  std::vector<std::optional<RouteGetResolution>> BatchRouteGet(
      const std::vector<std::string>& keys, const std::string& node_id,
      const std::unordered_set<std::string>& exclude_nodes);

  void SetLeaseDuration(std::chrono::system_clock::duration d) { lease_duration_ = d; }

 private:
  IMasterMetadataStore& store_;
  std::unique_ptr<RouteGetStrategy> get_strategy_;
  std::unique_ptr<RoutePutStrategy> put_strategy_;
  std::chrono::system_clock::duration lease_duration_{std::chrono::seconds{10}};
};

}  // namespace mori::umbp
