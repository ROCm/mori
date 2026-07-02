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

// KeySchema — every master metadata key, all sharing one deployment hash tag.
//
// The tag `{umbp:<namespace>}` is the Redis-cluster hash-tag delimiter, so all
// keys below hash to the same slot. That co-location is what lets a single Lua
// script mutate node + block + reverse-index keys atomically on any RESP store
// (Redis single/cluster, Dragonfly, Valkey).
//
// See design-redis-metadata-store.md §4 for the full schema table.

#pragma once

#include <string>

#include "umbp/distributed/types.h"

namespace mori::umbp::redis {

class KeySchema {
 public:
  explicit KeySchema(const std::string& ns) : tag_("{umbp:" + ns + "}") {}

  // The shared hash tag, e.g. "{umbp:default}". Passed to Lua as ARGV[1] so
  // scripts can compose auxiliary key names in the same slot.
  const std::string& Tag() const { return tag_; }

  // HASH: one client record.
  std::string Node(const std::string& node_id) const { return tag_ + ":node:" + node_id; }

  // SET: ALIVE node ids.
  std::string NodesAlive() const { return tag_ + ":nodes:alive"; }

  // HASH: node_id -> peer_address for ALIVE nodes only.
  std::string AlivePeers() const { return tag_ + ":alive_peers"; }

  // HASH: block locations ("l|<node>|<tier>" -> size) plus meta fields.
  std::string Block(const std::string& key) const { return tag_ + ":block:" + key; }

  // SET: the block keys a node owns (reverse index for node-scoped wipe).
  std::string NodeBlocks(const std::string& node_id) const {
    return tag_ + ":node:" + node_id + ":blocks";
  }

  // SET: the external-kv hashes a node registered (reverse index).
  std::string ExtKvNode(const std::string& node_id) const {
    return tag_ + ":extkv:node:" + node_id;
  }

 private:
  std::string tag_;
};

// Location hash-field prefix marker. A field named "l|<node>|<tier>" holds a
// location's size; any field beginning with this marker is a location, and
// "_"-prefixed fields (_lease/_lacc/_acnt/_created) are per-block metadata.
inline constexpr const char* kLocFieldMarker = "l|";

}  // namespace mori::umbp::redis
