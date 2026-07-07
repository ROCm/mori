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

// KeySchema — every master metadata key and the hash tags they hash under.
//
// Two families of keys, deliberately split across hash tags:
//
//   * Control-plane keys (client records, ALIVE set, peer projection, the
//     per-node reverse indexes) all share the control tag `{umbp:<ns>}`, so
//     they co-locate in one slot and a single Lua script can mutate them
//     atomically.
//
//   * Block-location keys are spread across `num_shards` shard tags
//     `{umbp:<ns>:b<shard>}` chosen by hashing the user key. Spreading blocks
//     over many slots is what lets the read hot path (RouteGet / Exists) run
//     one single-slot script per shard instead of piling every block lookup
//     onto one slot / one server thread. On a multi-threaded store (Dragonfly)
//     different shards land on different proactor threads and run in parallel.
//
// Backwards compatibility: `num_shards == 1` puts block keys back under the
// control tag, so the emitted key strings are byte-identical to the original
// single-tag schema (a deployment with sharding disabled is unchanged).
//
// The shard tag is derived from a stable hash (FNV-1a), NOT std::hash, so a key
// written by one build always resolves to the same shard after a rebuild.
// `num_shards` is fixed for a deployment's lifetime, exactly like the in-memory
// backend's UMBP_MASTER_INDEX_SHARDS: changing it with live data would strand
// keys under their old shard tag.
//
// See design-redis-metadata-store.md §4 for the full schema table.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp::redis {

class KeySchema {
 public:
  explicit KeySchema(const std::string& ns, std::size_t num_shards = 1)
      : control_tag_("{umbp:" + ns + "}"), num_shards_(num_shards == 0 ? 1 : num_shards) {
    block_tags_.reserve(num_shards_);
    if (num_shards_ == 1) {
      // Legacy layout: block keys live under the control tag.
      block_tags_.push_back(control_tag_);
    } else {
      for (std::size_t s = 0; s < num_shards_; ++s) {
        block_tags_.push_back("{umbp:" + ns + ":b" + std::to_string(s) + "}");
      }
    }
  }

  std::size_t NumShards() const { return num_shards_; }

  // The control-plane hash tag, e.g. "{umbp:default}". Passed to Lua as ARGV[1]
  // so scripts can compose auxiliary control-slot key names in the same slot.
  const std::string& Tag() const { return control_tag_; }

  // HASH: one client record.
  std::string Node(const std::string& node_id) const { return control_tag_ + ":node:" + node_id; }

  // SET: ALIVE node ids.
  std::string NodesAlive() const { return control_tag_ + ":nodes:alive"; }

  // HASH: node_id -> peer_address for ALIVE nodes only.
  std::string AlivePeers() const { return control_tag_ + ":alive_peers"; }

  // The shard a user key belongs to.
  std::size_t ShardOf(const std::string& key) const {
    return num_shards_ == 1 ? 0 : (StableHash(key) % num_shards_);
  }

  // The hash tag backing shard `shard` (0 <= shard < num_shards).
  const std::string& ShardTag(std::size_t shard) const { return block_tags_[shard]; }

  // HASH: block locations ("l|<node>|<tier>" -> size) plus meta fields. Placed
  // in the key's shard slot.
  std::string Block(const std::string& key) const {
    return ShardTag(ShardOf(key)) + ":block:" + key;
  }

  // SET: the block keys a node owns (reverse index for node-scoped wipe). Kept
  // on the control tag as one set per node; its members are full (already
  // sharded) block-key strings, so the wipe scripts can HDEL them directly
  // without recomputing the shard in Lua.
  std::string NodeBlocks(const std::string& node_id) const {
    return control_tag_ + ":node:" + node_id + ":blocks";
  }

  // SET: the external-kv hashes a node registered (reverse index).
  std::string ExtKvNode(const std::string& node_id) const {
    return control_tag_ + ":extkv:node:" + node_id;
  }

 private:
  // FNV-1a (32-bit): small, fast, and stable across builds/runs. std::hash is
  // avoided on purpose — it is implementation-defined, so its bucketing could
  // change under a toolchain upgrade and silently relocate every key's shard.
  static std::size_t StableHash(const std::string& key) {
    uint32_t hash = 2166136261u;
    for (const unsigned char c : key) {
      hash ^= c;
      hash *= 16777619u;
    }
    return static_cast<std::size_t>(hash);
  }

  std::string control_tag_;
  std::size_t num_shards_;
  std::vector<std::string> block_tags_;  // indexed by shard; size == num_shards_
};

// Location hash-field prefix marker. A field named "l|<node>|<tier>" holds a
// location's size; any field beginning with this marker is a location, and
// "_"-prefixed fields (_lease/_lacc/_acnt/_created) are per-block metadata.
inline constexpr const char* kLocFieldMarker = "l|";

}  // namespace mori::umbp::redis
