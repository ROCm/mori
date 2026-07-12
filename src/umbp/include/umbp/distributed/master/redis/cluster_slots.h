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

// Redis Cluster slot math, used ONLY at startup to place one block-shard hash
// tag on each master node (balanced placement). The redis-plus-plus client does
// the real per-command routing; this is just so we can pick tags whose slot
// lands on a chosen node so a RouteGet batch spreads evenly instead of piling
// onto whichever node the default {umbp:<ns>:bS} tags happen to hash to.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mori::umbp::redis {

// Number of hash slots in a Redis Cluster (fixed by the protocol).
inline constexpr int kNumSlots = 16384;

// CRC16-CCITT (XMODEM): poly 0x1021, init 0x0000, no reflection — the exact
// function Redis Cluster uses for CLUSTER KEYSLOT. Bitwise (no 256-entry table)
// because it only runs a few hundred times at startup for tag placement, so the
// table's speed is irrelevant and a hand-copied table would risk a typo.
inline uint16_t Crc16(const char* buf, std::size_t len) {
  uint16_t crc = 0;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(static_cast<unsigned char>(buf[i])) << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                           : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

// Slot for a key, honoring the hash-tag rule: if the key contains "{...}" with a
// non-empty body, only that body is hashed (so all keys sharing a tag share a
// slot). Matches Redis' keyHashSlot().
inline int SlotOfKey(const std::string& key) {
  const auto open = key.find('{');
  if (open != std::string::npos) {
    const auto close = key.find('}', open + 1);
    if (close != std::string::npos && close > open + 1) {
      return Crc16(key.data() + open + 1, close - open - 1) % kNumSlots;
    }
  }
  return Crc16(key.data(), key.size()) % kNumSlots;
}

// A master's owned slot ranges (inclusive), as returned by CLUSTER SLOTS.
using SlotRange = std::pair<uint16_t, uint16_t>;

inline bool SlotInRanges(int slot, const std::vector<SlotRange>& ranges) {
  for (const auto& r : ranges) {
    if (slot >= r.first && slot <= r.second) return true;
  }
  return false;
}

// Find a block-shard hash tag "{umbp:<ns>:b<k>}" whose slot lands in `ranges`
// (i.e. is served by a chosen master). Searches k = 0,1,2,... deterministically
// so the same topology yields the same tag across restarts (stable key
// placement). Returns false if none found within max_tries (each master owns
// ~1/N of the slots, so a hit is expected within a few N tries).
inline bool FindTagForRanges(const std::string& ns, const std::vector<SlotRange>& ranges,
                             std::string* out_tag, int max_tries = 200000) {
  for (int k = 0; k < max_tries; ++k) {
    std::string tag = "{umbp:" + ns + ":b" + std::to_string(k) + "}";
    if (SlotInRanges(SlotOfKey(tag), ranges)) {
      *out_tag = std::move(tag);
      return true;
    }
  }
  return false;
}

}  // namespace mori::umbp::redis
