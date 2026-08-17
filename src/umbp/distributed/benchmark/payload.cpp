// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/benchmark/payload.h"

#include <algorithm>
#include <stdexcept>

namespace mori::umbp::benchmark {
namespace {

uint64_t HashKey(std::string_view key) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : key) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t SplitMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t InitialState(std::string_view key, uint64_t operation_id, uint64_t seed) {
  return SplitMix64(HashKey(key) ^ SplitMix64(operation_id) ^ SplitMix64(seed));
}

}  // namespace

void FillDeterministicPayload(std::string_view key, uint64_t operation_id, uint64_t seed,
                              void* destination, size_t size) {
  if (size != 0 && destination == nullptr) {
    throw std::invalid_argument("payload destination must not be null");
  }
  auto* output = static_cast<uint8_t*>(destination);
  uint64_t state = InitialState(key, operation_id, seed);
  size_t offset = 0;
  while (offset < size) {
    state = SplitMix64(state);
    const size_t count = std::min(sizeof(state), size - offset);
    for (size_t byte = 0; byte < count; ++byte) {
      output[offset + byte] = static_cast<uint8_t>(state >> (byte * 8));
    }
    offset += count;
  }
}

std::vector<uint8_t> GenerateDeterministicPayload(std::string_view key, uint64_t operation_id,
                                                  uint64_t seed, size_t size) {
  std::vector<uint8_t> payload(size);
  FillDeterministicPayload(key, operation_id, seed, payload.data(), payload.size());
  return payload;
}

bool ValidateDeterministicPayload(std::string_view key, uint64_t operation_id, uint64_t seed,
                                  const void* data, size_t size) {
  if (size != 0 && data == nullptr) return false;
  const auto* input = static_cast<const uint8_t*>(data);
  uint64_t state = InitialState(key, operation_id, seed);
  size_t offset = 0;
  while (offset < size) {
    state = SplitMix64(state);
    const size_t count = std::min(sizeof(state), size - offset);
    for (size_t byte = 0; byte < count; ++byte) {
      if (input[offset + byte] != static_cast<uint8_t>(state >> (byte * 8))) {
        return false;
      }
    }
    offset += count;
  }
  return true;
}

}  // namespace mori::umbp::benchmark
