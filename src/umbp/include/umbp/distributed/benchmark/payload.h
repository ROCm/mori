// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mori::umbp::benchmark {

void FillDeterministicPayload(std::string_view key, uint64_t operation_id, uint64_t seed,
                              void* destination, size_t size);

std::vector<uint8_t> GenerateDeterministicPayload(std::string_view key, uint64_t operation_id,
                                                  uint64_t seed, size_t size);

bool ValidateDeterministicPayload(std::string_view key, uint64_t operation_id, uint64_t seed,
                                  const void* data, size_t size);

}  // namespace mori::umbp::benchmark
