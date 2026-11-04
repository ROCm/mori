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
/**
 * @file slot.hip.hpp
 * @brief Wave-slot packing/unpacking utilities for the host-to-device queues
 *
 * Each wave-slot is 64 bytes (8 x uint64_t), read cooperatively by 8 lanes:
 * +--------+--------+--------+--------+--------+--------+--------+---------+
 * | Lane 0 | Lane 1 | Lane 2 | Lane 3 | Lane 4 | Lane 5 | Lane 6 | Lane 7  |
 * | 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | 8B epoch|
 * +--------+--------+--------+--------+--------+--------+--------+---------+
 *
 * Total payload: 56 bytes per wave-slot (lanes 0-6). Lane 7 is a control
 * lane whose low byte is the epoch (SPSC) or which holds a sequence number
 * (MPMC). The device-to-host queue uses the chunk format of
 * detail/chunk.hip.hpp instead.
 */

#pragma once

#include <cstdint>
#include <cstring>

namespace mori::kiwi::queue {

/**
 * @brief Pack one lane's contribution to a 64-byte wave-slot
 *
 * Every lane does exactly one constant-size 8-byte operation.
 * Lanes 0-6 load 8 bytes of payload. Lane 7 writes the epoch.
 *
 * @param data Pointer to payload bytes (must be at least wave_slot_id * 56 + 56 bytes)
 * @param lane_id Lane within the wave-slot (0-7)
 * @param epoch The epoch counter (0-255)
 * @param wave_slot_id Which wave-slot in a multi-slot push (default 0)
 * @return Packed 64-bit value for this lane
 */
__host__ __device__ inline uint64_t pack_wave_lane(const void* data, int lane_id, uint8_t epoch,
                                                   int wave_slot_id = 0) {
  if (lane_id < 7) {
    uint64_t val;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(data);
    __builtin_memcpy(&val, src + wave_slot_id * 56 + lane_id * 8, 8);
    return val;
  }
  return static_cast<uint64_t>(epoch);
}

/**
 * @brief Unpack a wave-slot (8 uint64_t values) into a buffer
 *
 * Extracts 56 bytes of payload from lanes 0-6. Lane 7 is epoch-only.
 *
 * @param slot Array of 8 uint64_t values comprising the wave-slot
 * @param buf Output buffer (must be at least 56 bytes)
 */
__host__ __device__ inline void unpack_wave_slot(const uint64_t slot[8], void* buf) {
  unsigned char* dst = reinterpret_cast<unsigned char*>(buf);
  for (int lane = 0; lane < 7; ++lane) {
    __builtin_memcpy(dst + lane * 8, &slot[lane], 8);
  }
  // Lane 7 is epoch-only, no payload to extract
}

/**
 * @brief Extract only the epoch from a slot value (lane 7 of a wave-slot)
 *
 * The epoch is stored as a full uint64_t in lane 7; extract the low byte.
 *
 * @param slot_val The 64-bit value from lane 7
 * @return Epoch value (0-255)
 */
__host__ __device__ inline uint8_t get_epoch(uint64_t slot_val) {
  return static_cast<uint8_t>(slot_val & 0xFF);
}

}  // namespace mori::kiwi::queue
