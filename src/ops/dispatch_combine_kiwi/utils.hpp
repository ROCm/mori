#pragma once

#include "ep_config.hpp"

namespace mori::kiwi::ep {

// Inclusive prefix sum across one EP wave.  Lane i receives the sum of input
// values from lanes [0, i].  Callers can mask inactive lanes by passing zero.
__device__ inline int ep_wave_inclusive_prefix_sum(int value) {
    for (int offset = 1; offset < EP_WAVE_SIZE; offset <<= 1) {
        const int peer = __shfl_up(value, offset);
        const int lane = threadIdx.x % EP_WAVE_SIZE;
        if (lane >= offset) {
            value += peer;
        }
    }
    return value;
}

} // namespace kiwi::ep
