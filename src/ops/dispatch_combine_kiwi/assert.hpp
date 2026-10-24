#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

namespace mori::kiwi::ep {

[[noreturn]] __host__ __device__ inline void
kiwi_ep_assert_fail(const char* message) {
#if defined(__HIP_DEVICE_COMPILE__)
    printf("kiwi EP device assert failed: %s\n", message);
    __builtin_trap();
#else
    std::fprintf(stderr, "kiwi EP assert failed: %s\n", message);
    std::abort();
#endif
}

} // namespace kiwi::ep

#define KIWI_EP_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            kiwi_ep_assert_fail(message); \
        } \
    } while (0)
