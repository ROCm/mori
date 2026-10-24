#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace mori::kiwi::ep {
namespace internal {

inline void hip_check_abort(hipError_t error,
                            const char* file, int line) {
    if (error == hipSuccess) {
        return;
    }
    std::fprintf(stderr, "HIP error at %s:%d: %s\n",
                 file, line, hipGetErrorString(error));
    std::abort();
}

inline void hip_check_throw(hipError_t error,
                            const char* file, int line) {
    if (error == hipSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("HIP error at ") + file + ":"
        + std::to_string(line) + ": " + hipGetErrorString(error));
}

} // namespace internal
} // namespace kiwi::ep

#ifdef HIP_CHECK
#undef HIP_CHECK
#endif
#define HIP_CHECK(call)                                                   \
    ::mori::kiwi::ep::internal::hip_check_abort((call), __FILE__, __LINE__)

#define HIP_CHECK_THROW(call)                                             \
    ::mori::kiwi::ep::internal::hip_check_throw((call), __FILE__, __LINE__)
