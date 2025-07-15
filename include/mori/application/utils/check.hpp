#pragma once

#include <execinfo.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

namespace mori {
namespace application {

#define HIP_RUNTIME_CHECK(stmt)                                            \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      exit(-1);                                                            \
    }                                                                      \
    assert(hipSuccess == result);                                          \
  } while (0)

#define HIP_RUNTIME_CHECK_WITH_BACKTRACE(stmt)                             \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      void* array[20];                                                     \
      int size = backtrace(array, 20);                                     \
      backtrace_symbols_fd(array, size, STDERR_FILENO);                    \
      exit(-1);                                                            \
    }                                                                      \
    assert(hipSuccess == result);                                          \
  } while (0)

#define SYSCALL_RETURN_ZERO(stmt)                                                               \
  do {                                                                                          \
    auto _ret = (stmt);                                                                         \
    if (_ret != 0) {                                                                            \
      fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
      exit(-1);                                                                                 \
    }                                                                                           \
  } while (0)

#define SYSCALL_RETURN_ZERO_IGNORE_ERROR(stmt, ignored)                                           \
  do {                                                                                            \
    auto _ret = (stmt);                                                                           \
    if (_ret != 0) {                                                                              \
      int err = errno;                                                                            \
      if (err != ignored) {                                                                       \
        fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
        exit(-1);                                                                                 \
      }                                                                                           \
    }                                                                                             \
  } while (0)

}  // namespace application
}  // namespace mori