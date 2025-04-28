#pragma once

#include <hip/hip_runtime.h>

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

}  // namespace application
}  // namespace mori