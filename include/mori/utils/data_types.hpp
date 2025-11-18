#include <hip/hip_fp8.h>

#if defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1
#define MORI_FP8_TYPE_FNUZ_ENABLED
#endif

#if defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1
#define MORI_FP8_TYPE_OCP_ENABLED
#endif
