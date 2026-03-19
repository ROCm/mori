// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

namespace umbp {

enum class ErrorCode : int {
    OK = 0,
    SPDK_INIT_FAIL = -1,
    SPDK_IO_FAIL = -2,
    DMA_ALLOC_FAIL = -3,
    ALLOC_FAIL = -4,
    INVALID_ARG = -5,
};

}  // namespace umbp
