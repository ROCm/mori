// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdint>

#include "mori/collective/collective.hpp"

template double mori::collective::Allreduce_sdma<uint32_t>(uint32_t*, uint32_t*, size_t,
                                                            hipStream_t);
