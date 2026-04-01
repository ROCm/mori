// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — V1: simple chunked allreduce using existing RS+AG kernels.
// No single-kernel SDMA pipeline yet. This is the correctness-first baseline.
//
#pragma once

namespace mori {
namespace collective {
// V1 pipeline is implemented entirely in the host-side pipelined() method.
// No device kernel needed — it reuses SdmaReduceScatterKernel + AllGatherSdmaKernel.
}  // namespace collective
}  // namespace mori
