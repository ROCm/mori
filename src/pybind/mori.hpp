#pragma once

#include <torch/library.h>

namespace mori {
void RegisterMoriOps(torch::Library& m);
void RegisterMoriShmem(torch::Library& m);
}  // namespace mori
