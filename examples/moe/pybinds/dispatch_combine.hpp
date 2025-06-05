#pragma once

#include <torch/library.h>

namespace mori {
void register_dispatch_combine_ops(torch::Library& m);
}  // namespace mori
