#pragma once

#include <torch/library.h>

namespace mori {
void register_mori_ops(torch::Library& m);
}  // namespace mori
