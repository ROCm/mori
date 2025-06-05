#include <torch/library.h>

#include "pybinds/dispatch_combine.hpp"

TORCH_LIBRARY(mori_dispatch_combine, m) { mori::register_dispatch_combine_ops(m); }

// REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
