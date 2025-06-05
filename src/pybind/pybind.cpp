#include <torch/library.h>

#include "src/pybind/ops.hpp"

TORCH_LIBRARY(mori_ops, m) { mori::register_mori_ops(m); }
