#include <torch/library.h>

#include "src/pybind/mori.hpp"

TORCH_LIBRARY(mori_ops, m) { mori::RegisterMoriOps(m); }
TORCH_LIBRARY(mori_shmem, m) { mori::RegisterMoriShmem(m); }
