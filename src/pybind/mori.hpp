#pragma once

#include <pybind11/pybind11.h>

namespace mori {
void RegisterMoriOps(pybind11::module_& m);
void RegisterMoriShmem(pybind11::module_& m);
void RegisterMoriIo(pybind11::module_& m);
}  // namespace mori
