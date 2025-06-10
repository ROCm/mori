#include "src/pybind/mori.hpp"

PYBIND11_MODULE(libmori_pybinds, m) {
  mori::RegisterMoriOps(m);
  mori::RegisterMoriShmem(m);
}
