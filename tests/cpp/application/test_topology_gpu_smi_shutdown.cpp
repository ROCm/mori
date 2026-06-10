#include <cassert>

#include "mori/application/topology/gpu.hpp"

int main() {
  mori::application::TopoSystemGpu first;
  mori::application::TopoSystemGpu second;

  assert(first.NumGpus() == second.NumGpus());

  return 0;
}
